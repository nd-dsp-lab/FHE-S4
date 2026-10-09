#include "hescan.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

using namespace lbcrypto;

namespace hescan {

namespace {

bool IsPow2(uint32_t v) {
    return v && !(v & (v - 1));
}

double Seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Prefix-scan schedules over affine elements; `comb(early, late)` returns late o early.
// Shared by the encrypted kernel and the depth simulation so the two cannot drift apart.
template <class T, class F>
void RunSchedule(std::vector<T>& xs, ScanAlgo algo, F comb) {
    const int64_t L = static_cast<int64_t>(xs.size());
    switch (algo) {
        case ScanAlgo::Sequential:
            for (int64_t i = 1; i < L; ++i)
                xs[i] = comb(xs[i - 1], xs[i]);
            break;
        case ScanAlgo::HillisSteele:
            for (int64_t d = 1; d < L; d *= 2) {
                std::vector<T> next(xs);
#pragma omp parallel for schedule(dynamic)
                for (int64_t i = d; i < L; ++i)
                    next[i] = comb(xs[i - d], xs[i]);
                xs.swap(next);
            }
            break;
        case ScanAlgo::BrentKung: {
            int64_t dmax = 1;
            // up-sweep: xs[i] becomes the summary of a 2d-aligned block
            for (int64_t d = 1; 2 * d <= L; d *= 2) {
                dmax = d;
#pragma omp parallel for schedule(dynamic)
                for (int64_t i = 2 * d - 1; i < L; i += 2 * d)
                    xs[i] = comb(xs[i - d], xs[i]);
            }
            // down-sweep: fill in the remaining prefixes
            for (int64_t d = dmax; d >= 1; d /= 2) {
#pragma omp parallel for schedule(dynamic)
                for (int64_t i = 3 * d - 1; i < L; i += 2 * d)
                    xs[i] = comb(xs[i - d], xs[i]);
            }
            break;
        }
    }
}

// Seeds for chunk j of token k, in the layout documented in hescan.h.
struct Seeds {
    std::vector<std::vector<double>> x, a, B, C;  // [chunk][slot]
};

Seeds MakeSeeds(const Config& cfg, const PlainPacket& pkt, uint32_t k) {
    const uint32_t E = cfg.E(), Ks = cfg.Ks(), cs = cfg.cState(), ds = cfg.ds;
    const uint32_t hpg = cfg.H / cfg.G;
    Seeds s;
    for (auto* f : {&s.x, &s.a, &s.B, &s.C})
        f->assign(Ks, std::vector<double>(cfg.sState, 0.0));
    for (uint32_t j = 0; j < Ks; ++j) {
        for (uint32_t c = 0; c < cs; ++c) {
            const uint32_t e = j * cs + c;
            if (e >= E)
                break;
            const uint32_t h = e / cfg.P, g = h / hpg;
            s.x[j][c * ds] = pkt.x[k * E + e];
            if (c % cfg.runA() == 0)
                s.a[j][c * ds] = pkt.a[k * cfg.H + h];
            if (c % cfg.runG() == 0) {
                for (uint32_t n = 0; n < ds; ++n) {
                    s.B[j][c * ds + n] = pkt.B[(k * cfg.G + g) * ds + n];
                    s.C[j][c * ds + n] = pkt.C[(k * cfg.G + g) * ds + n];
                }
            }
        }
    }
    return s;
}

std::vector<Ct> EncryptChunks(const CC& cc, const PublicKey<DCRTPoly>& pk, const Config& cfg,
                              const std::vector<std::vector<double>>& chunks) {
    std::vector<Ct> out;
    if (cfg.complexPack) {
        for (uint32_t q = 0; 2 * q < chunks.size(); ++q) {
            std::vector<cplx> z(cfg.sState);
            for (uint32_t i = 0; i < cfg.sState; ++i)
                z[i] = cplx(chunks[2 * q][i], 2 * q + 1 < chunks.size() ? chunks[2 * q + 1][i] : 0.0);
            out.push_back(cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(z, 1, 0, nullptr, cfg.sState)));
        }
    }
    else {
        for (const auto& v : chunks)
            out.push_back(cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, cfg.sState)));
    }
    return out;
}

}  // namespace

ScanAlgo ParseScanAlgo(const std::string& s) {
    if (s == "seq" || s == "sequential")
        return ScanAlgo::Sequential;
    if (s == "hs" || s == "hillis-steele")
        return ScanAlgo::HillisSteele;
    if (s == "bk" || s == "brent-kung")
        return ScanAlgo::BrentKung;
    throw std::invalid_argument("unknown scan algorithm: " + s + " (use seq|hs|bk)");
}

const char* ScanAlgoName(ScanAlgo a) {
    switch (a) {
        case ScanAlgo::Sequential:
            return "sequential";
        case ScanAlgo::HillisSteele:
            return "hillis-steele";
        case ScanAlgo::BrentKung:
            return "brent-kung";
    }
    return "?";
}

bool CheckSchedule(ScanAlgo algo, uint32_t L) {
    struct Af {
        double A, U;
    };
    std::vector<Af> xs(L);
    for (uint32_t i = 0; i < L; ++i)
        xs[i] = Af{0.5 + 0.4 * ((i * 7919) % 13) / 13.0, 1.0 + ((i * 104729) % 17)};
    std::vector<Af> ref(xs);
    for (uint32_t i = 1; i < L; ++i)
        ref[i] = Af{ref[i].A * ref[i - 1].A, ref[i].A * ref[i - 1].U + ref[i].U};
    RunSchedule(xs, algo, [](const Af& e, const Af& l) { return Af{l.A * e.A, l.A * e.U + l.U}; });
    for (uint32_t i = 0; i < L; ++i)
        if (std::abs(xs[i].A - ref[i].A) > 1e-9 * std::abs(ref[i].A) + 1e-12 ||
            std::abs(xs[i].U - ref[i].U) > 1e-9 * std::abs(ref[i].U) + 1e-12)
            return false;
    return true;
}

// ---------------------------------------------------------------- Config

void Config::Validate(uint32_t maxSlots) const {
    auto req = [](bool ok, const char* msg) {
        if (!ok)
            throw std::invalid_argument(msg);
    };
    req(H && P && G && ds && L, "all dimensions must be positive");
    req(H % G == 0, "G must divide H");
    req(IsPow2(ds) && IsPow2(P) && IsPow2(sState) && IsPow2(E() / G),
        "d_s, P, s_state and E/G must be powers of two");
    req(IsPow2(radix) && radix >= 2, "radix must be a power of two >= 2");
    req(sState >= ds, "s_state must hold at least one channel (s_state >= d_s)");
    req(sState <= maxSlots, "s_state exceeds the CKKS slot count (N/2)");
}

uint32_t Config::Depth() const {
    struct Lv {
        uint32_t a, u;
    };
    const uint32_t s0 = complexPack ? 1 : 0;  // the split multiplies by 1/2 and -i/2
    std::vector<Lv> xs(L, Lv{s0, s0 + 1});    // U = br(x) (.) br(B)
    RunSchedule(xs, algo, [](const Lv& e, const Lv& l) {
        return Lv{std::max(e.a, l.a) + 1, std::max(std::max(l.a, e.u) + 1, l.u)};
    });
    uint32_t depth = 0;
    for (const auto& v : xs) {
        uint32_t h = withH0 ? std::max(std::max(v.a, s0) + 1, v.u) : v.u;
        uint32_t m = std::max(h, s0) + 1;  // (.) br(C_k)
        if (maskOutput || complexPack)
            m += 1;
        depth = std::max(depth, m);
    }
    return depth;
}

std::vector<int32_t> Config::RotationIndices() const {
    std::vector<int32_t> idx;
    auto rep = [&](uint32_t stride, uint32_t count) {
        for (uint32_t span = 1; span < count;) {
            uint32_t k = std::min(radix, count / span);
            for (uint32_t j = 1; j < k; ++j)
                idx.push_back(-static_cast<int32_t>(j * span * stride));
            span *= k;
        }
    };
    rep(1, ds);           // br(x)
    rep(1, runA() * ds);  // br(a)
    rep(ds, runG());      // br(B), br(C)
    for (uint32_t r = 1; r < ds; r *= 2)
        idx.push_back(static_cast<int32_t>(r));  // Sum_{d_s}
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    return idx;
}

// ---------------------------------------------------------------- client side

EncPacket EncryptPacket(const CC& cc, const PublicKey<DCRTPoly>& pk, const Config& cfg, const PlainPacket& pkt) {
    EncPacket enc;
    for (auto* f : {&enc.x, &enc.a, &enc.B, &enc.C})
        f->resize(cfg.L);
    for (uint32_t k = 0; k < cfg.L; ++k) {
        Seeds s    = MakeSeeds(cfg, pkt, k);
        enc.x[k]   = EncryptChunks(cc, pk, cfg, s.x);
        enc.a[k]   = EncryptChunks(cc, pk, cfg, s.a);
        enc.B[k]   = EncryptChunks(cc, pk, cfg, s.B);
        enc.C[k]   = EncryptChunks(cc, pk, cfg, s.C);
    }
    if (cfg.withH0) {
        const uint32_t E = cfg.E(), cs = cfg.cState(), ds = cfg.ds;
        std::vector<std::vector<double>> h(cfg.Ks(), std::vector<double>(cfg.sState, 0.0));
        for (uint32_t e = 0; e < E; ++e)
            for (uint32_t n = 0; n < ds; ++n)
                h[e / cs][(e % cs) * ds + n] = pkt.h0[e * ds + n];
        enc.h0 = EncryptChunks(cc, pk, cfg, h);
    }
    return enc;
}

std::vector<double> DecryptOutput(const CC& cc, const PrivateKey<DCRTPoly>& sk, const Config& cfg,
                                  const std::vector<std::vector<Ct>>& out) {
    const uint32_t E = cfg.E(), cs = cfg.cState(), ds = cfg.ds;
    std::vector<double> m(static_cast<size_t>(cfg.L) * E, 0.0);
    for (uint32_t k = 0; k < cfg.L; ++k) {
        for (uint32_t q = 0; q < out[k].size(); ++q) {
            Plaintext pt;
            cc->Decrypt(sk, out[k][q], &pt);
            pt->SetLength(cfg.sState);
            const auto& v = pt->GetCKKSPackedValue();
            const uint32_t parts = cfg.complexPack ? 2 : 1;
            for (uint32_t part = 0; part < parts; ++part) {
                const uint32_t j = q * parts + part;
                for (uint32_t c = 0; c < cs; ++c) {
                    const uint32_t e = j * cs + c;
                    if (e < E)
                        m[k * E + e] = part ? v[c * ds].imag() : v[c * ds].real();
                }
            }
        }
    }
    return m;
}

std::vector<double> PlainReference(const Config& cfg, const PlainPacket& pkt) {
    const uint32_t E = cfg.E(), ds = cfg.ds, hpg = cfg.H / cfg.G;
    std::vector<double> h(static_cast<size_t>(E) * ds, 0.0);
    if (cfg.withH0)
        h = pkt.h0;
    std::vector<double> m(static_cast<size_t>(cfg.L) * E, 0.0);
    for (uint32_t k = 0; k < cfg.L; ++k) {
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t hd = e / cfg.P, g = hd / hpg;
            const double a    = pkt.a[k * cfg.H + hd];
            double acc        = 0;
            for (uint32_t n = 0; n < ds; ++n) {
                double& s = h[e * ds + n];
                s         = a * s + pkt.x[k * E + e] * pkt.B[(k * cfg.G + g) * ds + n];  // Eq. 2
                acc += s * pkt.C[(k * cfg.G + g) * ds + n];                             // Eq. 3
            }
            m[k * E + e] = acc;
        }
    }
    return m;
}

// ---------------------------------------------------------------- HE side

HEScan::HEScan(CC cc, Config cfg) : cc_(std::move(cc)), cfg_(cfg), m_(2 * cc_->GetRingDimension()) {
    cfg_.Validate(cc_->GetRingDimension() / 2);
    std::vector<double> mask(cfg_.sState, 0.0);
    for (uint32_t c = 0; c < cfg_.cState(); ++c)
        mask[c * cfg_.ds] = 1.0;
    std::vector<cplx> maskI(cfg_.sState);
    for (uint32_t i = 0; i < cfg_.sState; ++i)
        maskI[i] = cplx(0, mask[i]);
    mask_  = cc_->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, cfg_.sState);
    maskI_ = cc_->MakeCKKSPackedPlaintext(maskI, 1, 0, nullptr, cfg_.sState);
}

Ct HEScan::Replicate(const Ct& ct, uint32_t stride, uint32_t count) const {
    Ct cur = ct;
    for (uint32_t span = 1; span < count;) {
        const uint32_t k = std::min(cfg_.radix, count / span);
        // one digit decomposition, k-1 rotations of the same ciphertext (hoisting)
        auto digits = cc_->EvalFastRotationPrecompute(cur);
        Ct acc      = cur->Clone();
        for (uint32_t j = 1; j < k; ++j) {
            const int32_t r = -static_cast<int32_t>(j * span * stride);
            cc_->EvalAddInPlace(acc, cc_->EvalFastRotation(cur, static_cast<uint32_t>(r), m_, digits));
        }
        cur = acc;
        span *= k;
    }
    return cur;
}

Ct HEScan::ReduceDs(const Ct& ct) const {
    Ct acc = ct;
    for (uint32_t r = 1; r < cfg_.ds; r *= 2)
        acc = cc_->EvalAdd(acc, cc_->EvalRotate(acc, static_cast<int32_t>(r)));
    return acc;
}

std::vector<Ct> HEScan::Unpack(const std::vector<Ct>& boundary) const {
    if (!cfg_.complexPack)
        return boundary;
    const uint32_t conjIdx = m_ - 1;  // automorphism X -> X^{2N-1} == slot-wise conjugation
    std::vector<Ct> chunks;
    for (const auto& z : boundary) {
        auto keys = cc_->GetEvalAutomorphismKeyMap(z->GetKeyTag());
        Ct zc     = cc_->EvalAutomorphism(z, conjIdx, keys);
        chunks.push_back(cc_->EvalMult(cc_->EvalAdd(z, zc), 0.5));           // real part
        chunks.push_back(cc_->EvalMult(cc_->EvalSub(z, zc), cplx(0, -0.5)));  // imag part
    }
    chunks.resize(cfg_.Ks());
    return chunks;
}

HEScan::Elem HEScan::Compose(const Elem& early, const Elem& late) const {
    // Eq. 5 / Eq. 6. EvalMult relinearizes; FLEXIBLEAUTO/FIXEDAUTO rescale and align levels.
    return Elem{cc_->EvalMult(late.A, early.A), cc_->EvalAdd(cc_->EvalMult(late.A, early.U), late.U)};
}

void HEScan::Scan(std::vector<Elem>& xs) const {
    RunSchedule(xs, cfg_.algo, [this](const Elem& e, const Elem& l) { return Compose(e, l); });
}

std::vector<std::vector<Ct>> HEScan::Run(const EncPacket& in, Timings* t) const {
    const uint32_t L = cfg_.L, Ks = cfg_.Ks();
    Timings local;
    Timings& tm = t ? *t : local;
    auto t0     = std::chrono::steady_clock::now();

    // 1-2. per token: split complex-paired boundary ciphertexts (one conjugation each), then
    // broadcast A_k = br(a_k), U_k = br(x_k) (.) br(B_k) and br(C_k). Split chunks are
    // dropped as soon as they are broadcast so only one token's worth is live per thread.
    std::vector<std::vector<Elem>> el(Ks, std::vector<Elem>(L));
    std::vector<std::vector<Ct>> Cb(Ks, std::vector<Ct>(L));
#pragma omp parallel for schedule(dynamic)
    for (int64_t k = 0; k < L; ++k) {
        auto x = Unpack(in.x[k]);
        auto a = Unpack(in.a[k]);
        auto B = Unpack(in.B[k]);
        auto C = Unpack(in.C[k]);
        for (uint32_t j = 0; j < Ks; ++j) {
            Ct Bb      = Replicate(B[j], cfg_.ds, cfg_.runG());
            Ct Xb      = Replicate(x[j], 1, cfg_.ds);
            el[j][k].A = Replicate(a[j], 1, cfg_.runA() * cfg_.ds);
            el[j][k].U = cc_->EvalMult(Xb, Bb);
            Cb[j][k]   = Replicate(C[j], cfg_.ds, cfg_.runG());
        }
    }
    std::vector<Ct> h0 = cfg_.withH0 ? Unpack(in.h0) : std::vector<Ct>{};
    tm.broadcast = Seconds(t0);

    // 3. prefix scan over tokens, per state chunk; afterwards el[j][k].U == h_k
    t0 = std::chrono::steady_clock::now();
    for (uint32_t j = 0; j < Ks; ++j) {
        Scan(el[j]);
        if (cfg_.withH0) {
#pragma omp parallel for schedule(dynamic)
            for (int64_t k = 0; k < L; ++k)
                el[j][k].U = cc_->EvalAdd(cc_->EvalMult(el[j][k].A, h0[j]), el[j][k].U);
        }
        for (auto& e : el[j])
            e.A = nullptr;  // decay prefixes are dead once h_k is formed
    }
    tm.scan = Seconds(t0);

    // 4. contraction m_k = Sum_{d_s}(h_k (.) br(C_k)), then mask / re-pair for the boundary
    t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<Ct>> y(L, std::vector<Ct>(Ks));
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t k = 0; k < L; ++k) {
        for (int64_t j = 0; j < Ks; ++j) {
            y[k][j]    = ReduceDs(cc_->EvalMult(el[j][k].U, Cb[j][k]));
            el[j][k].U = nullptr;
            Cb[j][k]   = nullptr;
        }
    }

    std::vector<std::vector<Ct>> out(L);
    for (uint32_t k = 0; k < L; ++k) {
        if (cfg_.complexPack) {
            for (uint32_t j = 0; j < Ks; j += 2) {
                Ct o = cc_->EvalMult(y[k][j], mask_);
                if (j + 1 < Ks)
                    o = cc_->EvalAdd(o, cc_->EvalMult(y[k][j + 1], maskI_));
                out[k].push_back(o);
            }
        }
        else if (cfg_.maskOutput) {
            for (uint32_t j = 0; j < Ks; ++j)
                out[k].push_back(cc_->EvalMult(y[k][j], mask_));
        }
        else {
            out[k] = y[k];
        }
    }
    tm.contract = Seconds(t0);
    return out;
}

}  // namespace hescan
