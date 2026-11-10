// HEScan GPU backend on FIDESlib's low-level CKKS layer (FIDESlib::CKKS).
//
// The high-level fideslib:: API has no conjugation and loads the conjugation key only for
// bootstrapping, so this file drives FIDESlib::CKKS directly, the same way FIDESlib's own
// OpenFHE-interface tests do:
//
//   OpenFHE context --GetRawParams--> GenCryptoContextGPU          (device context)
//   OpenFHE keys    --Get*KeySwitchKey--> KeySwitchingKey          (relin, rotations, conj)
//   OpenFHE ct      --GetRawCipherText--> FIDESlib::CKKS::Ciphertext   (upload)
//   device ct       --store + GetOpenFHECipherText--> OpenFHE ct       (download)
//
// The operation sequence matches hescan.cpp one for one (see the comments there).
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include "hescan_gpu.h"
#include "schedule.h"

#include <cuda_runtime.h>

#include <chrono>
#include <complex>
#include <stdexcept>
#include <string>

namespace FC = FIDESlib::CKKS;

namespace hescan {

namespace {

double Seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Sync() {
    // FIDESlib launches asynchronously; phase timings are only meaningful after a sync.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}

}  // namespace

struct HEScanGPU::Impl {
    using G = std::shared_ptr<FC::Ciphertext>;
    struct Elem {
        G A, U;  // affine map h -> A (.) h + U
    };

    CC cc;
    Config cfg;
    uint32_t N;
    // FIDESlib keys and plaintexts keep `Context&` references to this shared_ptr, so it must
    // stay at a stable address for the lifetime of the Impl (it lives inside a unique_ptr).
    FC::Context g;
    std::unique_ptr<FC::Plaintext> mask, maskI;
    Ct shell;  // a fresh top-level OpenFHE ciphertext: the template for downloads

    G New(const FC::Ciphertext& like) const {
        // Results are created as copies so their limbs are allocated at the right level;
        // FIDESlib's own API wrapper does the same before rotate_hoisted.
        auto r = std::make_shared<FC::Ciphertext>(g);
        r->copy(like);
        return r;
    }

    G Upload(const Ct& c) {
        return std::make_shared<FC::Ciphertext>(g, FC::GetRawCipherText(cc, c));
    }

    Ct Download(FC::Ciphertext& c) const {
        FC::RawCipherText raw;
        c.store(raw);
        Ct res = shell->Clone();
        FC::GetOpenFHECipherText(res, raw);
        return res;
    }

    // u = (z + conj z)/2,  v = -i/2 (z - conj z), as HEScan::Unpack.
    std::vector<G> Unpack(const std::vector<G>& boundary) const {
        if (!cfg.complexPack)
            return boundary;
        std::vector<G> chunks;
        for (const auto& z : boundary) {
            G zc = New(*z);
            zc->conjugate(*z);
            G u = New(*z);
            u->add(*zc);
            u->multScalar(0.5);
            G v = New(*z);
            v->sub(*zc);
            v->multScalar(std::complex<double>(0, -0.5));
            chunks.push_back(std::move(u));
            chunks.push_back(std::move(v));
        }
        chunks.resize(cfg.Ks());
        return chunks;
    }

    // br(): one hoisted batch of up to radix-1 rotations per stage, as HEScan::Replicate.
    G Replicate(const FC::Ciphertext& ct, uint32_t stride, uint32_t count) const {
        G cur = New(ct);
        for (uint32_t span = 1; span < count;) {
            const uint32_t k = std::min(cfg.radix, count / span);
            std::vector<int> idx;
            std::vector<G> outs;
            std::vector<FC::Ciphertext*> ptrs;
            for (uint32_t j = 1; j < k; ++j) {
                idx.push_back(-static_cast<int>(j * span * stride));
                outs.push_back(New(*cur));
                ptrs.push_back(outs.back().get());
            }
            cur->rotate_hoisted(idx, ptrs, false);
            for (auto& o : outs)
                cur->add(*o);
            span *= k;
        }
        return cur;
    }

    // Sum_{d_s}: log2(d_s) rotate-and-add steps, as HEScan::ReduceDs.
    G ReduceDs(G acc) const {
        for (uint32_t r = 1; r < cfg.ds; r *= 2) {
            G t = New(*acc);
            t->rotate(static_cast<int>(r));
            acc->add(*t);
        }
        return acc;
    }

    G Mult(const FC::Ciphertext& a, const FC::Ciphertext& b) const {
        G r = New(a);
        r->mult(b);
        return r;
    }

    // Eq. 5 / Eq. 6, as HEScan::Compose.
    Elem Compose(const Elem& early, const Elem& late) const {
        G A = Mult(*late.A, *early.A);
        G U = Mult(*late.A, *early.U);
        U->add(*late.U);
        return Elem{std::move(A), std::move(U)};
    }
};

HEScanGPU::HEScanGPU(CC cc, Config cfg, const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk, int device)
    : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cc    = std::move(cc);
    I.cfg   = cfg;
    I.N     = I.cc->GetRingDimension();
    I.cfg.Validate(I.N / 2);

    // Fail with a readable message here: without a usable device FIDESlib aborts deep inside
    // context creation instead of throwing.
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("no usable CUDA device: ") + cudaGetErrorString(err) +
                                 " (check nvidia-smi, and that the driver supports this CUDA toolkit)");
    if (device < 0 || device >= count)
        throw std::runtime_error("CUDA device " + std::to_string(device) + " requested, but only " +
                                 std::to_string(count) + " visible (CUDA_VISIBLE_DEVICES?)");

    // Device context from the OpenFHE parameters (moduli, scaling factors, dnum, technique).
    FC::RawParams raw = FC::GetRawParams(I.cc);
    FC::Parameters base{};
    I.g = FC::GenCryptoContextGPU(base.adaptTo(raw), {device});

    // Relinearization key.
    {
        FC::KeySwitchingKey ksk(I.g);
        FC::RawKeySwitchKey rk = FC::GetEvalKeySwitchKey(pk);
        ksk.Initialize(rk);
        I.g->AddEvalKey(std::move(ksk));
    }
    // Rotation keys. FIDESlib looks a key up by modpow(5, index, 2N) with an unsigned exponent,
    // so HEScan's negative (right-shift) indices are passed as index mod N/2. 5 has order N/2
    // modulo 2N, so this names exactly the automorphism OpenFHE generated for the negative index.
    {
        std::vector<int> idx;
        for (int32_t r : I.cfg.RotationIndices()) {
            const int64_t half = I.N / 2;
            idx.push_back(static_cast<int>(((r % half) + half) % half));
        }
        FC::AddRotationKeys(pk, I.g, idx);
    }
    // Conjugation key (automorphism 2N-1), stored by FIDESlib as rotation "index" 2N-1.
    if (I.cfg.complexPack) {
        FC::KeySwitchingKey ksk(I.g);
        FC::RawKeySwitchKey rk = FC::GetConjugateKeySwitchKey(pk);
        ksk.Initialize(rk);
        I.g->AddRotationKey(2 * static_cast<int>(I.N) - 1, std::move(ksk));
    }

    // Output masks, encoded exactly as in HEScan's constructor.
    std::vector<double> mask(I.cfg.sState, 0.0);
    for (uint32_t c = 0; c < I.cfg.cState(); ++c)
        mask[c * I.cfg.ds] = 1.0;
    std::vector<cplx> maskI(I.cfg.sState);
    for (uint32_t i = 0; i < I.cfg.sState; ++i)
        maskI[i] = cplx(0, mask[i]);
    auto pm  = I.cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, I.cfg.sState);
    auto pmI = I.cc->MakeCKKSPackedPlaintext(maskI, 1, 0, nullptr, I.cfg.sState);
    I.mask   = std::make_unique<FC::Plaintext>(I.g, FC::GetRawPlainText(I.cc, pm));
    I.maskI  = std::make_unique<FC::Plaintext>(I.g, FC::GetRawPlainText(I.cc, pmI));

    // Download template: a top-level ciphertext under the same key, so the level arithmetic in
    // GetOpenFHECipherText starts from level 0.
    I.shell = I.cc->Encrypt(pk, pm);
    Sync();
}

HEScanGPU::~HEScanGPU() {
    if (impl_) {
        impl_->mask.reset();
        impl_->maskI.reset();
        if (impl_->g)
            FC::DeregisterCryptoContextGPU(impl_->g);
    }
}

size_t HEScanGPU::DeviceMemoryUsed(int device) {
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess || cudaSetDevice(device) != cudaSuccess)
        return 0;
    size_t freeB = 0, totalB = 0;
    cudaError_t err = cudaMemGetInfo(&freeB, &totalB);
    cudaSetDevice(prev);
    return err == cudaSuccess ? totalB - freeB : 0;
}

std::vector<std::vector<Ct>> HEScanGPU::Run(const EncPacket& in, Timings* t) const {
    using G    = Impl::G;
    using Elem = Impl::Elem;
    Impl& I    = *impl_;
    const Config& cfg = I.cfg;
    const uint32_t L = cfg.L, Ks = cfg.Ks();
    Timings local;
    Timings& tm = t ? *t : local;
    tm.upload = tm.download = 0;

    // 1-2. per token: upload, split, broadcast. Uploads are timed separately; split chunks are
    // dropped as soon as they are broadcast, so one token's worth of them is live at a time.
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<Elem>> el(Ks, std::vector<Elem>(L));
    std::vector<std::vector<G>> Cb(Ks, std::vector<G>(L));
    auto upload = [&](const std::vector<Ct>& v) {
        auto tu = std::chrono::steady_clock::now();
        std::vector<G> r;
        for (const auto& c : v)
            r.push_back(I.Upload(c));
        Sync();
        tm.upload += Seconds(tu);
        return r;
    };
    for (uint32_t k = 0; k < L; ++k) {
        auto x = I.Unpack(upload(in.x[k]));
        auto a = I.Unpack(upload(in.a[k]));
        auto B = I.Unpack(upload(in.B[k]));
        auto C = I.Unpack(upload(in.C[k]));
        for (uint32_t j = 0; j < Ks; ++j) {
            G Bb       = I.Replicate(*B[j], cfg.ds, cfg.runG());
            G Xb       = I.Replicate(*x[j], 1, cfg.ds);
            el[j][k].A = I.Replicate(*a[j], 1, cfg.runA() * cfg.ds);
            el[j][k].U = I.Mult(*Xb, *Bb);
            Cb[j][k]   = I.Replicate(*C[j], cfg.ds, cfg.runG());
        }
    }
    std::vector<G> h0 = cfg.withH0 ? I.Unpack(upload(in.h0)) : std::vector<G>{};
    Sync();
    tm.broadcast = Seconds(t0) - tm.upload;

    // 3. prefix scan over tokens, per state chunk; afterwards el[j][k].U == h_k
    t0 = std::chrono::steady_clock::now();
    for (uint32_t j = 0; j < Ks; ++j) {
        detail::RunSchedule(
            el[j], cfg.algo, [&I](const Elem& e, const Elem& l) { return I.Compose(e, l); },
            /*parallel=*/false);
        if (cfg.withH0) {
            for (uint32_t k = 0; k < L; ++k) {
                G h = I.Mult(*el[j][k].A, *h0[j]);
                h->add(*el[j][k].U);
                el[j][k].U = std::move(h);
            }
        }
        for (auto& e : el[j])
            e.A.reset();  // decay prefixes are dead once h_k is formed
    }
    h0.clear();
    Sync();
    tm.scan = Seconds(t0);

    // 4. contraction m_k = Sum_{d_s}(h_k (.) br(C_k)), then mask / re-pair, then download
    t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<Ct>> out(L);
    for (uint32_t k = 0; k < L; ++k) {
        std::vector<G> y(Ks);
        for (uint32_t j = 0; j < Ks; ++j) {
            y[j] = I.ReduceDs(I.Mult(*el[j][k].U, *Cb[j][k]));
            el[j][k].U.reset();
            Cb[j][k].reset();
        }
        std::vector<G> o;
        if (cfg.complexPack) {
            for (uint32_t j = 0; j < Ks; j += 2) {
                G r = I.New(*y[j]);
                r->multPt(*I.mask);
                if (j + 1 < Ks) {
                    G s = I.New(*y[j + 1]);
                    s->multPt(*I.maskI);
                    r->add(*s);
                }
                o.push_back(std::move(r));
            }
        }
        else if (cfg.maskOutput) {
            for (uint32_t j = 0; j < Ks; ++j) {
                G r = I.New(*y[j]);
                r->multPt(*I.mask);
                o.push_back(std::move(r));
            }
        }
        else {
            o = std::move(y);
        }
        Sync();
        auto td = std::chrono::steady_clock::now();
        for (auto& c : o)
            out[k].push_back(I.Download(*c));
        tm.download += Seconds(td);
    }
    tm.contract = Seconds(t0) - tm.download;
    return out;
}

}  // namespace hescan
