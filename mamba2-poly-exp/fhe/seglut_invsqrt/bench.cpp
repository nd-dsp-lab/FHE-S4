// One encrypted 4-segment 1/sqrt(v) lookup in stock OpenFHE (CKKS -> FHEW -> CKKS).
//
// This is the encrypted counterpart of real_mamba/seglut.py's SegLUT for the RMSNorm
// 1/sqrt(v) gate, built only from public OpenFHE 1.2.1 calls, to measure what precision
// (in ROSETTA's bits, -log2 RMSE) and latency the scheme-switching path actually gives.
//
// Pipeline, per CKKS ciphertext of v (one value per slot):
//   1. segment index: c_k = [v >= tau_k] for the 3 inner boundaries. Each is CKKS->FHEW
//      of (v - tau_k), a large-precision FHEW EvalSign, then FHEW->CKKS. The boundaries
//      span 6e6, more than one large-precision sign resolves, so each comparison gets its
//      own scale and is only trusted up to 1.2*tau_{k+1}; above that the next comparison
//      is 1 and the masks below ignore it.
//   2. table, per segment s (construct-then-select, like ROSETTA): an affine map puts the
//      segment on table positions [G, q/2 - G), CKKS->FHEW at the large modulus, a manual
//      modulus switch down to q = 2N, then EvalFunc with a hand-built NEGACYCLIC table of
//      q/2 entries (one bootstrap; OpenFHE's GenerateLUTviaFunction would allow only
//      q/256 entries). Two tables per segment: f(knot) and f'(knot)*w.
//   3. interpolation: the table index the blind rotation actually used is
//      phase(ct mod q) + beta. Its offset from the true position is a public-coefficient
//      linear function of the key (the modulus-switch rounding), so it is brought back to
//      CKKS exactly by FHEW->CKKS of a "rounding-error" LWE ciphertext. Then
//      out = f(knot) + f'(knot)*w * (x_true - x_used), and 'nearest' = f(knot) alone.
//   4. select: out = sum_s M_s * out_s with M_s products of the c_k.
//
// FHEW table outputs live mod q and are decoded by FHEW->CKKS as q*sin(2*pi*phase/q)/(2*pi),
// so each table's values are pre-distorted with asin and kept within +-0.2 q.
//
// Diagnostics use the secret keys (benchmark only): table-index noise, table-output noise,
// comparison errors, and the decoded values against the phases they came from.

#include "openfhe.h"
#include "binfhecontext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

using namespace lbcrypto;

namespace {

using Clock = std::chrono::steady_clock;
double Since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Args {
    bool toy = false;
    uint32_t slots = 128;
    uint32_t logQ = 25;          // FHEW large-precision modulus (comparisons)
    uint32_t multDepth = 19;
    double guardFrac = 0.05;     // table positions kept free at each end of a segment
    double phaseFrac = 0.2;      // table outputs use phases within +-phaseFrac*q
    uint32_t seed = 0;
    uint32_t newton = 2;         // Newton steps after the table, reported cumulatively
    uint32_t clean = 1;          // sharpening passes on each comparison bit
    int threads = 0;
    std::vector<double> tau{0.0012574287695376616, 0.07583814844682049, 4.573956711644746,
                            275.86485731083445, 16637.984199846484};
};

Args Parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (k == "--toy") a.toy = true;
        else if (k == "--slots") a.slots = std::stoul(next());
        else if (k == "--logq") a.logQ = std::stoul(next());
        else if (k == "--depth") a.multDepth = std::stoul(next());
        else if (k == "--guard") a.guardFrac = std::stod(next());
        else if (k == "--phase-frac") a.phaseFrac = std::stod(next());
        else if (k == "--seed") a.seed = std::stoul(next());
        else if (k == "--newton") a.newton = std::stoul(next());
        else if (k == "--clean") a.clean = std::stoul(next());
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--tau") { a.tau.clear(); for (int j = 0; j < 5; ++j) a.tau.push_back(std::stod(next())); }
        else { std::cerr << "unknown argument " << k << "\n"; std::exit(2); }
    }
    return a;
}

int64_t Centered(uint64_t x, uint64_t m) { x %= m; return x >= m / 2 ? int64_t(x) - int64_t(m) : int64_t(x); }

// b - <a, s> mod (ciphertext modulus), s ternary
uint64_t Phase(const LWECiphertext& ct, const LWEPrivateKey& sk) {
    const uint64_t m = ct->GetModulus().ConvertToInt();
    const auto& a = ct->GetA();
    const auto& s = sk->GetElement();
    const uint64_t sm = s.GetModulus().ConvertToInt();
    __int128 acc = static_cast<__int128>(ct->GetB().ConvertToInt());
    for (size_t j = 0; j < a.GetLength(); ++j)
        acc -= static_cast<__int128>(a[j].ConvertToInt()) * Centered(s[j].ConvertToInt(), sm);
    __int128 r = acc % static_cast<__int128>(m);
    if (r < 0) r += m;
    return static_cast<uint64_t>(r);
}

// round (a, b) from Q down to q; Q/q is a power of two
LWECiphertext ModSwitchDown(const LWECiphertext& ct, uint64_t q) {
    const uint64_t Q = ct->GetModulus().ConvertToInt(), f = Q / q;
    const auto& a = ct->GetA();
    NativeVector a2(a.GetLength(), NativeInteger(q));
    for (size_t j = 0; j < a.GetLength(); ++j)
        a2[j] = NativeInteger(((a[j].ConvertToInt() + f / 2) / f) % q);
    NativeInteger b2(((ct->GetB().ConvertToInt() + f / 2) / f) % q);
    return std::make_shared<LWECiphertextImpl>(std::move(a2), b2);
}

// (f*a' - a, f*b' - b) mod Q: its phase is f*phase(ct mod q) - phase(ct mod Q), i.e. the
// modulus-switch rounding scaled by f. Small, so it decodes linearly in FHEW->CKKS.
LWECiphertext ModSwitchError(const LWECiphertext& ct, const LWECiphertext& ctq) {
    const uint64_t Q = ct->GetModulus().ConvertToInt(), q = ctq->GetModulus().ConvertToInt(), f = Q / q;
    const auto& a = ct->GetA();
    const auto& aq = ctq->GetA();
    NativeVector e(a.GetLength(), NativeInteger(Q));
    for (size_t j = 0; j < a.GetLength(); ++j)
        e[j] = NativeInteger((f * aq[j].ConvertToInt() + Q - a[j].ConvertToInt()) % Q);
    NativeInteger eb((f * ctq->GetB().ConvertToInt() + Q - ct->GetB().ConvertToInt()) % Q);
    return std::make_shared<LWECiphertextImpl>(std::move(e), eb);
}

// A table of q/2 values placed on the negacyclic test vector, outputs pre-distorted for
// the sine decode of FHEW->CKKS. decode(D) = mid + halfw * D / Dmax.
struct Table {
    std::vector<NativeInteger> lut;  // length q
    std::vector<double> vals;        // length q/2, the exact values per position
    double mid = 0, halfw = 1, Dmax = 1;
    double Decode(double D) const { return mid + halfw * D / Dmax; }
    // what FHEW->CKKS returns for the phase stored at a position
    double DecodedAt(uint64_t q, uint64_t pos) const {
        double ph = static_cast<double>(Centered(lut[pos].ConvertToInt(), q));
        return q * std::sin(2 * M_PI * ph / q) / (2 * M_PI);
    }
};

Table MakeTable(const std::vector<double>& vals, uint64_t q, double phaseFrac) {
    Table t;
    t.vals = vals;
    const auto [lo, hi] = std::minmax_element(vals.begin(), vals.end());
    t.mid = 0.5 * (*hi + *lo);
    t.halfw = std::max(0.5 * (*hi - *lo), 1e-300);
    t.Dmax = q * std::sin(2 * M_PI * phaseFrac) / (2 * M_PI);
    const uint64_t half = q / 2;
    t.lut.assign(q, NativeInteger(0));
    for (uint64_t x = 0; x < half; ++x) {
        double D = (vals[x] - t.mid) / t.halfw * t.Dmax;
        int64_t ph = std::llround(q * std::asin(2 * M_PI * D / q) / (2 * M_PI));
        if (ph == 0) ph = 1;  // OpenFHE recognises a negacyclic table only if no entry is 0
        uint64_t u = static_cast<uint64_t>((ph % int64_t(q) + int64_t(q)) % int64_t(q));
        t.lut[x] = NativeInteger(u);
        t.lut[x + half] = NativeInteger(q - u);
    }
    return t;
}

std::vector<double> Dec(CryptoContext<DCRTPoly>& cc, const PrivateKey<DCRTPoly>& sk,
                        const Ciphertext<DCRTPoly>& ct, uint32_t n) {
    Plaintext pt;
    cc->Decrypt(sk, ct, &pt);
    pt->SetLength(n);
    return pt->GetRealPackedValue();
}

struct Stats {
    std::vector<double> x;
    void Add(double v) { x.push_back(v); }
    double Rms() const { double s = 0; for (double v : x) s += v * v; return x.empty() ? 0 : std::sqrt(s / x.size()); }
    double MaxAbs() const { double m = 0; for (double v : x) m = std::max(m, std::fabs(v)); return m; }
    double Mean() const { return x.empty() ? 0 : std::accumulate(x.begin(), x.end(), 0.0) / x.size(); }
};

double Bits(double rms) { return rms > 0 ? -std::log2(rms) : 99.0; }

}  // namespace

int main(int argc, char** argv) {
    const Args A = Parse(argc, argv);
    if (A.threads > 0) omp_set_num_threads(A.threads);
    const auto& tau = A.tau;
    const uint32_t S = A.slots, nseg = 4;
    std::map<std::string, double> T;  // wall-clock seconds per stage

    std::cout << std::setprecision(6);
    std::cout << "== seglut 1/sqrt(v), OpenFHE scheme switching, " << (A.toy ? "TOY (insecure)" : "128-bit")
              << ", slots " << S << ", threads " << omp_get_max_threads() << "\n";

    // ---- CKKS context
    auto t0 = Clock::now();
    CCParams<CryptoContextCKKSRNS> P;
    P.SetMultiplicativeDepth(A.multDepth);
    P.SetScalingModSize(50);
    P.SetFirstModSize(60);
    P.SetScalingTechnique(FLEXIBLEAUTO);
    P.SetSecurityLevel(A.toy ? HEStd_NotSet : HEStd_128_classic);
    if (A.toy) P.SetRingDim(8192);
    P.SetBatchSize(S);
    P.SetSecretKeyDist(UNIFORM_TERNARY);
    P.SetKeySwitchTechnique(HYBRID);
    P.SetNumLargeDigits(3);
    auto cc = GenCryptoContext(P);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);
    auto keys = cc->KeyGen();

    SchSwchParams sp;
    sp.SetSecurityLevelCKKS(A.toy ? HEStd_NotSet : HEStd_128_classic);
    sp.SetSecurityLevelFHEW(A.toy ? TOY : STD128);
    sp.SetCtxtModSizeFHEWLargePrec(A.logQ);
    sp.SetNumSlotsCKKS(S);
    sp.SetNumValues(S);
    auto skF = cc->EvalSchemeSwitchingSetup(sp);
    auto ccLWE = cc->GetBinCCForSchemeSwitch();
    ccLWE->BTKeyGen(skF);
    cc->EvalSchemeSwitchingKeyGen(keys, skF);
    T["keygen (offline)"] = Since(t0);

    const auto lp = ccLWE->GetParams()->GetLWEParams();
    const uint64_t q = lp->Getq().ConvertToInt(), Qlwe = uint64_t{1} << A.logQ;
    const uint64_t beta = ccLWE->GetBeta().ConvertToInt(), half = q / 2;
    const uint64_t pLarge = Qlwe / (2 * beta);
    const double G = std::floor(A.guardFrac * half);
    std::cout << "CKKS ring " << cc->GetRingDimension() << ", depth " << A.multDepth
              << " | FHEW n " << lp->Getn() << ", N " << lp->GetN() << ", q " << q << ", logQ " << A.logQ
              << ", beta " << beta << " | table " << half << " positions/segment, guard " << G << "\n";

    // ---- inputs: uniform inside each segment, as SegLUT.precision() samples
    std::mt19937_64 rng(A.seed);
    std::vector<double> v(S);
    std::vector<int> seg(S);
    for (uint32_t i = 0; i < S; ++i) {
        seg[i] = static_cast<int>(i * nseg / S);
        std::uniform_real_distribution<double> U(tau[seg[i]], tau[seg[i] + 1]);
        v[i] = U(rng);
    }
    auto ctV = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, S));
    auto f = [](double x) { return 1.0 / std::sqrt(x); };
    auto fp = [](double x) { return -0.5 / (x * std::sqrt(x)); };

    // ---- tables (plaintext, offline)
    std::vector<double> w(nseg);
    std::vector<Table> TF(nseg), TS(nseg);
    for (uint32_t s = 0; s < nseg; ++s) {
        w[s] = (tau[s + 1] - tau[s]) / (half - 2 * G);
        std::vector<double> fv(half), sv(half);
        for (uint64_t x = 0; x < half; ++x) {
            // positions are linear in v and a segment spans 60x, so the low guard would
            // reach v <= 0; index noise moves a few positions, so 0.8*tau_s is never read
            double vx = std::max(tau[s] + (x - G) * w[s], 0.8 * tau[s]);
            fv[x] = f(vx);
            sv[x] = fp(vx) * w[s];  // per table position
        }
        TF[s] = MakeTable(fv, q, A.phaseFrac);
        TS[s] = MakeTable(sv, q, A.phaseFrac);
    }

    // ---- 1. segment index
    std::vector<Ciphertext<DCRTPoly>> lt(nseg);  // lt[k] ~ [v < tau_k], k = 1..3
    std::vector<std::vector<double>> ltTrue(nseg, std::vector<double>(S));
    double tCmpPre = 0, tCmpSw = 0, tCmpSign = 0, tCmpBack = 0;
    for (uint32_t k = 1; k < nseg; ++k) {
        const double R = (k == nseg - 1) ? std::max(tau[nseg] - tau[k], tau[k] - tau[0])
                                         : std::max(1.2 * tau[k + 1] - tau[k], tau[k] - tau[0]);
        const double scaleSign = 0.9 * (pLarge / 2.0) / R;
        auto d = cc->EvalSub(ctV, tau[k]);
        auto t = Clock::now();
        cc->EvalCKKStoFHEWPrecompute(scaleSign / pLarge);
        tCmpPre += Since(t);
        t = Clock::now();
        auto lwe = cc->EvalCKKStoFHEW(d, S);
        tCmpSw += Since(t);
        t = Clock::now();
        std::vector<LWECiphertext> sg(S);
#pragma omp parallel for schedule(dynamic)
        for (uint32_t i = 0; i < S; ++i) sg[i] = ccLWE->EvalSign(lwe[i], true);
        tCmpSign += Since(t);
        t = Clock::now();
        lt[k] = cc->EvalFHEWtoCKKS(sg, S, S, 4, -1.0, 1.0);
        tCmpBack += Since(t);
        for (uint32_t i = 0; i < S; ++i) ltTrue[k][i] = v[i] < tau[k] ? 1.0 : 0.0;
    }
    T["1 cmp: switch precompute x3 (cacheable)"] = tCmpPre;
    T["1 cmp: CKKS->FHEW x3"] = tCmpSw;
    T["1 cmp: FHEW EvalSign x3"] = tCmpSign;
    T["1 cmp: FHEW->CKKS x3"] = tCmpBack;

    // ---- 2. tables, construct-then-select
    std::vector<Ciphertext<DCRTPoly>> cF(nseg), cS(nseg), cE(nseg);
    std::vector<std::vector<uint64_t>> xUsed(nseg, std::vector<uint64_t>(S));
    std::vector<std::vector<int64_t>> outPhaseErr(nseg, std::vector<int64_t>(S));
    double tLutPre = 0, tLutSw = 0, tLutF = 0, tLutBack = 0;
    {
        auto t = Clock::now();
        cc->EvalCKKStoFHEWPrecompute(1.0 / q);  // message m -> phase m*Qlwe/q
        tLutPre += Since(t);
    }
    for (uint32_t s = 0; s < nseg; ++s) {
        // table position x = G + (v - tau_s)/w_s; the bootstrap reads LUT[phase + beta]
        auto ctM = cc->EvalAdd(cc->EvalMult(ctV, 1.0 / w[s]), G - tau[s] / w[s] - static_cast<double>(beta));
        auto t = Clock::now();
        auto lwe = cc->EvalCKKStoFHEW(ctM, S);
        tLutSw += Since(t);
        t = Clock::now();
        std::vector<LWECiphertext> F(S), Sl(S), E(S);
#pragma omp parallel for schedule(dynamic)
        for (uint32_t i = 0; i < S; ++i) {
            auto ctq = ModSwitchDown(lwe[i], q);
            F[i] = ccLWE->EvalFunc(ctq, TF[s].lut);
            Sl[i] = ccLWE->EvalFunc(ctq, TS[s].lut);
            E[i] = ModSwitchError(lwe[i], ctq);
            xUsed[s][i] = (Phase(ctq, skF) + beta) % q;
            outPhaseErr[s][i] = Centered((Phase(F[i], skF) + q - TF[s].lut[xUsed[s][i]].ConvertToInt()) % q, q);
        }
        tLutF += Since(t);
        t = Clock::now();
        cF[s] = cc->EvalFHEWtoCKKS(F, S, S, q, 0.0, static_cast<double>(q));
        cS[s] = cc->EvalFHEWtoCKKS(Sl, S, S, q, 0.0, static_cast<double>(q));
        cE[s] = cc->EvalFHEWtoCKKS(E, S, S, q, 0.0, static_cast<double>(q));
        tLutBack += Since(t);
    }
    T["2 lut: switch precompute (cacheable)"] = tLutPre;
    T["2 lut: CKKS->FHEW x4"] = tLutSw;
    T["2 lut: FHEW EvalFunc x8 (+diag)"] = tLutF;
    T["2 lut: FHEW->CKKS x12"] = tLutBack;

    // ---- 3. interpolate, 4. select
    auto t = Clock::now();
    std::vector<Ciphertext<DCRTPoly>> outI(nseg), outN(nseg);
    for (uint32_t s = 0; s < nseg; ++s) {
        auto fv = cc->EvalAdd(cc->EvalMult(cF[s], TF[s].halfw / TF[s].Dmax), TF[s].mid);
        auto sv = cc->EvalAdd(cc->EvalMult(cS[s], TS[s].halfw / TS[s].Dmax), TS[s].mid);
        // x_true - x_used = -(decoded rounding error)
        outI[s] = cc->EvalSub(fv, cc->EvalMult(sv, cE[s]));
        outN[s] = fv;
    }
    auto one = [&](const Ciphertext<DCRTPoly>& c) { return cc->EvalSub(1.0, c); };
    // Sharpen the comparison bits, x <- x^2 (3 - 2x) (error e -> ~3e^2, 2 levels). Above
    // its trusted range a comparison may be garbage, so a mask can reduce to a single
    // bit; at 128-bit that bit's ~2.5e-4 error times segment 0's outputs (up to ~45)
    // swamps segment 3's outputs (~0.008).
    for (uint32_t k = 1; k < nseg; ++k)
        for (uint32_t r = 0; r < A.clean; ++r)
            lt[k] = cc->EvalMult(cc->EvalSquare(lt[k]), cc->EvalSub(3.0, cc->EvalMult(lt[k], 2.0)));
    auto lt23 = cc->EvalMult(lt[2], lt[3]);
    std::vector<Ciphertext<DCRTPoly>> M(nseg);
    M[0] = cc->EvalMult(lt[1], lt23);
    M[1] = cc->EvalMult(one(lt[1]), lt23);
    M[2] = cc->EvalMult(one(lt[2]), lt[3]);
    M[3] = one(lt[3]);
    auto yI = cc->EvalMult(M[0], outI[0]);
    auto yN = cc->EvalMult(M[0], outN[0]);
    for (uint32_t s = 1; s < nseg; ++s) {
        yI = cc->EvalAdd(yI, cc->EvalMult(M[s], outI[s]));
        yN = cc->EvalAdd(yN, cc->EvalMult(M[s], outN[s]));
    }
    T["3+4 interpolate + select (CKKS)"] = Since(t);

    // ---- 5. Newton on the selected output, y <- y (3 - v y^2) / 2. Newton alone was
    // dead (no starting guess holds over a 6e6 range); the table is that guess, and one
    // step roughly squares its relative error. 3 levels per step, CKKS only.
    t = Clock::now();
    std::vector<Ciphertext<DCRTPoly>> yNewton{yI};
    for (uint32_t k = 0; k < A.newton; ++k) {
        const auto& y = yNewton.back();
        auto vy2 = cc->EvalMult(cc->EvalSquare(y), ctV);
        yNewton.push_back(cc->EvalMult(y, cc->EvalSub(1.5, cc->EvalMult(vy2, 0.5))));
    }
    T["5 Newton steps (CKKS)"] = Since(t);

    // ---- single-thread cost of one FHEW op of each kind
    {
        auto d = cc->EvalSub(ctV, tau[1]);
        cc->EvalCKKStoFHEWPrecompute(1.0 / pLarge);
        auto lwe = cc->EvalCKKStoFHEW(d, S);
        const int reps = 4;
        auto ts = Clock::now();
        for (int r = 0; r < reps; ++r) ccLWE->EvalSign(lwe[r], true);
        T["unit: one EvalSign, 1 thread"] = Since(ts) / reps;
        auto ctq = ModSwitchDown(lwe[0], q);
        ts = Clock::now();
        for (int r = 0; r < reps; ++r) ccLWE->EvalFunc(ctq, TF[0].lut);
        T["unit: one EvalFunc, 1 thread"] = Since(ts) / reps;
    }

    // ---- results
    auto yi = Dec(cc, keys.secretKey, yI, S), yn = Dec(cc, keys.secretKey, yN, S);
    std::cout << "\nlevels left on the output: " << (A.multDepth - yI->GetLevel()) << " of " << A.multDepth
              << " (after " << A.newton << " Newton steps: " << (A.multDepth - yNewton.back()->GetLevel()) << ")\n";

    std::cout << "\n-- diagnostics (secret-key, own segment only)\n";
    for (uint32_t k = 1; k < nseg; ++k) {
        auto c = Dec(cc, keys.secretKey, lt[k], S);
        int wrong = 0;
        double worstGap = 0;
        Stats err;
        for (uint32_t i = 0; i < S; ++i) {
            bool trusted = v[i] < ((k == nseg - 1) ? tau[nseg] : 1.2 * tau[k + 1]);
            if (!trusted) continue;
            err.Add(c[i] - ltTrue[k][i]);
            if (std::fabs(c[i] - ltTrue[k][i]) > 0.5) { ++wrong; worstGap = std::max(worstGap, std::fabs(v[i] / tau[k] - 1)); }
        }
        std::cout << "  compare tau_" << k << " = " << tau[k] << ": wrong " << wrong << "/" << err.x.size()
                  << " (worst at " << 100 * worstGap << "% from the boundary), bit rms err " << err.Rms() << "\n";
    }
    for (uint32_t s = 0; s < nseg; ++s) {
        auto Fd = Dec(cc, keys.secretKey, cF[s], S), Ed = Dec(cc, keys.secretKey, cE[s], S);
        auto Sd = Dec(cc, keys.secretKey, cS[s], S), Id = Dec(cc, keys.secretKey, outI[s], S);
        double maxE = 0, maxSv = 0, maxOut = 0, maxSvIn = 0;
        for (uint32_t i = 0; i < S; ++i) {
            maxE = std::max(maxE, std::fabs(Ed[i]));
            maxSv = std::max(maxSv, std::fabs(TS[s].Decode(Sd[i])));
            maxOut = std::max(maxOut, std::fabs(Id[i]));
            if (seg[i] == static_cast<int>(s)) {
                double xTrue = G + (v[i] - tau[s]) / w[s];
                maxSvIn = std::max(maxSvIn, std::fabs(TS[s].Decode(Sd[i]) - fp(std::max(tau[s] + (xTrue - G) * w[s], 1e-300)) * w[s]));
            }
        }
        std::cout << "  segment " << s << " all slots: max|E| " << maxE << " max|slope| " << maxSv
                  << " max|out_interp| " << maxOut << " | own slots: max slope err " << maxSvIn
                  << " (slope range " << TS[s].mid - TS[s].halfw << ".." << TS[s].mid + TS[s].halfw << ")\n";
        Stats idx, outPh, decF, decE;
        for (uint32_t i = 0; i < S; ++i) {
            if (seg[i] != static_cast<int>(s)) continue;
            double xTrue = G + (v[i] - tau[s]) / w[s];
            double dx = static_cast<double>(Centered((xUsed[s][i] + q - static_cast<uint64_t>(std::llround(xTrue)) % q) % q, q))
                        + (std::llround(xTrue) - xTrue);
            idx.Add(dx);
            outPh.Add(static_cast<double>(outPhaseErr[s][i]));
            decF.Add(Fd[i] - TF[s].DecodedAt(q, xUsed[s][i]));
            decE.Add(Ed[i] - dx);
        }
        std::cout << "  segment " << s << " [" << tau[s] << ", " << tau[s + 1] << "]: index noise rms "
                  << idx.Rms() << " max " << idx.MaxAbs() << " positions; table-output phase noise rms "
                  << outPh.Rms() << " max " << outPh.MaxAbs() << " (of +-" << A.phaseFrac * q
                  << "); f decode err rms " << decF.Rms() << "; rounding-error decode err rms " << decE.Rms() << "\n";
    }

    std::cout << "\n-- precision vs exact 1/sqrt(v)  (bits = -log2 RMSE, ROSETTA's definition)\n";
    std::cout << "  segment             n   bits(interp)  bits(nearest)  max|err| interp  rel-bits interp\n";
    Stats allI, allN;
    for (uint32_t s = 0; s <= nseg; ++s) {
        Stats eI, eN, rI;
        for (uint32_t i = 0; i < S; ++i) {
            if (s < nseg && seg[i] != static_cast<int>(s)) continue;
            double ex = f(v[i]);
            eI.Add(yi[i] - ex);
            eN.Add(yn[i] - ex);
            rI.Add((yi[i] - ex) / ex);
        }
        std::cout << "  " << std::left << std::setw(16) << (s < nseg ? "seg " + std::to_string(s) : std::string("all"))
                  << std::right << std::setw(5) << eI.x.size() << std::setw(14) << Bits(eI.Rms()) << std::setw(15)
                  << Bits(eN.Rms()) << std::setw(17) << eI.MaxAbs() << std::setw(17) << Bits(rI.Rms()) << "\n";
    }

    {
        std::vector<std::vector<double>> Md(nseg), Od(nseg), Ld(nseg);
        for (uint32_t s = 0; s < nseg; ++s) {
            Md[s] = Dec(cc, keys.secretKey, M[s], S);
            Od[s] = Dec(cc, keys.secretKey, outI[s], S);
        }
        for (uint32_t k = 1; k < nseg; ++k) Ld[k] = Dec(cc, keys.secretKey, lt[k], S);
        std::vector<uint32_t> order(S);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return std::fabs(yi[a] / f(v[a]) - 1) > std::fabs(yi[b] / f(v[b]) - 1);
        });
        std::cout << "\n-- worst 5 values (interp, before Newton)\n";
        for (uint32_t r = 0; r < 5; ++r) {
            uint32_t i = order[r];
            std::cout << "  v " << v[i] << " seg " << seg[i] << " exact " << f(v[i]) << " got " << yi[i]
                      << " | lt1..3 " << Ld[1][i] << " " << Ld[2][i] << " " << Ld[3][i] << " | M*out:";
            for (uint32_t s = 0; s < nseg; ++s) std::cout << " " << Md[s][i] * Od[s][i];
            std::cout << " | own out " << Od[seg[i]][i] << "\n";
        }
    }

    std::cout << "\n-- + Newton steps on the interpolated output\n";
    std::cout << "  steps   bits(all)  rel-bits(all)  rel-bits worst segment  max rel err\n";
    for (uint32_t k = 0; k < yNewton.size(); ++k) {
        auto yk = Dec(cc, keys.secretKey, yNewton[k], S);
        Stats e, r;
        double worstSeg = 99;
        for (uint32_t s = 0; s < nseg; ++s) {
            Stats rs;
            for (uint32_t i = 0; i < S; ++i)
                if (seg[i] == static_cast<int>(s)) rs.Add((yk[i] - f(v[i])) / f(v[i]));
            worstSeg = std::min(worstSeg, Bits(rs.Rms()));
        }
        for (uint32_t i = 0; i < S; ++i) {
            e.Add(yk[i] - f(v[i]));
            r.Add((yk[i] - f(v[i])) / f(v[i]));
        }
        std::cout << "  " << std::setw(5) << k << std::setw(12) << Bits(e.Rms()) << std::setw(15) << Bits(r.Rms())
                  << std::setw(24) << worstSeg << std::setw(13) << r.MaxAbs() << "\n";
    }

    std::cout << "\n-- latency, wall clock, " << S << " values in one ciphertext\n";
    double online = 0;
    for (const auto& [k, sec] : T) {
        std::cout << "  " << std::left << std::setw(44) << k << std::right << std::setw(10) << sec << " s\n";
        if (k.rfind("1 ", 0) == 0 || k.rfind("2 ", 0) == 0 || k.rfind("3", 0) == 0)
            if (k.find("cacheable") == std::string::npos) online += sec;
    }
    std::cout << "  online total (excl. keygen and cacheable precomputes): " << online << " s = "
              << online / S << " s per value\n";
    std::cout << "  FHEW single-thread work per value: 3 EvalSign + 8 EvalFunc = "
              << 3 * T["unit: one EvalSign, 1 thread"] + 8 * T["unit: one EvalFunc, 1 thread"] << " s\n";
    return 0;
}
