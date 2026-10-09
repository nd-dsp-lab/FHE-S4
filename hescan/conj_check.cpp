// Early check of the two OpenFHE features HEScan relies on that are not
// "obviously native":
//   1. complex packing + conjugation via EvalAutomorphism(2N-1), and the
//      one-conjugation split of a packed ciphertext z = u + i*v into u and v;
//   2. hoisted rotations (EvalFastRotation) with negative (right) indices.
#include "openfhe.h"

#include <complex>
#include <iostream>
#include <random>

using namespace lbcrypto;
using cplx = std::complex<double>;

static double MaxErr(const std::vector<cplx>& got, const std::vector<cplx>& want) {
    double e = 0;
    for (size_t i = 0; i < want.size(); ++i)
        e = std::max(e, std::abs(got[i] - want[i]));
    return e;
}

int main() {
    const uint32_t slots = 16;
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(4);
    p.SetFirstModSize(60);
    p.SetScalingModSize(40);
    p.SetRingDim(1 << 12);
    p.SetSecurityLevel(HEStd_NotSet);  // toy size, correctness only
    p.SetBatchSize(slots);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    p.SetCKKSDataType(COMPLEX);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    const uint32_t N        = cc->GetRingDimension();
    const uint32_t conjIdx  = 2 * N - 1;
    cc->EvalAutomorphismKeyGen(keys.secretKey, {conjIdx});
    cc->EvalRotateKeyGen(keys.secretKey, {-1, -3, 2});

    std::mt19937 rng(1);
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> u(slots), v(slots);
    std::vector<cplx> z(slots);
    for (uint32_t i = 0; i < slots; ++i) {
        u[i] = U(rng);
        v[i] = U(rng);
        z[i] = cplx(u[i], v[i]);
    }
    auto ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(z));

    auto dec = [&](const Ciphertext<DCRTPoly>& c) {
        Plaintext pt;
        cc->Decrypt(keys.secretKey, c, &pt);
        pt->SetLength(slots);
        return pt->GetCKKSPackedValue();
    };

    bool ok = true;
    auto report = [&](const char* name, double err) {
        bool pass = err < 1e-5;
        ok &= pass;
        std::cout << (pass ? "[PASS] " : "[FAIL] ") << name << "  max|err| = " << err << "\n";
    };

    // 1a. conjugation
    auto keyMap = cc->GetEvalAutomorphismKeyMap(ct->GetKeyTag());
    auto ctConj = cc->EvalAutomorphism(ct, conjIdx, keyMap);
    std::vector<cplx> zc(slots);
    for (uint32_t i = 0; i < slots; ++i)
        zc[i] = std::conj(z[i]);
    report("EvalAutomorphism(2N-1) == conj", MaxErr(dec(ctConj), zc));

    // 1b. one-conjugation split: u = (z + conj z)/2, v = (z - conj z) * (-i/2)
    auto ctU = cc->EvalMult(cc->EvalAdd(ct, ctConj), 0.5);
    auto ctV = cc->EvalMult(cc->EvalSub(ct, ctConj), cplx(0, -0.5));
    std::vector<cplx> uc(u.begin(), u.end()), vc(v.begin(), v.end());
    report("split real part", MaxErr(dec(ctU), uc));
    report("split imag part", MaxErr(dec(ctV), vc));

    // 1c. re-pack after real arithmetic: (u*u) + i*(v*v)
    auto ctRe = cc->EvalAdd(cc->EvalMult(ctU, ctU), cc->EvalMult(cc->EvalMult(ctV, ctV), cplx(0, 1)));
    std::vector<cplx> want(slots);
    for (uint32_t i = 0; i < slots; ++i)
        want[i] = cplx(u[i] * u[i], v[i] * v[i]);
    report("repack u^2 + i v^2", MaxErr(dec(ctRe), want));

    // 2. hoisted rotations, negative and positive indices
    auto digits = cc->EvalFastRotationPrecompute(ct);
    for (int r : {-1, -3, 2}) {
        auto ctR = cc->EvalFastRotation(ct, r, 2 * N, digits);
        std::vector<cplx> w(slots);
        for (int i = 0; i < (int)slots; ++i)
            w[i] = z[((i + r) % (int)slots + slots) % slots];
        std::string name = "EvalFastRotation(" + std::to_string(r) + ")";
        report(name.c_str(), MaxErr(dec(ctR), w));
    }

    std::cout << (ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << std::endl;
    return ok ? 0 : 1;
}
