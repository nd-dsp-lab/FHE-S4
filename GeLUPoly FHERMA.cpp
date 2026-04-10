#include "pke/openfhe.h"
#include <iostream>

using namespace lbcrypto;

Ciphertext<DCRTPoly> App_GELU(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct) {
    // 1. [-8, 8] -> [-1, 1]
    auto ct_scaled = cc->EvalMult(ct, 0.125);

    // 2. Chebyshev 
    // T_2 = 2x^2 - 1
    auto x2 = cc->EvalMult(ct_scaled, ct_scaled);
    x2 = cc->EvalAdd(x2, x2);
    x2 = cc->EvalSub(x2, 1.0);

    // T_4 = 2*T_2^2 - 1
    auto x4 = cc->EvalMult(x2, x2);
    x4 = cc->EvalAdd(x4, x4);
    x4 = cc->EvalSub(x4, 1.0);

    // T_6 = 2 * T_2 * T_4 - T_2
    auto x6 = cc->EvalMult(x2, x4);
    x6 = cc->EvalAdd(x6, x6);
    x6 = cc->EvalSub(x6, x2);

    // T_8 = 2*T_4^2 - 1
    auto x8 = cc->EvalMult(x4, x4);
    x8 = cc->EvalAdd(x8, x8);
    x8 = cc->EvalSub(x8, 1.0);

    // T_16 = 2*T_8^2 - 1
    auto x16 = cc->EvalMult(x8, x8);
    x16 = cc->EvalAdd(x16, x16);
    x16 = cc->EvalSub(x16, 1.0);

    // 3. Linear Combination
    auto v1 = cc->EvalAddMany({
        cc->EvalMult(x2, 1.7207979409147327),
        cc->EvalMult(x4, -0.3394792285477949),
        cc->EvalMult(x6, 0.11301345036922686)
    });
    cc->EvalAddInPlace(v1, 2.526419570299282);

    auto v2 = cc->EvalAddMany({
        cc->EvalMult(x2, 0.11587962664425583),
        cc->EvalMult(x4, -0.06624630540449666),
        cc->EvalMult(x6, 0.03036273788018306)
    });
    cc->EvalAddInPlace(v2, -0.09757261019185769);

    auto v3 = cc->EvalAddMany({
        cc->EvalMult(x2, 0.015221456559998686),
        cc->EvalMult(x4, -0.008393631773617285),
        cc->EvalMult(x6, 0.0043045979715645)
    });
    cc->EvalAddInPlace(v3, -0.013309648347659766);

    // 4. Final combination
    auto v2_T8 = cc->EvalMult(v2, x8);
    auto v3_T16 = cc->EvalMult(v3, x16);

    // T24
    auto x8_scaled = cc->EvalMult(x8, -0.0012773772058335409);
    auto T24 = cc->EvalMult(x8_scaled, x16);
    T24 = cc->EvalSub(cc->EvalMult(T24, 2.0), x8_scaled);

    // P_even
    // std::vector<Ciphertext<DCRTPoly>> final_parts = {v1, v2_T8, v3_T16, T24};
    auto P_even = cc->EvalAddMany({v1, v2_T8, v3_T16, T24});

    cc->EvalAddInPlace(P_even, cc->EvalMult(ct_scaled, 4.0));

    return P_even;
}

int main() {

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(13);
    parameters.SetScalingModSize(59);
    parameters.SetBatchSize(256);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    //Test & check depth
    std::vector<double> input(256, 2.0);
    Plaintext pt = cc->MakeCKKSPackedPlaintext(input);
    auto ct = cc->Encrypt(keys.publicKey, pt);

    auto resultCT = App_GELU(cc, ct);
    std::cout << resultCT->GetLevel() << std::endl;

    Plaintext res;

    cc->Decrypt(keys.secretKey, resultCT, &res);
    res->SetLength(256);
    std::vector<double> finalValues = res->GetRealPackedValue();

    std::cout << "Input: " << input[0] << " -> Output: " << finalValues[0] << std::endl;

    return 0;
}

