#include "pke/openfhe.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace lbcrypto;
using namespace std;

struct ChannelData {
    vector<double> x;
    vector<double> coeffs;
    double D = 0.0;
    vector<double> y_skip_expected;
    vector<double> y_act;
};

struct TestData {
    int seq_len = 0;
    int d_model = 0;
    int toeplitz_K = 0;
    vector<double> decoder_weight;
    double decoder_bias = 0.0;
    double out_expected = 0.0;
    vector<ChannelData> channels;
};

TestData ParseTestData(const string& filename) {
    TestData data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open " << filename << endl;
        exit(1);
    }

    string line;
    ChannelData current_channel;
    bool in_x = false;
    bool in_coeffs = false;
    bool in_skip = false;
    bool in_act = false;
    bool in_dec_wt = false;

    while (getline(file, line)) {
        if (line.find("\"seq_len\":") != string::npos) {
            data.seq_len = stoi(line.substr(line.find(":") + 1));
        } else if (line.find("\"d_model\":") != string::npos) {
            data.d_model = stoi(line.substr(line.find(":") + 1));
        } else if (line.find("\"toeplitz_K\":") != string::npos) {
            data.toeplitz_K = stoi(line.substr(line.find(":") + 1));
        } else if (line.find("\"decoder_bias\":") != string::npos) {
            data.decoder_bias = stod(line.substr(line.find(":") + 1));
        } else if (line.find("\"out_expected\":") != string::npos) {
            data.out_expected = stod(line.substr(line.find(":") + 1));
        } else if (line.find("\"D\":") != string::npos) {
            current_channel.D = stod(line.substr(line.find(":") + 1));
        } else if (line.find("\"decoder_weight\": [") != string::npos) {
            in_dec_wt = true;
        } else if (line.find("\"x\": [") != string::npos) {
            in_x = true;
        } else if (line.find("\"coeffs\": [") != string::npos) {
            in_coeffs = true;
        } else if (line.find("\"y_skip_expected\": [") != string::npos) {
            in_skip = true;
        } else if (line.find("\"y_act\": [") != string::npos) {
            in_act = true;
        } else if (line.find("]") != string::npos) {
            if (in_dec_wt)
                in_dec_wt = false;
            else if (in_x)
                in_x = false;
            else if (in_coeffs)
                in_coeffs = false;
            else if (in_skip)
                in_skip = false;
            else if (in_act) {
                in_act = false;
                data.channels.push_back(current_channel);
                current_channel = ChannelData();
            }
        } else if (in_x || in_coeffs || in_skip || in_act || in_dec_wt) {
            if (!line.empty() && line.back() == ',')
                line.pop_back();
            double val = stod(line);
            if (in_x)
                current_channel.x.push_back(val);
            else if (in_coeffs)
                current_channel.coeffs.push_back(val);
            else if (in_skip)
                current_channel.y_skip_expected.push_back(val);
            else if (in_act)
                current_channel.y_act.push_back(val);
            else if (in_dec_wt)
                data.decoder_weight.push_back(val);
        }
    }

    return data;
}

vector<double> ToeplitzPlain(const vector<double>& x, const vector<double>& coeffs, int L) {
    vector<double> y(L, 0.0);
    const int K = min(static_cast<int>(coeffs.size()), L);
    for (int k = 0; k < K; ++k) {
        for (int t = k; t < L; ++t) {
            y[t] += coeffs[k] * x[t - k];
        }
    }
    return y;
}

Ciphertext<DCRTPoly> FHEToeplitzRotateSum(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ctxt_x,
    const vector<double>& coeffs,
    int batchSize) {
    Ciphertext<DCRTPoly> ctxt_y;
    bool first = true;

    for (int k = 0; k < static_cast<int>(coeffs.size()); ++k) {
        Ciphertext<DCRTPoly> ctxt_shifted = (k == 0) ? ctxt_x : cc->EvalAtIndex(ctxt_x, -k);

        // Mask the first k slots after right-shift so wraparound terms do not leak in.
        vector<double> coeff_mask(batchSize, coeffs[k]);
        for (int i = 0; i < k && i < batchSize; ++i) {
            coeff_mask[i] = 0.0;
        }

        Plaintext ptxt_coeff_mask = cc->MakeCKKSPackedPlaintext(coeff_mask);
        Ciphertext<DCRTPoly> ctxt_term = cc->EvalMult(ctxt_shifted, ptxt_coeff_mask);

        if (first) {
            ctxt_y = ctxt_term;
            first = false;
        } else {
            ctxt_y = cc->EvalAdd(ctxt_y, ctxt_term);
        }
    }

    return ctxt_y;
}

int main(int argc, char* argv[]) {
    string filename = "toeplitz_test_data.json";
    if (argc > 1) {
        filename = argv[1];
    }

    TestData data = ParseTestData(filename);
    cout << "Loaded rotate-sum protocol input:" << endl;
    cout << "seq_len=" << data.seq_len << ", d_model=" << data.d_model
         << ", toeplitz_K=" << data.toeplitz_K << endl;

    // ---------------------------------------------------------------------
    // Stage 1: Create CKKS context and rotation keys once.
    // This protocol is intentionally inference-only and uses Galois/rotation
    // keys because rotate-sum Toeplitz requires EvalAtIndex.
    // ---------------------------------------------------------------------
    auto t_ctx0 = chrono::high_resolution_clock::now();

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(5);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetRingDim(8192);
    parameters.SetBatchSize(data.seq_len);
    parameters.SetSecurityLevel(HEStd_NotSet);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    vector<int32_t> rotationIndices;
    for (int k = 1; k < data.toeplitz_K; ++k) {
        rotationIndices.push_back(-k);
    }
    cc->EvalAtIndexKeyGen(keyPair.secretKey, rotationIndices);

    auto t_ctx1 = chrono::high_resolution_clock::now();
    cout << "[rotate_sum] context_creation_time_s="
         << chrono::duration<double>(t_ctx1 - t_ctx0).count() << endl;

    double overall_max_err = 0.0;
    double total_encrypt_s = 0.0;
    double total_rotate_sum_s = 0.0;
    double total_decrypt_s = 0.0;
    double total_plain_ref_s = 0.0;

    // ---------------------------------------------------------------------
    // Stage 2: One ciphertext per channel.
    // For each channel:
    //   1) encrypt plaintext sequence
    //   2) apply explicit rotate-sum Toeplitz convolution
    //   3) decrypt result
    //   4) compare against plaintext Toeplitz reference
    // ---------------------------------------------------------------------
    for (int c = 0; c < data.d_model; ++c) {
        const auto& chan = data.channels[c];

        auto t_enc0 = chrono::high_resolution_clock::now();
        Plaintext ptxt_x = cc->MakeCKKSPackedPlaintext(chan.x);
        Ciphertext<DCRTPoly> ctxt_x = cc->Encrypt(keyPair.publicKey, ptxt_x);
        auto t_enc1 = chrono::high_resolution_clock::now();

        auto t_ref0 = chrono::high_resolution_clock::now();
        vector<double> y_plain = ToeplitzPlain(chan.x, chan.coeffs, data.seq_len);
        auto t_ref1 = chrono::high_resolution_clock::now();

        auto t_op0 = chrono::high_resolution_clock::now();
        Ciphertext<DCRTPoly> ctxt_y = FHEToeplitzRotateSum(cc, ctxt_x, chan.coeffs, data.seq_len);
        auto t_op1 = chrono::high_resolution_clock::now();

        auto t_dec0 = chrono::high_resolution_clock::now();
        Plaintext ptxt_y;
        cc->Decrypt(keyPair.secretKey, ctxt_y, &ptxt_y);
        ptxt_y->SetLength(data.seq_len);
        vector<double> y_dec = ptxt_y->GetRealPackedValue();
        auto t_dec1 = chrono::high_resolution_clock::now();

        double max_err = 0.0;
        for (int i = 0; i < data.seq_len; ++i) {
            max_err = max(max_err, abs(y_dec[i] - y_plain[i]));
        }
        overall_max_err = max(overall_max_err, max_err);

        double enc_s = chrono::duration<double>(t_enc1 - t_enc0).count();
        double ref_s = chrono::duration<double>(t_ref1 - t_ref0).count();
        double op_s = chrono::duration<double>(t_op1 - t_op0).count();
        double dec_s = chrono::duration<double>(t_dec1 - t_dec0).count();

        total_encrypt_s += enc_s;
        total_plain_ref_s += ref_s;
        total_rotate_sum_s += op_s;
        total_decrypt_s += dec_s;

        cout << "[rotate_sum] channel=" << c
             << " encrypt_time_s=" << enc_s
             << " plain_ref_time_s=" << ref_s
             << " op_time_s=" << op_s
             << " decrypt_time_s=" << dec_s
             << " max_abs_diff=" << scientific << max_err << endl;
    }

    cout << "[rotate_sum] total_encrypt_s=" << total_encrypt_s << endl;
    cout << "[rotate_sum] total_plain_ref_s=" << total_plain_ref_s << endl;
    cout << "[rotate_sum] total_op_s=" << total_rotate_sum_s << endl;
    cout << "[rotate_sum] total_decrypt_s=" << total_decrypt_s << endl;
    cout << "[rotate_sum] overall_max_abs_diff=" << scientific << overall_max_err << endl;

    return 0;
}
