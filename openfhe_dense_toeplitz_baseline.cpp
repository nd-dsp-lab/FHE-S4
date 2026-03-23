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

vector<vector<double>> BuildDenseToeplitzMatrix(const vector<double>& coeffs, int L) {
    vector<vector<double>> T(L, vector<double>(L, 0.0));
    const int K = min(static_cast<int>(coeffs.size()), L);
    for (int k = 0; k < K; ++k) {
        for (int row = k; row < L; ++row) {
            T[row][row - k] = coeffs[k];
        }
    }
    return T;
}

Ciphertext<DCRTPoly> FHEDenseToeplitzBaseline(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ctxt_x,
    const vector<vector<double>>& T,
    int batchSize) {
    // Dense baseline:
    // 1) treat Toeplitz as a generic dense matrix
    // 2) compute one encrypted inner product per output row
    // 3) merge those row outputs into a packed ciphertext
    //
    // This intentionally does NOT exploit Toeplitz structure the way rotate-sum
    // does. It is a generic dense linear transform baseline.
    vector<Ciphertext<DCRTPoly>> row_outputs;
    row_outputs.reserve(batchSize);

    for (int row = 0; row < batchSize; ++row) {
        Plaintext ptxt_row = cc->MakeCKKSPackedPlaintext(T[row]);
        Ciphertext<DCRTPoly> ctxt_row = cc->EvalInnerProduct(ctxt_x, ptxt_row, batchSize);
        row_outputs.push_back(ctxt_row);
    }

    return cc->EvalMerge(row_outputs);
}

int main(int argc, char* argv[]) {
    string filename = "toeplitz_test_data.json";
    if (argc > 1) {
        filename = argv[1];
    }

    TestData data = ParseTestData(filename);
    cout << "Loaded dense Toeplitz baseline input:" << endl;
    cout << "seq_len=" << data.seq_len << ", d_model=" << data.d_model
         << ", toeplitz_K=" << data.toeplitz_K << endl;

    auto t_ctx0 = chrono::high_resolution_clock::now();

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(8);
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
    cc->EvalSumKeyGen(keyPair.secretKey);

    vector<int32_t> mergeRotations;
    for (int i = 1; i < data.seq_len; ++i) {
        mergeRotations.push_back(i);
        mergeRotations.push_back(-i);
    }
    cc->EvalAtIndexKeyGen(keyPair.secretKey, mergeRotations);

    auto t_ctx1 = chrono::high_resolution_clock::now();
    cout << "[dense] context_creation_time_s="
         << chrono::duration<double>(t_ctx1 - t_ctx0).count() << endl;

    double overall_max_err = 0.0;
    double total_encrypt_s = 0.0;
    double total_plain_ref_s = 0.0;
    double total_dense_op_s = 0.0;
    double total_decrypt_s = 0.0;

    for (int c = 0; c < data.d_model; ++c) {
        const auto& chan = data.channels[c];

        auto t_ref0 = chrono::high_resolution_clock::now();
        vector<double> y_plain = ToeplitzPlain(chan.x, chan.coeffs, data.seq_len);
        vector<vector<double>> T = BuildDenseToeplitzMatrix(chan.coeffs, data.seq_len);
        auto t_ref1 = chrono::high_resolution_clock::now();

        auto t_enc0 = chrono::high_resolution_clock::now();
        Plaintext ptxt_x = cc->MakeCKKSPackedPlaintext(chan.x);
        Ciphertext<DCRTPoly> ctxt_x = cc->Encrypt(keyPair.publicKey, ptxt_x);
        auto t_enc1 = chrono::high_resolution_clock::now();

        auto t_op0 = chrono::high_resolution_clock::now();
        Ciphertext<DCRTPoly> ctxt_y = FHEDenseToeplitzBaseline(cc, ctxt_x, T, data.seq_len);
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

        double ref_s = chrono::duration<double>(t_ref1 - t_ref0).count();
        double enc_s = chrono::duration<double>(t_enc1 - t_enc0).count();
        double op_s = chrono::duration<double>(t_op1 - t_op0).count();
        double dec_s = chrono::duration<double>(t_dec1 - t_dec0).count();

        total_plain_ref_s += ref_s;
        total_encrypt_s += enc_s;
        total_dense_op_s += op_s;
        total_decrypt_s += dec_s;

        cout << "[dense] channel=" << c
             << " plain_ref_time_s=" << ref_s
             << " encrypt_time_s=" << enc_s
             << " op_time_s=" << op_s
             << " decrypt_time_s=" << dec_s
             << " max_abs_diff=" << scientific << max_err << endl;
    }

    cout << "[dense] total_plain_ref_s=" << total_plain_ref_s << endl;
    cout << "[dense] total_encrypt_s=" << total_encrypt_s << endl;
    cout << "[dense] total_op_s=" << total_dense_op_s << endl;
    cout << "[dense] total_decrypt_s=" << total_decrypt_s << endl;
    cout << "[dense] overall_max_abs_diff=" << scientific << overall_max_err << endl;

    return 0;
}
