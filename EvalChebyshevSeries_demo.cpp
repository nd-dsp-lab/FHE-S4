#include "pke/openfhe.h"
#include <algorithm>
#include <exception>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace lbcrypto;
using namespace std;

// A simple structure to hold the parsed JSON datA
struct ChannelData {
    vector<double> x;
    vector<double> coeffs;
    double D;
    vector<double> y_skip_expected;
    vector<double> y_act;
};

struct TestData {
    int seq_len;
    int d_model;
    int toeplitz_K;
    vector<double> decoder_weight;
    double decoder_bias;
    double out_expected;
    double unencrypted_forward_time;
    vector<ChannelData> channels;
    vector<double> gelu_domain;
    vector<double> gelu_cheb;
    vector<vector<double>> conv_weight;
    vector<double> conv_bias;
    vector<vector<double>> post_gelu;
    vector<vector<double>> pre_gate;
    vector<vector<double>> gate;
    vector<vector<double>> gated;
    vector<double> pooled;
};

// Simple JSON parser for the specific output format of the python export script
TestData ParseTestData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open " << filename << endl;
        exit(1);
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::exception& e) {
        cerr << "Invalid JSON in " << filename << ": " << e.what() << endl;
        exit(1);
    }

    TestData data;
    try {
        data.seq_len = root.at("seq_len").get<int>();
        data.d_model = root.at("d_model").get<int>();
        data.toeplitz_K = root.at("toeplitz_K").get<int>();
        data.decoder_weight = root.at("decoder_weight").get<vector<double>>();
        data.decoder_bias = root.at("decoder_bias").get<double>();
        data.out_expected = root.at("out_expected").get<double>();
        data.unencrypted_forward_time = root.at("unencrypted_forward_time").get<double>();
        data.gelu_domain = root.at("gelu_domain").get<vector<double>>();
        data.gelu_cheb = root.at("gelu_cheb").get<vector<double>>();
        data.conv_weight = root.at("conv_weight").get<vector<vector<double>>>();
        data.post_gelu = root.at("post_gelu").get<vector<vector<double>>>();
        data.pre_gate = root.at("pre_gate").get<vector<vector<double>>>();
        data.gate = root.at("gate").get<vector<vector<double>>>();
        data.gated = root.at("gated").get<vector<vector<double>>>();
        data.pooled = root.at("pooled").get<vector<double>>();

        const auto& conv_bias_json = root.at("conv_bias");
        if (conv_bias_json.is_array()) {
            data.conv_bias = conv_bias_json.get<vector<double>>();
        } else {
            // Backward-compatible fallback for old exports that wrote scalar conv_bias.
            data.conv_bias = {conv_bias_json.get<double>()};
        }

        const auto& channels_json = root.at("channels");
        if (!channels_json.is_array()) {
            throw runtime_error("\"channels\" must be a JSON array");
        }
        for (const auto& channel_json : channels_json) {
            ChannelData channel;
            channel.x = channel_json.at("x").get<vector<double>>();
            channel.coeffs = channel_json.at("coeffs").get<vector<double>>();
            channel.D = channel_json.at("D").get<double>();
            channel.y_skip_expected = channel_json.at("y_skip_expected").get<vector<double>>();
            channel.y_act = channel_json.at("y_act").get<vector<double>>();
            data.channels.push_back(std::move(channel));
        }
    } catch (const exception& e) {
        cerr << "JSON schema mismatch in " << filename << ": " << e.what() << endl;
        exit(1);
    }

    return data;
}

static double MaxAbsDiff(const vector<double>& a, const vector<double>& b) {
    const size_t n = min(a.size(), b.size());
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_err = max(max_err, abs(a[i] - b[i]));
    }
    return max_err;
}


int main(int argc, char* argv[]) {
    string filename = "forward_pass_data.json";
    if (argc > 1) {
        filename = argv[1];
    }
    
    TestData data = ParseTestData(filename);
    cout << "Loaded test data:" << endl;
    cout << "seq_len: " << data.seq_len << ", d_model: " << data.d_model << ", toeplitz_K: " << data.toeplitz_K << endl;

    if (data.channels.size() != static_cast<size_t>(data.d_model)) {
        cerr << "channel count mismatch: got " << data.channels.size()
             << ", expected " << data.d_model << endl;
        return 1;
    }
    if (data.conv_bias.size() != static_cast<size_t>(2 * data.d_model)) {
        cerr << "conv_bias size mismatch: got " << data.conv_bias.size()
             << ", expected " << 2 * data.d_model << endl;
        return 1;
    }
    if (data.post_gelu.size() != static_cast<size_t>(data.d_model) ||
        data.gate.size() != static_cast<size_t>(data.d_model) ||
        data.gated.size() != static_cast<size_t>(data.d_model) ||
        data.pooled.size() != static_cast<size_t>(data.d_model) ||
        data.pre_gate.size() != static_cast<size_t>(2 * data.d_model)) {
        cerr << "stage reference size mismatch in exported JSON" << endl;
        return 1;
    }
    
    // 1. Setup OpenFHE Context
    auto t_start = chrono::high_resolution_clock::now();
    uint32_t multDepth = 13;     // Extra headroom for GELU poly + GLU + decoder.
    uint32_t scaleModSize = 59;  // Larger scale for better post-chain decode precision.
    uint32_t firstModSize = 60;
    uint32_t ringDim = 16384;    // More modulus capacity for deeper CKKS chains.
    uint32_t batchSize = data.seq_len;
    
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    
    // Generate Keys
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    
    // Generate Rotation Keys for Toeplitz Convolution
    vector<int32_t> indexList;
    for (int k = 1; k < data.toeplitz_K; ++k) {
        indexList.push_back(-k); // shift right (delay)
    }
    cc->EvalAtIndexKeyGen(keyPair.secretKey, indexList);
    
    // Generate EvalSum Keys for Mean Reduction (Phase 3)
    cc->EvalSumKeyGen(keyPair.secretKey);
    
    auto t_end = chrono::high_resolution_clock::now();
    cout << "[fhe] context_creation_time_s=" << chrono::duration<double>(t_end - t_start).count() << endl;
    cout << "[fhe] params multDepth=" << multDepth
         << " scaleModSize=" << scaleModSize
         << " firstModSize=" << firstModSize
         << " ringDim=" << ringDim
         << " batchSize=" << batchSize << endl;

    auto decryptVector = [&](const Ciphertext<DCRTPoly>& ct, size_t outLen) {
        Plaintext p;
        cc->Decrypt(keyPair.secretKey, ct, &p);
        p->SetLength(outLen);
        auto raw = p->GetRealPackedValue();
        return vector<double>(raw.begin(), raw.end());
    };

    auto printProbe = [&](const string& tag, const vector<double>& v) {
        double minv = v.empty() ? 0.0 : v[0];
        double maxv = v.empty() ? 0.0 : v[0];
        for (double val : v) {
            minv = min(minv, val);
            maxv = max(maxv, val);
        }
        double first0 = v.empty() ? 0.0 : v[0];
        double first1 = v.size() > 1 ? v[1] : first0;
        cout << "[probe] " << tag
             << " ok len=" << v.size()
             << " min=" << scientific << minv
             << " max=" << scientific << maxv
             << " first2={" << first0 << ", " << first1 << "}" << endl;
    };

    auto probeDecrypt = [&](const string& tag, const Ciphertext<DCRTPoly>& ct, size_t outLen) {
        try {
            vector<double> v = decryptVector(ct, outLen);
            printProbe(tag, v);
            return v;
        } catch (const exception& e) {
            cerr << "[probe] " << tag << " decrypt_failed: " << e.what() << endl;
            throw;
        }
    };

    auto compareStage = [&](const string& tag, const vector<double>& actual, const vector<double>& expected) {
        double err = MaxAbsDiff(actual, expected);
        cout << "[diff] " << tag << "_max_abs_diff=" << scientific << err << endl;
        return err;
    };
    
    double phase1_max_err = 0.0;
    
    // ==========================================
    // PHASE 1: FHE TOEPLITZ CONV + SKIP
    // ==========================================
    vector<Ciphertext<DCRTPoly>> ctxt_y_all(data.d_model);

    // time:
    auto t0_fhe = chrono::high_resolution_clock::now();
    
    for (int c = 0; c < data.d_model; ++c) {
        auto& chan = data.channels[c];
        
        // --- ENCRYPT ---
        Plaintext ptxt_x = cc->MakeCKKSPackedPlaintext(chan.x);
        Ciphertext<DCRTPoly> ctxt_x = cc->Encrypt(keyPair.publicKey, ptxt_x);
        
        // --- TOEPLITZ ---
        Ciphertext<DCRTPoly> ctxt_y; 
        
        for (int k = 0; k < data.toeplitz_K; ++k) {
            double coeff = chan.coeffs[k];
            
            vector<double> mask(batchSize, 1.0);
            for(int i=0; i<k; ++i){
                mask[i] = 0.0;
            }
            Plaintext ptxt_mask = cc->MakeCKKSPackedPlaintext(mask);
            
            Ciphertext<DCRTPoly> ctxt_shifted;
            if (k == 0) {
                ctxt_shifted = ctxt_x;
            } else {
                ctxt_shifted = cc->EvalAtIndex(ctxt_x, -k); 
            }
            
            vector<double> coeff_mask(batchSize, 0.0);
            for(size_t i=0; i<batchSize; ++i) {
                coeff_mask[i] = coeff * mask[i];
            }
            Plaintext ptxt_coeff_mask = cc->MakeCKKSPackedPlaintext(coeff_mask);
            
            Ciphertext<DCRTPoly> ctxt_term = cc->EvalMult(ctxt_shifted, ptxt_coeff_mask);
            
            if (k == 0) {
                ctxt_y = ctxt_term;
            } else {
                ctxt_y = cc->EvalAdd(ctxt_y, ctxt_term);
            }
        }
        
        // --- SKIP CONNECTION ---
        Plaintext ptxt_D = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, chan.D));
        Ciphertext<DCRTPoly> ctxt_skip_term = cc->EvalMult(ctxt_x, ptxt_D);
        ctxt_y_all[c] = cc->EvalAdd(ctxt_y, ctxt_skip_term);
        
        //vector<double> skip_dec = decryptVector(ctxt_y_all[c], batchSize);
        //double max_err = compareStage("phase1/skip[c=" + to_string(c) + "]", skip_dec, chan.y_skip_expected);
        //phase1_max_err = max(phase1_max_err, max_err);
    }
    //cout << "[phase1] overall_skip_max_abs_diff=" << scientific << phase1_max_err << endl;
    //cout << "[phase1] complete" << endl;


    // ==========================================
    // PHASE 2: NON-LINEAR ACTIVATION
    // ==========================================
    // The model has been trained in linear approximations of GELU
    // EvalChebyshevSeries expects the original approximation interval [lo, hi].
    double lo = data.gelu_domain[0];
    double hi = data.gelu_domain[1];

    // OpenFHE's Chebyshev-series evaluator uses the conventional c0/2 term,
    // while the PyTorch exporter stores coefficients for c0*T0 + c1*T1 + ...
    vector<double> cheb_coeffs = data.gelu_cheb;
    if (!cheb_coeffs.empty()) {
        cheb_coeffs[0] *= 2.0;
    }
    vector<Ciphertext<DCRTPoly>> ctxt_act(data.d_model);
    double phase2_max_err = 0.0;
    for (int c = 0; c < data.d_model; ++c) {
        ctxt_act[c] = cc->EvalChebyshevSeries(ctxt_y_all[c], cheb_coeffs, lo, hi);
        vector<double> act_dec = probeDecrypt("phase2/post_gelu[c=" + to_string(c) + "]", ctxt_act[c], batchSize);
        double err = compareStage("phase2/post_gelu[c=" + to_string(c) + "]", act_dec, data.post_gelu[c]);
        phase2_max_err = max(phase2_max_err, err);
    }

    cout << "[phase2- Using Chebyshev] overall_post_gelu_max_abs_diff=" << scientific << phase2_max_err << endl;
    cout << "[phase2 - Using Chebyshev] complete" << endl;
}