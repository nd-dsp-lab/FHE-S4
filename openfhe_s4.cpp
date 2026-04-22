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
    vector<ChannelData> channels;
    vector<double> gelu_domain;
    vector<double> gelu_cheb;
    vector<vector<double>> conv_weight;
    vector<double> conv_bias;
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
        data.gelu_domain = root.at("gelu_domain").get<vector<double>>();
        data.gelu_cheb = root.at("gelu_cheb").get<vector<double>>();
        data.conv_weight = root.at("conv_weight").get<vector<vector<double>>>();

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

    auto probeDecrypt = [&](const string& tag, const Ciphertext<DCRTPoly>& ct, size_t outLen) {
        try {
            Plaintext p;
            cc->Decrypt(keyPair.secretKey, ct, &p);
            p->SetLength(outLen);
            auto v = p->GetRealPackedValue();
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
        } catch (const exception& e) {
            cerr << "[probe] " << tag << " decrypt_failed: " << e.what() << endl;
            throw;
        }
    };
    
    double phase1_max_err = 0.0;
    
    // ==========================================
    // PHASE 1: FHE TOEPLITZ CONV + SKIP
    // ==========================================
    vector<Ciphertext<DCRTPoly>> ctxt_y_all(data.d_model);
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
        
        // --- PHASE 1 DECRYPT FOR VERIFICATION ---
        //Plaintext ptxt_result;
        //cc->Decrypt(keyPair.secretKey, ctxt_y, &ptxt_result);
        //ptxt_result->SetLength(batchSize);
        //auto dec_vec = ptxt_result->GetRealPackedValue();
        
        //double max_err = 0.0;
        //for (size_t i = 0; i < batchSize; ++i) {
        //    max_err = max(max_err, abs(dec_vec[i] - chan.y_skip_expected[i]));
        //}
        //phase1_max_err = max(phase1_max_err, max_err);
        //cout << "[phase1] channel=" << c << " skip_max_abs_diff=" << scientific << max_err << endl;
    }
    cout << "[phase1] complete" << endl;


    // ==========================================
    // PHASE 2: NON-LINEAR ACTIVATION
    // ==========================================
    // The model has been trained in linear approximations of GELU
    // EvalChebyshevSeries expects the original approximation interval [lo, hi].
    double lo = data.gelu_domain[0];
    double hi = data.gelu_domain[1];

    // use EvalChebyshevSeries over each channel
    auto& cheb_coeffs = data.gelu_cheb;
    vector<Ciphertext<DCRTPoly>> ctxt_act(data.d_model);
    for (int c = 0; c < data.d_model; ++c) {
        ctxt_act[c] = cc->EvalChebyshevSeries(ctxt_y_all[c], cheb_coeffs, lo, hi);
        probeDecrypt("phase2/act[c=" + to_string(c) + "]", ctxt_act[c], batchSize);
    }

    cout << "[phase2] complete" << endl;

    
    // ==========================================
    // PHASE 2.5: CONV LAYER AND GLU
    // ==========================================
    // apply convolution by over convolution dimesions d_model and 2*d_model (from Python)
    if (data.conv_weight.size() != 2 * data.d_model) {
        cerr << "conv_weight row mismatch: got " << data.conv_weight.size() << ", expected " << 2 * data.d_model << endl;
        exit(1);
    }

    for (size_t i = 0; i < data.conv_weight.size(); ++i) {
        if (data.conv_weight[i].size() != data.d_model) {
            cerr << "conv_weight[" << i << "] size mismatch: got " << data.conv_weight[i].size() << ", expected " << data.d_model << endl;
            exit(1);
        }
    }


    vector<Ciphertext<DCRTPoly>> ctxt_pre_gate(2*data.d_model);
    // loop over convolution channels
    for (int i = 0; i < 2*data.d_model; ++i) {
        // get weight and bias
        Plaintext ptxt_conv_w0 = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, data.conv_weight[i][0]));
        Plaintext ptxt_conv_bias = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, data.conv_bias[i]));

        // multiply the post-gelu ciphertext for first term
        ctxt_pre_gate[i] = cc->EvalMult(ctxt_act[0], ptxt_conv_w0);

        // Sum over channels
        for (int j = 1; j < data.d_model; ++j) {
            Plaintext ptxt_conv_w = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, data.conv_weight[i][j]));
            auto tmp = cc->EvalMult(ctxt_act[j], ptxt_conv_w);
            ctxt_pre_gate[i] = cc->EvalAdd(ctxt_pre_gate[i], tmp);
        }
        // add bias
        ctxt_pre_gate[i] = cc->EvalAdd(ctxt_pre_gate[i], ptxt_conv_bias);
        probeDecrypt("conv/pre_gate[i=" + to_string(i) + "]", ctxt_pre_gate[i], batchSize);
    }
    cout << "[conv] complete" << endl;
    
    // GLU
    // split into a and b
    vector<Ciphertext<DCRTPoly>> a_enc(ctxt_pre_gate.begin(), ctxt_pre_gate.begin() + data.d_model);
    vector<Ciphertext<DCRTPoly>> b_enc(ctxt_pre_gate.begin() + data.d_model, ctxt_pre_gate.end());
    vector<Ciphertext<DCRTPoly>> ctxt_gated(data.d_model);
    // gate each channel
    Plaintext p_025 = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, 0.25));
    Plaintext p_05  = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, 0.5));
    for (int c = 0; c < data.d_model; ++c) {
        // Build linearized gate in ciphertext space.
        Ciphertext<DCRTPoly> gate = cc->EvalMult(b_enc[c], p_025); // 0.25 * b
        gate = cc->EvalAdd(gate, p_05); // 0.25 * b + 0.5 

        // Debug/interop path requested: decrypt, clamp in plaintext, then re-encrypt.
        Plaintext ptxt_gate_pre_clamp;
        cc->Decrypt(keyPair.secretKey, gate, &ptxt_gate_pre_clamp);
        ptxt_gate_pre_clamp->SetLength(batchSize);
        auto gate_values = ptxt_gate_pre_clamp->GetRealPackedValue();
        for (auto& g : gate_values) {
            g = clamp(g, 0.0, 1.0);
        }
        Plaintext ptxt_gate_clamped = cc->MakeCKKSPackedPlaintext(gate_values);
        gate = cc->Encrypt(keyPair.publicKey, ptxt_gate_clamped);

        // use the gate
        ctxt_gated[c] = cc->EvalMult(a_enc[c], gate);
        probeDecrypt("glu/gate[c=" + to_string(c) + "]", gate, batchSize);
        probeDecrypt("glu/gated[c=" + to_string(c) + "]", ctxt_gated[c], batchSize);
    }
    cout << "[glu] complete" << endl;


    // ==========================================
    // PHASE 3: FHE MEAN REDUCTION AND DECODER
    // ==========================================
    //////////////////////////////
    // Check: dimensions of the weights/biases
    //////////////////////////////
    Ciphertext<DCRTPoly> ctxt_final_out;
    bool first = true;
    
    // The Python code groups the final decoder after the `.mean(dim=-1)` reduction
    // So for each channel c, we summarize L elements.
    for (int c = 0; c < data.d_model; ++c) {
        auto& chan = data.channels[c];
        
        // Re-encrypt the activated plaintext from Phase 2
        //Plaintext ptxt_act = cc->MakeCKKSPackedPlaintext(chan.y_act);
        //Ciphertext<DCRTPoly> ctxt_act = cc->Encrypt(keyPair.publicKey, ptxt_act);
        
        // Mean reduction over L elements (EvalSum adds all elements, we mult by 1/L)
        // Note: EvalSum replaces all elements with the sum of the whole vector
        Ciphertext<DCRTPoly> ctxt_sum = cc->EvalSum(ctxt_gated[c], batchSize);
        Plaintext ptxt_div = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, 1.0 / data.seq_len));
        Ciphertext<DCRTPoly> ctxt_mean = cc->EvalMult(ctxt_sum, ptxt_div);
        probeDecrypt("phase3/mean[c=" + to_string(c) + "]", ctxt_mean, batchSize);
        
        // Linear Decoder Weight Application
        Plaintext ptxt_w = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, data.decoder_weight[c]));
        Ciphertext<DCRTPoly> ctxt_decoder_term = cc->EvalMult(ctxt_mean, ptxt_w);
        
        if (first) {
            ctxt_final_out = ctxt_decoder_term;
            first = false;
        } else {
            ctxt_final_out = cc->EvalAdd(ctxt_final_out, ctxt_decoder_term);
        }
    }
    
    // Add decoder bias
    Plaintext ptxt_bias = cc->MakeCKKSPackedPlaintext(vector<double>(batchSize, data.decoder_bias));
    ctxt_final_out = cc->EvalAdd(ctxt_final_out, ptxt_bias);
    
    // --- PHASE 4: FINAL PLAINTEXT DECRYPTION ---
    Plaintext ptxt_final;
    cc->Decrypt(keyPair.secretKey, ctxt_final_out, &ptxt_final);
    ptxt_final->SetLength(1); // The mean is copied across all slots, just read index 0.
    
    double final_decrypted = ptxt_final->GetRealPackedValue()[0];
    cout << "\n[phase3] OpenFHE Output: " << scientific << final_decrypted << endl;
    cout << "[phase3] Expected Output: " << scientific << data.out_expected << endl;
    cout << "[phase3] Global Error: " << scientific << abs(final_decrypted - data.out_expected) << endl;
    
    return 0;
}
