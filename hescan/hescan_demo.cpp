// HEScan end-to-end demo: random factorized packet -> encrypt -> HEScan -> decrypt,
// compared against Eq. 2-3 evaluated in double precision.
//
//   ./hescan_demo [--H 4] [--P 4] [--G 2] [--ds 8] [--L 16] [--sstate 64]
//                 [--scan bk|hs|seq] [--radix 2] [--complex 1] [--h0 0]
//                 [--scaling flex|fixed] [--logn 15] [--secure 1] [--tol 1e-3] [--seed 1]
//                 [--gpu 0] [--device 0] [--compare 0]
//
//   --gpu 1      run the kernel on the GPU through FIDESlib (needs a -DHESCAN_GPU=ON build)
//   --compare 1  with --gpu 1: also run the CPU kernel on the same packet and report the
//                CPU/GPU difference (decrypted, and whether the ciphertexts are bit-identical)
#include "hescan.h"
#ifdef HESCAN_WITH_GPU
#include "hescan_gpu.h"
#endif

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

using namespace lbcrypto;
using namespace hescan;

namespace {

double Seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

PlainPacket RandomPacket(const Config& cfg, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> sym(-1.0, 1.0), decay(0.5, 1.0);
    PlainPacket p;
    auto fill = [&](std::vector<double>& v, size_t n, std::uniform_real_distribution<double>& d) {
        v.resize(n);
        for (auto& e : v)
            e = d(rng);
    };
    fill(p.x, size_t(cfg.L) * cfg.E(), sym);
    fill(p.a, size_t(cfg.L) * cfg.H, decay);  // a = exp(A * delta) in (0, 1)
    fill(p.B, size_t(cfg.L) * cfg.G * cfg.ds, sym);
    fill(p.C, size_t(cfg.L) * cfg.G * cfg.ds, sym);
    if (cfg.withH0)
        fill(p.h0, size_t(cfg.E()) * cfg.ds, sym);
    // keep |B|,|C| ~ 1/sqrt(d_s) so m_k stays O(1) like a normalized model
    for (auto* v : {&p.B, &p.C})
        for (auto& e : *v)
            e /= std::sqrt(double(cfg.ds));
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> opt;
    for (int i = 1; i + 1 < argc; i += 2)
        opt[std::string(argv[i]).substr(2)] = argv[i + 1];
    auto get = [&](const char* k, const char* d) { return opt.count(k) ? opt[k] : std::string(d); };

    Config cfg;
    cfg.H           = std::stoul(get("H", "4"));
    cfg.P           = std::stoul(get("P", "4"));
    cfg.G           = std::stoul(get("G", "2"));
    cfg.ds          = std::stoul(get("ds", "8"));
    cfg.L           = std::stoul(get("L", "16"));
    cfg.sState      = std::stoul(get("sstate", "64"));
    cfg.radix       = std::stoul(get("radix", "2"));
    cfg.algo        = ParseScanAlgo(get("scan", "bk"));
    cfg.complexPack = get("complex", "1") == "1";
    cfg.withH0      = get("h0", "0") == "1";
    const bool fixed  = get("scaling", "flex") == "fixed";
    const uint32_t logN = std::stoul(get("logn", "15"));
    const bool secure = get("secure", "1") == "1";
    const double tol  = std::stod(get("tol", "1e-3"));
    const bool useGpu  = get("gpu", "0") == "1";
    const int device   = std::stoi(get("device", "0"));
    const bool compare = useGpu && get("compare", "0") == "1";
#ifndef HESCAN_WITH_GPU
    if (useGpu) {
        std::cerr << "--gpu 1 needs a build with -DHESCAN_GPU=ON (FIDESlib); this binary is CPU-only\n";
        return 2;
    }
#endif

    for (uint32_t L : {1u, 2u, 3u, 5u, 6u, 7u, 8u, 13u, 16u, 31u, 64u, 100u}) {
        if (!CheckSchedule(cfg.algo, L)) {
            std::cerr << "scan schedule self-check failed for L=" << L << "\n";
            return 1;
        }
    }

    // Chain [60, 40 x D, 60]: firstModSize 60, scalingModSize 40, D levels; the trailing
    // 60-bit-ish moduli are the key-switching (P) moduli OpenFHE adds for HYBRID switching.
    const uint32_t D = cfg.Depth();
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(D);
    params.SetFirstModSize(60);
    params.SetScalingModSize(40);
    params.SetRingDim(1u << logN);
    params.SetBatchSize(cfg.sState);
    params.SetScalingTechnique(fixed ? FIXEDAUTO : FLEXIBLEAUTO);
    params.SetSecurityLevel(secure ? HEStd_128_classic : HEStd_NotSet);
    params.SetCKKSDataType(cfg.complexPack ? COMPLEX : REAL);

    auto t0 = std::chrono::steady_clock::now();
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cfg.Validate(cc->GetRingDimension() / 2);

    std::cout << "HEScan (FESC) on OpenFHE " << GetOPENFHEVersion() << "\n"
              << "  H=" << cfg.H << " P=" << cfg.P << " G=" << cfg.G << " d_s=" << cfg.ds << " L=" << cfg.L
              << "  E=" << cfg.E() << "  s_state=" << cfg.sState << " c_state=" << cfg.cState()
              << " K_s=" << cfg.Ks() << "\n"
              << "  scan=" << ScanAlgoName(cfg.algo) << " radix=" << cfg.radix
              << " complexPack=" << cfg.complexPack << " h0=" << cfg.withH0
              << " scaling=" << (fixed ? "FIXEDAUTO" : "FLEXIBLEAUTO")
              << " backend=" << (useGpu ? "GPU (FIDESlib), device " + std::to_string(device) : std::string("CPU"))
              << "\n"
              << "  N=" << cc->GetRingDimension() << " depth D=" << D
              << " log2(Q)=" << cc->GetModulus().GetMSB()
              << " security=" << (secure ? "128-bit classic" : "NOT SET (toy)") << "\n";

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    auto rot = cfg.RotationIndices();
    cc->EvalRotateKeyGen(keys.secretKey, rot);
    if (cfg.complexPack)
        cc->EvalAutomorphismKeyGen(keys.secretKey, {2 * cc->GetRingDimension() - 1});
    std::cout << "  rotation keys: " << rot.size() << (cfg.complexPack ? " (+1 conjugation key)" : "")
              << "   setup+keygen " << std::fixed << std::setprecision(2) << Seconds(t0) << " s\n";

    PlainPacket pkt = RandomPacket(cfg, std::stoul(get("seed", "1")));
    t0              = std::chrono::steady_clock::now();
    EncPacket enc   = EncryptPacket(cc, keys.publicKey, cfg, pkt);
    std::cout << "  boundary ciphertexts per token: " << 4 * cfg.BoundaryCts() << " in (x,a,B,C), "
              << cfg.BoundaryCts() << " out   [unpaired: " << 4 * cfg.Ks() << " in, " << cfg.Ks()
              << " out]\n"
              << "  encrypt " << Seconds(t0) << " s\n";

    Timings tm;
    std::vector<std::vector<Ct>> out, outCpu;
    if (!useGpu || compare) {
        HEScan kernel(cc, cfg);
        t0           = std::chrono::steady_clock::now();
        outCpu       = kernel.Run(enc, &tm);
        double total = Seconds(t0);
        std::cout << "  HEScan CPU " << total << " s  (split+br " << tm.broadcast << ", scan "
                  << tm.scan << ", contract " << tm.contract << ")\n";
    }
    if (!useGpu) {
        out = outCpu;
    }
#ifdef HESCAN_WITH_GPU
    else {
        t0 = std::chrono::steady_clock::now();
        HEScanGPU gpuKernel(cc, cfg, keys.publicKey, device);
        std::cout << "  GPU setup (context + keys to device) " << Seconds(t0) << " s\n";
        t0           = std::chrono::steady_clock::now();
        out          = gpuKernel.Run(enc, &tm);
        double total = Seconds(t0);
        std::cout << "  HEScan GPU " << total << " s  (upload " << tm.upload << ", split+br " << tm.broadcast
                  << ", scan " << tm.scan << ", contract " << tm.contract << ", download " << tm.download
                  << ")   device memory in use " << std::setprecision(2)
                  << HEScanGPU::DeviceMemoryUsed(device) / double(1u << 30) << " GiB\n";
    }
#endif

    uint32_t maxLevel = 0;
    for (auto& row : out)
        for (auto& c : row)
            maxLevel = std::max<uint32_t>(maxLevel, c->GetLevel() + (c->GetNoiseScaleDeg() > 1 ? 1 : 0));
    std::cout << "  levels consumed: " << maxLevel << " / " << D << "\n";

    auto got  = DecryptOutput(cc, keys.secretKey, cfg, out);
    auto want = PlainReference(cfg, pkt);
    if (compare) {
        auto cpu      = DecryptOutput(cc, keys.secretKey, cfg, outCpu);
        double diff   = 0;
        for (size_t i = 0; i < cpu.size(); ++i)
            diff = std::max(diff, std::abs(got[i] - cpu[i]));
        size_t same = 0, total = 0;
        for (size_t k = 0; k < out.size(); ++k)
            for (size_t q = 0; q < out[k].size(); ++q, ++total)
                same += (*out[k][q] == *outCpu[k][q]) ? 1 : 0;
        std::cout << std::scientific << std::setprecision(3) << "  max|m_GPU - m_CPU| = " << diff
                  << "   bit-identical ciphertexts: " << same << " / " << total << "\n";
    }
    double maxErr = 0, maxRef = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        maxErr = std::max(maxErr, std::abs(got[i] - want[i]));
        maxRef = std::max(maxRef, std::abs(want[i]));
    }
    std::cout << std::scientific << std::setprecision(3) << "  max|m_HE - m_ref| = " << maxErr
              << "   (max|m_ref| = " << maxRef << ", log2 rel err = " << std::fixed << std::setprecision(1)
              << std::log2(maxErr / maxRef) << ")\n";
    const bool ok = maxErr <= tol;
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
