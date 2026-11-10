// HEScan on the GPU, through FIDESlib (https://github.com/CAPS-UMU/FIDESlib).
//
// OpenFHE stays the client/MPC side: it builds the context, generates every key, encrypts the
// packet and decrypts the result. HEScanGPU::Run takes the same OpenFHE EncPacket as
// HEScan::Run and returns OpenFHE ciphertexts in the same layout. In between, everything runs
// on the device: the packet is copied up once per token, and only the outputs come back.
//
// The kernel issues the same operation sequence as the CPU HEScan, so Config::Depth() holds
// unchanged. FIDESlib mirrors OpenFHE's FLEXIBLEAUTO / FIXEDAUTO level handling, so no explicit
// rescales are needed.
//
// FIDESlib links a patched OpenFHE 1.6.1 (static). Build with -DHESCAN_GPU=ON; see README.md.
// The FIDESlib headers are kept out of this header so callers need no CUDA include paths.
#pragma once

#include "hescan.h"

#include <memory>
#include <vector>

namespace hescan {

class HEScanGPU {
public:
    // `cc` must already hold the relinearization key, the rotation keys for
    // cfg.RotationIndices() and, if cfg.complexPack, the conjugation key (automorphism 2N-1);
    // they are copied to device `device` here.
    HEScanGPU(CC cc, Config cfg, const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk, int device = 0);
    ~HEScanGPU();
    HEScanGPU(const HEScanGPU&)            = delete;
    HEScanGPU& operator=(const HEScanGPU&) = delete;

    // Runs the full kernel; returns out[k][q] (paired if complexPack), exactly like HEScan::Run.
    std::vector<std::vector<Ct>> Run(const EncPacket& in, Timings* t = nullptr) const;

    // Device memory currently held by FIDESlib on `device`, in bytes (0 if unknown).
    static size_t DeviceMemoryUsed(int device = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hescan
