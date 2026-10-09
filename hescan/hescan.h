// HEScan: factorized encrypted selective-scan kernel (FESC, arXiv 2608.17442) on OpenFHE CKKS.
//
// Logical recurrence and output contraction (FESC Eq. 2-3):
//
//     h_k = br(a_k) (.) h_{k-1} + x_k (x)_g B_k          h_k in R^{E x d_s},  E = H*P
//     m_k = Sum_{d_s}( h_k (.) br(C_k) )                 m_k in R^{H x P}
//
// Per token the MPC side hands over only the compact factorized packet
// (x_k in R^E, a_k in R^H, B_k, C_k in R^{G x d_s}); HEScan never ships expanded
// state across the boundary and returns only the contracted m_k.
//
// The scan composes affine maps  T_k : h -> A_k (.) h + U_k  with
// A_k = br(a_k),  U_k = br(x_k) (.) br(B_k).  Composition (early, then late):
//
//     A_{late o early} = A_late (.) A_early                       (Eq. 5, decay term)
//     U_{late o early} = A_late (.) U_early + U_late              (Eq. 6, state term)
//
// Each composition is one ct-ct multiplicative level, so a Brent-Kung prefix scan gives
// O(L) work and ~2 log2 L depth instead of the sequential chain's L.
//
// Slot layout of one state chunk (s_state slots, c_state = s_state / d_s channels):
//     slot(c, n) = c * d_s + n,   global channel e = j * c_state + c,  K_s = E*d_s / s_state chunks.
//
// Packet "seed" layouts (zeros everywhere else); br() replicates a seed with
// hoisted rotations of one ciphertext:
//     x seed : x_e         at slot c*d_s                    br: stride 1,   count d_s
//     a seed : a_head(e)   at slot c*d_s, c % runA == 0     br: stride 1,   count runA*d_s
//     B/C seed: B_{g(e),n} at slot c*d_s+n, c % runG == 0   br: stride d_s, count runG
// with runA = min(P, c_state), runG = min(E/G, c_state).
//
// Complex pairing: chunks 2q and 2q+1 travel as one ciphertext z = u + i v (halving
// boundary ciphertexts). One conjugation splits them: u = (z + conj z)/2, v = -i/2 (z - conj z).
// Outputs are re-paired as m_{2q} + i m_{2q+1} through the output mask (a plaintext mult).
#pragma once

#include "openfhe.h"

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace hescan {

using Ct   = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using CC   = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using cplx = std::complex<double>;

enum class ScanAlgo { Sequential, HillisSteele, BrentKung };

ScanAlgo ParseScanAlgo(const std::string& s);
const char* ScanAlgoName(ScanAlgo a);

struct Config {
    uint32_t H      = 4;   // heads
    uint32_t P      = 4;   // head dim
    uint32_t G      = 2;   // B/C groups (G divides H)
    uint32_t ds     = 8;   // state dim d_s
    uint32_t L      = 16;  // tokens scanned in one block
    uint32_t sState = 64;  // active slots per state chunk (s_state)
    uint32_t radix  = 2;   // br() rotations hoisted per stage (2 = doubling; larger = fewer stages)
    ScanAlgo algo   = ScanAlgo::BrentKung;
    bool complexPack = true;   // pair chunks in real/imag parts across the boundary
    bool withH0      = false;  // carry an encrypted initial state h_0 into the block
    bool maskOutput  = true;   // zero out non-output slots of m_k (always on with complexPack)

    uint32_t E() const { return H * P; }
    uint32_t cState() const { return sState / ds; }
    uint32_t Ks() const { return (E() * ds + sState - 1) / sState; }
    uint32_t runA() const { return std::min(P, cState()); }
    uint32_t runG() const { return std::min(E() / G, cState()); }
    // ciphertexts per field that cross the hybrid boundary per token
    uint32_t BoundaryCts() const { return complexPack ? (Ks() + 1) / 2 : Ks(); }

    void Validate(uint32_t maxSlots) const;
    // multiplicative depth HEScan consumes (exact for the chosen scan schedule)
    uint32_t Depth() const;
    std::vector<int32_t> RotationIndices() const;
};

// Plaintext packet for one block of L tokens (row-major).
struct PlainPacket {
    std::vector<double> x;   // L x E
    std::vector<double> a;   // L x H
    std::vector<double> B;   // L x G x ds
    std::vector<double> C;   // L x G x ds
    std::vector<double> h0;  // E x ds (only if withH0)
};

// Encrypted packet: field[k][q], q over boundary ciphertexts (chunk pairs if complexPack).
struct EncPacket {
    std::vector<std::vector<Ct>> x, a, B, C;
    std::vector<Ct> h0;  // per boundary ciphertext, full state layout (no br needed)
};

// ---- MPC / client side helpers (plaintext layout + encryption + decoding) ----
EncPacket EncryptPacket(const CC& cc, const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk, const Config& cfg,
                        const PlainPacket& pkt);
// out[k][q] -> m (L x E)
std::vector<double> DecryptOutput(const CC& cc, const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk,
                                  const Config& cfg, const std::vector<std::vector<Ct>>& out);
// Runs the scan schedule on scalar affine maps and compares with a sequential fold.
bool CheckSchedule(ScanAlgo algo, uint32_t L);
// Eq. 2-3 evaluated directly in double precision.
std::vector<double> PlainReference(const Config& cfg, const PlainPacket& pkt);

// ---- HE side ----
struct Timings {
    double broadcast = 0;  // complex split + br(), wall seconds
    double scan      = 0;  // prefix scan (+ h_0 term)
    double contract  = 0;  // Sum_{d_s} contraction + output mask / re-pairing
};

class HEScan {
public:
    HEScan(CC cc, Config cfg);

    // Runs the full kernel; returns out[k][q] (paired if complexPack).
    std::vector<std::vector<Ct>> Run(const EncPacket& in, Timings* t = nullptr) const;

    // br(): replicate a seed `count` times at multiples of `stride` (count a power of two).
    Ct Replicate(const Ct& ct, uint32_t stride, uint32_t count) const;
    // Sum_{d_s}: segmented rotate-and-add reduction tree; result at slot c*d_s.
    Ct ReduceDs(const Ct& ct) const;

private:
    struct Elem {
        Ct A, U;
    };
    std::vector<Ct> Unpack(const std::vector<Ct>& boundary) const;
    void Scan(std::vector<Elem>& xs) const;
    Elem Compose(const Elem& early, const Elem& late) const;

    CC cc_;
    Config cfg_;
    uint32_t m_;  // cyclotomic order 2N
    lbcrypto::Plaintext mask_, maskI_;
};

}  // namespace hescan
