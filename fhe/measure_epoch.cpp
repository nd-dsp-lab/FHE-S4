// STEP 1 -- measure the usable CKKS EPOCH after bootstrapping.
//
// EPOCH = the number of multiplicative levels a ciphertext has available between
// two bootstraps, at 128-bit security, on parameters we could deploy. Every depth
// estimate in this project so far (2 for the exp gate, ~10 for the scan, ~14 per
// block) is counted off an evaluation graph. This program is the first place a
// level is consumed by a real CKKS implementation.
//
// ONE CONFIGURATION PER PROCESS. fhe/run_epoch_sweep.py runs the sweep. Running
// each configuration in its own process means an out-of-memory kill during
// bootstrapping keygen is recorded by the driver as a result, instead of taking
// the rest of the sweep down with it. For the same reason the cheap search result
// is written to --out BEFORE the expensive keygen starts.
//
// What it does, numbered as in the STEP 1 brief:
//   1   CKKS context at HEStd_128_classic (or HEStd_NotSet with --insecure, for
//       smoke tests only -- such a row is never an EPOCH).
//   --  SEARCH: the largest levelsAvailableAfterBootstrap whose context the library
//       accepts at a FIXED ring dimension AND whose log2(QP) is within the
//       HE-standard bound for that ring dimension. Context generation only, no
//       keys, so this part costs seconds even at N = 2^17.
//   2   Bootstrapping setup + keys, at that maximum.
//   3   Encrypt a known real vector of full slot count, uniform in [-1, 1].
//   4,5 Level before and after one bootstrap.
//   6   Empirical check: multiply the refreshed ciphertext by an encryption of 1.0,
//       one ct x ct multiply at a time, decrypting after each, until the library
//       refuses. Reported separately: multiplies the library ACCEPTED, and
//       multiplies that still DECRYPT CORRECTLY. Those two can differ by one at the
//       last level, which is exactly why both are recorded.
//   7   Precision -log2(max abs error) after 1 bootstrap and after each of up to 5
//       consecutive bootstraps. Also the mean-abs version, which is what OpenFHE's
//       own iterative-ckks-bootstrapping example reports, so numbers can be
//       compared with the literature.
//   8   Peak host memory at each stage, and the serialized size of the evaluation
//       key material alone.
//   9   log2(Q), log2(P), log2(QP) and the bound, with the margin.
//  10   Wall-clock times, labelled CPU-ONLY / NON-TRANSFERABLE in the output.
//
// Every API used here was checked against OpenFHE v1.1.4 and v1.6.0 sources before
// being written; see fhe/EPOCH.md.

#include "openfhe.h"

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace lbcrypto;

// ----------------------------------------------------------------------------
// A flat JSON object, written by hand so the program depends on nothing but
// OpenFHE. Keys keep insertion order.
// ----------------------------------------------------------------------------
class Json {
public:
    void str(const std::string& k, const std::string& v) {
        kv_.emplace_back(k, "\"" + esc(v) + "\"");
    }
    void num(const std::string& k, double v) {
        kv_.emplace_back(k, fmt(v));
    }
    void integer(const std::string& k, long long v) {
        kv_.emplace_back(k, std::to_string(v));
    }
    void boolean(const std::string& k, bool v) {
        kv_.emplace_back(k, v ? "true" : "false");
    }
    void nums(const std::string& k, const std::vector<double>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i)
            s += (i ? ", " : "") + fmt(v[i]);
        kv_.emplace_back(k, s + "]");
    }
    void ints(const std::string& k, const std::vector<long long>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i)
            s += (i ? ", " : "") + std::to_string(v[i]);
        kv_.emplace_back(k, s + "]");
    }
    std::string dump() const {
        std::string s = "{\n";
        for (size_t i = 0; i < kv_.size(); ++i)
            s += "  \"" + esc(kv_[i].first) + "\": " + kv_[i].second + (i + 1 < kv_.size() ? ",\n" : "\n");
        return s + "}\n";
    }
    void write(const std::string& path) const {
        if (path.empty())
            return;
        std::ofstream f(path + ".tmp");
        f << dump();
        f.close();
        std::rename((path + ".tmp").c_str(), path.c_str());  // never leave a half-written row
    }

private:
    static std::string fmt(double v) {
        if (!std::isfinite(v))
            return "null";
        std::ostringstream o;
        o.precision(10);
        o << v;
        return o.str();
    }
    static std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            switch (c) {
                case '"':  o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        o += buf;
                    }
                    else {
                        o += c;
                    }
            }
        }
        return o;
    }
    std::vector<std::pair<std::string, std::string>> kv_;
};

// ----------------------------------------------------------------------------
// Peak resident set size. ru_maxrss is KILOBYTES on Linux and BYTES on macOS.
// It is a high-water mark, so "after keygen" includes keygen's temporaries:
// treat stage deltas as upper bounds, and use the serialized size for the keys.
// ----------------------------------------------------------------------------
static double peak_rss_mb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
}

// Counts bytes written and stores none, so measuring multi-GB key material does
// not itself need multi-GB of memory.
class CountingBuf : public std::streambuf {
public:
    unsigned long long n = 0;

protected:
    int_type overflow(int_type c) override {
        if (!traits_type::eq_int_type(c, traits_type::eof()))
            ++n;
        return traits_type::not_eof(c);
    }
    std::streamsize xsputn(const char*, std::streamsize k) override {
        n += static_cast<unsigned long long>(k);
        return k;
    }
};

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------
struct Cfg {
    uint32_t logN     = 16;
    uint32_t scale    = 50;  // scaling modulus bits (the "Delta" of CKKS)
    // Bits of the first prime q0. 0 means scale + 1, OpenFHE's own convention in
    // every bootstrapping example. A FIXED 60 was wrong: at scale 40 the q0/Delta
    // gap (20 bits) exceeds the bootstrap correction factor and the library throws,
    // and at scale 50 with a uniform secret it silently returned garbage
    // (-2.4 bits) without raising anything.
    uint32_t firstMod = 0;
    std::string skdist = "uniform";
    std::vector<uint32_t> budget = {4, 4};  // {CoeffsToSlots, SlotsToCoeffs} level budget
    int levelsAfter  = -1;                  // -1 = search for the maximum that respects 128-bit
    int maxSearch    = 80;
    int nBoots       = 5;
    int maxMults     = 80;
    double correctTol = 0.01;  // a decrypted slot counts as correct if |err| < this
    bool insecure    = false;
    bool searchOnly  = false;
    bool skipKeySize = false;
    uint32_t seed    = 12345;
    std::string out;
};

static SecretKeyDist dist_of(const Cfg& c) {
    if (c.skdist == "uniform")
        return UNIFORM_TERNARY;
    if (c.skdist == "sparse")
        return SPARSE_TERNARY;
    throw std::invalid_argument("--skdist must be 'uniform' or 'sparse', got '" + c.skdist + "'");
}

static std::vector<uint32_t> parse_budget(const std::string& s) {
    std::vector<uint32_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        v.push_back(static_cast<uint32_t>(std::stoul(tok)));
    if (v.size() != 2)
        throw std::invalid_argument("--budget must look like 4,4");
    return v;
}

static Cfg parse_args(int argc, char** argv) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next     = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--logn")              c.logN = std::stoul(next());
        else if (a == "--scale")        c.scale = std::stoul(next());
        else if (a == "--first-mod")    c.firstMod = std::stoul(next());
        else if (a == "--skdist")       c.skdist = next();
        else if (a == "--budget")       c.budget = parse_budget(next());
        else if (a == "--levels-after") c.levelsAfter = std::stoi(next());
        else if (a == "--max-search")   c.maxSearch = std::stoi(next());
        else if (a == "--boots")        c.nBoots = std::stoi(next());
        else if (a == "--max-mults")    c.maxMults = std::stoi(next());
        else if (a == "--correct-tol")  c.correctTol = std::stod(next());
        else if (a == "--seed")         c.seed = std::stoul(next());
        else if (a == "--out")          c.out = next();
        else if (a == "--insecure")     c.insecure = true;
        else if (a == "--search-only")  c.searchOnly = true;
        else if (a == "--skip-key-size") c.skipKeySize = true;
        else if (a == "-h" || a == "--help") {
            std::cout << "measure_epoch --logn 16 --scale 50 --skdist uniform|sparse --budget 4,4\n"
                         "              [--levels-after N] [--search-only] [--boots 5] [--out row.json]\n"
                         "              [--insecure]   (HEStd_NotSet: smoke tests only, never an EPOCH)\n";
            std::exit(0);
        }
        else
            throw std::invalid_argument("unknown argument " + a);
    }
    if (c.firstMod == 0)
        c.firstMod = c.scale + 1;
    if (c.insecure && c.levelsAfter < 0)
        throw std::invalid_argument("--insecure has no security bound to search against; give --levels-after");
    return c;
}

// ----------------------------------------------------------------------------
// Context construction and modulus accounting
// ----------------------------------------------------------------------------
static CryptoContext<DCRTPoly> make_cc(const Cfg& c, uint32_t depth) {
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(dist_of(c));
    p.SetSecurityLevel(c.insecure ? HEStd_NotSet : HEStd_128_classic);
    p.SetRingDim(1u << c.logN);  // FIXED: the library must refuse rather than grow N
    p.SetScalingModSize(c.scale);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    p.SetFirstModSize(c.firstMod);
    p.SetMultiplicativeDepth(depth);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
    return cc;
}

struct ModInfo {
    uint32_t ringDim  = 0;
    uint32_t logQ     = 0;  // bit length of Q = q0 * q1 * ... (the ciphertext modulus)
    uint32_t logP     = 0;  // bit length of P (the key-switching modulus)
    uint32_t nQ       = 0;
    uint32_t nP       = 0;
    uint32_t maxLogQP = 0;  // HE-standard maximum for this ring dimension, 128-bit classic
};

static ModInfo moduli(const CryptoContext<DCRTPoly>& cc) {
    ModInfo m;
    m.ringDim  = cc->GetRingDimension();
    auto elem  = cc->GetCryptoParameters()->GetElementParams();
    m.logQ     = elem->GetModulus().GetMSB();
    m.nQ       = static_cast<uint32_t>(elem->GetParams().size());
    auto rns   = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    if (rns && rns->GetParamsP()) {
        m.logP = rns->GetParamsP()->GetModulus().GetMSB();
        m.nP   = static_cast<uint32_t>(rns->GetParamsP()->GetParams().size());
    }
    // The HE-standard tables OpenFHE checks against are for a uniform ternary
    // secret. OpenFHE applies the same table to SPARSE_TERNARY; whether that bound
    // really gives 128-bit security for a sparse secret is NOT established here.
    m.maxLogQP = StdLatticeParm::FindMaxQ(HEStd_ternary, HEStd_128_classic, m.ringDim);
    return m;
}

// ----------------------------------------------------------------------------
// Precision of a decryption against the original vector
// ----------------------------------------------------------------------------
struct Prec {
    double maxAbs    = NAN;
    double meanAbs   = NAN;
    double bitsMax   = NAN;  // -log2(max abs error): the brief's definition
    double bitsMean  = NAN;  // -log2(mean abs error): OpenFHE's example's definition
    long long nWrong = 0;    // slots with |err| >= correctTol
};

static Prec precision(const std::vector<double>& truth, Plaintext& pt, double tol) {
    pt->SetLength(truth.size());
    const auto& v = pt->GetCKKSPackedValue();
    Prec r;
    double mx = 0.0, acc = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        double e = std::abs(v[i].real() - truth[i]);
        mx = std::max(mx, e);
        acc += e;
        if (!(e < tol))
            ++r.nWrong;
    }
    r.maxAbs   = mx;
    r.meanAbs  = acc / static_cast<double>(truth.size());
    r.bitsMax  = -std::log2(mx);
    r.bitsMean = -std::log2(r.meanAbs);
    return r;
}

static long long levels_remaining(const Ciphertext<DCRTPoly>& ct, uint32_t depth) {
    // Same expression as OpenFHE's simple-ckks-bootstrapping example. The
    // NoiseScaleDeg term matters under FLEXIBLEAUTO: after a multiply, a rescale is
    // owed and that level is already spoken for.
    return static_cast<long long>(depth) - static_cast<long long>(ct->GetLevel()) -
           (static_cast<long long>(ct->GetNoiseScaleDeg()) - 1);
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Json row;
    Cfg c;
    try {
        c = parse_args(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << "argument error: " << e.what() << "\n";
        return 2;
    }

    const uint32_t N = 1u << c.logN;
    row.str("step", "STEP 1 -- usable CKKS epoch after bootstrapping");
    row.integer("logN", c.logN);
    row.integer("ring_dim_requested", N);
    row.integer("scaling_mod_bits", c.scale);
    row.integer("first_mod_bits", c.firstMod);
    row.str("secret_key_dist", c.skdist);
    row.ints("level_budget", {static_cast<long long>(c.budget[0]), static_cast<long long>(c.budget[1])});
    row.str("scaling_technique", "FLEXIBLEAUTO");
    row.str("security_level", c.insecure ? "HEStd_NotSet (SMOKE TEST -- NOT AN EPOCH)" : "HEStd_128_classic");
    row.integer("native_int_bits", NATIVEINT);
#ifdef _OPENMP
    row.integer("omp_max_threads", omp_get_max_threads());
#else
    row.integer("omp_max_threads", 1);
#endif
    row.str("timing_label", "CPU-ONLY, NON-TRANSFERABLE: says nothing about GPU latency");
    row.str("status", "started");
    row.write(c.out);

    try {
        const SecretKeyDist skd = dist_of(c);
        const uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(c.budget, skd);
        row.integer("bootstrap_depth", bootDepth);

        // ------------------------------------------------------------ SEARCH
        // Linear, not binary, so the whole ladder is visible: the first rejection
        // is the answer, and its message says which bound was hit.
        int epoch = c.levelsAfter;
        std::string firstRejection;
        std::vector<long long> ladderLogQP;
        if (c.levelsAfter < 0) {
            auto t0 = Clock::now();
            epoch   = 0;
            for (int L = 1; L <= c.maxSearch; ++L) {
                bool ok = false;
                std::string why;
                try {
                    auto cc   = make_cc(c, static_cast<uint32_t>(L) + bootDepth);
                    ModInfo m = moduli(cc);
                    ladderLogQP.push_back(static_cast<long long>(m.logQ + m.logP));
                    if (m.ringDim != N)
                        why = "library raised ring dimension to " + std::to_string(m.ringDim);
                    else if (m.maxLogQP == 0)
                        why = "no HE-standard row for ring dimension " + std::to_string(m.ringDim);
                    else if (m.logQ + m.logP > m.maxLogQP)
                        why = "log2(QP) = " + std::to_string(m.logQ + m.logP) + " exceeds bound " +
                              std::to_string(m.maxLogQP);
                    else
                        ok = true;
                }
                catch (const std::exception& e) {
                    why = std::string("library refused: ") + e.what();
                }
                CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
                if (!ok) {
                    firstRejection = "levelsAfter=" + std::to_string(L) + ": " + why;
                    break;
                }
                epoch = L;
            }
            row.num("search_seconds_cpu_only", secs_since(t0));
            row.ints("search_ladder_logQP", ladderLogQP);
            row.str("search_first_rejection", firstRejection);
            if (epoch == 0)
                throw std::runtime_error("no levelsAfter >= 1 fits at this ring dimension: " + firstRejection);
            if (firstRejection.empty())
                row.str("search_warning", "hit --max-search without a rejection; EPOCH may be larger");
        }
        row.integer("epoch_declared", epoch);
        const uint32_t depth = static_cast<uint32_t>(epoch) + bootDepth;
        row.integer("multiplicative_depth_total", depth);

        // Modulus accounting at the chosen configuration.
        {
            auto cc   = make_cc(c, depth);
            ModInfo m = moduli(cc);
            row.integer("ring_dim_actual", m.ringDim);
            row.integer("log2_Q", m.logQ);
            row.integer("log2_P", m.logP);
            row.integer("log2_QP", m.logQ + m.logP);
            row.integer("num_Q_primes", m.nQ);
            row.integer("num_P_primes", m.nP);
            row.integer("max_log2_QP_128bit", m.maxLogQP);
            row.integer("modulus_margin_bits", static_cast<long long>(m.maxLogQP) - (m.logQ + m.logP));
            CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
        }
        row.str("status", c.searchOnly ? "search_only_done" : "search_done");
        row.write(c.out);
        if (c.searchOnly) {
            std::cout << row.dump();
            return 0;
        }

        // ------------------------------------------------------------ KEYS
        const double rssBase = peak_rss_mb();
        auto t0              = Clock::now();
        auto cc              = make_cc(c, depth);
        const uint32_t slots = cc->GetRingDimension() / 2;  // full packing
        cc->EvalBootstrapSetup(c.budget, {0, 0}, slots);
        row.num("time_bootstrap_setup_s_cpu_only", secs_since(t0));
        const double rssSetup = peak_rss_mb();

        t0      = Clock::now();
        auto kp = cc->KeyGen();
        cc->EvalMultKeyGen(kp.secretKey);
        cc->EvalBootstrapKeyGen(kp.secretKey, slots);
        row.num("time_keygen_s_cpu_only", secs_since(t0));
        const double rssKeys = peak_rss_mb();
        row.integer("slots", slots);
        row.num("peak_rss_mb_before_setup", rssBase);
        row.num("peak_rss_mb_after_bootstrap_setup", rssSetup);
        row.num("peak_rss_mb_after_keygen", rssKeys);

        if (!c.skipKeySize) {
            CountingBuf mb, ab;
            std::ostream mos(&mb), aos(&ab);
            bool okM = cc->SerializeEvalMultKey(mos, SerType::BINARY);
            bool okA = cc->SerializeEvalAutomorphismKey(aos, SerType::BINARY);
            row.boolean("key_serialize_ok", okM && okA);
            row.num("key_bytes_relin_mb", mb.n / (1024.0 * 1024.0));
            row.num("key_bytes_rotation_mb", ab.n / (1024.0 * 1024.0));
            row.num("key_bytes_total_mb", (mb.n + ab.n) / (1024.0 * 1024.0));
        }
        row.str("status", "keys_done");
        row.write(c.out);

        // ------------------------------------------------------------ DATA
        std::mt19937_64 rng(c.seed);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        std::vector<double> x(slots);
        for (auto& v : x)
            v = U(rng);

        // Encrypt with ONE level left, i.e. a ciphertext that needs refreshing.
        Plaintext pt = cc->MakeCKKSPackedPlaintext(x, 1, depth - 1);
        auto ct      = cc->Encrypt(kp.publicKey, pt);
        row.integer("levels_before_bootstrap", levels_remaining(ct, depth));
        {
            // Baseline: the SAME vector, encrypted and decrypted with no bootstrap.
            // If this is already bad, the fault is in encoding or in this
            // program's measurement -- not in bootstrapping.
            Plaintext o;
            cc->Decrypt(kp.secretKey, ct, &o);
            Prec p0 = precision(x, o, c.correctTol);
            row.num("precision_bits_max_fresh_no_boot", p0.bitsMax);
            row.num("precision_bits_mean_fresh_no_boot", p0.bitsMean);
        }

        // ------------------------------------------------------------ BOOTSTRAP x1
        t0      = Clock::now();
        auto b1 = cc->EvalBootstrap(ct);
        row.num("time_one_bootstrap_s_cpu_only", secs_since(t0));
        const long long declared = levels_remaining(b1, depth);
        row.integer("levels_after_bootstrap_declared", declared);
        row.num("peak_rss_mb_after_one_bootstrap", peak_rss_mb());

        Plaintext out;
        cc->Decrypt(kp.secretKey, b1, &out);
        Prec p1 = precision(x, out, c.correctTol);
        row.num("precision_bits_max_after_1_boot", p1.bitsMax);
        row.num("precision_bits_mean_after_1_boot", p1.bitsMean);
        row.num("max_abs_err_after_1_boot", p1.maxAbs);

        // ------------------------------------------------------------ BOOTSTRAP xK
        // No computation in between. A 24-layer model refreshes at least 24 times;
        // the question is whether precision decays across refreshes.
        std::vector<double> bootBitsMax{p1.bitsMax}, bootBitsMean{p1.bitsMean};
        std::string bootErr;
        {
            auto cur = b1;
            for (int k = 2; k <= c.nBoots; ++k) {
                try {
                    cur = cc->EvalBootstrap(cur);
                    Plaintext o;
                    cc->Decrypt(kp.secretKey, cur, &o);
                    Prec pk = precision(x, o, c.correctTol);
                    bootBitsMax.push_back(pk.bitsMax);
                    bootBitsMean.push_back(pk.bitsMean);
                }
                catch (const std::exception& e) {
                    bootErr = "bootstrap #" + std::to_string(k) + ": " + e.what();
                    break;
                }
            }
        }
        row.nums("precision_bits_max_per_bootstrap", bootBitsMax);
        row.nums("precision_bits_mean_per_bootstrap", bootBitsMean);
        if (!bootErr.empty())
            row.str("repeated_bootstrap_error", bootErr);
        row.num("precision_bits_max_after_last_boot", bootBitsMax.back());
        row.integer("bootstraps_completed", static_cast<long long>(bootBitsMax.size()));
        row.str("status", "bootstraps_done");
        row.write(c.out);

        // ------------------------------------------------------------ EMPIRICAL EPOCH
        // Multiply by an encryption of 1.0: a genuine ct x ct multiply (relinearised,
        // one level consumed) that leaves the VALUE unchanged, so every step can be
        // checked against x. Squaring would also consume a level per step, but would
        // drive |x|^(2^k) to zero and make "still correct" meaningless.
        std::vector<double> ones(slots, 1.0);
        auto onesCt = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(ones));
        long long accepted = 0, correct = 0;
        bool stillCorrect  = true;
        std::string refusal;
        std::vector<double> stepBits;
        std::vector<long long> stepLevels;
        {
            auto cur = b1;
            for (int s = 1; s <= c.maxMults; ++s) {
                try {
                    cur = cc->EvalMult(cur, onesCt);
                }
                catch (const std::exception& e) {
                    refusal = std::string("EvalMult #") + std::to_string(s) + ": " + e.what();
                    break;
                }
                ++accepted;
                stepLevels.push_back(levels_remaining(cur, depth));
                try {
                    Plaintext o;
                    cc->Decrypt(kp.secretKey, cur, &o);
                    Prec ps = precision(x, o, c.correctTol);
                    stepBits.push_back(ps.bitsMax);
                    if (stillCorrect && ps.nWrong == 0)
                        ++correct;
                    else
                        stillCorrect = false;
                }
                catch (const std::exception& e) {
                    stepBits.push_back(NAN);
                    stillCorrect = false;
                    if (refusal.empty())
                        refusal = std::string("Decrypt after EvalMult #") + std::to_string(s) + ": " + e.what();
                }
            }
        }
        if (refusal.empty())
            refusal = "no refusal within --max-mults";
        row.integer("empirical_mults_accepted", accepted);
        row.integer("empirical_mults_correct", correct);
        row.num("correct_tolerance_abs", c.correctTol);
        row.str("empirical_stop_reason", refusal);
        row.nums("empirical_precision_bits_max_per_mult", stepBits);
        row.ints("empirical_levels_remaining_per_mult", stepLevels);
        row.boolean("declared_equals_empirical_correct", declared == correct);
        row.boolean("declared_equals_empirical_accepted", declared == accepted);

        row.num("peak_rss_mb_final", peak_rss_mb());
        row.str("status", "ok");
        row.write(c.out);
        std::cout << row.dump();
        cc->ClearStaticMapsAndVectors();
        return 0;
    }
    catch (const std::exception& e) {
        row.str("status", "error");
        row.str("error", e.what());
        row.num("peak_rss_mb_at_error", peak_rss_mb());
        row.write(c.out);
        std::cout << row.dump();
        return 3;
    }
}
