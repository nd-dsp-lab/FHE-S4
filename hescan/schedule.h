// Prefix-scan schedules over affine elements, shared by the CPU kernel, the GPU kernel and the
// depth simulation in Config::Depth() so the three cannot drift apart.
#pragma once

#include "hescan.h"

#include <cstdint>
#include <vector>

namespace hescan::detail {

// `comb(early, late)` returns late o early. With `parallel`, independent combines inside one
// scan level run as an OpenMP loop (CPU backend). The GPU backend passes false: one FIDESlib
// context shares key-switching scratch buffers, so its calls must stay on one host thread.
template <class T, class F>
void RunSchedule(std::vector<T>& xs, ScanAlgo algo, F comb, bool parallel = true) {
    const int64_t L = static_cast<int64_t>(xs.size());
    (void)parallel;
    switch (algo) {
        case ScanAlgo::Sequential:
            for (int64_t i = 1; i < L; ++i)
                xs[i] = comb(xs[i - 1], xs[i]);
            break;
        case ScanAlgo::HillisSteele:
            for (int64_t d = 1; d < L; d *= 2) {
                std::vector<T> next(xs);
#pragma omp parallel for schedule(dynamic) if (parallel)
                for (int64_t i = d; i < L; ++i)
                    next[i] = comb(xs[i - d], xs[i]);
                xs.swap(next);
            }
            break;
        case ScanAlgo::BrentKung: {
            int64_t dmax = 1;
            // up-sweep: xs[i] becomes the summary of a 2d-aligned block
            for (int64_t d = 1; 2 * d <= L; d *= 2) {
                dmax = d;
#pragma omp parallel for schedule(dynamic) if (parallel)
                for (int64_t i = 2 * d - 1; i < L; i += 2 * d)
                    xs[i] = comb(xs[i - d], xs[i]);
            }
            // down-sweep: fill in the remaining prefixes
            for (int64_t d = dmax; d >= 1; d /= 2) {
#pragma omp parallel for schedule(dynamic) if (parallel)
                for (int64_t i = 3 * d - 1; i < L; i += 2 * d)
                    xs[i] = comb(xs[i - d], xs[i]);
            }
            break;
        }
    }
}

}  // namespace hescan::detail
