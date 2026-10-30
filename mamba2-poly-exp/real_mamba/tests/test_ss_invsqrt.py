"""The plaintext OpenFHE-pipeline 1/sqrt must reproduce what bench.cpp measured.

Run:  python -m pytest real_mamba/tests/test_ss_invsqrt.py -q
"""

import unittest

import torch

from real_mamba.ss_invsqrt import SchemeSwitchInvSqrt

# bench.cpp's default boundaries, and fhe/seglut_invsqrt/logs/std128_final.log
TAU = [0.0012574287695376616, 0.07583814844682049, 4.573956711644746,
       275.86485731083445, 16637.984199846484]
STD128_REL_BITS = {0: 5.54, 1: 9.20, 2: 15.24}


class TestAgainstBenchmark(unittest.TestCase):
    def test_std128_rel_bits(self):
        # 9.7 = the worst per-segment noise the benchmark measured
        m = SchemeSwitchInvSqrt(TAU, index_noise=9.7, phase_noise=9.7)
        for steps, want in STD128_REL_BITS.items():
            got = m.precision(per_segment=4096, newton=steps)["rel_bits"][0]
            # 128 samples in the benchmark: its RMS after one step sits ~0.5 bit low
            self.assertAlmostEqual(got, want, delta=0.6, msg=f"{steps} Newton steps")

    def test_noiseless_table_is_the_table(self):
        m = SchemeSwitchInvSqrt(TAU, index_noise=0, phase_noise=0, ckks_bits=None)
        self.assertGreater(m.precision(newton=0)["rel_bits"][0], 10.0)

    def test_past_the_guard_is_garbage(self):
        # negacyclic: the phase is negated, so the value is reflected about the
        # table's midpoint -- far too large -- and Newton then diverges from it
        m = SchemeSwitchInvSqrt(TAU, index_noise=0, phase_noise=0, ckks_bits=None)
        m.track = True
        v = torch.tensor([2.0 * TAU[-1], 0.5 * TAU[-1]], dtype=torch.float64)
        rel0 = m(v, newton=0) * v.sqrt() - 1
        self.assertGreater(float(rel0[0]), 5.0)
        self.assertLess(abs(float(rel0[1])), 0.05)
        self.assertEqual(m.n_wrap, 1)
        rel2 = m(v, newton=2) * v.sqrt() - 1
        self.assertGreater(abs(float(rel2[0])), 1.0)
        self.assertLess(abs(float(rel2[1])), 1e-3)


if __name__ == "__main__":
    unittest.main()
