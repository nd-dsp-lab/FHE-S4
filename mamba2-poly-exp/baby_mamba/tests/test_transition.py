"""Tests that the baby transition really is the Mamba-2 transition.

Run:  python -m pytest baby_mamba/tests -q
   or python -m unittest discover -s baby_mamba/tests
"""

import math
import unittest

import torch
import torch.nn.functional as F

from baby_mamba.polynomial import build_poly
from baby_mamba.transition import (
    BabyConfig,
    BabyMamba2Transition,
    ExactExp,
    reference_recurrence,
)


class TestShapesAndSigns(unittest.TestCase):
    def setUp(self):
        self.cfg = BabyConfig(batch=2, seqlen=8)
        self.model = BabyMamba2Transition(self.cfg, ExactExp(), seed=0)
        self.u = torch.randn(2, 8, self.cfg.d_model,
                             generator=torch.Generator().manual_seed(1))
        self.out = self.model(self.u)

    def test_shapes_match_mamba2_semantics(self):
        c = self.cfg
        self.assertEqual(tuple(self.out["delta"].shape), (c.batch, c.seqlen, c.nheads))
        self.assertEqual(tuple(self.out["A"].shape), (c.nheads,))
        self.assertEqual(tuple(self.out["z"].shape), (c.batch, c.seqlen, c.nheads))
        self.assertEqual(tuple(self.out["a"].shape), (c.batch, c.seqlen, c.nheads))
        self.assertEqual(tuple(self.out["h"].shape),
                         (c.batch, c.seqlen, c.nheads, c.headdim, c.d_state))
        self.assertEqual(tuple(self.out["y"].shape),
                         (c.batch, c.seqlen, c.nheads, c.headdim))

    def test_A_is_one_scalar_per_head_and_negative(self):
        """The single most important shape fact: A is (nheads,), not (d_inner, d_state)."""
        self.assertEqual(self.out["A"].numel(), self.cfg.nheads)
        self.assertTrue((self.out["A"] < 0).all())

    def test_delta_positive_and_z_nonpositive(self):
        self.assertTrue((self.out["delta"] > 0).all())
        self.assertTrue((self.out["z"] <= 0).all())

    def test_exact_a_in_unit_interval(self):
        """exp(z) for z <= 0 lies in (0, 1]. This is what keeps the scan stable."""
        a = self.out["a"]
        self.assertTrue((a > 0).all())
        self.assertTrue((a <= 1.0 + 1e-6).all())

    def test_A_equals_minus_exp_A_log(self):
        torch.testing.assert_close(self.model.A(), -torch.exp(self.model.A_log.float()))

    def test_dt_bias_init_is_softplus_inverse(self):
        """mamba2.py:118-127 stores softplus^-1(dt), so softplus(dt_bias) must land
        back inside [dt_min, dt_max]."""
        dt = F.softplus(self.model.dt_bias)
        self.assertTrue((dt >= self.cfg.dt_min * 0.999).all())
        self.assertTrue((dt <= self.cfg.dt_max * 1.001).all())


class TestRecurrence(unittest.TestCase):
    def test_forward_matches_standalone_recurrence(self):
        cfg = BabyConfig(batch=2, seqlen=16)
        m = BabyMamba2Transition(cfg, ExactExp(), seed=3)
        u = torch.randn(2, 16, cfg.d_model, generator=torch.Generator().manual_seed(4))
        out = m(u)
        h_ref = reference_recurrence(out["a"], out["b"])
        torch.testing.assert_close(out["h"], h_ref)

    def test_recurrence_is_the_closed_form_sum(self):
        """h_t = sum_{s<=t} (prod_{k=s+1..t} a_k) * b_s, checked by brute force."""
        cfg = BabyConfig(batch=1, seqlen=6)
        m = BabyMamba2Transition(cfg, ExactExp(), seed=5)
        u = torch.randn(1, 6, cfg.d_model, generator=torch.Generator().manual_seed(6))
        out = m(u)
        a, b, h = out["a"], out["b"], out["h"]
        for t in range(cfg.seqlen):
            acc = torch.zeros_like(b[:, 0])
            for s in range(t + 1):
                decay = torch.ones_like(a[:, 0])
                for k in range(s + 1, t + 1):
                    decay = decay * a[:, k]
                acc = acc + decay[:, :, None, None] * b[:, s]
            torch.testing.assert_close(h[:, t], acc, rtol=1e-5, atol=1e-6)

    def test_a_equal_one_means_pure_accumulation(self):
        """Sanity check on the semantics: a == 1 must turn the scan into a cumsum."""
        cfg = BabyConfig(batch=1, seqlen=5)
        m = BabyMamba2Transition(cfg, ExactExp(), seed=7)
        u = torch.randn(1, 5, cfg.d_model, generator=torch.Generator().manual_seed(8))
        out = m(u)
        h_ones = reference_recurrence(torch.ones_like(out["a"]), out["b"])
        torch.testing.assert_close(h_ones, out["b"].cumsum(dim=1))

    def test_a_equal_zero_means_pure_forgetting(self):
        cfg = BabyConfig(batch=1, seqlen=5)
        m = BabyMamba2Transition(cfg, ExactExp(), seed=9)
        u = torch.randn(1, 5, cfg.d_model, generator=torch.Generator().manual_seed(10))
        out = m(u)
        h_zeros = reference_recurrence(torch.zeros_like(out["a"]), out["b"])
        torch.testing.assert_close(h_zeros, out["b"])


class TestSwapIsTheOnlyChange(unittest.TestCase):
    def test_only_a_differs_when_swapping_transition(self):
        """Swapping the transition must leave delta, A, z, b, x, B, C untouched.

        If this test ever fails, the polynomial experiment has been contaminated
        by a change to something other than exp.
        """
        cfg = BabyConfig(batch=2, seqlen=12)
        u = torch.randn(2, 12, cfg.d_model, generator=torch.Generator().manual_seed(11))
        exact = BabyMamba2Transition(cfg, ExactExp(), seed=12)(u)
        poly = BabyMamba2Transition(cfg, build_poly(4, -8, 0), seed=12)(u)
        for key in ("delta", "A", "z", "b", "x", "B", "C", "gate_z"):
            torch.testing.assert_close(exact[key], poly[key], msg=f"{key} changed!")
        self.assertFalse(torch.allclose(exact["a"], poly["a"]))

    def test_poly_that_is_exp_reproduces_exp(self):
        """A high-degree Chebyshev fit is near-exact (max err 8e-8 on [-8, 0]), so
        if the polynomial machinery itself were buggy this test would catch it.
        Any error we report later therefore comes from the LOW degree, not from
        the plumbing."""
        cfg = BabyConfig(batch=2, seqlen=12)
        u = torch.randn(2, 12, cfg.d_model, generator=torch.Generator().manual_seed(13))
        with torch.no_grad():
            exact = BabyMamba2Transition(cfg, ExactExp(), seed=14)(u)
            good = build_poly(12, -8.0, 0.0)
            got = BabyMamba2Transition(cfg, good, seed=14)(u)
        zmin = float(exact["z"].min())
        self.assertLess(zmin, 0.0)
        self.assertGreater(zmin, -8.0)                  # inside the fit interval
        # The floor here is float32 COEFFICIENT storage, not the evaluation:
        # PolyExp keeps coefficients in fp32, which caps a degree-12 fit at about
        # 1e-5 no matter how well numpy fitted it. That is fine -- 1e-5 is three
        # orders of magnitude below the degree-4 error we actually care about.
        torch.testing.assert_close(exact["a"], got["a"], rtol=0, atol=1e-4)
        torch.testing.assert_close(exact["h"], got["h"], rtol=0, atol=1e-4)


class TestDeltaScaleKnob(unittest.TestCase):
    def test_delta_scale_pushes_z_more_negative(self):
        u = torch.randn(2, 32, 4, generator=torch.Generator().manual_seed(15))
        z1 = BabyMamba2Transition(BabyConfig(batch=2, seqlen=32), ExactExp(), seed=16)(u)["z"]
        z4 = BabyMamba2Transition(BabyConfig(batch=2, seqlen=32, delta_scale=4.0),
                                  ExactExp(), seed=16)(u)["z"]
        torch.testing.assert_close(z4, z1 * 4.0)


if __name__ == "__main__":
    unittest.main()
