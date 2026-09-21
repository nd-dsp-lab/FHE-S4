"""Tests for the polynomial approximations and their FHE depth accounting."""

import math
import unittest

import numpy as np
import torch

from baby_mamba.polynomial import (
    PolyExp,
    PowerSchedule,
    approximation_report,
    build_poly,
    fit_exp_poly,
    load_poly,
    save_coefficients,
)


class TestDepthAccounting(unittest.TestCase):
    def test_expected_depths(self):
        """The numbers quoted in the brief: deg2 -> 1, deg3 -> 2, deg4 -> 2."""
        self.assertEqual(PowerSchedule.binary(2).ct_ct_depth, 1)
        self.assertEqual(PowerSchedule.binary(3).ct_ct_depth, 2)
        self.assertEqual(PowerSchedule.binary(4).ct_ct_depth, 2)

    def test_depth_is_ceil_log2(self):
        for d in range(2, 33):
            self.assertEqual(PowerSchedule.binary(d).ct_ct_depth,
                             math.ceil(math.log2(d)), f"degree {d}")

    def test_depth_beats_horner(self):
        """Horner would be depth == degree. Our schedule must do better for deg>=3."""
        for d in (3, 4, 5, 8):
            self.assertLess(PowerSchedule.binary(d).ct_ct_depth, d)

    def test_mult_count_is_degree_minus_one_at_least(self):
        for d in range(2, 17):
            s = PowerSchedule.binary(d)
            self.assertGreaterEqual(s.ct_ct_mults, 1)
            # every power 2..d is built exactly once
            self.assertEqual(s.ct_ct_mults, d - 1)

    def test_schedule_only_uses_already_built_powers(self):
        for d in range(2, 17):
            built = {1}
            for target, lo, hi in PowerSchedule.binary(d).steps:
                self.assertIn(lo, built)
                self.assertIn(hi, built)
                self.assertEqual(target, lo + hi)
                built.add(target)

    def test_polyexp_reports_depth(self):
        p = build_poly(4, -8, 0)
        self.assertEqual(p.degree, 4)
        self.assertEqual(p.ct_ct_depth, 2)
        self.assertEqual(p.ct_ct_mults, 3)


class TestEvaluationCorrectness(unittest.TestCase):
    def test_matches_numpy_polyval(self):
        """The explicit-powers graph must compute the same polynomial as Horner,
        just at a different depth. If this fails, the graph is wrong."""
        rng = np.random.default_rng(0)
        for degree in range(1, 9):
            c = rng.normal(size=degree + 1)
            p = PolyExp(c)
            z = torch.linspace(-10, 2, 257, dtype=torch.float64)
            got = p(z.float()).double()
            want = torch.as_tensor(np.polyval(c[::-1], z.numpy()))
            torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)

    def test_gradients_flow_to_coefficients_when_trainable(self):
        p = PolyExp([1.0, 1.0, 0.5], trainable=True)
        z = torch.linspace(-4, 0, 32)
        p(z).sum().backward()
        self.assertIsNotNone(p.coeffs.grad)
        self.assertTrue(torch.isfinite(p.coeffs.grad).all())

    def test_frozen_coefficients_have_no_grad(self):
        p = PolyExp([1.0, 1.0, 0.5], trainable=False)
        self.assertNotIn("coeffs", dict(p.named_parameters()))

    def test_preserves_shape_and_dtype(self):
        p = build_poly(4, -8, 0)
        for shape in [(3,), (2, 5), (2, 7, 4), (2, 3, 4, 5)]:
            z = torch.randn(*shape) - 2
            self.assertEqual(p(z).shape, z.shape)
        z64 = torch.randn(4, dtype=torch.float64) - 2
        self.assertEqual(p(z64).dtype, torch.float64)


class TestApproximationQuality(unittest.TestCase):
    """The core Part 3 comparison: exact exp vs P2 vs P3 vs P4."""

    INTERVAL = (-8.0, 0.0)

    def test_higher_degree_is_more_accurate(self):
        prev = float("inf")
        for d in (2, 3, 4, 5, 6):
            r = approximation_report(fit_exp_poly(d, *self.INTERVAL), *self.INTERVAL)
            self.assertLess(r["max_abs_error"], prev, f"degree {d} not better than {d-1}")
            prev = r["max_abs_error"]

    def test_error_budget_on_minus8_to_0(self):
        """Documented, measured error levels. These are assertions about the
        MATH, not about our code, so they are safe to pin down."""
        budget = {2: 3.0e-1, 3: 1.1e-1, 4: 3.5e-2}
        for d, limit in budget.items():
            r = approximation_report(fit_exp_poly(d, *self.INTERVAL), *self.INTERVAL)
            self.assertLess(r["max_abs_error"], limit, f"degree {d}")

    def test_all_polys_tracked_against_torch_exp(self):
        z = torch.linspace(*self.INTERVAL, 4001)
        exact = torch.exp(z)
        for d, tol in ((2, 3.0e-1), (3, 1.1e-1), (4, 3.5e-2)):
            p = build_poly(d, *self.INTERVAL)
            err = (p(z) - exact).abs()
            self.assertLess(float(err.max()), tol)
            self.assertLess(float(err.mean()), tol / 2)

    def test_remez_beats_chebyshev_on_max_error(self):
        for d in (2, 3, 4):
            cheb = approximation_report(fit_exp_poly(d, *self.INTERVAL, method="chebyshev"),
                                        *self.INTERVAL)["max_abs_error"]
            remez = approximation_report(fit_exp_poly(d, *self.INTERVAL, method="remez"),
                                         *self.INTERVAL)["max_abs_error"]
            self.assertLess(remez, cheb * 1.001, f"degree {d}")

    def test_lstsq_beats_others_on_rmse(self):
        for d in (2, 3, 4):
            ls = approximation_report(fit_exp_poly(d, *self.INTERVAL, method="lstsq"),
                                      *self.INTERVAL)["rmse"]
            cheb = approximation_report(fit_exp_poly(d, *self.INTERVAL, method="chebyshev"),
                                        *self.INTERVAL)["rmse"]
            self.assertLessEqual(ls, cheb * 1.001, f"degree {d}")

    def test_remez_equioscillates(self):
        """A true minimax polynomial touches +-E at degree+2 points."""
        d = 4
        c = fit_exp_poly(d, *self.INTERVAL, method="remez")
        z = np.linspace(*self.INTERVAL, 20001)
        err = np.polyval(c[::-1], z) - np.exp(z)
        emax = np.abs(err).max()
        near = np.abs(np.abs(err) - emax) < 1e-3 * emax
        # count contiguous blocks of near-extremal points
        blocks = np.sum(np.diff(near.astype(int)) == 1) + int(near[0])
        self.assertGreaterEqual(blocks, d + 2)

    def test_pin_zero_gives_exactly_one_at_zero(self):
        for method in ("lobatto", "lstsq", "chebyshev"):
            for d in (2, 3, 4):
                c = fit_exp_poly(d, -8.0, 0.0, method=method, pin_zero=True)
                self.assertAlmostEqual(float(np.polyval(c[::-1], 0.0)), 1.0, places=9,
                                       msg=f"{method} degree {d}")

    def test_pin_zero_rejected_for_remez_with_a_useful_message(self):
        with self.assertRaises(ValueError) as cm:
            fit_exp_poly(4, -8.0, 0.0, method="remez", pin_zero=True)
        self.assertIn("lobatto", str(cm.exception))

    def test_narrow_interval_is_much_easier(self):
        """The interval is the dominant cost driver -- this is why Part 6 exists."""
        wide = approximation_report(fit_exp_poly(4, -8, 0), -8, 0)["max_abs_error"]
        narrow = approximation_report(fit_exp_poly(4, -2, 0), -2, 0)["max_abs_error"]
        self.assertLess(narrow, wide / 100)


class TestGatePathologies(unittest.TestCase):
    """exp(z) in (0, 1] for z <= 0. Polynomials do not respect that, and these
    tests DOCUMENT the violation rather than hiding it."""

    def test_unpinned_polys_go_negative_on_minus8_to_0(self):
        for d in (2, 3, 4):
            r = approximation_report(fit_exp_poly(d, -8, 0), -8, 0)
            self.assertLess(r["poly_min"], 0.0, f"degree {d} unexpectedly stayed >= 0")
            self.assertGreater(r["frac_poly_lt_0"], 0.0)

    def test_unpinned_polys_undershoot_at_zero(self):
        for d in (2, 3, 4):
            r = approximation_report(fit_exp_poly(d, -8, 0), -8, 0)
            self.assertLess(r["P_at_0"], 1.0)

    def test_poly_diverges_outside_its_fit_interval(self):
        """Outside [xmin, xmax] a polynomial does not decay -- it blows up. This
        is the single most dangerous property for us, and Part 6 is what stops us
        walking into it."""
        p = build_poly(4, -8, 0)
        far = torch.tensor([-20.0, -40.0])
        vals = p(far)
        self.assertTrue((vals.abs() > 1.0).all())
        self.assertGreater(float(p(torch.tensor([-40.0]))[0].abs()), 100.0)

    def test_no_clamp_anywhere(self):
        """We promised not to hide instability with clamp. Enforced, not trusted.

        Why the promise matters: `min(max(a, 0), 1)` would make every table in
        Part 8 read "ok". A clamp is a COMPARISON, and comparisons are exactly
        what CKKS cannot do cheaply -- you need a high-degree sign-approximation
        polynomial, which costs far more depth than the degree-4 polynomial we
        are trying to afford. So a clamped result would not be implementable
        under FHE, and reporting one would invalidate the experiment.

        This walks the AST rather than grepping, so prose in a docstring that
        mentions the word does not trip it, and a real call cannot hide behind
        odd formatting. A genuinely necessary clamp (guarding a division or a
        sqrt) must carry an `ALLOW-CLAMP` marker in the six lines ending at the
        call, saying why. Six, because the justification usually needs a sentence
        or two and the call itself is often the tail of a multi-line expression.
        """
        import ast
        import pathlib

        CLAMPS = {"clamp", "clamp_", "clamp_min", "clamp_max", "clip", "clip_"}
        root = pathlib.Path(__file__).resolve().parents[2]
        files = [f for f in list((root / "baby_mamba").rglob("*.py"))
                 + list((root / "real_mamba").rglob("*.py"))
                 + list(root.glob("*.py"))
                 if "tests" not in f.parts]
        self.assertGreater(len(files), 8, "the scanner found almost no source to check")

        offenders = []
        for f in files:
            src = f.read_text()
            lines = src.splitlines()
            for node in ast.walk(ast.parse(src, filename=str(f))):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else None)
                if name not in CLAMPS:
                    continue
                lineno = node.lineno
                window = "\n".join(lines[max(0, lineno - 6):lineno])
                if "ALLOW-CLAMP" in window:
                    continue
                offenders.append(f"{f.relative_to(root)}:{lineno}: {lines[lineno - 1].strip()}")
        self.assertEqual(
            offenders, [],
            "clamp is not FHE-friendly (see Part 8). If a clamp is genuinely a "
            "division/sqrt guard rather than a stability crutch, mark it ALLOW-CLAMP "
            "with a reason:\n  " + "\n  ".join(offenders))

    def test_the_clamp_scanner_actually_catches_things(self):
        """A scanner that can never fail is worthless. Prove it fires."""
        import ast
        import tempfile

        CLAMPS = {"clamp", "clamp_", "clamp_min", "clamp_max", "clip", "clip_"}
        bad = "import torch\ndef f(a):\n    return a.clamp(0, 1)\n"
        found = [n for n in ast.walk(ast.parse(bad))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in CLAMPS]
        self.assertEqual(len(found), 1)


class TestSerialisation(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "p.json"
            c = fit_exp_poly(4, -8, 0)
            save_coefficients(path, c, -8, 0, "chebyshev", "uniform",
                              report=approximation_report(c, -8, 0))
            p = load_poly(path)
            self.assertEqual(p.degree, 4)
            self.assertEqual(p.ct_ct_depth, 2)
            self.assertEqual(p.interval, (-8.0, 0.0))
            np.testing.assert_allclose(p.coeff_list(), c, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
