"""The reference SSD must be the real SSD. These tests are the whole basis for
trusting every number in Parts 6-10.

Run:  python -m pytest real_mamba/tests/test_reference_ssd.py -q
"""

import unittest

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from baby_mamba.polynomial import build_poly
from baby_mamba.transition import ExactExp
from real_mamba.reference_ssd import (
    segprod,
    ssd_product_form,
    ssd_sequential,
    suffix_prod_exclusive,
)


# ---------------------------------------------------------------------------
# Verbatim copy of the official cumsum-form reference, so we can compare against
# it without importing mamba_ssm (which needs CUDA + Triton at import time).
# Source: third_party/mamba/mamba_ssm/modules/ssd_minimal.py:23-78
# ---------------------------------------------------------------------------
def segsum(x):
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


def make_inputs(B=2, L=96, H=4, P=8, N=6, G=1, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, H, P, generator=g, dtype=dtype)
    dt = F.softplus(torch.randn(B, L, H, generator=g, dtype=dtype) - 2)
    A = -torch.exp(torch.rand(H, generator=g, dtype=dtype) * 2)
    Bm = torch.randn(B, L, G, N, generator=g, dtype=dtype)
    Cm = torch.randn(B, L, G, N, generator=g, dtype=dtype)
    D = torch.randn(H, generator=g, dtype=dtype)
    return x, dt, A, Bm, Cm, D


class TestSegProd(unittest.TestCase):
    def test_segprod_matches_exp_of_segsum(self):
        """segprod(exp(z)) must equal exp(segsum(z)). This is the identity the
        whole per-step reformulation rests on."""
        z = -torch.rand(2, 3, 8, dtype=torch.float64) * 2
        torch.testing.assert_close(segprod(torch.exp(z)), torch.exp(segsum(z)))

    def test_segprod_entries_by_brute_force(self):
        a = torch.rand(7, dtype=torch.float64) + 0.5
        M = segprod(a)
        for i in range(7):
            for j in range(7):
                if i < j:
                    self.assertEqual(float(M[i, j]), 0.0)
                else:
                    want = torch.tensor(1.0, dtype=torch.float64)
                    for k in range(j + 1, i + 1):
                        want = want * a[k]
                    self.assertAlmostEqual(float(M[i, j]), float(want), places=12)

    def test_suffix_prod_exclusive(self):
        a = torch.rand(6, dtype=torch.float64) + 0.5
        r = suffix_prod_exclusive(a)
        self.assertAlmostEqual(float(r[-1]), 1.0, places=14)
        for i in range(6):
            want = torch.tensor(1.0, dtype=torch.float64)
            for k in range(i + 1, 6):
                want = want * a[k]
            self.assertAlmostEqual(float(r[i]), float(want), places=12)

    def test_segprod_works_for_negative_a(self):
        """Polynomials can produce a < 0. The algebra must not care."""
        a = torch.randn(9, dtype=torch.float64)
        M = segprod(a)
        for i in range(9):
            want = torch.tensor(1.0, dtype=torch.float64)
            for k in range(1, i + 1):
                want = want * a[k]
            self.assertAlmostEqual(float(M[i, 0]), float(want), places=10)


class TestAgainstOfficialReference(unittest.TestCase):
    """With transition = exp, our product form must equal the OFFICIAL
    cumsum-form `ssd_minimal_discrete`. This is the load-bearing test."""

    def test_matches_ssd_minimal_discrete(self):
        x, dt, A, Bm, Cm, D = make_inputs()
        B, L, H, P = x.shape
        N = Bm.shape[-1]
        # upstream's calling convention: X = x*dt, and its "A" argument is z = A*dt
        Bh = Bm.expand(B, L, H, N).contiguous()
        Ch = Cm.expand(B, L, H, N).contiguous()
        for chunk in (16, 32, 48):
            want, _ = ssd_minimal_discrete(x * dt.unsqueeze(-1), dt * A, Bh, Ch, chunk)
            got = ssd_product_form(x, dt, A, Bm, Cm, ExactExp(), D=None, chunk_size=chunk)
            torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)

    def test_matches_with_D_skip(self):
        x, dt, A, Bm, Cm, D = make_inputs()
        B, L, H, N = x.shape[0], x.shape[1], x.shape[2], Bm.shape[-1]
        Bh, Ch = Bm.expand(B, L, H, N).contiguous(), Cm.expand(B, L, H, N).contiguous()
        want, _ = ssd_minimal_discrete(x * dt.unsqueeze(-1), dt * A, Bh, Ch, 32)
        want = want + x * D.unsqueeze(-1)
        got = ssd_product_form(x, dt, A, Bm, Cm, ExactExp(), D=D, chunk_size=32)
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)


class TestChunkedEqualsSequential(unittest.TestCase):
    """The chunked product form is an algebraic identity, so it must hold for ANY
    transition -- including polynomials that go negative or exceed 1."""

    def _check(self, transition, name):
        x, dt, A, Bm, Cm, D = make_inputs()
        want = ssd_sequential(x, dt, A, Bm, Cm, transition, D=D)
        for chunk in (8, 16, 32, 96):
            got = ssd_product_form(x, dt, A, Bm, Cm, transition, D=D, chunk_size=chunk)
            torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-11,
                                       msg=f"{name} chunk={chunk}")

    def test_exact(self):
        self._check(ExactExp(), "exp")

    def test_poly2(self):
        self._check(build_poly(2, -8, 0), "P2")

    def test_poly3(self):
        self._check(build_poly(3, -8, 0), "P3")

    def test_poly4(self):
        self._check(build_poly(4, -8, 0), "P4")

    def test_poly_with_negative_outputs(self):
        """Force a into the negative region and check the algebra still holds."""
        x, dt, A, Bm, Cm, D = make_inputs()
        A = A * 8.0                                       # pushes z well below -8
        p = build_poly(3, -8, 0)
        z = dt * A
        self.assertLess(float(p(z).min()), 0.0, "test did not reach the negative region")
        want = ssd_sequential(x, dt, A, Bm, Cm, p, D=D)
        got = ssd_product_form(x, dt, A, Bm, Cm, p, D=D, chunk_size=32)
        torch.testing.assert_close(got, want, rtol=1e-7, atol=1e-9)

    def test_ragged_length_is_padded_correctly(self):
        """seqlen not divisible by chunk_size must still match the sequential scan."""
        for L in (70, 100, 129):
            x, dt, A, Bm, Cm, D = make_inputs(L=L)
            want = ssd_sequential(x, dt, A, Bm, Cm, ExactExp(), D=D)
            got = ssd_product_form(x, dt, A, Bm, Cm, ExactExp(), D=D, chunk_size=64)
            self.assertEqual(got.shape, want.shape)
            torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)

    def test_multiple_groups(self):
        x, dt, A, Bm, Cm, D = make_inputs(H=6, G=3)
        want = ssd_sequential(x, dt, A, Bm, Cm, ExactExp(), D=D)
        got = ssd_product_form(x, dt, A, Bm, Cm, ExactExp(), D=D, chunk_size=32)
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)

    def test_initial_states(self):
        x, dt, A, Bm, Cm, D = make_inputs()
        h0 = torch.randn(x.shape[0], x.shape[2], x.shape[3], Bm.shape[-1], dtype=x.dtype)
        want = ssd_sequential(x, dt, A, Bm, Cm, ExactExp(), D=D, initial_states=h0)
        got = ssd_product_form(x, dt, A, Bm, Cm, ExactExp(), D=D, chunk_size=32,
                               initial_states=h0)
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)


class TestGradients(unittest.TestCase):
    def test_gradients_flow_through_the_transition(self):
        """Part 9 needs gradients w.r.t. dt_bias-like inputs and, in MODE B, w.r.t.
        the polynomial coefficients themselves."""
        x, dt, A, Bm, Cm, D = make_inputs(L=32, dtype=torch.float32)
        dt = dt.detach().requires_grad_()
        A = A.detach().requires_grad_()
        p = build_poly(4, -8, 0, trainable=True)
        y = ssd_product_form(x, dt, A, Bm, Cm, p, D=D, chunk_size=16)
        y.sum().backward()
        for t, n in ((dt.grad, "dt"), (A.grad, "A"), (p.coeffs.grad, "coeffs")):
            self.assertIsNotNone(t, n)
            self.assertTrue(torch.isfinite(t).all(), n)
            self.assertGreater(float(t.abs().sum()), 0.0, n)


if __name__ == "__main__":
    unittest.main()
