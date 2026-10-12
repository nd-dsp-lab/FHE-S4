"""CPU tests for the all-polynomial emulator. Run: python -m pytest allpoly/tests -q"""
import math

import numpy as np
import pytest
import torch

from allpoly.ops import ChebPerChannel, PolyInvSqrt, Refresher, cheb_depth, clenshaw, fit_cheb_interp


def test_cheb_depth_matches_openfhe_table():
    assert cheb_depth(4) == 4 and cheb_depth(13) == 5 and cheb_depth(27) == 6 and cheb_depth(59) == 7
    assert cheb_depth(27, unit_interval=True) == 5


def test_clenshaw_matches_numpy():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(3, 30))
    t = rng.uniform(-1, 1, size=(5, 3))
    ref = np.stack([np.polynomial.chebyshev.chebval(t[:, j], c[j]) for j in range(3)], axis=1)
    got = clenshaw(torch.tensor(c), torch.tensor(t)).numpy()
    assert np.allclose(got, ref, atol=1e-10)


def test_silu_fit_converges_with_degree():
    lo, hi = np.array([-20.0, -3.0]), np.array([15.0, 4.0])
    x = torch.linspace(0, 1, 2001)[:, None] * torch.tensor(hi - lo) + torch.tensor(lo)
    ref = x * torch.sigmoid(x)
    errs = []
    for d in (7, 15, 31):
        m = ChebPerChannel(fit_cheb_interp(lambda v: v / (1 + np.exp(-v)), lo, hi, d), lo, hi, d)
        errs.append(float((m(x.float()) - ref).abs().max()))
        assert m.frac_out == 0.0
    # a 35-wide interval converges slowly: measured 0.65 / 0.17 / 0.0075
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.02, errs


def test_out_of_interval_is_counted_not_clamped():
    lo, hi = np.array([-1.0]), np.array([1.0])
    m = ChebPerChannel(fit_cheb_interp(np.tanh, lo, hi, 9), lo, hi, 9)
    m(torch.tensor([[0.0], [3.0]]))
    assert m.frac_out == 0.5


def test_softplus_square_form_is_nonnegative():
    lo, hi = np.array([-12.0]), np.array([6.0])
    m = ChebPerChannel(fit_cheb_interp(lambda x: np.sqrt(np.logaddexp(0, x)), lo, hi, 13),
                       lo, hi, 13, square=True)
    x = torch.linspace(-12, 6, 4001)[:, None]
    y = m(x)
    assert float(y.min()) >= 0.0
    assert float((y - torch.nn.functional.softplus(x)).abs().max()) < 1e-2


@pytest.mark.parametrize("lo,hi", [(0.05, 1e4), (1e-3, 1e5)])
def test_invsqrt_newton_is_accurate_inside_range(lo, hi):
    m = PolyInvSqrt(lo, hi, degree=27, target_rel=2.0 ** -12)
    assert m.init_in_basin
    v = torch.tensor(np.geomspace(lo, hi, 2001), dtype=torch.float32)
    rel = (m(v) * torch.sqrt(v) - 1).abs().max().item()
    print(f"[{lo:g},{hi:g}] newton={m.newton} depth={m.depth} final rel {rel:.2e}")
    assert rel < 2.0 ** -11      # fp32 slack on top of the 2^-12 grid target


def test_refresher_noise_level():
    r = Refresher(bits=10.0, scales={"block_in:0": torch.tensor(4.0)}, sites=("block_in",), seed=1)
    x = torch.zeros(200_000)
    e = r(0, "block_in", x)
    assert abs(float(e.abs().mean()) - 4.0 * 2 ** -10) / (4.0 * 2 ** -10) < 0.02
    with pytest.raises(KeyError):
        r(1, "block_in", x)


def test_exact_matches_model():
    """The explicit forward IS the model: exact ops must reproduce model(ids).logits."""
    from real_mamba.model import load_model
    from allpoly.forward import ExactOps, allpoly_forward
    model, cfg, _ = load_model(backend="local", device="cpu", dtype=torch.float32)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 96))
    with torch.no_grad():
        ref = model(ids).logits.float()
        got = allpoly_forward(model, ids, ExactOps(), chunk_size=32)
    err = (got - ref).abs().max().item() / ref.abs().max().item()
    print(f"relative max diff exact-vs-model: {err:.2e}")
    assert err < 1e-4


def test_refresher_is_per_channel():
    """Noise is relative to each channel's own range, as a per-slot scaling would make it."""
    r = Refresher(bits=8.0, scales={"scan_in_dt:0": torch.tensor([1e-3, 10.0])}, sites=("scan_in_dt",), seed=2)
    e = r(0, "scan_in_dt", torch.zeros(100_000, 2))
    small, big = float(e[:, 0].abs().mean()), float(e[:, 1].abs().mean())
    assert abs(small / (1e-3 * 2 ** -8) - 1) < 0.03 and abs(big / (10.0 * 2 ** -8) - 1) < 0.03
