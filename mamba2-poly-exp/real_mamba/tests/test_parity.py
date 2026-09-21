"""Does our instrumentation compute the same thing as the official model?

Three layers of check, in increasing cost:

1. ALWAYS RUNS. Structural checks on the patch: it installs and restores
   cleanly, it only changes `z -> a`, and per-layer transitions land on the right
   layers. No download, no GPU.

2. NEEDS THE CHECKPOINT (set FHEMAMBA_TEST_CHECKPOINT=1). Loads the real
   state-spaces/mamba2-130m weights into the local backend, checks the shapes the
   Part 0 report claims, and checks that a short wikitext-2 perplexity is in a
   plausible range for a pretrained 130M model. A backend bug would show up here
   as a perplexity in the thousands.

3. NEEDS mamba-ssm + CUDA (skipped otherwise). The real parity test: the
   OFFICIAL fused forward vs our reference forward with ExactExp, on the same
   weights and the same input. Run this on the GPU box.

    python -m pytest real_mamba/tests/test_parity.py -q
    FHEMAMBA_TEST_CHECKPOINT=1 python -m pytest real_mamba/tests/test_parity.py -q
"""

import importlib.util
import os
import unittest

import torch

from baby_mamba.polynomial import build_poly
from baby_mamba.transition import ExactExp
from real_mamba.model import LiteConfig, MambaLMHeadModelLite, iter_mixers
from real_mamba.patch import patch_transition, patched, set_transition

HAVE_MAMBA_SSM = importlib.util.find_spec("mamba_ssm") is not None
HAVE_CUDA = torch.cuda.is_available()
HAVE_CAUSAL_CONV1D = importlib.util.find_spec("causal_conv1d") is not None
WANT_CHECKPOINT = os.environ.get("FHEMAMBA_TEST_CHECKPOINT") == "1"

# Why causal_conv1d gates some tests but not all:
#
# `Mamba2.forward` has two branches and BOTH need the causal_conv1d CUDA
# extension in practice:
#   * the default fused path calls `causal_conv1d_cuda.causal_conv1d_fwd`
#     directly (ssd_combined.py), so without the package it raises
#     AttributeError: 'NoneType' object has no attribute 'causal_conv1d_fwd';
#   * the `use_mem_eff_path=False` branch is supposed to fall back to a plain
#     nn.Conv1d, but in mamba_ssm 2.2.2 that line reads `self.dconv` (a typo for
#     `self.d_conv`) and slices `[:, -(self.dconv - 1):]` where current upstream
#     has `[:, :-(self.d_conv - 1)]`. It is dead code that has never run.
#
# So a WHOLE-MIXER comparison needs causal_conv1d installed. The SSD scan itself
# -- the only thing this project replaces -- does not: `mamba_chunk_scan_combined`
# is pure Triton and takes x/dt/A/B/C directly. TestOfficialSSDParity below is
# therefore the load-bearing test, and it runs with mamba_ssm + CUDA alone.
_NEEDS_CONV = "needs the causal_conv1d CUDA extension (see the note in this file)"


def tiny_model(seed=0):
    """A 2-layer, d_model=32 model with the same structure as the real one.

    Random weights: these tests are about plumbing, not about quality.
    """
    torch.manual_seed(seed)
    cfg = LiteConfig(d_model=32, n_layer=2, d_intermediate=0, vocab_size=64,
                     pad_vocab_size_multiple=16, d_state=8, headdim=8, ngroups=1,
                     chunk_size=16)
    return MambaLMHeadModelLite(cfg, dtype=torch.float32).eval(), cfg


class TestPatchMechanics(unittest.TestCase):
    def setUp(self):
        self.model, self.cfg = tiny_model()
        self.ids = torch.randint(0, self.cfg.vocab_size, (2, 32))

    def test_finds_every_mixer(self):
        self.assertEqual(len(list(iter_mixers(self.model))), self.cfg.n_layer)

    def test_patch_and_restore_leaves_no_trace(self):
        with torch.no_grad():
            before = self.model(self.ids).logits.clone()
        h = patch_transition(self.model, ExactExp(), chunk_size=16)
        self.assertEqual(h.n_patched, self.cfg.n_layer)
        self.assertTrue(hasattr(self.model, "fhe_transitions"))
        h.restore()
        self.assertFalse(hasattr(self.model, "fhe_transitions"))
        with torch.no_grad():
            after = self.model(self.ids).logits
        torch.testing.assert_close(before, after)

    def test_exact_patch_is_a_no_op_on_the_local_backend(self):
        """The local backend's own forward IS the reference forward, so patching it
        with ExactExp must change nothing at all. If this drifts, the two paths
        have diverged and every later comparison is suspect."""
        with torch.no_grad():
            before = self.model(self.ids).logits.clone()
            with patched(self.model, ExactExp(), chunk_size=16):
                after = self.model(self.ids).logits.clone()
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_chunk_size_does_not_change_the_answer(self):
        outs = []
        for cs in (8, 16, 32):
            with torch.no_grad(), patched(self.model, ExactExp(), chunk_size=cs):
                outs.append(self.model(self.ids).logits.clone())
        for o in outs[1:]:
            torch.testing.assert_close(outs[0], o, rtol=1e-4, atol=1e-4)

    def test_polynomial_changes_the_output_but_exact_does_not(self):
        with torch.no_grad():
            with patched(self.model, ExactExp(), chunk_size=16):
                a = self.model(self.ids).logits.clone()
            with patched(self.model, build_poly(2, -8, 0), chunk_size=16):
                b = self.model(self.ids).logits.clone()
        self.assertFalse(torch.allclose(a, b))

    def test_per_layer_transitions_land_on_the_right_layer(self):
        per_layer = {0: build_poly(2, -8, 0), 1: build_poly(4, -8, 0)}
        h = patch_transition(self.model, per_layer, chunk_size=16)
        try:
            got = {i: m._transition.degree for i, m in iter_mixers(self.model)}
            self.assertEqual(got, {0: 2, 1: 4})
        finally:
            h.restore()

    def test_set_transition_swaps_in_place(self):
        h = patch_transition(self.model, ExactExp(), chunk_size=16)
        try:
            p = build_poly(3, -8, 0)
            set_transition(h, p)
            for _, m in iter_mixers(self.model):
                self.assertIs(m._transition, p)
        finally:
            h.restore()

    def test_trainable_coefficients_are_visible_to_the_optimiser(self):
        """MODE B in Part 9 depends on this: the polynomial's parameters must show
        up in model.parameters() once patched."""
        p = build_poly(4, -8, 0, trainable=True)
        n_before = sum(1 for _ in self.model.parameters())
        h = patch_transition(self.model, p, chunk_size=16)
        try:
            ids = [id(q) for q in self.model.parameters()]
            self.assertIn(id(p.coeffs), ids)
            self.assertEqual(sum(1 for _ in self.model.parameters()), n_before + 1)
        finally:
            h.restore()

    def test_inference_params_are_refused_not_silently_wrong(self):
        with patched(self.model, ExactExp(), chunk_size=16):
            _, mixer = next(iter_mixers(self.model))
            with self.assertRaises(NotImplementedError):
                mixer(torch.randn(1, 1, self.cfg.d_model), inference_params=object())

    def test_collector_sees_every_layer(self):
        seen = []
        with torch.no_grad(), patched(self.model, ExactExp(), chunk_size=16,
                                      collector=lambda i, z, a: seen.append(i)):
            self.model(self.ids)
        self.assertEqual(sorted(seen), list(range(self.cfg.n_layer)))


@unittest.skipUnless(WANT_CHECKPOINT,
                     "set FHEMAMBA_TEST_CHECKPOINT=1 to download and check the real weights")
class TestRealCheckpoint(unittest.TestCase):
    """Integrity of the LOCAL loader against the real weights.

    backend is pinned to "local" on purpose, and not left on "auto":

      * this class exists to check that OUR module tree reads the official
        checkpoint correctly, which is exactly what "auto" would stop testing the
        moment mamba-ssm is installed;
      * the official model cannot run on CPU at all. `MambaLMHeadModel` is built
        with fused_add_norm=True, so `Block.forward` calls the Triton
        `layer_norm_fn` even when the mixer has been patched away, and Triton
        needs CUDA. Leaving this on "auto" passes on a laptop (no mamba-ssm ->
        falls back to local) and fails on the cluster (mamba-ssm present ->
        picks official -> Triton on CPU tensors). That is what it did.

    The official backend is covered by TestOfficialParity, which requires CUDA.
    """

    @classmethod
    def setUpClass(cls):
        from real_mamba.model import load_model
        device = "cuda" if HAVE_CUDA else "cpu"
        cls.model, cls.cfg, cls.backend = load_model(backend="local", device=device,
                                                     dtype=torch.float32, verbose=False)
        cls.device = device

    def test_backend_is_the_one_we_asked_for(self):
        self.assertEqual(self.backend, "local")

    def test_shapes_match_the_part0_report(self):
        mixers = list(iter_mixers(self.model))
        self.assertEqual(len(mixers), 24)
        _, m = mixers[0]
        self.assertEqual(m.nheads, 24)
        self.assertEqual(m.headdim, 64)
        self.assertEqual(m.d_state, 128)
        self.assertEqual(m.d_inner, 1536)
        self.assertEqual(tuple(m.A_log.shape), (24,))
        self.assertEqual(tuple(m.dt_bias.shape), (24,))
        self.assertEqual(tuple(m.in_proj.weight.shape), (3352, 768))

    def test_A_is_negative_everywhere(self):
        for _, m in iter_mixers(self.model):
            self.assertTrue((-torch.exp(m.A_log.float()) < 0).all())

    def test_perplexity_is_plausible_for_a_pretrained_130m(self):
        from real_mamba.data import get_blocks
        from real_mamba.eval_lm import evaluate
        blocks, _, _ = get_blocks("wikitext2", "validation", 1024, verbose=False)
        with patched(self.model, ExactExp(), chunk_size=128):
            res = evaluate(self.model, blocks[:4], device=self.device,
                           vocab_size=self.cfg.vocab_size)
        # A working 130M model lands around 15-30 here. A loading or SSD bug puts
        # it in the thousands (or at ~50k, i.e. uniform over the vocab).
        self.assertGreater(res["perplexity"], 5.0)
        self.assertLess(res["perplexity"], 40.0)


@unittest.skipUnless(HAVE_MAMBA_SSM and HAVE_CUDA,
                     "needs the mamba-ssm package and a CUDA device")
class TestOfficialSSDParity(unittest.TestCase):
    """THE load-bearing test: our scan vs the OFFICIAL Triton scan kernel.

    This compares `real_mamba.reference_ssd.ssd_product_form` (per-step product
    form, transition = exp) against `mamba_chunk_scan_combined` (the fused
    cumsum-form Triton kernel the real model actually runs) on identical inputs.

    It deliberately bypasses conv1d / RMSNorm / projections, because those are
    not what this project changes, and because comparing them would drag in the
    causal_conv1d dependency for no scientific gain. If this passes, our
    reformulation of the SSD is the same function as upstream's.
    """

    # WHY float16 AND NOT float32.
    # Measured on the CRC Quadro RTX 6000 (sm_75), 2026-09-19: the official
    # `mamba_chunk_scan_combined` compiles in fp16 ONLY. fp32 and bf16 both die
    # with `IndexError: map::at` inside Triton's IR translation -- Turing has
    # fp16 tensor cores and no bf16 ones, and the fp32 tl.dot path has no valid
    # MMA layout there. (Plain Triton kernels and `layer_norm_fn` compile fine,
    # so this is specific to the SSD kernel, not a broken Triton install.)
    # So on this hardware the comparison has to happen in fp16, and the
    # tolerance has to reflect fp16 arithmetic rather than our algebra.
    DTYPE = torch.float16

    def _inputs(self, batch=2, seqlen=256, nheads=8, headdim=32, dstate=64,
                ngroups=1, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        d = dict(device="cuda", dtype=self.DTYPE)
        x = torch.randn(batch, seqlen, nheads, headdim, generator=g, **d)
        # dt and A stay fp32: that is what the real model does too
        # (mamba2.py:182 forces A through .float()).
        dt = torch.nn.functional.softplus(
            torch.randn(batch, seqlen, nheads, generator=g, device="cuda",
                        dtype=torch.float32) - 2.0)
        A = -torch.exp(torch.rand(nheads, generator=g, device="cuda",
                                  dtype=torch.float32) * 2.0)
        B = torch.randn(batch, seqlen, ngroups, dstate, generator=g, **d)
        C = torch.randn(batch, seqlen, ngroups, dstate, generator=g, **d)
        D = torch.randn(nheads, generator=g, device="cuda", dtype=torch.float32)
        return x, dt, A, B, C, D

    # fp16 carries ~3 decimal digits, and a length-L scan accumulates that over
    # many multiply-adds. 3e-2 relative is the realistic agreement between an
    # fp16 kernel and an fp32 reference on the SAME numbers. If our algebra were
    # actually wrong the error would be O(1), not O(1e-2) -- so this tolerance
    # is loose in absolute terms and still decisive for the question being asked.
    REL_TOL = 3e-2

    def _compare(self, chunk_size, with_D, seed=0, seqlen=256):
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        from baby_mamba.transition import ExactExp
        from real_mamba.reference_ssd import ssd_product_form

        x, dt, A, B, C, D = self._inputs(seqlen=seqlen, seed=seed)
        with torch.no_grad():
            want = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=chunk_size,
                                             D=(D if with_D else None), z=None)
            # Our reference sees the SAME numbers, widened to fp32 -- exactly what
            # real_mamba/patch.py does in the model.
            got = ssd_product_form(x.float(), dt, A, B.float(), C.float(), ExactExp(),
                                   D=(D if with_D else None), chunk_size=chunk_size)
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(torch.isfinite(got).all(), "our output is not finite")
        self.assertTrue(torch.isfinite(want).all(), "the official kernel output is not finite")
        w = want.float()
        rel = float((got - w).norm() / w.norm())
        self.assertLess(rel, self.REL_TOL,
                        f"relative error {rel:.3e} at chunk_size={chunk_size}, "
                        f"seqlen={seqlen} -- too large to be fp16 rounding")
        # A shuffled control: if the test would pass on the WRONG answer it is
        # worthless. Compare against a time-reversed reference and require it to
        # be far worse.
        bad = float((got.flip(1) - w).norm() / w.norm())
        self.assertGreater(bad, 10 * rel,
                           "the tolerance is so loose that a wrong answer passes")
        return rel

    def test_matches_official_kernel_chunk64(self):
        print(f"\n  rel error vs mamba_chunk_scan_combined (chunk 64): "
              f"{self._compare(64, with_D=False):.3e}")

    def test_matches_official_kernel_chunk128(self):
        print(f"\n  rel error vs mamba_chunk_scan_combined (chunk 128): "
              f"{self._compare(128, with_D=False):.3e}")

    def test_matches_official_kernel_chunk256(self):
        print(f"\n  rel error vs mamba_chunk_scan_combined (chunk 256): "
              f"{self._compare(256, with_D=False):.3e}")

    def test_matches_official_kernel_with_D_skip(self):
        print(f"\n  rel error with the D skip term: "
              f"{self._compare(128, with_D=True):.3e}")

    def test_matches_official_kernel_long_sequence(self):
        print(f"\n  rel error at seqlen 2048: "
              f"{self._compare(128, with_D=True, seqlen=2048):.3e}")

    def test_our_chunk_size_does_not_matter_but_theirs_is_fixed(self):
        """Our chunked product form is an algebraic identity, so every chunk size
        must land on the same answer as the one official run.

        Same fp16-in / fp32-reference convention as _compare (see DTYPE above):
        the kernel gets fp16 because that is all it compiles on sm_75, our
        reference gets the identical numbers widened to fp32.
        """
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        from baby_mamba.transition import ExactExp
        from real_mamba.reference_ssd import ssd_product_form

        x, dt, A, B, C, D = self._inputs(seed=3)
        with torch.no_grad():
            want = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=128,
                                             D=D, z=None).float()
            for cs in (32, 64, 128, 256):
                got = ssd_product_form(x.float(), dt, A, B.float(), C.float(),
                                       ExactExp(), D=D, chunk_size=cs)
                rel = float((got - want).norm() / want.norm())
                print(f"\n  our chunk_size={cs:<4d} vs official chunk_size=128: "
                      f"rel error {rel:.3e}")
                self.assertLess(rel, self.REL_TOL, f"our chunk_size={cs}")


@unittest.skipUnless(HAVE_MAMBA_SSM and HAVE_CUDA,
                     "needs the mamba-ssm package and a CUDA device")
@unittest.skipUnless(HAVE_CAUSAL_CONV1D, _NEEDS_CONV)
class TestOfficialParity(unittest.TestCase):
    """THE parity test. Run this on the GPU box.

    Builds one official `Mamba2` module, runs its fused forward, then runs our
    reference forward on the SAME module with ExactExp, and compares.
    """

    def test_reference_forward_matches_the_fused_kernel(self):
        from mamba_ssm.modules.mamba2 import Mamba2

        from real_mamba.patch import mamba2_reference_forward

        torch.manual_seed(0)
        mixer = Mamba2(d_model=256, d_state=64, headdim=32, ngroups=1, chunk_size=64,
                       layer_idx=0, device="cuda", dtype=torch.float32).eval()
        u = torch.randn(2, 256, 256, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            want = mixer(u)                                   # fused Triton path
            got = mamba2_reference_forward(mixer, u, ExactExp(), chunk_size=64)
        # fp32 Triton vs fp32 PyTorch over a 256-step scan: 1e-3 relative is the
        # realistic agreement level, dominated by reduction order.
        torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)

    def test_unfused_path_also_matches(self):
        from mamba_ssm.modules.mamba2 import Mamba2

        from real_mamba.patch import mamba2_reference_forward

        torch.manual_seed(1)
        mixer = Mamba2(d_model=256, d_state=64, headdim=32, ngroups=1, chunk_size=64,
                       layer_idx=0, use_mem_eff_path=False,
                       device="cuda", dtype=torch.float32).eval()
        u = torch.randn(1, 128, 256, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            want = mixer(u)
            got = mamba2_reference_forward(mixer, u, ExactExp(), chunk_size=64)
        torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)

    def test_official_and_local_backends_agree_end_to_end(self):
        from real_mamba.model import load_local, load_official
        off, _ = load_official(device="cuda", dtype=torch.float32)
        loc, _ = load_local(device="cuda", dtype=torch.float32)
        ids = torch.randint(0, 50277, (1, 256), device="cuda")
        with torch.no_grad():
            a = off(ids).logits.float()
            with patched(loc, ExactExp(), chunk_size=128):
                b = loc(ids).logits.float()
        torch.testing.assert_close(b, a, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    unittest.main()
