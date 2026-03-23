import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import toeplitz, block_diag

import tenseal

GLU_GATE_DOMAIN = (-142.0, 142.0)
GELU_DOMAIN = (-5.0, 15.0)
POLY4_GATE_CHEB = np.array(
    [0.5, 1.03580642, 0.0, -0.273067994, 0.0],
    dtype=np.float64,
)
POLY6_GATE_CHEB = np.array(
    [0.5, 0.927977957, 0.0, -0.35367563, 0.0, 0.168670784, 0.0],
    dtype=np.float64,
)
POLY4_GELU_CHEB = np.array(
    [5.977405989737, 7.988856571172, 1.530634414283, -0.667904507889, 0.246138181485],
    dtype=np.float64,
)
POLY6_GELU_CHEB = np.array(
    [
        6.04831644882,
        8.094460228971,
        1.371512430522,
        -0.740775295685,
        0.134773988002,
        0.126019270103,
        -0.239569101196,
    ],
    dtype=np.float64,
)


def chebyshev_eval_torch(
    x: torch.Tensor,
    coeffs: torch.Tensor,
    domain: tuple[float, float] = GLU_GATE_DOMAIN,
) -> torch.Tensor:
    lo, hi = domain
    t = (2.0 * x - (hi + lo)) / (hi - lo)
    if coeffs.numel() == 1:
        return coeffs[0] * torch.ones_like(t)

    t0 = torch.ones_like(t)
    result = coeffs[0] * t0
    t1 = t
    result = result + coeffs[1] * t1
    for i in range(2, coeffs.numel()):
        ti = 2.0 * t * t1 - t0
        result = result + coeffs[i] * ti
        t0, t1 = t1, ti
    return result


def apply_glu_mode(
    x: torch.Tensor,
    glu_mode: str,
    split_dim: int,
    poly4_coeffs: torch.Tensor | None = None,
    poly6_coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    a, b = torch.chunk(x, 2, dim=split_dim)
    if glu_mode == "exact":
        gate = torch.sigmoid(b)
    elif glu_mode == "poly4_gate":
        if poly4_coeffs is None:
            raise RuntimeError("poly4 coefficients are required for poly4_gate mode.")
        gate = chebyshev_eval_torch(b, poly4_coeffs)
    elif glu_mode == "poly6_gate":
        if poly6_coeffs is None:
            raise RuntimeError("poly6 coefficients are required for poly6_gate mode.")
        gate = chebyshev_eval_torch(b, poly6_coeffs)
    elif glu_mode == "linear_gate":
        gate = torch.clamp(0.25 * b + 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported glu_mode: {glu_mode}")
    return a * gate


class SelectableGLU(nn.Module):
    def __init__(self, glu_mode: str = "exact", split_dim: int = -2):
        super().__init__()
        self.glu_mode = glu_mode
        self.split_dim = split_dim
        self.register_buffer("poly4_coeffs", torch.tensor(POLY4_GATE_CHEB, dtype=torch.float32))
        self.register_buffer("poly6_coeffs", torch.tensor(POLY6_GATE_CHEB, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_glu_mode(
            x,
            glu_mode=self.glu_mode,
            split_dim=self.split_dim,
            poly4_coeffs=self.poly4_coeffs,
            poly6_coeffs=self.poly6_coeffs,
        )


def apply_gelu_mode(
    x: torch.Tensor,
    gelu_mode: str,
    poly4_coeffs: torch.Tensor | None = None,
    poly6_coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    if gelu_mode == "exact":
        return torch.nn.functional.gelu(x)
    if gelu_mode == "poly4":
        if poly4_coeffs is None:
            raise RuntimeError("poly4 coefficients are required for poly4 GELU mode.")
        return chebyshev_eval_torch(x, poly4_coeffs, domain=GELU_DOMAIN)
    if gelu_mode == "poly6":
        if poly6_coeffs is None:
            raise RuntimeError("poly6 coefficients are required for poly6 GELU mode.")
        return chebyshev_eval_torch(x, poly6_coeffs, domain=GELU_DOMAIN)
    raise ValueError(f"Unsupported gelu_mode: {gelu_mode}")


class SelectableGELU(nn.Module):
    def __init__(self, gelu_mode: str = "exact"):
        super().__init__()
        self.gelu_mode = gelu_mode
        self.register_buffer("poly4_coeffs", torch.tensor(POLY4_GELU_CHEB, dtype=torch.float32))
        self.register_buffer("poly6_coeffs", torch.tensor(POLY6_GELU_CHEB, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_gelu_mode(
            x,
            gelu_mode=self.gelu_mode,
            poly4_coeffs=self.poly4_coeffs,
            poly6_coeffs=self.poly6_coeffs,
        )

class S4DKernel(nn.Module):
    """
    100% Faithful S4D Kernel Generation.
    Uses ZOH discretization and standard S4D initialization.
    """
    def __init__(self, d_model, d_state=8, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.N = d_state

        # 1. Initialize dt (Timescale)
        # Log-uniform initialization (Standard S4)
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # 2. Initialize A (State Matrix) - S4D-Lin Initialization
        # A is diagonal. Real part is -0.5, Imag part is spaced by pi.
        # This is the "HiPPO" approximation for diagonal matrices.
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.A_imag = nn.Parameter(torch.pi * torch.arange(d_state).float().repeat(d_model, 1))

        # 3. Initialize B (Input Projection)
        # Standard S4D fixes B to ones (redundant with C), but we make it learnable to be rigorous
        self.B = nn.Parameter(torch.ones(d_model, d_state, dtype=torch.cfloat))

        # 4. Initialize C (Output Projection)
        # Random normal initialization
        self.C = nn.Parameter(torch.randn(d_model, d_state, dtype=torch.cfloat))

    def forward(self, L):
        """
        Returns the Discrete Convolution Kernel (K) using ZOH.
        Shape: (d_model, L)
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt)          # (H)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H, N)
        
        # --- The Faithful ZOH Discretization ---
        # A_bar = exp(A * dt)
        # B_bar = (A_bar - I) * A^{-1} * B * dt (approx) or exact ZOH integration
        
        # Since A is diagonal, we can do element-wise ops.
        # dt must be broadcast to (H, N)
        dt_broad = dt.unsqueeze(-1)
        
        # 1. Discretize A -> A_bar (State transition)
        A_bar = torch.exp(A * dt_broad)  # (H, N)

        # 2. Discretize B -> B_bar (Input mixing)
        # Exact ZOH: B_bar = A^{-1} (exp(A*dt) - I) * B
        B_bar = (1/A) * (A_bar - 1.0) * self.B # (H, N)

        # --- Compute Kernel (Power Series) ---
        # K = [C B_bar, C A_bar B_bar, C A_bar^2 B_bar, ...]
        # We compute this explicitly for small L (Faithful to the math, distinct from FFT optimization)
        
        # Create vandermonde-like powers: A_bar^t
        # Range t = [0, ..., L-1]
        t = torch.arange(L, device=A.device).unsqueeze(0).unsqueeze(0) # (1, 1, L)
        A_bar_pow = torch.pow(A_bar.unsqueeze(-1), t) # (H, N, L)
        
        # Combine: C * (A_bar^t) * B_bar
        # Sum over state dimension N (since y = C x)
        # Shape: (H, N, L) -> (H, L)
        K = (self.C.unsqueeze(-1) * A_bar_pow * B_bar.unsqueeze(-1)).sum(dim=1)
        
        return K.real # Return real part (SISO system assumption)

class MiniS4D(nn.Module):
    """
    S4D block faithful to official state-spaces/s4: includes D (skip), dropout,
    and output_linear (Conv1d + GLU). For a minimal/FHE-friendly variant that
    omits these, see README.
    """
    def __init__(self, d_model=4, d_state=8, L=64, dropout=0.0, glu_mode="exact", gelu_mode="exact"):
        super().__init__()
        self.d_model = d_model
        self.L = L
        self.glu_mode = glu_mode
        self.gelu_mode = gelu_mode

        # D term (skip connection) — official S4D
        self.D = nn.Parameter(torch.randn(d_model))

        self.kernel_gen = S4DKernel(d_model, d_state)
        
        # non-linear GELU
        self.activation = SelectableGELU(gelu_mode=gelu_mode)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            SelectableGLU(glu_mode=glu_mode, split_dim=-2),
        )
        self.decoder = nn.Linear(d_model, 1)  # Simple regression head for demo

    def forward(self, u, context = None):
        """
        u: (Batch, d_model, L)
        """
        # 1. Generate Kernel (ZOH based)
        if context is None:
            K = self.kernel_gen(self.L)  # (d_model, L)

            # 2. Convolution (causal)
            k_conv = K.unsqueeze(1)  # (H, 1, L)
            y = torch.nn.functional.conv1d(u, k_conv, padding=self.L - 1, groups=self.d_model)
            y = y[:, :, : self.L]  # Causal slice

            # 3. D term (skip connection) — official S4D
            y = y + u * self.D.unsqueeze(-1)
            
            # 4. Activation + dropout + output_linear (Conv1d + GLU)
            
            # GLU non-linear

            y = self.dropout(self.activation(y))
            y = self.output_linear(y)

            return self.decoder(y.mean(dim=-1))

        else:
            T = block_diag(*[self.export_toeplitz(h) for h in range(self.d_model)])
            # ERROR: need Galois keys for rotation -- way to avoid?
            y = u.matmul(T.tolist())
            y = y + (u * self.D.detach().cpu().numpy().repeat(self.L).tolist())

            y_plain = torch.tensor(y.decrypt())
            y_activated = self.activation(y_plain)
            y = tenseal.ckks_vector(context, y_activated.tolist())

            conv_weight = self.output_linear[0].weight.detach().cpu().squeeze(-1).numpy().T()
            conv_bias = self.output_linear[0].bias.detach().cpu().numpy().tolist()
            # ERROR: shapes of y and conv_weight don't match -- where is the disparity??
            y = y.matmul(conv_weight.tolist()) + conv_bias

            y_plain = torch.tensor(y.decrypt(), dtype=torch.float64).reshape(2 * self.d_model, self.L)
            y_plain = apply_glu_mode(
                y_plain,
                glu_mode=self.glu_mode,
                split_dim=0,
                poly4_coeffs=self.output_linear[1].poly4_coeffs.to(dtype=y_plain.dtype),
                poly6_coeffs=self.output_linear[1].poly6_coeffs.to(dtype=y_plain.dtype),
            )
            y_plain = y_plain.reshape(self.d_model, self.L)
            out_plain = self.decoder(y_plain.mean(dim=-1))
            out = tenseal.ckks_vector(context, out_plain.detach().flatten().tolist())
            return out


    def export_toeplitz(self, head_idx=0):
        """
        Exports the EXACT kernel matrix for FHE usage.
        """
        with torch.no_grad():
            K = self.kernel_gen(self.L)
            k_np = K[head_idx].cpu().numpy()

            # Construct Toeplitz
            T = np.zeros((self.L, self.L))
            for i in range(self.L):
                for j in range(i + 1):
                    T[i, j] = k_np[i-j]
            return T


if __name__ == "__main__":
    # CPU is enough for this tiny model; GPU used automatically if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MiniS4D(d_model=4, d_state=8, L=64, dropout=0.1).to(device)
    u = torch.randn(4, 4, 64, device=device)  # (B, d_model, L)

    # Forward
    out = model(u)
    print(f"Input shape: {u.shape} -> Output shape: {out.shape}")

    # Optional: a few training steps on random targets
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(3):
        opt.zero_grad()
        out = model(u)
        loss = out.squeeze(-1).pow(2).mean()  # dummy loss
        loss.backward()
        opt.step()
        print(f"Step {step + 1}, loss: {loss.item():.4f}")
    print("Done.")