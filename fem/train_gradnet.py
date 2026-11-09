"""
Training of the softmax gradient networks of the bearingless machine.

The architecture is that of `motulator_gradnet.GradNet` with one module. The current map
(`--map current`) is the gradient of the energy with respect to the per-unit flux
linkages and displacement, and the flux map (`--map flux`) the gradient of the co-energy
with respect to the per-unit currents and displacement. The periodic features of the
rotor angle (`--k`) are additional inputs. A network without them is a weighted
least-squares fit over the rotor angle, which approximates the average over the angle.

The output is averaged over the symmetry group of the machine. At the checked operating
points, the energy of the finite-element model is concave in the displacement, so that
the network of the current map approximates the energy added with `k_q*abs(q_M)**2/2`.
The linear term of the currents in the flux map has a lower bound, which represents the
leakage inductance remaining in deep saturation and keeps the map invertible also
outside the range of the data.

Usage: python train_gradnet.py dataset.npz out.npz [options]

"""

import argparse
import time
from dataclasses import asdict

import numpy as np
import torch
from motulator_gradnet import GradNet, Softmax, to_function
from torch import nn

from motulator_bearingless.gradnet import GradNetFunction, symmetry_group
from motulator_bearingless.utils import BaseValues, NominalValues, torque

NOM = NominalValues(U=220, I=12, f=50, P=2.6e3, tau=14.5)
BASE = BaseValues.from_nominal(NOM, n_p=2)
I_B, PSI_B = float(BASE.i), float(BASE.psi)
E_B = 3 * I_B * PSI_B  # Base value (J) of the state function
Q_B = 1e-3  # Base value (m) of the displacement, the air gap
L_MIN = 2e-3  # Lower bound (H) of the incremental inductance of the flux map
MU_MIN = L_MIN * I_B / PSI_B  # In per-unit quantities
TRAIN_SETS, TEST_SETS = (0, 6, 7), (1,)


# %%
class SymmetrizedNet(nn.Module):
    """Gradient network averaged over the symmetry group, with the fixed linear term."""

    def __init__(self, net: GradNet, group: np.ndarray, mu_min: float) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("group", torch.tensor(group, dtype=torch.float32))
        currents = torch.zeros(group.shape[-1])
        currents[:4] = mu_min
        self.register_buffer("mu_min", currents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the gradient, `x` of shape (batch, n)."""
        g = self.group
        y = (self.net(x[None, ...] @ g.transpose(1, 2)) @ g).mean(dim=0)
        return y + self.mu_min * x


# %%
def load_tensors(
    path: str, sets: tuple[int, ...], is_flux_map: bool, k: tuple[int, ...], k_q: float
) -> dict[str, torch.Tensor]:
    """Inputs and targets of the given sets in per-unit quantities."""
    with np.load(path) as data:
        mask = np.isin(data["set"], sets)
        d = {key: data[key][mask] for key in data.files}
    i1, i2 = d["i_s1"] / I_B, d["i_s2"] / I_B
    psi1, psi2 = d["psi_s1"] / PSI_B, d["psi_s2"] / PSI_B
    q, f = d["q_M"] / Q_B, d["f_M"] * Q_B / E_B
    if is_flux_map:
        u1, u2, y1, y2, y_f, sign = i1, i2, psi1, psi2, f, 1.0
    else:
        u1, u2, y1, y2, y_f, sign = psi1, psi2, i1, i2, k_q * q - f, -1.0
    inputs = [u1, u2, q, *(np.exp(1j * order * d["theta_M"]) for order in k)]
    tau_theta = d["tau_M"] - torque(
        d["i_s1"], d["i_s2"], d["psi_s1"], d["psi_s2"], d["q_M"], d["f_M"]
    )

    def real(*z: np.ndarray) -> torch.Tensor:
        v = np.stack([g(x) for x in z for g in (np.real, np.imag)], axis=-1)
        return torch.tensor(v, dtype=torch.float32)

    return {
        "x": real(*inputs),
        "y": real(y1, y2, y_f),
        "t": torch.tensor(sign * tau_theta / E_B, dtype=torch.float32),
        "theta": torch.tensor(d["theta_M"], dtype=torch.float32),
        "f_abs": torch.tensor(np.abs(f), dtype=torch.float32),
    }


def angle_derivative(
    y: torch.Tensor, theta: torch.Tensor, k: tuple[int, ...]
) -> torch.Tensor:
    """Derivative of the state function with respect to the rotor angle."""
    dw = torch.zeros_like(theta)
    for m, order in enumerate(k):
        c, s = torch.cos(order * theta), torch.sin(order * theta)
        dw = dw + order * (c * y[:, 7 + 2 * m] - s * y[:, 6 + 2 * m])
    return dw


class Loss:
    """
    Weighted mean squared error, the error scales in per-unit quantities.

    The error scales of the two-pole output and the force grow with their magnitudes,
    so that their relative accuracy is emphasized in the normal operating range.

    """

    def __init__(self, k: tuple[int, ...], args: argparse.Namespace) -> None:
        self.k = k
        self.s_1, self.y_1 = args.s_1, args.y_1
        self.s_2 = args.s_2
        self.s_f, self.f_0 = args.s_f * Q_B / E_B, args.f_0 * Q_B / E_B
        self.s_t = args.s_t / E_B

    def __call__(self, model: nn.Module, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the loss of a batch."""
        out = model(b["x"])
        e = (out[:, :6] - b["y"]) ** 2
        scale_1 = self.s_1 * (1 + torch.hypot(b["y"][:, 0], b["y"][:, 1]) / self.y_1)
        scale_f = self.s_f * (1 + b["f_abs"] / self.f_0)
        loss = (
            ((e[:, 0] + e[:, 1]) / scale_1**2).mean()
            + ((e[:, 2] + e[:, 3]) / self.s_2**2).mean()
            + ((e[:, 4] + e[:, 5]) / scale_f**2).mean()
        )
        if self.k:
            dw = angle_derivative(out, b["theta"], self.k)
            loss = loss + ((dw - b["t"]) ** 2).mean() / self.s_t**2
        return loss


def errors(model: nn.Module, t: dict[str, torch.Tensor], k: tuple[int, ...]) -> str:
    """RMS errors of the outputs in per-unit quantities, and of the force and torque."""
    with torch.no_grad():
        out = model(t["x"])
        e = (out[:, :6] - t["y"]).numpy()
        tau = (angle_derivative(out, t["theta"], k) - t["t"]).numpy()
    rms = [float(np.sqrt(2 * np.mean(e[:, m : m + 2] ** 2))) for m in (0, 2, 4)]
    tau_rms = float(np.sqrt(np.mean(tau**2)))
    return f"{rms[0]:.4f} {rms[1]:.4f} {rms[2] * E_B / Q_B:.1f}N {tau_rms * E_B:.2f}Nm"


def save(args: argparse.Namespace, net: GradNet, mu_min: float, k_q: float) -> None:
    """Save the network, the fixed linear term included in the trained one."""
    fcn = to_function(net)
    par = asdict(fcn) | {"mu_log": np.log(np.exp(fcn.mu_log) + mu_min)}
    GradNetFunction(
        **par, map_type=f"{args.map}_map", k=tuple(args.k), q_base=Q_B, k_q=k_q
    ).save(args.out)


def train(args: argparse.Namespace) -> None:
    """Train the network and save it."""
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    k = tuple(args.k)
    is_flux_map = args.map == "flux"
    if is_flux_map and k:
        raise ValueError("The flux map of the controller has no spatial harmonics")
    k_q = 0.0 if is_flux_map else 1e6 * args.k_q * Q_B**2 / E_B
    mu_min = MU_MIN if is_flux_map else 0.0
    t_train = load_tensors(args.dataset, TRAIN_SETS, is_flux_map, k, k_q)
    t_test = load_tensors(args.dataset, TEST_SETS, is_flux_map, k, k_q)
    n = t_train["x"].shape[0]
    print(f"Training points {n}, test points {t_test['x'].shape[0]}")
    net = GradNet(
        6 + 2 * len(k),
        4,
        embed_dim=args.embed,
        activation=Softmax,
        mu_log0=-1.0,
        psi_base=PSI_B,
        i_base=I_B,
    )
    model = SymmetrizedNet(net, symmetry_group(k), mu_min)
    loss_fcn = Loss(k, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * (n // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, args.lr / 50)
    t_0 = time.time()

    def report(label: str) -> None:
        print(
            f"{label} {time.time() - t_0:6.0f}s train {errors(model, t_train, k)} "
            f"test {errors(model, t_test, k)}",
            flush=True,
        )

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        for m in range(0, n - args.batch + 1, args.batch):
            batch = {key: v[perm[m : m + args.batch]] for key, v in t_train.items()}
            opt.zero_grad()
            loss_fcn(model, batch).backward()
            opt.step()
            sched.step()
        if epoch % 100 == 0:
            report(f"{epoch:5d}")
    # Final polishing of the full-batch loss
    opt_l = torch.optim.LBFGS(
        model.parameters(),
        max_iter=args.lbfgs,
        history_size=50,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt_l.zero_grad()
        loss = loss_fcn(model, t_train)
        loss.backward()
        return loss

    opt_l.step(closure)
    report("L-BFGS")
    save(args, net, mu_min, k_q)


# %%
def get_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("out")
    p.add_argument("--map", choices=("current", "flux"), required=True)
    p.add_argument("--k", type=int, nargs="*", default=[], help="Periodic features")
    p.add_argument("--embed", type=int, default=64, help="Hidden units")
    p.add_argument("--k_q", type=float, default=10.0, help="Quadratic term (N/µm)")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lbfgs", type=int, default=300, help="L-BFGS iterations")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--s_1", type=float, default=0.002, help="Error scale of the h = 1")
    p.add_argument("--y_1", type=float, default=0.1, help="Growth of the h = 1 scale")
    p.add_argument("--s_2", type=float, default=0.01, help="Error scale of the h = 2")
    p.add_argument("--s_f", type=float, default=5.0, help="Error scale of the force")
    p.add_argument("--f_0", type=float, default=167.0, help="Growth of the force scale")
    p.add_argument("--s_t", type=float, default=0.2, help="Error scale of the torque")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
