"""
Checks of the FEMM model.

The checks print the following results:

- boundary: vector potential on the outer boundary and the effect of the shaft model
- mesh: comparison with finer meshes
- symmetry: the symmetries used in the gradient networks
- hessian: reciprocity and convexity of the co-energy from central differences

Usage: python check_model.py [boundary] [mesh] [symmetry] [hessian] [field out.png]

"""

import sys

import bsyrm_femm as bf
import field_plot as fp
import matplotlib.pyplot as plt
import numpy as np

from motulator_bearingless.utils import set_style

PORT = 48990


def check_boundary() -> None:
    """Outer boundary and the shaft models."""
    print("Shaft models, inductances (mH) and force constants (H/m) at theta_M = 0")
    print(f"{'shaft':8s}{'L_1':>8s}{'L_d':>8s}{'L_q':>8s}{'m_d':>8s}{'m_q':>8s}")
    for shaft in ("steel", 1.0, "A0"):
        m = bf.Model(bf.Params(shaft=shaft), PORT)
        L_1 = m.solve(5 + 0j, 0j)["psi_s1"].real / 5
        L_d = m.solve(0j, 5 + 0j)["psi_s2"].real / 5
        L_q = m.solve(0j, 5j)["psi_s2"].imag / 5
        m_d = m.solve(1 + 0j, 5 + 0j)["f_M"].real / 15
        m_q = m.solve(1j, 5j)["f_M"].real / 15
        print(
            f"{shaft!s:8s}{1e3 * L_1:8.2f}{1e3 * L_d:8.2f}{1e3 * L_q:8.2f}"
            f"{m_d:8.2f}{m_q:8.2f}"
        )
        if shaft == "steel":
            xy, a, _ = fp.read_solution(m.fem.replace(".fem", ".ans"))
            r = np.hypot(xy[:, 0], xy[:, 1])
            outer = r > m.P.R_so - 1e-3
            print(
                f"  |A| on the {outer.sum()} outer nodes: {np.abs(a[outer]).max():.1e}"
                f" Wb/m, largest |A| of the solution {np.abs(a).max():.1e} Wb/m"
            )
        m.close()


POINTS = (
    (0j, 10 + 5j, 0j, 0.0),
    (2 + 1j, 12 + 12j, 100e-6 - 50e-6j, 0.1222),
    (5 - 3j, 5 + 20j, 250e-6 + 100e-6j, 0.384),
    (0.5j, 3 + 1j, -30e-6j, 0.2269),
    (10 + 5j, 25 + 10j, 50e-6j, 0.0524),
)


def check_mesh() -> None:
    """Differences from a mesh that has about 12 times more elements."""
    m = bf.Model(bf.Params(), PORT)
    fine = bf.Params(
        seg_gap=0.0625,
        mesh_gap=0.07,
        mesh_slot=1.5,
        mesh_s=1.5,
        mesh_r=0.5,
        mesh_bar=0.8,
        mesh_sh=2.0,
        seg_barrier=0.3,
        seg_fillet=4.0,
        seg_tip=0.08,
        gap_rings=0.1,
        precision=1e-8,
    )
    m_f = bf.Model(fine, PORT + 1)
    for point in POINTS:
        r, r_f = m.solve(*point), m_f.solve(*point)
        print(
            f"{point}: elements {r['n_elem']} and {r_f['n_elem']}, "
            f"psi_s1 {abs(r['psi_s1'] - r_f['psi_s1']) * 1e3:.2f} mVs, "
            f"psi_s2 {abs(r['psi_s2'] - r_f['psi_s2']) * 1e3:.2f} mVs, "
            f"f_M {abs(r['f_M'] - r_f['f_M']):.2f} N of {abs(r_f['f_M']):.0f} N, "
            f"tau_M {abs(r['tau_M'] - r_f['tau_M']):.3f} Nm of {r_f['tau_M']:.2f} Nm"
        )
    m.close()
    m_f.close()


def check_symmetry() -> None:
    """Symmetries of the magnetic model."""
    m = bf.Model(bf.Params(), PORT)
    i_1, i_2, q, th = 2 + 1j, 12 + 7j, 120e-6 - 80e-6j, np.deg2rad(7.0)
    ref = m.solve(i_1, i_2, q, th)

    def report(name, r, g_1, g_2, g_f, g_t):
        print(
            f"{name:42s} psi_s1 {abs(g_1(r['psi_s1']) - ref['psi_s1']) * 1e3:.3f} mVs, "
            f"psi_s2 {abs(g_2(r['psi_s2']) - ref['psi_s2']) * 1e3:.3f} mVs, "
            f"f_M {abs(g_f(r['f_M']) - ref['f_M']):.3f} N, "
            f"tau_M {abs(g_t(r['tau_M']) - ref['tau_M']):.4f} Nm"
        )

    same = lambda x: x  # noqa: E731
    report("theta_M + 60 deg", m.solve(i_1, i_2, q, th + np.pi / 3), *[same] * 4)
    report(
        "(-i_s1, -i_s2, q_M)",
        m.solve(-i_1, -i_2, q, th),
        lambda x: -x,
        lambda x: -x,
        same,
        same,
    )
    report(
        "(-i_s1, i_s2, -q_M)",
        m.solve(-i_1, i_2, -q, th),
        lambda x: -x,
        same,
        lambda x: -x,
        same,
    )
    report(
        "(-j*i_s1, -i_s2, -j*q_M, theta_M + 30 deg)",
        m.solve(-1j * i_1, -i_2, -1j * q, th + np.pi / 6),
        lambda x: 1j * x,
        lambda x: -x,
        lambda x: 1j * x,
        same,
    )
    report(
        "(conj(i_s1), conj(i_s2), conj(q_M), -theta_M)",
        m.solve(np.conj(i_1), np.conj(i_2), np.conj(q), -th),
        np.conj,
        np.conj,
        np.conj,
        lambda x: -x,
    )
    m.close()


def check_hessian() -> None:
    """Reciprocity and convexity of the co-energy."""
    m = bf.Model(bf.Params(), PORT)
    steps = np.array([0.3, 0.3, 0.3, 0.3, 30.0, 30.0])
    base = bf.Params()
    print(base.N_series, "series turns per phase")
    points = (
        ([0, 0, 1, 0, 0, 0], 0.0),
        ([0, 0, 5, 0, 0, 0], 0.0),
        ([0, 0, 12, 12, 0, 0], 0.0873),
        ([1, 0.5, 12, 12, 50, -20], 0.0873),
        ([3, -2, 20, 20, 100, 50], 0.0873),
        ([5, 5, 5, 20, 200, 0], 0.0873),
        ([0, 0, 30, 5, 0, 0], 0.0873),
        ([10, 0, 5, 0, 200, 100], 0.0873),
        ([0, 0, 36, 0, 300, 0], 0.0),
    )

    def grad(x, th):
        r = m.solve(x[0] + 1j * x[1], x[2] + 1j * x[3], (x[4] + 1j * x[5]) * 1e-6, th)
        return np.array(
            [
                3 * r["psi_s1"].real,
                3 * r["psi_s1"].imag,
                3 * r["psi_s2"].real,
                3 * r["psi_s2"].imag,
                r["f_M"].real * 1e-6,
                r["f_M"].imag * 1e-6,
            ]
        )

    for point, th in points:
        x_0 = np.array(point, float)
        H = np.zeros((6, 6))
        for k in range(6):
            e = np.zeros(6)
            e[k] = steps[k]
            H[:, k] = (grad(x_0 + e, th) - grad(x_0 - e, th)) / (2 * steps[k])
        asym = np.abs(H - H.T).max() / np.abs(H).max()
        H = 0.5 * (H + H.T)
        scale = np.diag([17.0, 17.0, 17.0, 17.0, 1000.0, 1000.0])
        eig = np.linalg.eigvalsh(scale @ H @ scale / (3 * 17 * 0.57))
        schur = H[4:, 4:] - H[4:, :4] @ np.linalg.solve(H[:4, :4], H[:4, 4:])
        print(
            f"{x_0}: asymmetry {asym:.1e}, smallest eigenvalue {eig[0]:.4f} p.u., "
            f"stiffness at constant flux linkages "
            f"{1e6 * np.linalg.eigvalsh(schur)} N/µm"
        )
    m.close()


def plot_fields(out: str) -> None:
    """Flux lines and flux density of the planes separately and together."""
    set_style()
    m = bf.Model(bf.Params(), PORT)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    cases = ((5 + 0j, 0j), (0j, 5 + 0j), (2 + 0j, 12 + 12j))
    for ax, (i_1, i_2), label in zip(axes, cases, ("(a)", "(b)", "(c)"), strict=True):
        m.solve(i_1, i_2)
        pc = fp.plot_field(m.fem.replace(".fem", ".ans"), ax)
        ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top")
    fig.colorbar(pc, ax=axes, label="Flux density (T)", shrink=0.8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    m.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if "boundary" in args:
        check_boundary()
    if "mesh" in args:
        check_mesh()
    if "symmetry" in args:
        check_symmetry()
    if "hessian" in args:
        check_hessian()
    if "field" in args:
        plot_fields(args[args.index("field") + 1])
