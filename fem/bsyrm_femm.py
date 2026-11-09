"""
FEMM model of the 2.6-kW six-phase bearingless synchronous reluctance machine.

The whole cross-section is modeled, since the two-pole and four-pole fields and the
radial rotor displacement rule out any periodicity. The rotor center stays at the
origin, and the stator is translated by `-q_M`, so that the torque integral refers to
the rotor center.

Conventions, as in Ahmed et al. (https://doi.org/10.1109/ECCE53617.2023.10361945):

- Phase k has the magnetic axis at the mechanical angle `(k - 1)*60` deg for both the
  two-pole and four-pole components of its winding function. Each phase is a coil
  group spanning 90 deg, with the conductors of positive turns (out of the plane) at
  the axis + 45 deg and those of negative turns at the axis - 45 deg, so that a positive
  current gives the outward flux at the axis.
- The rotor d-axis (no barriers) is at the angle `theta_M` and the q-axis (center of the
  barriers) at `theta_M + 45` deg.
- Peak-valued space vectors `x_sh = (1/3)*sum(x_k*alpha**(h*(k - 1)))` and rotor
  coordinates `i_s1 = exp(-1j*theta_M)*i_s1_ab`, `i_s2 = exp(-2j*theta_M)*i_s2_ab`.

Boundary conditions: A = 0 on the outer surface of the stator. There is no other outer
boundary: the shaft is a meshed part of the rotor, through which the two-pole flux can
close.

"""

import os
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares

FEMM_QT = os.environ.get("FEMM_QT", os.path.expanduser("~/GitHub/femm-qt"))
sys.path.insert(0, FEMM_QT + "/pyfemm")
os.environ.setdefault("FEMM_DIR", FEMM_QT + "/build/bin")
import femm  # noqa: E402

ALPHA = np.exp(1j * np.pi / 3)
DEG = np.pi / 180
MU_0 = 4e-7 * np.pi

# Steel data (B in T, H in A/m) of the stator and rotor laminations and the solid shaft
B_TABLE = np.arange(21) / 10
# fmt: off
H_STATOR = np.array([
    0, 19, 39, 58, 77, 96, 116, 135, 154, 173, 195, 228, 266, 328, 518, 1051, 2300,
    4834, 9268, 16215, 26966,
], dtype=float)
H_ROTOR = np.array([
    0, 44, 87, 131, 175, 218, 262, 306, 349, 393, 440, 509, 594, 694, 860, 1390, 2891,
    5882, 10755, 16993, 26719,
], dtype=float)
H_SHAFT = np.array([
    0, 48, 97, 145, 194, 242, 291, 339, 388, 436, 492, 601, 758, 980, 1379, 2137, 3585,
    6458, 10373, 15274, 23176,
], dtype=float)
# fmt: on
BH_STATOR = np.array([B_TABLE, H_STATOR])
BH_ROTOR = np.array([B_TABLE, H_ROTOR])
BH_SHAFT = np.array([B_TABLE, H_SHAFT])


def extend_bh(bh: np.ndarray) -> np.ndarray:
    """Extend a BH curve above its last point, with a constant polarization."""
    b, h = bh[0], bh[1]
    j_s = b[-1] - MU_0 * h[-1]
    h_ext = h[-1] * np.array([1.5, 2, 3, 5, 10, 20, 40])
    b_ext = j_s + MU_0 * h_ext
    return np.array([np.r_[b, b_ext], np.r_[h, h_ext]])


@dataclass
class Params:
    """Dimensions (mm), materials, and meshing of the machine."""

    # Stator
    Q_s: int = 36  # Slots
    n_s: int = 16  # Conductors per slot (single-layer winding)
    R_si: float = 72.5  # Inner radius
    R_so: float = 117.5  # Outer radius
    H_1: float = 20.0  # Slot depth from the bore
    H_11: float = 1.3  # Tooth-tip height at the slot opening
    H_13: float = 13.9  # Height of the trapezoidal part of the slot
    B_11: float = 3.5  # Slot opening
    B_12: float = 5.5  # Slot width at the wedge line
    B_13: float = 7.9  # Slot width at the bottom rounding
    # Rotor
    p: int = 2  # Pole pairs
    L: float = 195.25  # Stack length
    R_sh: float = 27.0  # Shaft radius
    g: float = 1.0  # Air gap
    h_rib: float = 1.5  # Rib thickness at the barrier ends
    # Radii of the barrier edges along the q-axis, from the outermost inward
    r_bar: tuple[float, ...] = (
        65.014,
        63.324,
        58.186,
        54.470,
        47.929,
        42.846,
        37.030,
        30.351,
    )
    n_seg: int = 15  # Points per barrier edge
    # Lamination fill factors
    k_lam_s: float = 0.975
    k_lam_r: float = 0.98
    # Shaft: "steel" (meshed solid steel), a number (relative permeability of a meshed
    # linear shaft, 1.0 for a nonmagnetic shaft), or "A0" (hole with A = 0 on its
    # surface, i.e., no flux in the shaft)
    shaft: str | float = "steel"
    # Mesh
    gap_rings: float = 0.15  # Distance of the auxiliary gap rings from the surfaces
    seg_gap: float = 0.3  # Maximum segment of the gap arcs (deg)
    mesh_gap: float = 0.35  # Element size in the air gap
    mesh_slot: float = 6.0
    mesh_s: float = 6.0  # Stator yoke and teeth
    mesh_r: float = 3.0  # Rotor iron
    mesh_bar: float = 4.0  # Barrier air
    mesh_sh: float = 10.0  # Shaft
    seg_barrier: float = 2.0  # Maximum side along the barrier edges
    seg_fillet: float = 25.0  # Maximum segment of the barrier-end fillets (deg)
    seg_tip: float = 0.5  # Maximum side at the slot opening
    precision: float = 1e-7
    min_angle: float = 30.0

    @property
    def R_r(self) -> float:
        """Rotor outer radius."""
        return self.R_si - self.g

    @property
    def N_series(self) -> int:
        """Series turns per phase."""
        return self.Q_s // 12 * self.n_s


# %%
def _streamline(P: Params):
    """Functions of the flux-barrier streamlines around the shaft circle."""
    p, c = P.p, P.R_sh ** (2 * P.p)

    def rho(xi, phi):
        return (phi + np.sqrt(phi**2 + 4 * c * np.sin(p * xi) ** 2)) / (
            2 * np.sin(p * xi)
        )

    def rad(xi, phi):
        return rho(xi, phi) ** (1 / p)

    def tangent(xi, phi):
        r = rad(xi, phi)
        k = p * (r ** (p - 1) - c / r ** (p + 1))
        vx = k * np.cos(p * xi) * np.cos(xi) + k * np.sin(p * xi) * np.sin(xi)
        vy = k * np.cos(p * xi) * np.sin(xi) - k * np.sin(p * xi) * np.cos(xi)
        return vx, vy

    def dlen(xi, phi):
        r_p = rho(xi, phi)
        dr_p = (
            -p
            * phi
            * r_p
            / (np.sqrt(phi**2 + 4 * c * np.sin(p * xi) ** 2) * np.tan(p * xi))
        )
        dr = (1 / p) * r_p ** ((1 - p) / p) * dr_p
        return np.sqrt(dr**2 + r_p ** (2 / p))

    return rad, tangent, dlen


def barrier_edges(P: Params) -> list[dict]:
    """
    Edges of one pole's barriers, with the q-axis on the angle `180/(2p)` deg.

    Each barrier follows two streamlines from the q-axis toward the rotor surface and
    ends with two circular fillets meeting at the point E on the middle streamline.

    Returns
    -------
    list[dict]
        For each barrier, `top` and `bot` (edge points, from the q-axis to the fillet
        start, shape (n_seg, 2)), `E` (fillet tip), and the fillet sweep angles
        `sweep_top` and `sweep_bot` (deg).

    """
    p, c = P.p, P.R_sh ** (2 * P.p)
    rad, tangent, dlen = _streamline(P)
    r_bar = np.array(P.r_bar)
    r_mid = 0.5 * (r_bar[::2] + r_bar[1::2])
    phi_bar = r_bar**p - c / r_bar**p
    phi_mid = r_mid**p - c / r_mid**p
    R_e = P.R_r - P.h_rib
    phi_e = R_e**p - c / R_e**p
    xi_q = np.pi / (2 * p)

    barriers = []
    for nb, phi_m in enumerate(phi_mid):
        xi_E = np.arcsin(phi_m / phi_e) / p
        E = R_e * np.array([np.cos(xi_E), np.sin(xi_E)])
        v_E = np.array(tangent(xi_E, phi_m))
        v_E = v_E / np.hypot(*v_E)
        edges = {}
        for name, phi in (("top", phi_bar[2 * nb]), ("bot", phi_bar[2 * nb + 1])):
            # Fillet: a circle through E, normal to the middle streamline at E and
            # tangent to the edge streamline at the fillet start C
            xi_CD = np.arcsin(phi / phi_e) / p
            CD = R_e * np.array([np.cos(xi_CD), np.sin(xi_CD)])
            d = np.hypot(*(CD - E))

            def eqs(x, phi=phi, E=E, v_E=v_E):
                xi = np.arctan2(x[1], x[0])
                r = rad(xi, phi)
                vx, vy = tangent(xi, phi)
                return [
                    x[0] - r * np.cos(xi),
                    x[1] - r * np.sin(xi),
                    (x[0] - x[2]) ** 2
                    + (x[1] - x[3]) ** 2
                    - ((x[2] - E[0]) ** 2 + (x[3] - E[1]) ** 2),
                    (x[2] - x[0]) * vx + (x[3] - x[1]) * vy,
                    (E[0] - x[2]) * v_E[1] - (E[1] - x[3]) * v_E[0],
                ]

            x_0 = np.r_[CD - d * v_E, E - d * v_E]
            sol = least_squares(eqs, x_0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if np.linalg.norm(sol.fun) > 1e-6:
                raise RuntimeError(f"Barrier {nb}: no fillet found for {name}")
            C, center = sol.x[:2], sol.x[2:]
            # Edge points at equal arc lengths from the q-axis to C
            xi_C = np.arctan2(C[1], C[0])

            def arc_len(xi, phi=phi):
                return quad(dlen, xi, xi_q, args=(phi,), limit=200)[0]

            l_tot = arc_len(xi_C)
            pts = np.zeros((P.n_seg, 2))
            r_q = rad(xi_q, phi)
            pts[0] = r_q * np.array([np.cos(xi_q), np.sin(xi_q)])
            pts[-1] = C
            for k in range(1, P.n_seg - 1):
                l_k = l_tot * k / (P.n_seg - 1)
                xi_k = brentq(
                    lambda xi, l_k=l_k: arc_len(xi) - l_k, xi_C, xi_q, xtol=1e-12
                )
                r_k = rad(xi_k, phi)
                pts[k] = r_k * np.array([np.cos(xi_k), np.sin(xi_k)])
            edges[name] = pts
            # Sweep angle of the fillet (CCW positive)
            a_C = np.arctan2(C[1] - center[1], C[0] - center[0])
            a_E = np.arctan2(E[1] - center[1], E[0] - center[0])
            sweep = (a_C - a_E if name == "top" else a_E - a_C) / DEG
            edges["sweep_" + name] = (sweep + 180) % 360 - 180
        edges["E"] = E
        barriers.append(edges)
    return barriers


def _mirror_q(pts: np.ndarray, p: int) -> np.ndarray:
    """Mirror points about the q-axis at the angle `180/(2p)` deg."""
    a = 2 * np.pi / (2 * p)
    c, s = np.cos(a), np.sin(a)
    return np.stack(
        [c * pts[..., 0] + s * pts[..., 1], s * pts[..., 0] - c * pts[..., 1]], axis=-1
    )


# %%
@dataclass
class Geometry:
    """Planar straight-line graph of the machine, by parts that move together."""

    nodes: list = field(default_factory=list)  # (x, y, part)
    segs: list = field(default_factory=list)  # (n0, n1, max_side, bdry)
    arcs: list = field(default_factory=list)  # (n0, n1, sweep, max_seg, bdry)
    labels: list = field(default_factory=list)  # (x, y, part, material, size, circ...)

    def node(self, x: float, y: float, part: str) -> int:
        """Add a node, reusing an existing one at the same position in the part."""
        key = (round(x, 9), round(y, 9), part)
        if not hasattr(self, "_index"):
            self._index = {}
        if key in self._index:
            return self._index[key]
        self.nodes.append((x, y, part))
        self._index[key] = len(self.nodes) - 1
        return self._index[key]

    def seg(self, a, b, part: str, max_side: float = -1, bdry: int = 0) -> None:
        """Add a segment between the points `a` and `b`."""
        self.segs.append(
            (self.node(*a, part), self.node(*b, part), max_side, bdry, part)
        )

    def arc(self, a, b, sweep: float, part: str, max_seg: float, bdry: int = 0) -> None:
        """Add an arc, CCW from `a` to `b`, sweeping `sweep` deg."""
        self.arcs.append(
            (self.node(*a, part), self.node(*b, part), sweep, max_seg, bdry, part)
        )

    def label(self, x, y, part, block, size=-1.0, circuit=0, group=0, turns=1):
        """Add a block label."""
        self.labels.append((x, y, part, block, size, circuit, group, turns))


def _pol(r: float, a: float) -> tuple[float, float]:
    return (r * np.cos(a * DEG), r * np.sin(a * DEG))


def _rot(pt, a: float) -> tuple[float, float]:
    c, s = np.cos(a * DEG), np.sin(a * DEG)
    return (c * pt[0] - s * pt[1], s * pt[0] + c * pt[1])


def phase_slots(P: Params) -> dict[int, list[tuple[float, int]]]:
    """Slot angles (deg) and turns of each phase, see the module docstring."""
    q = P.Q_s // 12
    tau = 360.0 / P.Q_s
    slots = {}
    for k in range(6):
        axis = 60.0 * k
        side = []
        for j in range(q):
            offset = (j - (q - 1) / 2) * tau
            side.append(((axis + 45.0 + offset) % 360, P.n_s))
            side.append(((axis - 45.0 + offset) % 360, -P.n_s))
        slots[k] = side
    return slots


# Block types of the .fem file (1-based), the boundary and circuit tables are fixed
BLOCKS = {"air": 1, "copper": 2, "stator": 3, "rotor": 4, "shaft": 5}
BDRY_A0 = 1
GROUPS = {"gap": 10, "rotor": 1, "barrier": 2, "shaft": 3, "stator": 20, "slot": 21}


def _add_stator(g: Geometry, P: Params) -> None:
    """Add the stator slots centered at the angles `tau/2 + n*tau` and the outer arc."""
    tau = 360.0 / P.Q_s
    x_2 = np.sqrt(P.R_si**2 - P.B_11**2 / 4)
    x_3 = x_2 + P.H_11
    x_4 = x_2 + P.H_1 - P.B_13 / 2 - P.H_13
    x_5 = x_4 + P.H_13
    x_6 = x_2 + P.H_1
    a_o = np.arctan2(P.B_11 / 2, x_2) / DEG
    for n in range(P.Q_s):
        a = tau / 2 + n * tau
        A1 = _rot((x_2, -P.B_11 / 2), a)
        A2 = _rot((x_3, -P.B_11 / 2), a)
        A3 = _rot((x_4, -P.B_12 / 2), a)
        A4 = _rot((x_5, -P.B_13 / 2), a)
        B1 = _rot((x_2, P.B_11 / 2), a)
        B2 = _rot((x_3, P.B_11 / 2), a)
        B3 = _rot((x_4, P.B_12 / 2), a)
        B4 = _rot((x_5, P.B_13 / 2), a)
        C = _rot((x_6, 0.0), a)
        g.seg(A1, A2, "s", P.seg_tip)
        g.seg(A2, A3, "s", P.seg_tip)
        g.seg(A3, A4, "s")
        g.seg(B1, B2, "s", P.seg_tip)
        g.seg(B2, B3, "s", P.seg_tip)
        g.seg(B3, B4, "s")
        g.seg(A3, B3, "s")  # Wedge line: the slot opening and the wedge are air
        g.arc(A4, C, 90.0, "s", 30.0)
        g.arc(C, B4, 90.0, "s", 30.0)
        # Tooth arc at the bore, from this slot to the next one
        next_A1 = _rot((x_2, -P.B_11 / 2), a + tau)
        g.arc(B1, next_A1, tau - 2 * a_o, "s", P.seg_gap)
    # Outer boundary with A = 0
    for n in range(4):
        a_0, a_1 = 90.0 * n, 90.0 * (n + 1)
        g.arc(_pol(P.R_so, a_0), _pol(P.R_so, a_1), 90.0, "s", 2.0, BDRY_A0)


def _add_gap_rings(g: Geometry, P: Params) -> None:
    """Add the rotor surface and the auxiliary rings regularizing the gap mesh."""
    R_ring_s = P.R_si - P.gap_rings
    R_ring_r = P.R_r + P.gap_rings
    for n in range(4):
        a_0, a_1 = 90.0 * n, 90.0 * (n + 1)
        g.arc(_pol(R_ring_s, a_0), _pol(R_ring_s, a_1), 90.0, "s", P.seg_gap)
        g.arc(_pol(P.R_r, a_0), _pol(P.R_r, a_1), 90.0, "r", P.seg_gap)
        g.arc(_pol(R_ring_r, a_0), _pol(R_ring_r, a_1), 90.0, "r", P.seg_gap)


def _add_barrier(g: Geometry, P: Params, b: dict, a_pole: float) -> None:
    """Add the both halves of a flux barrier of the pole at the angle `a_pole` (deg)."""

    def turned(pts):
        return _rot(pts, a_pole)

    s_t, s_u = abs(b["sweep_top"]), abs(b["sweep_bot"])
    for mirrored in (False, True):
        t = _mirror_q(b["top"], P.p) if mirrored else b["top"]
        u = _mirror_q(b["bot"], P.p) if mirrored else b["bot"]
        e = _mirror_q(b["E"], P.p) if mirrored else b["E"]
        for k in range(P.n_seg - 1):
            g.seg(turned(t[k]), turned(t[k + 1]), "r", P.seg_barrier)
            g.seg(turned(u[k]), turned(u[k + 1]), "r", P.seg_barrier)
        # Fillets, CCW from the first to the second point
        if mirrored:
            g.arc(turned(t[-1]), turned(e), s_t, "r", P.seg_fillet)
            g.arc(turned(e), turned(u[-1]), s_u, "r", P.seg_fillet)
        else:
            g.arc(turned(e), turned(t[-1]), s_t, "r", P.seg_fillet)
            g.arc(turned(u[-1]), turned(e), s_u, "r", P.seg_fillet)
    x_m, y_m = turned(0.5 * (b["top"][0] + b["bot"][0]))
    g.label(x_m, y_m, "r", "air", P.mesh_bar, group=GROUPS["barrier"])


def _add_labels(g: Geometry, P: Params) -> None:
    """Add the block labels of the iron parts, the air gap, and the slots."""
    x_6 = np.sqrt(P.R_si**2 - P.B_11**2 / 4) + P.H_1
    x_4 = x_6 - P.B_13 / 2 - P.H_13
    x_rib = P.R_r - P.h_rib / 2
    g.label(x_rib, 0.0, "r", "rotor", P.mesh_r, group=GROUPS["rotor"])
    g.label(P.R_r + P.gap_rings / 2, 0.0, "r", "air", P.mesh_gap, group=GROUPS["gap"])
    r_mid = 0.5 * (P.R_si + P.R_r)
    g.label(r_mid, 0.0, "m", "air", P.mesh_gap, group=GROUPS["gap"])
    g.label(P.R_si - P.gap_rings / 2, 0.0, "s", "air", P.mesh_gap, group=GROUPS["gap"])
    g.label(0.5 * (x_6 + P.R_so), 0.0, "s", "stator", P.mesh_s, group=GROUPS["stator"])
    x_slot = 0.5 * (x_6 + x_4)
    for k, side in phase_slots(P).items():
        for a, turns in side:
            x, y = _pol(x_slot, a)
            g.label(x, y, "s", "copper", P.mesh_slot, k + 1, GROUPS["slot"], turns)


def build_geometry(P: Params) -> Geometry:
    """Build the machine with a centric rotor at the rotor angle of zero."""
    g = Geometry()
    _add_stator(g, P)
    _add_gap_rings(g, P)
    barriers = barrier_edges(P)
    for pole in range(2 * P.p):
        for b in barriers:
            _add_barrier(g, P, b, pole * 180.0 / P.p)
    # The shaft is meshed, or a hole with A = 0 on its surface
    bdry = BDRY_A0 if P.shaft == "A0" else 0
    for n in range(4):
        a_0, a_1 = 90.0 * n, 90.0 * (n + 1)
        g.arc(_pol(P.R_sh, a_0), _pol(P.R_sh, a_1), 90.0, "r", 3.0, bdry)
    if P.shaft != "A0":
        g.label(0.0, 0.0, "r", "shaft", P.mesh_sh, group=GROUPS["shaft"])
    _add_labels(g, P)
    return g


# %%
def _fem_header(P: Params) -> str:
    """Header of a magnetostatic .fem file: materials, boundaries, and circuits."""

    def block(name, mu=1.0, bh=None, fill=1.0, sigma=0.0):
        s = (
            f'  <BeginBlock>\n    <BlockName> = "{name}"\n    <Mu_x> = {mu}\n'
            f"    <Mu_y> = {mu}\n    <H_c> = 0\n    <H_cAngle> = 0\n    <J_re> = 0\n"
            f"    <J_im> = 0\n    <Sigma> = {sigma}\n    <d_lam> = 0\n    <Phi_h> = 0\n"
            f"    <Phi_hx> = 0\n    <Phi_hy> = 0\n    <LamType> = 0\n"
            f"    <LamFill> = {fill}\n    <NStrands> = 0\n    <WireD> = 0\n"
        )
        if bh is None:
            s += "    <BHPoints> = 0\n"
        else:
            s += f"    <BHPoints> = {bh.shape[1]}\n"
            for b, h in bh.T:
                s += f"      {b:.10g}\t{h:.10g}\n"
        return s + "  <EndBlock>\n"

    shaft_mu = 1.0
    shaft_bh = extend_bh(BH_SHAFT)
    if P.shaft != "steel":
        shaft_bh = None
        shaft_mu = 1.0 if P.shaft == "A0" else float(P.shaft)
    txt = (
        "[Format]      =  4.0\n[Frequency]   =  0\n"
        f"[Precision]   =  {P.precision}\n[MinAngle]    =  {P.min_angle}\n"
        f"[DoSmartMesh] =  0\n[Depth]       =  {P.L}\n[LengthUnits] =  millimeters\n"
        "[ProblemType] =  planar\n[Coordinates] =  cartesian\n[ACSolver]    =  1\n"
        '[PrevType]    =  0\n[PrevSoln]    =  ""\n[Comment]     =  ""\n'
        "[PointProps]   = 0\n[BdryProps]   = 1\n"
        '  <BeginBdry>\n    <BdryName> = "A0"\n    <BdryType> = 0\n    <A_0> = 0\n'
        "    <A_1> = 0\n    <A_2> = 0\n    <Phi> = 0\n    <c0> = 0\n    <c0i> = 0\n"
        "    <c1> = 0\n    <c1i> = 0\n    <Mu_ssd> = 0\n    <Sigma_ssd> = 0\n"
        "    <innerangle> = 0\n    <outerangle> = 0\n  <EndBdry>\n"
        "[BlockProps]  = 5\n"
    )
    txt += block("air")
    txt += block("copper")
    txt += block("stator", bh=extend_bh(BH_STATOR), fill=P.k_lam_s)
    txt += block("rotor", bh=extend_bh(BH_ROTOR), fill=P.k_lam_r)
    txt += block("shaft", mu=shaft_mu, bh=shaft_bh)
    txt += "[CircuitProps]  = 6\n"
    return txt


def _phase_currents(i_s1_ab: complex, i_s2_ab: complex) -> np.ndarray:
    """Phase currents (A) from the space vectors in stator coordinates."""
    k = np.arange(6)
    return (i_s1_ab * ALPHA**-k).real + (i_s2_ab * ALPHA ** (-2 * k)).real


def write_fem(
    path: str,
    P: Params,
    geo: Geometry,
    i_k: np.ndarray,
    theta_M: float = 0.0,
    q_M_ab: complex = 0j,
) -> None:
    """
    Write a .fem file.

    Parameters
    ----------
    path : str
        File path.
    P : Params
        Parameters.
    geo : Geometry
        Geometry from `build_geometry`.
    i_k : ndarray, shape (6,)
        Phase currents (A).
    theta_M : float, optional
        Rotor angle (rad).
    q_M_ab : complex, optional
        Rotor displacement (mm) in stator coordinates.

    """
    c, s = np.cos(theta_M), np.sin(theta_M)
    sx, sy = -q_M_ab.real, -q_M_ab.imag  # The stator moves, the rotor center stays

    def move(x, y, part):
        if part == "s":
            return x + sx, y + sy
        if part == "r":
            return c * x - s * y, s * x + c * y
        # Middle of the gap, halfway between the rings of the stator and the rotor
        return x + 0.5 * sx, y + 0.5 * sy

    out = [_fem_header(P)]
    for k, i in enumerate(i_k):
        out.append(
            f'  <BeginCircuit>\n    <CircuitName> = "P{k + 1}"\n'
            f"    <TotalAmps_re> = {i:.15g}\n    <TotalAmps_im> = 0\n"
            "    <CircuitType> = 1\n  <EndCircuit>\n"
        )
    out.append(f"[NumPoints] = {len(geo.nodes)}\n")
    for x, y, part in geo.nodes:
        x_m, y_m = move(x, y, part)
        out.append(f"{x_m:.14g}\t{y_m:.14g}\t0\t0\n")
    out.append(f"[NumSegments] = {len(geo.segs)}\n")
    for n0, n1, max_side, bdry, _ in geo.segs:
        out.append(f"{n0}\t{n1}\t{max_side}\t{bdry}\t0\t0\n")
    out.append(f"[NumArcSegments] = {len(geo.arcs)}\n")
    for n0, n1, sweep, max_seg, bdry, _ in geo.arcs:
        out.append(f"{n0}\t{n1}\t{sweep:.14g}\t{max_seg}\t{bdry}\t0\t0\t1\n")
    if P.shaft == "A0":
        out.append(f"[NumHoles] = 1\n0\t0\t{GROUPS['shaft']}\n")
    else:
        out.append("[NumHoles] = 0\n")
    out.append(f"[NumBlockLabels] = {len(geo.labels)}\n")
    for x, y, part, block, size, circuit, group, turns in geo.labels:
        x_m, y_m = move(x, y, part)
        out.append(
            f"{x_m:.14g}\t{y_m:.14g}\t{BLOCKS[block]}\t{size}\t{circuit}\t0\t{group}"
            f"\t{turns}\t0\n"
        )
    with open(path, "w") as f:
        f.write("".join(out))


# %%
class Model:
    """A FEMM instance solving the machine at given currents and rotor positions."""

    def __init__(
        self,
        P: Params,
        port: int,
        workdir: str | None = None,
        geo: Geometry | None = None,
    ) -> None:
        self.P = P
        self.geo = build_geometry(P) if geo is None else geo
        self.f = femm.FemmSession(port=port, headless=True)
        self.f.newdocument(0)
        base = workdir or os.environ.get("BSYRM_FEMM_WORK", "/dev/shm/bsyrm_femm")
        os.makedirs(base, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix=f"p{port}_", dir=base)
        self.fem = os.path.join(self.dir, "m.fem")

    def close(self) -> None:
        """Close the FEMM instance and remove the temporary files."""
        self.f.close()
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def solve(
        self, i_s1: complex, i_s2: complex, q_M: complex = 0j, theta_M: float = 0.0
    ) -> dict:
        """
        Solve one operating point.

        Parameters
        ----------
        i_s1, i_s2 : complex
            Stator currents (A) of the two planes in rotor coordinates.
        q_M : complex, optional
            Rotor displacement (m) in rotor coordinates.
        theta_M : float, optional
            Rotor angle (rad).

        Returns
        -------
        dict
            `psi_s1`, `psi_s2` (Vs, rotor coordinates), `f_M` (N, rotor coordinates),
            `tau_M` (Nm, about the rotor center), `W`, `W_co` (J), `psi_k` (Vs, phases),
            `i_k` (A, phases), and the mesh size `n_elem`.

        """
        e = np.exp(1j * theta_M)
        i_k = _phase_currents(i_s1 * e, i_s2 * e**2)
        q_M_ab = complex(q_M * e) * 1e3  # mm
        write_fem(self.fem, self.P, self.geo, i_k, theta_M, q_M_ab)
        f = self.f
        f.opendocument(self.fem)
        f.mi_analyze(1)
        f.mi_loadsolution()
        psi_k = np.array([f.mo_getcircuitproperties(f"P{k + 1}")[2] for k in range(6)])
        f.mo_clearblock()
        for group in (GROUPS["rotor"], GROUPS["barrier"], GROUPS["shaft"]):
            f.mo_groupselectblock(group)
        f_x = f.mo_blockintegral(18)
        f_y = f.mo_blockintegral(19)
        tau = f.mo_blockintegral(22)
        f.mo_clearblock()
        f.mo_groupselectblock()
        W = f.mo_blockintegral(2)
        W_co = f.mo_blockintegral(17)
        n_elem = f.mo_numelements()
        f.mo_close()
        f.mi_close()
        k = np.arange(6)
        psi_s1_ab = np.sum(psi_k * ALPHA**k) / 3
        psi_s2_ab = np.sum(psi_k * ALPHA ** (2 * k)) / 3
        f_ab = complex(_first(f_x), _first(f_y))
        return {
            "psi_s1": psi_s1_ab * np.conj(e),
            "psi_s2": psi_s2_ab * np.conj(e) ** 2,
            "f_M": f_ab * np.conj(e),
            "tau_M": _first(tau),
            "W": _first(W),
            "W_co": _first(W_co),
            "psi_k": psi_k,
            "i_k": i_k,
            "n_elem": int(_first(n_elem)),
        }


def _first(x):
    """First element of a value returned by FEMM, a scalar or a sequence."""
    if isinstance(x, list | tuple):
        return x[0]
    return x
