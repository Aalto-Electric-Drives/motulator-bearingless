"""Tests of the gradient-network magnetic maps and the saturated machine parameters."""

import pickle
from pathlib import Path

import numpy as np
import pytest
from motulator.drive.gradnet import GradNetFunction as BaseFunction

from motulator_bearingless import control, utils
from motulator_bearingless import gradnet as gn
from motulator_bearingless import model as system

Q_BASE = 1e-3
I_BASE, PSI_BASE = 17.0, 0.57
K_Q = 0.3  # Per-unit coefficient of the quadratic term of the current map
PSI_1, PSI_2, Q_M, THETA_M = 0.03 - 0.01j, 0.3 + 0.12j, 80e-6 - 30e-6j, 0.3
EXP = np.exp(1j * THETA_M)
I_1, I_2 = 0.8 - 0.3j, 9 + 6j


def random_net(map_type: str, k: tuple[int, ...] = ()) -> gn.GradNetFunction:
    """Random softmax network of the given type."""
    rng = np.random.default_rng(1)
    n = 6 + 2 * len(k)
    return gn.GradNetFunction(
        W=rng.normal(scale=0.5, size=(1, 12, n)),
        b=rng.normal(scale=0.2, size=(1, 12)),
        beta_log=np.array([0.5]),
        mu_log=np.full(4, -1.0),
        bias=rng.normal(size=n),
        activation="Softmax",
        p=0,
        psi_base=PSI_BASE,
        i_base=I_BASE,
        map_type=map_type,
        k=k,
        q_base=Q_BASE,
        k_q=K_Q if map_type == "current_map" else 0.0,
    )


@pytest.fixture(params=[(), (6,), (6, 36)], ids=["no_harm", "k6", "k6_k36"])
def machine_par(request) -> utils.BaseBearinglessMachinePars:
    """Machine parameters with a current map, without and with spatial harmonics."""
    fcn = random_net("current_map", request.param)
    if not request.param:
        return system.SaturatedBearinglessSynchronousMachinePars(
            R_s=0.2, i_s_fcn=gn.CurrentMap(fcn)
        )
    return system.SpatialSaturatedBearinglessSynchronousMachinePars(
        R_s=0.2, magnetic_map_fcn=gn.CurrentMapWithHarmonics(fcn)
    )


@pytest.fixture
def control_par() -> system.SaturatedBearinglessSynchronousMachinePars:
    """Machine parameters with a flux map."""
    return system.SaturatedBearinglessSynchronousMachinePars(
        R_s=0.2, psi_s_fcn=gn.FluxMap(random_net("flux_map"))
    )


# %%
def test_group_is_closed() -> None:
    """The symmetry elements form a group of orthogonal matrices."""
    G = gn.symmetry_group((6, 12, 36))
    assert np.allclose(G @ G.transpose(0, 2, 1), np.eye(12))
    for g_a in G:
        for g_b in G:
            assert np.any(np.all(np.isclose(G, g_a @ g_b), axis=(1, 2)))


def test_symmetry_group_requires_multiples_of_six() -> None:
    """The maps are periodic in the rotor angle with the period of 60 deg."""
    with pytest.raises(ValueError):
        gn.symmetry_group((5,))


@pytest.mark.parametrize(
    "map_class, map_type, k",
    [
        (gn.CurrentMap, "flux_map", ()),
        (gn.CurrentMap, "current_map", (6,)),
        (gn.CurrentMapWithHarmonics, "current_map", ()),
        (gn.CurrentMapWithHarmonics, "flux_map", (6,)),
        (gn.FluxMap, "current_map", ()),
        (gn.FluxMap, "flux_map", (6,)),
    ],
)
def test_map_must_fit_the_network(map_class, map_type, k) -> None:
    """A map refuses a network of another type."""
    with pytest.raises(ValueError):
        map_class(random_net(map_type, k))


def test_missing_maps_are_detected() -> None:
    """The parameters need a map, and the control system needs the flux map."""
    with pytest.raises(ValueError):
        system.SaturatedBearinglessSynchronousMachinePars(R_s=0.2)
    par = system.SaturatedBearinglessSynchronousMachinePars(
        R_s=0.2, i_s_fcn=gn.CurrentMap(random_net("current_map"))
    )
    with pytest.raises(ValueError):
        par.psi_s(I_1, I_2, Q_M)


def test_save_and_load(tmp_path: Path) -> None:
    """A network is restored from a file with the displacement quantities."""
    fcn = random_net("current_map", (6,))
    fcn.save(tmp_path / "net.npz")
    loaded = gn.load_gradnet(tmp_path / "net.npz")
    assert (loaded.map_type, loaded.k) == ("current_map", (6,))
    assert (loaded.q_base, loaded.k_q) == (Q_BASE, K_Q)
    x = np.random.default_rng(2).normal(size=(3, 8))
    assert np.allclose(fcn(x)[0], loaded(x)[0])


@pytest.mark.parametrize("shape", [(6,), (16, 6), (5, 16, 6)])
def test_jacobian_equals_base_class(shape: tuple[int, ...]) -> None:
    """The fast Jacobian equals that of the numpy function of motulator."""
    fcn = random_net("flux_map")
    x = np.random.default_rng(2).normal(size=shape)
    y, jac = fcn(x, True)
    y_ref, jac_ref = BaseFunction.__call__(fcn, x, True)
    assert jac is not None
    assert jac_ref is not None
    assert np.allclose(y, y_ref)
    assert np.allclose(jac, jac_ref)


# %%
# Current maps of the machine model
def test_current_map_symmetries(machine_par) -> None:
    """The current map has the symmetries of the machine by construction."""
    map_fcn = machine_par.magnetic_map
    i_1, i_2, f, tau = map_fcn(PSI_1, PSI_2, Q_M, EXP)
    expected = pytest.approx((i_1, i_2, f, tau))
    assert map_fcn(PSI_1, PSI_2, Q_M, EXP * np.exp(1j * np.pi / 3)) == expected
    assert map_fcn(-PSI_1, -PSI_2, Q_M, EXP) == pytest.approx((-i_1, -i_2, f, tau))
    mirrored = map_fcn(np.conj(PSI_1), np.conj(PSI_2), np.conj(Q_M), np.conj(EXP))
    assert mirrored == pytest.approx((np.conj(i_1), np.conj(i_2), np.conj(f), -tau))
    turned = map_fcn(-1j * PSI_1, -PSI_2, -1j * Q_M, EXP * np.exp(1j * np.pi / 6))
    assert turned == pytest.approx((-1j * i_1, -i_2, -1j * f, tau))
    # No force and no two-pole current without the two-pole flux at a centric rotor
    i_1c, _, f_c, _ = map_fcn(0j, PSI_2, 0j, EXP)
    assert abs(f_c) < 1e-9
    assert abs(i_1c) < 1e-9


def test_current_map_is_conservative(machine_par) -> None:
    """The currents, the force, and the torque follow from one energy function."""
    eps_p, eps_q, eps_t = 1e-6, 1e-8, 1e-6

    def at(psi_1=PSI_1, psi_2=PSI_2, q=Q_M, theta=THETA_M):
        return machine_par.magnetic_map(psi_1, psi_2, q, np.exp(1j * theta))

    # 3*di_s/dq_M = -df_M/dpsi_s
    di_q = (at(q=Q_M + eps_q)[1] - at(q=Q_M - eps_q)[1]) / (2 * eps_q)
    df_p = (at(psi_2=PSI_2 + eps_p)[2] - at(psi_2=PSI_2 - eps_p)[2]) / (2 * eps_p)
    assert 3 * di_q.real == pytest.approx(-df_p.real, rel=1e-5)
    # The Jacobian of the currents is symmetric between the planes
    di_1 = (at(psi_2=PSI_2 + eps_p)[0] - at(psi_2=PSI_2 - eps_p)[0]) / (2 * eps_p)
    di_2 = (at(psi_1=PSI_1 + eps_p)[1] - at(psi_1=PSI_1 - eps_p)[1]) / (2 * eps_p)
    assert di_1.real == pytest.approx(di_2.real, rel=1e-5)

    # The torque follows from the energy: 3*di_s/dtheta_M = -dtau_h/dpsi_s, where the
    # torque term tau_h is what remains after subtracting the part given by the currents
    def tau_h(psi_2):
        i_1, i_2, f, tau = at(psi_2=psi_2)
        return tau - utils.torque(i_1, i_2, PSI_1, psi_2, Q_M, f)

    di_t = (at(theta=THETA_M + eps_t)[1] - at(theta=THETA_M - eps_t)[1]) / (2 * eps_t)
    dtau = (tau_h(PSI_2 + eps_p) - tau_h(PSI_2 - eps_p)) / (2 * eps_p)
    assert 3 * di_t.real == pytest.approx(-dtau, rel=1e-4, abs=1e-6)


def test_long_time_series() -> None:
    """A long time series gives the same results as its parts."""
    current_map = gn.CurrentMapWithHarmonics(random_net("current_map", (6, 36)))
    n = gn.CHUNK + 5
    theta = np.linspace(0, 1, n)
    i_1, _, _, tau = current_map(PSI_1, PSI_2, Q_M, np.exp(1j * theta))
    assert i_1.shape == (n,)
    last = current_map(PSI_1, PSI_2, Q_M, np.exp(1j * theta[-1]))
    assert (i_1[-1], tau[-1]) == pytest.approx((last[0], last[3]))
    flux_map = gn.FluxMap(random_net("flux_map"))
    i_s2 = I_2 * np.linspace(0, 1, n)
    assert flux_map.jacobian(I_1, i_s2, Q_M).shape == (n, 6, 6)


def test_current_map_arrays(machine_par) -> None:
    """Arrays give the same results as scalars."""
    psi_2 = PSI_2 * np.array([0.1, 1.0, 2.5])
    i_1, i_2, f, tau = machine_par.magnetic_map(PSI_1, psi_2, Q_M, EXP)
    for k, value in enumerate(psi_2):
        scalar = machine_par.magnetic_map(PSI_1, value, Q_M, EXP)
        assert scalar == pytest.approx((i_1[k], i_2[k], f[k], tau[k]))


# %%
# Flux map of the control system
def test_flux_map_symmetries(control_par) -> None:
    """The flux map has the symmetries of the machine by construction."""
    par = control_par
    psi_1, psi_2 = par.psi_s(I_1, I_2, Q_M)
    f = par.f_M(I_1, I_2, Q_M)
    assert par.psi_s(-I_1, -I_2, Q_M) == pytest.approx((-psi_1, -psi_2))
    assert par.f_M(-I_1, -I_2, Q_M) == pytest.approx(f)
    mirrored = (np.conj(I_1), np.conj(I_2), np.conj(Q_M))
    assert par.psi_s(*mirrored) == pytest.approx((np.conj(psi_1), np.conj(psi_2)))
    assert par.f_M(*mirrored) == pytest.approx(np.conj(f))
    turned = (-1j * I_1, -I_2, -1j * Q_M)
    assert par.psi_s(*turned) == pytest.approx((-1j * psi_1, -psi_2))
    assert par.f_M(*turned) == pytest.approx(-1j * f)
    assert abs(par.f_M(0j, I_2, 0j)) < 1e-9
    assert abs(par.psi_s(0j, I_2, 0j)[0]) < 1e-12


def test_flux_map_reciprocity(control_par) -> None:
    """The flux linkages and the force are gradients of one co-energy function."""
    par = control_par
    eps_i, eps_q = 1e-5, 1e-9
    df_i = (par.f_M(I_1 + eps_i, I_2, Q_M) - par.f_M(I_1 - eps_i, I_2, Q_M)) / (
        2 * eps_i
    )
    dpsi_q = (
        par.psi_s(I_1, I_2, Q_M + eps_q)[0] - par.psi_s(I_1, I_2, Q_M - eps_q)[0]
    ) / (2 * eps_q)
    assert df_i.real == pytest.approx(3 * dpsi_q.real, rel=1e-5)


def test_flux_map_jacobians(control_par) -> None:
    """The Jacobians agree with central differences."""
    par = control_par
    eps = 1e-5
    L_s = par.incr_ind_mat(I_1, I_2, Q_M)
    p_1p, p_2p = par.psi_s(I_1, I_2 + eps, Q_M)
    p_1m, p_2m = par.psi_s(I_1, I_2 - eps, Q_M)
    assert L_s[0, 2] == pytest.approx((p_1p - p_1m).real / (2 * eps), rel=1e-6)
    assert L_s[2, 2] == pytest.approx((p_2p - p_2m).real / (2 * eps), rel=1e-6)
    assert np.allclose(L_s, L_s.T)
    assert np.all(np.linalg.eigvalsh(L_s) > 0)
    # Sensitivities at constant flux linkages from the central differences of the flux
    # and force maps at constant currents, di/dq = -L_s^-1*dpsi/dq
    i_s = utils.c2v(I_1, I_2)

    def outputs(i: np.ndarray, q: complex) -> np.ndarray:
        i_1, i_2 = utils.v2c(i)
        return utils.c2v(*par.psi_s(i_1, i_2, q), par.f_M(i_1, i_2, q))

    eps_q = 1e-6
    d_q = np.stack(
        [
            (outputs(i_s, Q_M + eps_q * d) - outputs(i_s, Q_M - eps_q * d))
            / (2 * eps_q)
            for d in (1, 1j)
        ],
        axis=-1,
    )
    d_i = np.stack(
        [
            (outputs(i_s + eps * e, Q_M) - outputs(i_s - eps * e, Q_M)) / (2 * eps)
            for e in np.eye(4)
        ],
        axis=-1,
    )
    G = -np.linalg.solve(d_i[:4], d_q[:4])
    K = d_q[4:] + d_i[4:] @ G
    tol_G, tol_K = 1e-5 * np.abs(G).max(), 1e-5 * np.abs(K).max()
    assert par.di_dq(I_1, I_2, Q_M) == pytest.approx(G, rel=1e-4, abs=tol_G)
    assert par.df_dq(I_1, I_2, Q_M) == pytest.approx(K, rel=1e-4, abs=tol_K)


def test_flux_map_scalar_changed_in_place() -> None:
    """A zero-dimensional input changed in place gives a new result."""
    flux_map = gn.FluxMap(random_net("flux_map"))
    i_s1 = np.array(I_1)
    flux_map(i_s1, I_2, Q_M)
    i_s1[...] = 2 * I_1
    psi_1, _, f = flux_map(i_s1, I_2, Q_M)
    psi_1_ref, _, f_ref = gn.FluxMap(random_net("flux_map"))(2 * I_1, I_2, Q_M)
    assert psi_1 == pytest.approx(psi_1_ref)
    assert f == pytest.approx(f_ref)


def test_flux_map_inverse(control_par) -> None:
    """The four-pole flux map is inverted by the equivalent machine of motulator."""
    par = utils.fourpole_pars(control_par)
    for i_s_dq in 2 * I_2 * np.array([0.1, 1.0, 2.5]):
        psi_s_dq = complex(par.psi_s_dq(i_s_dq))
        assert par.iterate_i_s_dq(psi_s_dq) == pytest.approx(i_s_dq, rel=1e-6)
    # The machine model needs the current map
    with pytest.raises(ValueError):
        control_par.i_s(PSI_1, PSI_2, Q_M)


def test_linearized_parameters(control_par) -> None:
    """The linear parameters reproduce the slopes of the maps at the operating point."""
    par = control_par
    lin = par.linearized(I_2)
    _, psi_2 = par.psi_s(0j, I_2, 0j)
    assert lin.psi_s(0j, I_2, 0j)[1] == pytest.approx(psi_2)
    assert lin.R_s == par.R_s
    eps = 1e-5

    def slopes(m) -> tuple[complex, complex]:
        """Sensitivities of the force to the two-pole current components."""
        df_d = (m.f_M(eps, I_2, 0j) - m.f_M(-eps, I_2, 0j)) / (2 * eps)
        df_q = (m.f_M(1j * eps, I_2, 0j) - m.f_M(-1j * eps, I_2, 0j)) / (2 * eps)
        return df_d, df_q

    # The fitted components of the sensitivity, 6*m_d*i_s2d and 6*m_q*i_s2q
    df_d, df_q = slopes(par)
    df_d_lin, df_q_lin = slopes(lin)
    assert df_d_lin.real - df_q_lin.imag == pytest.approx(df_d.real - df_q.imag)
    assert df_q_lin.real + df_d_lin.imag == pytest.approx(df_q.real + df_d.imag)


def test_pickle(machine_par, control_par) -> None:
    """The parameters can be saved with the simulation results."""
    args = (PSI_1, PSI_2, Q_M, EXP)
    copy = pickle.loads(pickle.dumps(machine_par))
    assert copy.magnetic_map(*args) == pytest.approx(machine_par.magnetic_map(*args))
    copy = pickle.loads(pickle.dumps(control_par))
    assert copy.f_M(I_1, I_2, Q_M) == pytest.approx(control_par.f_M(I_1, I_2, Q_M))


# %%
# Trained networks
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MODELS = EXAMPLES / "trained_models"


def trained_flux_par() -> system.SaturatedBearinglessSynchronousMachinePars:
    """Machine parameters with the trained flux map."""
    fcn = gn.load_gradnet(MODELS / "bsyrm_fem_flux_map_softmax_d128.npz")
    return system.SaturatedBearinglessSynchronousMachinePars(
        0.215, psi_s_fcn=gn.FluxMap(fcn)
    )


def trained_current_par(harmonics: bool) -> utils.BaseBearinglessMachinePars:
    """Machine parameters with a trained current map."""
    if harmonics:
        fcn = gn.load_gradnet(MODELS / "bsyrm_fem_curr_map_harm_softmax_d128.npz")
        return system.SpatialSaturatedBearinglessSynchronousMachinePars(
            0.215, gn.CurrentMapWithHarmonics(fcn)
        )
    fcn = gn.load_gradnet(MODELS / "bsyrm_fem_curr_map_softmax_d64.npz")
    return system.SaturatedBearinglessSynchronousMachinePars(
        0.215, i_s_fcn=gn.CurrentMap(fcn)
    )


def rms(e: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(np.abs(e) ** 2)))


@pytest.fixture(scope="module")
def test_points() -> dict[str, np.ndarray]:
    """Held-out points of the finite-element dataset."""
    with np.load(EXAMPLES / "datasets" / "bsyrm_fem.npz") as data:
        mask = data["set"] == 1
        return {key: data[key][mask] for key in data.files}


@pytest.mark.parametrize(
    "harmonics, i_max, f_max, tau_max",
    [(True, 0.2, 75.0, 0.45), (False, 0.45, 180.0, 1.8)],
    ids=["harmonics", "no_harmonics"],
)
def test_trained_current_maps(test_points, harmonics, i_max, f_max, tau_max) -> None:
    """The trained current maps reproduce the finite-element test points."""
    d = test_points
    i_1, i_2, f, tau = trained_current_par(harmonics).magnetic_map(
        d["psi_s1"], d["psi_s2"], d["q_M"], np.exp(1j * d["theta_M"])
    )
    assert rms(i_1 - d["i_s1"]) < i_max
    assert rms(i_2 - d["i_s2"]) < i_max
    assert rms(f - d["f_M"]) < f_max
    assert rms(tau - d["tau_M"]) < tau_max


def test_trained_flux_map(test_points) -> None:
    """The trained flux map reproduces the finite-element test points."""
    d = test_points
    par = trained_flux_par()
    psi_1, psi_2 = par.psi_s(d["i_s1"], d["i_s2"], d["q_M"])
    assert rms(psi_1 - d["psi_s1"]) < 8e-3
    assert rms(psi_2 - d["psi_s2"]) < 5e-3
    assert rms(par.f_M(d["i_s1"], d["i_s2"], d["q_M"]) - d["f_M"]) < 200


def test_trained_flux_map_is_invertible() -> None:
    """The four-pole flux map can be inverted also outside the range of the data."""
    par = utils.fourpole_pars(trained_flux_par())
    for psi_s_dq in (0.15j, 0.4j, 0.8j, 1.0 + 0j, 0.5 + 0.5j):
        i_s_dq = par.iterate_i_s_dq(psi_s_dq)
        assert par.psi_s_dq(i_s_dq) == pytest.approx(psi_s_dq, abs=1e-6)


def test_four_pole_plane_of_trained_flux_map() -> None:
    """The equivalent three-phase machine has the torque of the four-pole plane."""
    par = utils.fourpole_pars(trained_flux_par())
    i_s_dq = 2 * (12 + 12j)
    tau_M = 1.5 * par.n_p * np.imag(i_s_dq * np.conj(par.psi_s_dq(i_s_dq)))
    assert tau_M == pytest.approx(19.5, rel=0.03)


def _simulate(par_machine, par_ctrl, t_stop: float):
    """Direct force control at a fixed rotor speed with a force reference step."""
    machine = system.BearinglessSynchronousMachine(par_machine)
    mechanics = system.ExternalRotorSpeed()
    mechanics.set_external_rotor_speed(2 * np.pi * 1700 / 60)
    radial = system.ExternalRotorPosition()
    converter = system.MultiThreePhaseConverter(u_dc=540)
    sensors = system.BearinglessDriveSensors(
        w_M=system.Sensor(), theta_M=system.Sensor()
    )
    mdl = system.BearinglessDrive(machine, mechanics, radial, converter, sensors)
    cfg = control.DirectForceControllerCfg(i_s_max=25.5, psi_s_min=0.4)
    ctrl = control.DirectForceControlSystem(
        control.DirectForceController(par_ctrl, cfg)
    )
    ctrl.set_torque_ref(lambda t: 0.0)
    ctrl.set_force_ref(lambda t: (t > 0.5 * t_stop) * 100 + 0j)
    return system.Simulation(mdl, ctrl, show_progress=False).simulate(t_stop=t_stop)


@pytest.mark.parametrize("harmonics", [False, True], ids=["no_harmonics", "harmonics"])
def test_closed_loop_with_trained_maps(harmonics: bool) -> None:
    """The force follows its reference with the trained maps."""
    res = _simulate(trained_current_par(harmonics), trained_flux_par(), 0.1)
    t, machine = res.mdl.t, res.mdl.machine
    assert np.mean(np.abs(machine.psi_s2_dq[t > 0.04])) == pytest.approx(0.4, abs=0.02)
    assert abs(np.mean(machine.f_M_ab[t > 0.08]) - 100) < 15
    # No force during the magnetization, when the force matrix is nearly singular
    assert np.max(np.abs(machine.f_M_ab[t < 0.05])) < 1
