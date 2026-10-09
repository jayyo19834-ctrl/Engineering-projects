import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# DOUBLE PENDULUM CHAOS TEST
#
# Goal:
# Compare sensitivity to small initial perturbations for:
#   1. A high-energy configuration
#   2. A low-energy control configuration
#
# This measures finite-time trajectory separation.
# It is NOT, by itself, a rigorous Lyapunov exponent calculation.
# ============================================================

g = 9.81
m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0


def deriv(t, y):
    th1, w1, th2, w2 = y

    delta = th1 - th2

    den = (
        2 * m1
        + m2
        - m2 * np.cos(2 * th1 - 2 * th2)
    )

    dw1 = (
        -g * (2 * m1 + m2) * np.sin(th1)
        - m2 * g * np.sin(th1 - 2 * th2)
        - 2
        * np.sin(delta)
        * m2
        * (
            w2**2 * L2
            + w1**2 * L1 * np.cos(delta)
        )
    ) / (L1 * den)

    dw2 = (
        2
        * np.sin(delta)
        * (
            w1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th1)
            + w2**2 * L2 * m2 * np.cos(delta)
        )
    ) / (L2 * den)

    return [w1, dw1, w2, dw2]


def total_energy(y):
    th1, w1, th2, w2 = y

    kinetic = (
        0.5 * m1 * (L1 * w1) ** 2
        + 0.5
        * m2
        * (
            (L1 * w1) ** 2
            + (L2 * w2) ** 2
            + 2
            * L1
            * L2
            * w1
            * w2
            * np.cos(th1 - th2)
        )
    )

    potential = (
        -(m1 + m2) * g * L1 * np.cos(th1)
        - m2 * g * L2 * np.cos(th2)
    )

    return kinetic + potential


def state_separation(sol_a, sol_b):
    """
    Full-state separation.

    Angular differences are wrapped to [-pi, pi] so two physically
    identical orientations separated by 2*pi are not counted as far apart.
    """

    dth1 = np.arctan2(
        np.sin(sol_a.y[0] - sol_b.y[0]),
        np.cos(sol_a.y[0] - sol_b.y[0]),
    )

    dw1 = sol_a.y[1] - sol_b.y[1]

    dth2 = np.arctan2(
        np.sin(sol_a.y[2] - sol_b.y[2]),
        np.cos(sol_a.y[2] - sol_b.y[2]),
    )

    dw2 = sol_a.y[3] - sol_b.y[3]

    return np.sqrt(
        dth1**2
        + dw1**2
        + dth2**2
        + dw2**2
    )


def run_pair(base_state, perturbation, t_eval):
    perturbed_state = np.array(base_state, dtype=float)

    # Perturb theta1 only.
    perturbed_state[0] += perturbation

    sol_a = solve_ivp(
        deriv,
        (t_eval[0], t_eval[-1]),
        base_state,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    sol_b = solve_ivp(
        deriv,
        (t_eval[0], t_eval[-1]),
        perturbed_state,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol_a.success:
        raise RuntimeError(sol_a.message)

    if not sol_b.success:
        raise RuntimeError(sol_b.message)

    separation = state_separation(sol_a, sol_b)

    return sol_a, sol_b, separation


# ------------------------------------------------------------
# TEST CONDITIONS
# ------------------------------------------------------------

t_eval = np.linspace(0.0, 20.0, 8000)

perturbations = [
    1e-3,
    1e-4,
    1e-5,
    1e-6,
]

# High-energy configuration similar to our previous experiment.
high_energy_state = [
    0.9 * np.pi,
    0.0,
    0.9 * np.pi,
    0.0,
]

# Low-energy control: small angles near the stable downward position.
low_energy_state = [
    0.10,
    0.0,
    0.10,
    0.0,
]


# ------------------------------------------------------------
# ENERGY CHECK
# ------------------------------------------------------------

reference_solution = solve_ivp(
    deriv,
    (t_eval[0], t_eval[-1]),
    high_energy_state,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
)

energy = total_energy(reference_solution.y)

energy_range = np.max(energy) - np.min(energy)
relative_energy_range = energy_range / max(abs(energy[0]), 1e-15)

print("============================================")
print("DOUBLE PENDULUM CHAOS TEST")
print("============================================")
print()
print("Numerical energy check:")
print(f"Initial energy:       {energy[0]:.12f} J")
print(f"Final energy:         {energy[-1]:.12f} J")
print(f"Energy range:         {energy_range:.6e} J")
print(f"Relative energy range:{relative_energy_range:.6e}")
print()


# ------------------------------------------------------------
# PERTURBATION TESTS
# ------------------------------------------------------------

high_results = {}
low_results = {}

for eps in perturbations:

    _, _, high_sep = run_pair(
        high_energy_state,
        eps,
        t_eval,
    )

    _, _, low_sep = run_pair(
        low_energy_state,
        eps,
        t_eval,
    )

    high_results[eps] = high_sep
    low_results[eps] = low_sep

    print(f"Perturbation = {eps:.1e}")
    print(
        f"  High-energy final separation: "
        f"{high_sep[-1]:.6e}"
    )
    print(
        f"  Low-energy final separation:  "
        f"{low_sep[-1]:.6e}"
    )
    print()


# ------------------------------------------------------------
# PLOT 1
# RAW STATE SEPARATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for eps in perturbations:
    plt.semilogy(
        t_eval,
        high_results[eps],
        label=f"High energy, eps={eps:.0e}",
    )

plt.xlabel("Time (s)")
plt.ylabel("Full-state separation")
plt.title("High-Energy Double Pendulum: Perturbation Growth")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "double_pendulum_high_energy_separation.png",
    dpi=200,
)
plt.close()


# ------------------------------------------------------------
# PLOT 2
# LOW-ENERGY CONTROL
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for eps in perturbations:
    plt.semilogy(
        t_eval,
        low_results[eps],
        label=f"Low energy, eps={eps:.0e}",
    )

plt.xlabel("Time (s)")
plt.ylabel("Full-state separation")
plt.title("Low-Energy Double Pendulum Control")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "double_pendulum_low_energy_separation.png",
    dpi=200,
)
plt.close()


# ------------------------------------------------------------
# NORMALIZED PERTURBATION GROWTH
#
# If separation approximately behaves as
#
#       delta(t) = delta_0 * exp(lambda * t)
#
# then
#
#       ln(delta(t) / delta_0)
#
# should contain an approximately linear region.
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for eps in perturbations:

    sep = high_results[eps]

    # Prevent log(0).
    normalized = np.maximum(
        sep / eps,
        1e-300,
    )

    plt.plot(
        t_eval,
        np.log(normalized),
        label=f"eps={eps:.0e}",
    )

plt.xlabel("Time (s)")
plt.ylabel("ln(delta / delta_0)")
plt.title(
    "High-Energy Perturbation Growth\n"
    "Candidate Exponential-Growth Region"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "double_pendulum_log_growth.png",
    dpi=200,
)
plt.close()


print("Plots saved:")
print("  double_pendulum_high_energy_separation.png")
print("  double_pendulum_low_energy_separation.png")
print("  double_pendulum_log_growth.png")
print()
print(
    "IMPORTANT: Positive-looking growth is evidence of "
    "sensitivity, not yet a rigorous Lyapunov exponent."
)
