import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# DOUBLE PENDULUM — FINITE-TIME LYAPUNOV TEST
#
# Gate 2:
# Estimate the exponential divergence rate using repeated
# perturbation + renormalization.
#
# This is stronger than simply measuring final separation.
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
        2 * m1 + m2
        - m2 * np.cos(2 * th1 - 2 * th2)
    )

    dw1 = (
        -g * (2 * m1 + m2) * np.sin(th1)
        - m2 * g * np.sin(th1 - 2 * th2)
        - 2 * np.sin(delta) * m2
        * (
            w2**2 * L2
            + w1**2 * L1 * np.cos(delta)
        )
    ) / (L1 * den)

    dw2 = (
        2 * np.sin(delta)
        * (
            w1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th1)
            + w2**2 * L2 * m2 * np.cos(delta)
        )
    ) / (L2 * den)

    return np.array([w1, dw1, w2, dw2])


def wrapped_difference(a, b):
    d = a - b

    # Wrap angular coordinates.
    d[0] = np.arctan2(np.sin(d[0]), np.cos(d[0]))
    d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))

    return d


def lyapunov_test(
    initial_state,
    delta0=1e-8,
    dt_renorm=0.05,
    total_time=20.0,
):

    reference = np.array(initial_state, dtype=float)

    # Initial perturbation in theta1.
    perturbation = np.array(
        [delta0, 0.0, 0.0, 0.0]
    )

    nearby = reference + perturbation

    t = 0.0
    log_growth_sum = 0.0

    times = []
    estimates = []
    local_growth = []

    while t < total_time:

        t_next = min(t + dt_renorm, total_time)

        sol_ref = solve_ivp(
            deriv,
            (t, t_next),
            reference,
            rtol=1e-10,
            atol=1e-12,
        )

        sol_near = solve_ivp(
            deriv,
            (t, t_next),
            nearby,
            rtol=1e-10,
            atol=1e-12,
        )

        if not sol_ref.success:
            raise RuntimeError(sol_ref.message)

        if not sol_near.success:
            raise RuntimeError(sol_near.message)

        reference = sol_ref.y[:, -1]
        nearby_end = sol_near.y[:, -1]

        difference = wrapped_difference(
            nearby_end.copy(),
            reference.copy(),
        )

        distance = np.linalg.norm(difference)

        if distance <= 0:
            raise RuntimeError(
                "Trajectory separation collapsed to zero."
            )

        growth = np.log(distance / delta0)

        log_growth_sum += growth

        elapsed = t_next

        estimate = log_growth_sum / elapsed

        times.append(elapsed)
        estimates.append(estimate)
        local_growth.append(
            growth / (t_next - t)
        )

        # Renormalize perturbation back to delta0.
        direction = difference / distance

        nearby = reference + delta0 * direction

        t = t_next

    return (
        np.array(times),
        np.array(estimates),
        np.array(local_growth),
    )


# ============================================================
# CONDITIONS
# ============================================================

high_energy_state = [
    0.9 * np.pi,
    0.0,
    0.9 * np.pi,
    0.0,
]

low_energy_state = [
    0.10,
    0.0,
    0.10,
    0.0,
]

delta0 = 1e-8
dt_renorm = 0.05
total_time = 20.0


# ============================================================
# RUN TESTS
# ============================================================

print("============================================")
print("DOUBLE PENDULUM — LYAPUNOV GATE")
print("============================================")
print()

print("Running high-energy case...")

high_t, high_ftle, high_local = lyapunov_test(
    high_energy_state,
    delta0,
    dt_renorm,
    total_time,
)

print("Running low-energy control...")

low_t, low_ftle, low_local = lyapunov_test(
    low_energy_state,
    delta0,
    dt_renorm,
    total_time,
)


print()
print("RESULTS")
print("--------------------------------------------")

print(
    f"High-energy finite-time exponent: "
    f"{high_ftle[-1]:.6f} 1/s"
)

print(
    f"Low-energy finite-time exponent:  "
    f"{low_ftle[-1]:.6f} 1/s"
)

if high_ftle[-1] > 0:
    doubling_time = np.log(2) / high_ftle[-1]

    print(
        f"High-energy perturbation doubling time: "
        f"{doubling_time:.6f} s"
    )

print()
print(
    "Interpretation requires convergence with time, "
    "perturbation size, and renormalization interval."
)


# ============================================================
# PLOT — RUNNING EXPONENT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    high_t,
    high_ftle,
    label="High-energy case",
)

plt.plot(
    low_t,
    low_ftle,
    label="Low-energy control",
)

plt.axhline(
    0.0,
    linewidth=1,
)

plt.xlabel("Time (s)")
plt.ylabel("Running finite-time exponent (1/s)")
plt.title(
    "Double Pendulum — Finite-Time Lyapunov Estimate"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "double_pendulum_lyapunov.png",
    dpi=200,
)

plt.close()


# ============================================================
# PLOT — LOCAL GROWTH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    high_t,
    high_local,
    label="High-energy local growth",
)

plt.plot(
    low_t,
    low_local,
    label="Low-energy local growth",
)

plt.axhline(
    0.0,
    linewidth=1,
)

plt.xlabel("Time (s)")
plt.ylabel("Local growth rate (1/s)")
plt.title(
    "Double Pendulum — Local Perturbation Growth"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "double_pendulum_local_growth.png",
    dpi=200,
)

plt.close()


print()
print("Plots saved:")
print("  double_pendulum_lyapunov.png")
print("  double_pendulum_local_growth.png")
