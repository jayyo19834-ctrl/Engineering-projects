import numpy as np
from scipy.integrate import solve_ivp
from time import perf_counter

# ============================================================
# DOUBLE PENDULUM — LYAPUNOV CONVERGENCE TEST V2
#
# Gate 3B
#
# Same convergence question as Gate 3, but reference and
# perturbed trajectories are integrated simultaneously as
# one 8-state system.
#
# Tests:
#   1. Total integration time
#   2. Initial perturbation magnitude
#   3. Renormalization interval
#
# Desired behavior:
#   - High-energy exponent remains positive and converges.
#   - Low-energy control trends toward zero.
#   - Results become insensitive to delta0 and dt_renorm.
# ============================================================

g = 9.81
m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0


def pendulum_deriv(y):
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

    return np.array(
        [w1, dw1, w2, dw2]
    )


def combined_deriv(t, y):
    reference = y[:4]
    nearby = y[4:]

    return np.concatenate(
        (
            pendulum_deriv(reference),
            pendulum_deriv(nearby),
        )
    )


def wrapped_difference(a, b):
    d = a - b

    d[0] = np.arctan2(
        np.sin(d[0]),
        np.cos(d[0]),
    )

    d[2] = np.arctan2(
        np.sin(d[2]),
        np.cos(d[2]),
    )

    return d


def lyapunov_estimate(
    initial_state,
    delta0,
    dt_renorm,
    total_time,
):
    reference = np.array(
        initial_state,
        dtype=float,
    )

    nearby = reference.copy()
    nearby[0] += delta0

    t = 0.0
    log_growth_sum = 0.0

    while t < total_time - 1e-12:

        t_next = min(
            t + dt_renorm,
            total_time,
        )

        combined_state = np.concatenate(
            (reference, nearby)
        )

        sol = solve_ivp(
            combined_deriv,
            (t, t_next),
            combined_state,
            rtol=1e-10,
            atol=1e-12,
            method="DOP853",
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        reference = sol.y[:4, -1]
        nearby_end = sol.y[4:, -1]

        difference = wrapped_difference(
            nearby_end.copy(),
            reference.copy(),
        )

        distance = np.linalg.norm(
            difference
        )

        if not np.isfinite(distance):
            raise RuntimeError(
                "Non-finite trajectory separation."
            )

        if distance <= 0.0:
            raise RuntimeError(
                "Trajectory separation collapsed to zero."
            )

        growth = np.log(
            distance / delta0
        )

        log_growth_sum += growth

        direction = (
            difference / distance
        )

        nearby = (
            reference
            + delta0 * direction
        )

        t = t_next

    return log_growth_sum / total_time


# ============================================================
# INITIAL CONDITIONS
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


# ============================================================
# CONVERGENCE MATRIX
# ============================================================

times = [
    20.0,
    40.0,
    80.0,
    160.0,
]

delta_values = [
    1e-6,
    1e-8,
    1e-10,
]

renorm_values = [
    0.025,
    0.05,
    0.10,
]


print("=" * 78)
print(
    "DOUBLE PENDULUM — LYAPUNOV CONVERGENCE GATE 3B"
)
print("=" * 78)

print(
    "time | delta0 | renorm | "
    "high exponent | low exponent"
)

print("-" * 78)


results = []

start_clock = perf_counter()


for total_time in times:

    for delta0 in delta_values:

        for dt_renorm in renorm_values:

            case_start = perf_counter()

            high = lyapunov_estimate(
                high_energy_state,
                delta0,
                dt_renorm,
                total_time,
            )

            low = lyapunov_estimate(
                low_energy_state,
                delta0,
                dt_renorm,
                total_time,
            )

            runtime = (
                perf_counter()
                - case_start
            )

            results.append(
                (
                    total_time,
                    delta0,
                    dt_renorm,
                    high,
                    low,
                    runtime,
                )
            )

            print(
                f"{total_time:5.0f} | "
                f"{delta0:8.1e} | "
                f"{dt_renorm:6.3f} | "
                f"{high:12.6f} | "
                f"{low:12.6f} | "
                f"{runtime:7.2f}s"
            )


total_runtime = (
    perf_counter()
    - start_clock
)


# ============================================================
# SUMMARY BY INTEGRATION TIME
# ============================================================

print()
print("=" * 78)
print("SUMMARY BY TOTAL TIME")
print("=" * 78)


for total_time in times:

    subset = [
        row
        for row in results
        if row[0] == total_time
    ]

    high_values = np.array(
        [row[3] for row in subset]
    )

    low_values = np.array(
        [row[4] for row in subset]
    )

    print()
    print(
        f"Integration time: "
        f"{total_time:.0f} s"
    )

    print(
        "  HIGH:"
        f" mean={np.mean(high_values):.6f},"
        f" std={np.std(high_values):.6f},"
        f" min={np.min(high_values):.6f},"
        f" max={np.max(high_values):.6f}"
    )

    print(
        "  LOW: "
        f" mean={np.mean(low_values):.6f},"
        f" std={np.std(low_values):.6f},"
        f" min={np.min(low_values):.6f},"
        f" max={np.max(low_values):.6f}"
    )


# ============================================================
# LONGEST-RUN CONVERGENCE DIAGNOSTIC
# ============================================================

max_time = max(times)

longest = [
    row
    for row in results
    if row[0] == max_time
]

high_long = np.array(
    [row[3] for row in longest]
)

low_long = np.array(
    [row[4] for row in longest]
)


high_mean = np.mean(high_long)
high_std = np.std(high_long)

low_mean = np.mean(low_long)
low_std = np.std(low_long)


print()
print("=" * 78)
print("LONGEST-RUN DIAGNOSTIC")
print("=" * 78)

print(
    f"High-energy mean: "
    f"{high_mean:.6f} 1/s"
)

print(
    f"High-energy std:  "
    f"{high_std:.6f} 1/s"
)

print(
    f"Low-energy mean:  "
    f"{low_mean:.6f} 1/s"
)

print(
    f"Low-energy std:   "
    f"{low_std:.6f} 1/s"
)


if high_mean != 0.0:

    relative_spread = (
        high_std
        / abs(high_mean)
    )

    print(
        f"High-energy relative spread: "
        f"{100 * relative_spread:.3f}%"
    )


# ============================================================
# TIME TREND
# ============================================================

print()
print("=" * 78)
print("MEAN EXPONENT TREND")
print("=" * 78)


for total_time in times:

    subset = [
        row
        for row in results
        if row[0] == total_time
    ]

    high_values = np.array(
        [row[3] for row in subset]
    )

    low_values = np.array(
        [row[4] for row in subset]
    )

    print(
        f"{total_time:5.0f} s : "
        f"high={np.mean(high_values):.6f} "
        f"low={np.mean(low_values):.6f}"
    )


print()
print("=" * 78)
print("RUNTIME")
print("=" * 78)

print(
    f"Total sweep runtime: "
    f"{total_runtime:.2f} seconds"
)


print()
print("GATE 3B INTERPRETATION:")
print(
    "Strong convergence requires the high-energy "
    "estimate to remain positive and become relatively "
    "insensitive to delta0 and renormalization interval."
)

print(
    "The low-energy control should move toward zero "
    "as total integration time increases."
)

print(
    "No automatic PASS is issued. "
    "The numerical table must be inspected."
  )
print(
    "No automatic PASS is issued. "
    "The numerical table must be inspected."
)
