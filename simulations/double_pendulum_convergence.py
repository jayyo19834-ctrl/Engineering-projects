import numpy as np
from scipy.integrate import solve_ivp

# ============================================================
# DOUBLE PENDULUM — LYAPUNOV CONVERGENCE TEST
#
# Gate 3:
# Test sensitivity of the finite-time Lyapunov estimate to:
#
#   1. Total integration time
#   2. Initial perturbation magnitude
#   3. Renormalization interval
#
# PASS behavior:
#   High-energy estimate approaches a stable positive value.
#   Low-energy control trends toward zero.
#   Result is reasonably insensitive to delta0 and dt_renorm.
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

    nearby = reference + np.array(
        [delta0, 0.0, 0.0, 0.0]
    )

    t = 0.0
    log_growth_sum = 0.0

    while t < total_time - 1e-12:

        t_next = min(
            t + dt_renorm,
            total_time,
        )

        ref_sol = solve_ivp(
            deriv,
            (t, t_next),
            reference,
            rtol=1e-10,
            atol=1e-12,
        )

        near_sol = solve_ivp(
            deriv,
            (t, t_next),
            nearby,
            rtol=1e-10,
            atol=1e-12,
        )

        if not ref_sol.success:
            raise RuntimeError(
                ref_sol.message
            )

        if not near_sol.success:
            raise RuntimeError(
                near_sol.message
            )

        reference = ref_sol.y[:, -1]

        nearby_end = near_sol.y[:, -1]

        difference = wrapped_difference(
            nearby_end.copy(),
            reference.copy(),
        )

        distance = np.linalg.norm(
            difference
        )

        if not np.isfinite(distance):
            raise RuntimeError(
                "Non-finite separation."
            )

        if distance <= 0.0:
            raise RuntimeError(
                "Separation collapsed to zero."
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
# PARAMETER SWEEP
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


print("=" * 72)
print("DOUBLE PENDULUM — LYAPUNOV CONVERGENCE GATE")
print("=" * 72)

print()
print(
    "Columns: time | delta0 | renorm | "
    "high exponent | low exponent"
)
print("-" * 72)


results = []

for total_time in times:

    for delta0 in delta_values:

        for dt_renorm in renorm_values:

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

            results.append(
                (
                    total_time,
                    delta0,
                    dt_renorm,
                    high,
                    low,
                )
            )

            print(
                f"{total_time:6.1f} | "
                f"{delta0:8.1e} | "
                f"{dt_renorm:6.3f} | "
                f"{high:12.6f} | "
                f"{low:12.6f}"
            )


# ============================================================
# SUMMARY BY TOTAL TIME
# ============================================================

print()
print("=" * 72)
print("SUMMARY BY TOTAL TIME")
print("=" * 72)

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
        f"Time = {total_time:.1f} s"
    )

    print(
        "  High-energy:"
        f" mean={np.mean(high_values):.6f},"
        f" std={np.std(high_values):.6f},"
        f" min={np.min(high_values):.6f},"
        f" max={np.max(high_values):.6f}"
    )

    print(
        "  Low-energy: "
        f" mean={np.mean(low_values):.6f},"
        f" std={np.std(low_values):.6f},"
        f" min={np.min(low_values):.6f},"
        f" max={np.max(low_values):.6f}"
    )


# ============================================================
# SIMPLE CONVERGENCE INDICATORS
# ============================================================

longest = [
    row
    for row in results
    if row[0] == max(times)
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
print("=" * 72)
print("LONGEST-RUN DIAGNOSTIC")
print("=" * 72)

print(
    f"High-energy mean exponent: "
    f"{high_mean:.6f} 1/s"
)

print(
    f"High-energy std:           "
    f"{high_std:.6f} 1/s"
)

print(
    f"Low-energy mean exponent:  "
    f"{low_mean:.6f} 1/s"
)

print(
    f"Low-energy std:            "
    f"{low_std:.6f} 1/s"
)


if high_mean > 0:
    relative_spread = (
        high_std / abs(high_mean)
    )

    print(
        f"High-energy relative spread: "
        f"{100 * relative_spread:.3f}%"
    )


print()
print("INTERPRETATION RULE:")
print(
    "A robust chaotic result should retain a "
    "positive high-energy exponent across numerical "
    "settings while the regular control trends "
    "toward zero with increasing integration time."
)

print()
print(
    "Do not declare convergence solely from this "
    "script's summary. Inspect the full table."
)
