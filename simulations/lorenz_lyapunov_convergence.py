# ============================================================
# GATE 4 — LORENZ LYAPUNOV BENCHMARK
# ============================================================
#
# Purpose:
# Validate the Lyapunov/convergence measurement pipeline against
# the canonical Lorenz system before relying further on the
# double-pendulum chaos measurements.
#
# Canonical Lorenz parameters:
#   sigma = 10
#   rho   = 28
#   beta  = 8/3
#
# Literature/reference expectation:
# Largest Lyapunov exponent is approximately 0.9 per Lorenz
# time unit for the canonical chaotic attractor.
#
# IMPORTANT:
# Acceptance criteria below are defined BEFORE execution.
# Do not change them after seeing the results.
# ============================================================

import numpy as np
from scipy.integrate import solve_ivp
from time import perf_counter


# ============================================================
# LORENZ PARAMETERS
# ============================================================

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

RTOL = 1e-10
ATOL = 1e-12

TOTAL_TIMES = [20.0, 40.0, 80.0, 160.0]

DELTA_VALUES = [
    1e-6,
    1e-8,
    1e-10,
]

RENORM_INTERVALS = [
    0.025,
    0.050,
    0.100,
]


# ============================================================
# PREDEFINED GATE CRITERIA
# ============================================================
#
# These thresholds are frozen BEFORE execution.
#
# Canonical Lorenz largest Lyapunov exponent is expected near
# 0.9 per Lorenz time unit.
#
# Gate 4 PASS requires:
#
# 1. Longest-run mean exponent between 0.80 and 1.00
#
# 2. Longest-run relative standard deviation <= 10%
#
# 3. All longest-run parameter combinations remain positive
#
# These are experiment-specific validation criteria.
# They are not universal mathematical definitions of chaos.
# ============================================================

EXPECTED_LOW = 0.80
EXPECTED_HIGH = 1.00

MAX_RELATIVE_STD_PERCENT = 10.0


# ============================================================
# LORENZ EQUATIONS
# ============================================================

def lorenz_deriv(state):
    x, y, z = state

    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z

    return np.array(
        [dx, dy, dz],
        dtype=float
    )


# ============================================================
# COMBINED REFERENCE + PERTURBED SYSTEM
# ============================================================
#
# Integrating both trajectories in one solve_ivp call helps
# ensure that they experience the same adaptive integration
# intervals and numerical tolerances.
# ============================================================

def combined_deriv(t, state):
    reference = state[:3]
    perturbed = state[3:]

    d_reference = lorenz_deriv(reference)
    d_perturbed = lorenz_deriv(perturbed)

    return np.concatenate(
        [d_reference, d_perturbed]
    )


# ============================================================
# LYAPUNOV ESTIMATOR
# ============================================================

def lyapunov_estimate(
    initial_state,
    delta0,
    dt_renorm,
    total_time
):
    reference = np.array(
        initial_state,
        dtype=float
    )

    # Initial perturbation along x.
    direction = np.array(
        [1.0, 0.0, 0.0],
        dtype=float
    )

    perturbed = reference + delta0 * direction

    log_growth_sum = 0.0
    elapsed = 0.0

    while elapsed < total_time:

        interval = min(
            dt_renorm,
            total_time - elapsed
        )

        combined_state = np.concatenate(
            [reference, perturbed]
        )

        solution = solve_ivp(
            combined_deriv,
            [0.0, interval],
            combined_state,
            method="DOP853",
            rtol=RTOL,
            atol=ATOL
        )

        if not solution.success:
            raise RuntimeError(
                "solve_ivp integration failed."
            )

        final_state = solution.y[:, -1]

        reference = final_state[:3]
        perturbed = final_state[3:]

        difference = perturbed - reference

        distance = np.linalg.norm(
            difference
        )

        if (
            not np.isfinite(distance)
            or distance <= 0.0
        ):
            raise RuntimeError(
                "Invalid trajectory separation."
            )

        log_growth_sum += np.log(
            distance / delta0
        )

        # Renormalize perturbation.
        difference /= distance

        perturbed = (
            reference
            + delta0 * difference
        )

        elapsed += interval

    return log_growth_sum / total_time


# ============================================================
# INITIAL CONDITION
# ============================================================
#
# Standard non-equilibrium Lorenz starting point.
# ============================================================

INITIAL_STATE = np.array(
    [1.0, 1.0, 1.0],
    dtype=float
)


# ============================================================
# RUN CONVERGENCE MATRIX
# ============================================================

results = []

print()
print("=" * 72)
print("GATE 4 — LORENZ LYAPUNOV BENCHMARK")
print("=" * 72)

print()
print("Canonical parameters:")
print(f"sigma = {SIGMA}")
print(f"rho   = {RHO}")
print(f"beta  = {BETA:.12f}")

print()
print("Initial state:")
print(INITIAL_STATE)

print()
print("Frozen acceptance criteria:")
print(
    f"Longest-run mean exponent: "
    f"{EXPECTED_LOW:.2f} to "
    f"{EXPECTED_HIGH:.2f}"
)

print(
    f"Longest-run relative std <= "
    f"{MAX_RELATIVE_STD_PERCENT:.1f}%"
)

print(
    "All longest-run parameter "
    "combinations must remain positive."
)

print()
print("=" * 72)


for total_time in TOTAL_TIMES:

    print()
    print(
        f"TOTAL TIME = "
        f"{total_time:.0f}"
    )

    print("-" * 72)

    for delta0 in DELTA_VALUES:

        for dt_renorm in RENORM_INTERVALS:

            start = perf_counter()

            exponent = lyapunov_estimate(
                INITIAL_STATE,
                delta0,
                dt_renorm,
                total_time
            )

            runtime = (
                perf_counter() - start
            )

            results.append(
                {
                    "total_time": total_time,
                    "delta0": delta0,
                    "dt_renorm": dt_renorm,
                    "exponent": exponent,
                    "runtime": runtime,
                }
            )

            print(
                f"T={total_time:6.1f}  "
                f"delta0={delta0:.0e}  "
                f"dt={dt_renorm:5.3f}  "
                f"lambda={exponent: .8f}  "
                f"runtime={runtime:6.2f}s"
            )


# ============================================================
# SUMMARY BY TOTAL TIME
# ============================================================

print()
print("=" * 72)
print("CONVERGENCE SUMMARY")
print("=" * 72)


for total_time in TOTAL_TIMES:

    values = np.array(
        [
            r["exponent"]
            for r in results
            if r["total_time"] == total_time
        ]
    )

    mean_value = np.mean(values)
    std_value = np.std(values)

    relative_std = (
        100.0
        * std_value
        / abs(mean_value)
    )

    print()
    print(
        f"T = {total_time:.0f}"
    )

    print(
        f"Mean exponent     = "
        f"{mean_value:.8f}"
    )

    print(
        f"Standard deviation = "
        f"{std_value:.8f}"
    )

    print(
        f"Minimum exponent  = "
        f"{np.min(values):.8f}"
    )

    print(
        f"Maximum exponent  = "
        f"{np.max(values):.8f}"
    )

    print(
        f"Relative std      = "
        f"{relative_std:.3f}%"
    )


# ============================================================
# LONGEST-RUN DIAGNOSTIC
# ============================================================

longest_time = max(
    TOTAL_TIMES
)

longest_values = np.array(
    [
        r["exponent"]
        for r in results
        if r["total_time"] == longest_time
    ]
)

long_mean = np.mean(
    longest_values
)

long_std = np.std(
    longest_values
)

long_relative_std = (
    100.0
    * long_std
    / abs(long_mean)
)

long_min = np.min(
    longest_values
)

long_max = np.max(
    longest_values
)


print()
print("=" * 72)
print("LONGEST-RUN DIAGNOSTIC")
print("=" * 72)

print(
    f"Integration time = "
    f"{longest_time:.0f}"
)

print(
    f"Mean exponent = "
    f"{long_mean:.8f}"
)

print(
    f"Standard deviation = "
    f"{long_std:.8f}"
)

print(
    f"Minimum exponent = "
    f"{long_min:.8f}"
)

print(
    f"Maximum exponent = "
    f"{long_max:.8f}"
)

print(
    f"Relative std = "
    f"{long_relative_std:.3f}%"
)


# ============================================================
# AUTOMATIC GATE VERDICT
# ============================================================

mean_pass = (
    EXPECTED_LOW
    <= long_mean
    <= EXPECTED_HIGH
)

spread_pass = (
    long_relative_std
    <= MAX_RELATIVE_STD_PERCENT
)

positive_pass = np.all(
    longest_values > 0.0
)


if (
    mean_pass
    and spread_pass
    and positive_pass
):
    verdict = (
        "PASS - Gate 4 Lorenz benchmark "
        "criteria satisfied"
    )

elif positive_pass:
    verdict = (
        "INCONCLUSIVE - positive exponent "
        "recovered, but benchmark convergence "
        "criteria not fully satisfied"
    )

else:
    verdict = (
        "FAIL - Lorenz benchmark did not "
        "recover a consistently positive "
        "largest exponent"
    )


print()
print("=" * 72)
print("VERDICT")
print("=" * 72)

print(verdict)

print()
print("Criterion checks:")

print(
    f"Mean in "
    f"[{EXPECTED_LOW:.2f}, "
    f"{EXPECTED_HIGH:.2f}] "
    f"= {mean_pass}"
)

print(
    f"Relative std <= "
    f"{MAX_RELATIVE_STD_PERCENT:.1f}% "
    f"= {spread_pass}"
)

print(
    "All longest-run exponents "
    f"positive = {positive_pass}"
)


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

print()
print("=" * 72)
print("INTERPRETATION")
print("=" * 72)

print(
    "Gate 4 tests the Lyapunov measurement "
    "pipeline against the canonical Lorenz "
    "system."
)

print(
    "A PASS supports the numerical method's "
    "ability to recover a known positive "
    "largest Lyapunov exponent."
)

print(
    "It does not mathematically prove chaos "
    "and does not independently prove the "
    "double-pendulum Gate 3B result."
)

print(
    "A failure or inconclusive result should "
    "trigger investigation of the estimator "
    "before increasing confidence in results "
    "from less standardized systems."
)

print()
print("=" * 72)
