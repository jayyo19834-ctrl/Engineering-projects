# ============================================================
# GATE 5 — DOUBLE PENDULUM VARIATIONAL LYAPUNOV VALIDATION
# ============================================================
#
# PURPOSE
# Independently test the positive largest Lyapunov exponent
# measured by the two-trajectory/renormalization method used
# in Gates 2 and 3B.
#
# Gate 5 uses a tangent-space variational method:
#
#       d(delta)/dt = J(y) delta
#
# where J(y) is the Jacobian of the double-pendulum vector
# field.
#
# The Jacobian is evaluated numerically using complex-step
# differentiation.
#
# IMPORTANT
# Gate 3B is NOT modified by this test.
#
# Acceptance criteria below are frozen BEFORE execution.
# Do not change them after seeing the results.
# ============================================================

import numpy as np
from scipy.integrate import solve_ivp
from time import perf_counter


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

G = 9.81

M1 = 1.0
M2 = 1.0

L1 = 1.0
L2 = 1.0


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

RTOL = 1e-10
ATOL = 1e-12

TOTAL_TIMES = [
    20.0,
    40.0,
    80.0,
    160.0,
]

RENORM_INTERVALS = [
    0.025,
    0.050,
    0.100,
]


# ============================================================
# COMPLEX-STEP JACOBIAN SETTING
# ============================================================
#
# Complex-step differentiation avoids the subtractive
# cancellation associated with ordinary finite differences.
#
# The double-pendulum derivative function below is written
# so that complex-valued intermediate states are preserved.
# ============================================================

COMPLEX_STEP = 1e-20


# ============================================================
# FROZEN GATE 5 ACCEPTANCE CRITERIA
# ============================================================
#
# Gate 3B longest-run mean:
#
#       approximately 1.2445 1/s
#
# Gate 5 is intentionally NOT required to reproduce the
# exact same number.
#
# PASS requires:
#
# 1. Longest-run mean variational exponent > 0.50 1/s
#
# 2. Longest-run relative standard deviation <= 10%
#
# 3. All longest-run estimates remain positive
#
# 4. Longest-run mean must be within 25% of the frozen
#    Gate 3B mean.
#
# These criteria are defined BEFORE Gate 5 execution.
#
# They are experiment-specific numerical-validation criteria,
# not universal definitions or mathematical proof of chaos.
# ============================================================

GATE3B_REFERENCE = 1.244525852944515

MIN_POSITIVE_EXPONENT = 0.50

MAX_RELATIVE_STD_PERCENT = 10.0

MAX_RELATIVE_DIFFERENCE_PERCENT = 25.0


# ============================================================
# INITIAL CONDITION
# ============================================================

INITIAL_STATE = np.array(
    [
        0.9 * np.pi,
        0.0,
        0.9 * np.pi,
        0.0,
    ],
    dtype=float
)


# ============================================================
# INITIAL TANGENT VECTOR
# ============================================================
#
# Unit perturbation initially aligned with theta1.
#
# Periodic renormalization allows the tangent vector to
# converge toward the dominant expanding direction.
# ============================================================

INITIAL_TANGENT = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float
)


# ============================================================
# DOUBLE-PENDULUM VECTOR FIELD
# ============================================================
#
# State:
#
# y = [theta1, omega1, theta2, omega2]
#
# Standard ideal planar double pendulum with point masses and
# massless rods.
#
# IMPORTANT:
# Do not cast the state to float inside this function.
# Complex-step differentiation requires complex values to
# propagate through the equations.
# ============================================================

def pendulum_deriv(state):

    theta1, omega1, theta2, omega2 = state

    delta = theta1 - theta2

    denominator1 = (
        L1
        * (
            2.0 * M1
            + M2
            - M2 * np.cos(2.0 * delta)
        )
    )

    denominator2 = (
        L2
        * (
            2.0 * M1
            + M2
            - M2 * np.cos(2.0 * delta)
        )
    )

    domega1 = (
        -G
        * (2.0 * M1 + M2)
        * np.sin(theta1)

        - M2
        * G
        * np.sin(theta1 - 2.0 * theta2)

        - 2.0
        * np.sin(delta)
        * M2
        * (
            omega2**2
            * L2

            + omega1**2
            * L1
            * np.cos(delta)
        )
    ) / denominator1

    domega2 = (
        2.0
        * np.sin(delta)
        * (
            omega1**2
            * L1
            * (M1 + M2)

            + G
            * (M1 + M2)
            * np.cos(theta1)

            + omega2**2
            * L2
            * M2
            * np.cos(delta)
        )
    ) / denominator2

    return np.array(
        [
            omega1,
            domega1,
            omega2,
            domega2,
        ]
    )


# ============================================================
# COMPLEX-STEP JACOBIAN
# ============================================================

def jacobian_complex_step(state):

    state = np.asarray(
        state,
        dtype=float
    )

    dimension = len(state)

    jacobian = np.zeros(
        (dimension, dimension),
        dtype=float
    )

    for column in range(dimension):

        complex_state = state.astype(
            complex
        )

        complex_state[column] += (
            1j * COMPLEX_STEP
        )

        derivative = pendulum_deriv(
            complex_state
        )

        jacobian[:, column] = (
            np.imag(derivative)
            / COMPLEX_STEP
        )

    return jacobian


# ============================================================
# COMBINED STATE + TANGENT EQUATIONS
# ============================================================
#
# Combined vector:
#
# [physical state (4), tangent vector (4)]
#
# Total dimension = 8
# ============================================================

def variational_deriv(t, combined_state):

    state = combined_state[:4]

    tangent = combined_state[4:]

    state_derivative = pendulum_deriv(
        state
    )

    jacobian = jacobian_complex_step(
        state
    )

    tangent_derivative = (
        jacobian @ tangent
    )

    return np.concatenate(
        [
            state_derivative,
            tangent_derivative,
        ]
    )


# ============================================================
# VARIATIONAL LYAPUNOV ESTIMATOR
# ============================================================

def variational_lyapunov(
    initial_state,
    initial_tangent,
    dt_renorm,
    total_time
):

    state = np.array(
        initial_state,
        dtype=float
    )

    tangent = np.array(
        initial_tangent,
        dtype=float
    )

    tangent_norm = np.linalg.norm(
        tangent
    )

    if tangent_norm <= 0.0:
        raise RuntimeError(
            "Initial tangent vector has zero norm."
        )

    tangent /= tangent_norm

    log_growth_sum = 0.0

    elapsed = 0.0

    while elapsed < total_time:

        interval = min(
            dt_renorm,
            total_time - elapsed
        )

        combined_initial = np.concatenate(
            [
                state,
                tangent,
            ]
        )

        solution = solve_ivp(
            variational_deriv,
            [0.0, interval],
            combined_initial,
            method="DOP853",
            rtol=RTOL,
            atol=ATOL
        )

        if not solution.success:
            raise RuntimeError(
                "Variational integration failed."
            )

        final_combined = (
            solution.y[:, -1]
        )

        state = final_combined[:4]

        tangent = final_combined[4:]

        norm = np.linalg.norm(
            tangent
        )

        if (
            not np.isfinite(norm)
            or norm <= 0.0
        ):
            raise RuntimeError(
                "Invalid tangent-vector norm."
            )

        log_growth_sum += np.log(
            norm
        )

        tangent /= norm

        elapsed += interval

    return (
        log_growth_sum
        / total_time
    )


# ============================================================
# PRE-RUN INFORMATION
# ============================================================

print()
print("=" * 76)
print(
    "GATE 5 — DOUBLE PENDULUM "
    "VARIATIONAL LYAPUNOV VALIDATION"
)
print("=" * 76)

print()
print("Initial state:")
print(INITIAL_STATE)

print()
print("Initial tangent:")
print(INITIAL_TANGENT)

print()
print("Frozen Gate 3B reference:")
print(
    f"{GATE3B_REFERENCE:.12f} 1/s"
)

print()
print("Frozen Gate 5 criteria:")

print(
    f"Mean exponent > "
    f"{MIN_POSITIVE_EXPONENT:.2f} 1/s"
)

print(
    f"Relative std <= "
    f"{MAX_RELATIVE_STD_PERCENT:.1f}%"
)

print(
    "All longest-run estimates "
    "must remain positive."
)

print(
    f"Mean must be within "
    f"{MAX_RELATIVE_DIFFERENCE_PERCENT:.1f}% "
    f"of Gate 3B."
)

print()
print(
    f"Complex-step size = "
    f"{COMPLEX_STEP:.1e}"
)

print()
print("=" * 76)


# ============================================================
# RUN CONVERGENCE MATRIX
# ============================================================

results = []

for total_time in TOTAL_TIMES:

    print()
    print(
        f"TOTAL TIME = "
        f"{total_time:.0f} s"
    )

    print("-" * 76)

    for dt_renorm in RENORM_INTERVALS:

        start = perf_counter()

        exponent = variational_lyapunov(
            INITIAL_STATE,
            INITIAL_TANGENT,
            dt_renorm,
            total_time
        )

        runtime = (
            perf_counter()
            - start
        )

        results.append(
            {
                "total_time": total_time,
                "dt_renorm": dt_renorm,
                "exponent": exponent,
                "runtime": runtime,
            }
        )

        print(
            f"T={total_time:6.1f}  "
            f"dt={dt_renorm:5.3f}  "
            f"lambda={exponent: .8f}  "
            f"runtime={runtime:7.2f}s"
        )


# ============================================================
# CONVERGENCE SUMMARY
# ============================================================

print()
print("=" * 76)
print("CONVERGENCE SUMMARY")
print("=" * 76)

for total_time in TOTAL_TIMES:

    values = np.array(
        [
            r["exponent"]
            for r in results
            if r["total_time"]
            == total_time
        ]
    )

    mean_value = np.mean(
        values
    )

    std_value = np.std(
        values
    )

    relative_std = (
        100.0
        * std_value
        / abs(mean_value)
    )

    print()
    print(
        f"T = {total_time:.0f} s"
    )

    print(
        f"Mean exponent      = "
        f"{mean_value:.8f}"
    )

    print(
        f"Standard deviation = "
        f"{std_value:.8f}"
    )

    print(
        f"Minimum exponent   = "
        f"{np.min(values):.8f}"
    )

    print(
        f"Maximum exponent   = "
        f"{np.max(values):.8f}"
    )

    print(
        f"Relative std       = "
        f"{relative_std:.3f}%"
    )


# ============================================================
# LONGEST-RUN STATISTICS
# ============================================================

longest_time = max(
    TOTAL_TIMES
)

longest_values = np.array(
    [
        r["exponent"]
        for r in results
        if r["total_time"]
        == longest_time
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

relative_difference = (
    100.0
    * abs(
        long_mean
        - GATE3B_REFERENCE
    )
    / abs(
        GATE3B_REFERENCE
    )
)


# ============================================================
# LONGEST-RUN DIAGNOSTIC
# ============================================================

print()
print("=" * 76)
print("LONGEST-RUN DIAGNOSTIC")
print("=" * 76)

print(
    f"Integration time = "
    f"{longest_time:.0f} s"
)

print(
    f"Mean variational exponent = "
    f"{long_mean:.8f} 1/s"
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

print(
    f"Gate 3B reference = "
    f"{GATE3B_REFERENCE:.8f} 1/s"
)

print(
    f"Difference from Gate 3B = "
    f"{relative_difference:.3f}%"
)


# ============================================================
# FROZEN CRITERION CHECKS
# ============================================================

positive_mean_pass = (
    long_mean
    > MIN_POSITIVE_EXPONENT
)

spread_pass = (
    long_relative_std
    <= MAX_RELATIVE_STD_PERCENT
)

all_positive_pass = np.all(
    longest_values > 0.0
)

agreement_pass = (
    relative_difference
    <= MAX_RELATIVE_DIFFERENCE_PERCENT
)


# ============================================================
# AUTOMATIC VERDICT
# ============================================================

if (
    positive_mean_pass
    and spread_pass
    and all_positive_pass
    and agreement_pass
):

    verdict = (
        "PASS - Gate 5 independent "
        "variational validation criteria "
        "satisfied"
    )

elif (
    positive_mean_pass
    and all_positive_pass
):

    verdict = (
        "INCONCLUSIVE - independent "
        "variational method recovered a "
        "positive exponent, but convergence "
        "or Gate 3B agreement criteria were "
        "not fully satisfied"
    )

else:

    verdict = (
        "FAIL - independent variational "
        "method did not recover the required "
        "positive Lyapunov behavior"
    )


# ============================================================
# VERDICT OUTPUT
# ============================================================

print()
print("=" * 76)
print("VERDICT")
print("=" * 76)

print(verdict)

print()
print("Criterion checks:")

print(
    f"Mean > "
    f"{MIN_POSITIVE_EXPONENT:.2f} "
    f"= {positive_mean_pass}"
)

print(
    f"Relative std <= "
    f"{MAX_RELATIVE_STD_PERCENT:.1f}% "
    f"= {spread_pass}"
)

print(
    "All longest-run estimates "
    f"positive = {all_positive_pass}"
)

print(
    f"Difference from Gate 3B <= "
    f"{MAX_RELATIVE_DIFFERENCE_PERCENT:.1f}% "
    f"= {agreement_pass}"
)


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

print()
print("=" * 76)
print("INTERPRETATION")
print("=" * 76)

print(
    "Gate 5 uses tangent-space dynamics "
    "rather than the two-nearby-trajectory "
    "estimator used in Gate 3B."
)

print(
    "Agreement therefore provides an "
    "independent numerical cross-check of "
    "the positive largest Lyapunov exponent."
)

print(
    "A PASS increases confidence in the "
    "double-pendulum numerical chaos "
    "classification."
)

print(
    "A PASS is not a mathematical proof of "
    "chaos and is not experimental validation "
    "of a physical double pendulum."
)

print(
    "Gate 3B remains frozen regardless of "
    "the Gate 5 outcome."
)

print()
print("=" * 76)
