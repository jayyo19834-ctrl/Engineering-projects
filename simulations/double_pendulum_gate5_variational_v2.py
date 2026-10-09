# ============================================================
# GATE 5B — DOUBLE PENDULUM VARIATIONAL LYAPUNOV VALIDATION
# OPTIMIZED ANALYTIC-JACOBIAN EXECUTION
# ============================================================
#
# STATUS / PROVENANCE
#
# Gate 5:
#   Frozen complex-step variational implementation.
#   Execution incomplete because of runtime/resource limit.
#   Gate 5 remains unchanged.
#
# Gate 5B:
#   Execution-optimized implementation using an analytic
#   Jacobian of the SAME double-pendulum vector field.
#
# IMPORTANT:
# Before any Lyapunov calculation is accepted, the analytic
# Jacobian must agree with the frozen complex-step Jacobian
# at predefined validation states.
#
# The physical model, initial condition, integrator,
# tolerances, convergence times, renormalization intervals,
# Gate 3B reference, and acceptance criteria are unchanged.
#
# Do not change acceptance criteria after execution.
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
# FROZEN COMPLEX-STEP SETTING
# ============================================================

COMPLEX_STEP = 1e-20


# ============================================================
# JACOBIAN VALIDATION THRESHOLD
# ============================================================
#
# This criterion is defined before execution.
#
# The analytic Jacobian must agree with complex-step to:
#
#     max absolute element error <= 1e-10
#
# at every predefined validation state.
#
# Failure blocks Gate 5B execution.
# ============================================================

JACOBIAN_MAX_ABS_ERROR = 1e-10


# ============================================================
# FROZEN GATE 5 / GATE 3B ACCEPTANCE CRITERIA
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
# [theta1, omega1, theta2, omega2]
#
# This is the SAME vector field used by frozen Gate 5.
#
# Do not cast to float inside this function because the
# complex-step validation requires complex arithmetic.
# ============================================================

def pendulum_deriv(state):

    theta1, omega1, theta2, omega2 = state

    delta = theta1 - theta2

    D = (
        2.0 * M1
        + M2
        - M2 * np.cos(2.0 * delta)
    )

    denominator1 = L1 * D
    denominator2 = L2 * D

    Q = (
        omega2**2 * L2
        + omega1**2 * L1 * np.cos(delta)
    )

    numerator1 = (
        -G
        * (2.0 * M1 + M2)
        * np.sin(theta1)

        - M2
        * G
        * np.sin(theta1 - 2.0 * theta2)

        - 2.0
        * M2
        * np.sin(delta)
        * Q
    )

    R = (
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

    numerator2 = (
        2.0
        * np.sin(delta)
        * R
    )

    domega1 = (
        numerator1
        / denominator1
    )

    domega2 = (
        numerator2
        / denominator2
    )

    return np.array(
        [
            omega1,
            domega1,
            omega2,
            domega2,
        ]
    )


# ============================================================
# FROZEN COMPLEX-STEP JACOBIAN
# ============================================================
#
# Used ONLY for validation in Gate 5B.
# It is not used during the production Lyapunov integrations.
# ============================================================

def jacobian_complex_step(state):

    state = np.asarray(
        state,
        dtype=float
    )

    dimension = len(state)

    J = np.zeros(
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

        J[:, column] = (
            np.imag(derivative)
            / COMPLEX_STEP
        )

    return J


# ============================================================
# ANALYTIC JACOBIAN
# ============================================================
#
# The derivatives below are algebraic derivatives of the
# exact same vector field defined above.
#
# Jacobian ordering:
#
# rows:
#   theta1_dot
#   omega1_dot
#   theta2_dot
#   omega2_dot
#
# columns:
#   theta1
#   omega1
#   theta2
#   omega2
# ============================================================

def jacobian_analytic(state):

    theta1, omega1, theta2, omega2 = state

    delta = theta1 - theta2

    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    sin_2d = np.sin(2.0 * delta)

    D = (
        2.0 * M1
        + M2
        - M2 * np.cos(2.0 * delta)
    )

    # --------------------------------------------------------
    # Derivatives of D
    # --------------------------------------------------------

    dD_dt1 = (
        2.0
        * M2
        * sin_2d
    )

    dD_dt2 = (
        -2.0
        * M2
        * sin_2d
    )

    # --------------------------------------------------------
    # Q for omega1_dot numerator
    # --------------------------------------------------------

    Q = (
        omega2**2 * L2
        + omega1**2 * L1 * cos_d
    )

    dQ_dt1 = (
        -omega1**2
        * L1
        * sin_d
    )

    dQ_dt2 = (
        omega1**2
        * L1
        * sin_d
    )

    dQ_dw1 = (
        2.0
        * omega1
        * L1
        * cos_d
    )

    dQ_dw2 = (
        2.0
        * omega2
        * L2
    )

    # --------------------------------------------------------
    # Numerator N1
    # --------------------------------------------------------

    N1 = (
        -G
        * (2.0 * M1 + M2)
        * np.sin(theta1)

        - M2
        * G
        * np.sin(theta1 - 2.0 * theta2)

        - 2.0
        * M2
        * sin_d
        * Q
    )

    dN1_dt1 = (
        -G
        * (2.0 * M1 + M2)
        * np.cos(theta1)

        - M2
        * G
        * np.cos(theta1 - 2.0 * theta2)

        - 
