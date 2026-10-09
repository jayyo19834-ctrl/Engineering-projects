# ============================================================
# GATE 5B REVISION 2
# DOUBLE PENDULUM VARIATIONAL LYAPUNOV VALIDATION
# ============================================================
#
# PROVENANCE
#
# Previous Gate 5B blob:
# 027a1352a277fe6d5a920f152969dfea5fd51d9f
#
# Status of previous blob:
# INVALID / INCOMPLETE IMPLEMENTATION SNAPSHOT.
# The committed file ended inside jacobian_analytic().
# It was never scientifically executed.
#
# This revision preserves the intended Gate 5B method and all
# frozen scientific acceptance criteria.
#
# METHOD
# - Same double-pendulum vector field as Gate 5
# - Tangent-space variational dynamics
# - Analytic Jacobian for production integration
# - Complex-step Jacobian used as independent implementation
#   cross-check before Lyapunov execution
# - DOP853
# - rtol = 1e-10
# - atol = 1e-12
#
# IMPORTANT
# Do not change acceptance criteria after seeing results.
# ============================================================

import numpy as np
from scipy.integrate import solve_ivp
from time import perf_counter


# ============================================================
# PHYSICAL MODEL
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

TOTAL_TIMES = [20.0, 40.0, 80.0, 160.0]

RENORM_INTERVALS = [0.025, 0.050, 0.100]

COMPLEX_STEP = 1e-20


# ============================================================
# FROZEN JACOBIAN VALIDATION CRITERION
# ============================================================

JACOBIAN_MAX_ABS_ERROR = 1e-10


# ============================================================
# FROZEN GATE 5B ACCEPTANCE CRITERIA
# ============================================================

GATE3B_REFERENCE = 1.244525852944515

MIN_POSITIVE_EXPONENT = 0.50

MAX_RELATIVE_STD_PERCENT = 10.0

MAX_RELATIVE_DIFFERENCE_PERCENT = 25.0


# ============================================================
# INITIAL CONDITIONS
# ============================================================

INITIAL_STATE = np.array(
    [0.9 * np.pi, 0.0, 0.9 * np.pi, 0.0],
    dtype=float,
)

INITIAL_TANGENT = np.array(
    [1.0, 0.0, 0.0, 0.0],
    dtype=float,
)


# ============================================================
# DOUBLE-PENDULUM VECTOR FIELD
# ============================================================

def pendulum_deriv(state):

    t1, w1, t2, w2 = state

    d = t1 - t2

    D = (
        2.0 * M1
        + M2
        - M2 * np.cos(2.0 * d)
    )

    Q = (
        w2**2 * L2
        + w1**2 * L1 * np.cos(d)
    )

    N1 = (
        -G * (2.0 * M1 + M2) * np.sin(t1)
        - M2 * G * np.sin(t1 - 2.0 * t2)
        - 2.0 * M2 * np.sin(d) * Q
    )

    R = (
        w1**2 * L1 * (M1 + M2)
        + G * (M1 + M2) * np.cos(t1)
        + w2**2 * L2 * M2 * np.cos(d)
    )

    N2 = (
        2.0 * np.sin(d) * R
    )

    return np.array(
        [
            w1,
            N1 / (L1 * D),
            w2,
            N2 / (L2 * D),
        ]
    )


# ============================================================
# COMPLEX-STEP JACOBIAN
# ============================================================
#
# This reproduces the Gate 5 Jacobian method and is used here
# only to validate the analytic implementation.
# ============================================================

def jacobian_complex_step(state):

    x = np.asarray(state, dtype=float)

    J = np.zeros((4, 4), dtype=float)

    for j in range(4):

        z = x.astype(complex)

        z[j] += 1j * COMPLEX_STEP

        J[:, j] = (
            np.imag(pendulum_deriv(z))
            / COMPLEX_STEP
        )

    return J


# ============================================================
# ANALYTIC JACOBIAN
# ============================================================

def jacobian_analytic(state):

    t1, w1, t2, w2 = state

    d = t1 - t2

    sd = np.sin(d)
    cd = np.cos(d)
    s2d = np.sin(2.0 * d)

    D = (
        2.0 * M1
        + M2
        - M2 * np.cos(2.0 * d)
    )

    dD_t1 = 2.0 * M2 * s2d
    dD_t2 = -dD_t1

    # --------------------------------------------------------
    # First angular-acceleration numerator
    # --------------------------------------------------------

    Q = (
        w2**2 * L2
        + w1**2 * L1 * cd
    )

    dQ_t1 = -w1**2 * L1 * sd
    dQ_t2 = +w1**2 * L1 * sd

    dQ_w1 = 2.0 * w1 * L1 * cd
    dQ_w2 = 2.0 * w2 * L2

    N1 = (
        -G * (2.0 * M1 + M2) * np.sin(t1)
        - M2 * G * np.sin(t1 - 2.0 * t2)
        - 2.0 * M2 * sd * Q
    )

    dN1_t1 = (
        -G * (2.0 * M1 + M2) * np.cos(t1)
        - M2 * G * np.cos(t1 - 2.0 * t2)
        - 2.0 * M2 * (
            cd * Q
            + sd * dQ_t1
        )
    )

    dN1_t2 = (
        2.0 * M2 * G * np.cos(t1 - 2.0 * t2)
        - 2.0 * M2 * (
            -cd * Q
            + sd * dQ_t2
        )
    )

    dN1_w1 = (
        -2.0 * M2 * sd * dQ_w1
    )

    dN1_w2 = (
        -2.0 * M2 * sd * dQ_w2
    )

    # --------------------------------------------------------
    # Second angular-acceleration numerator
    # --------------------------------------------------------

    R = (
        w1**2 * L1 * (M1 + M2)
        + G * (M1 + M2) * np.cos(t1)
        + w2**2 * L2 * M2 * cd
    )

    dR_t1 = (
        -G * (M1 + M2) * np.sin(t1)
        - w2**2 * L2 * M2 * sd
    )

    dR_t2 = (
        w2**2 * L2 * M2 * sd
    )

    dR_w1 = (
        2.0 * w1 * L1 * (M1 + M2)
    )

    dR_w2 = (
        2.0 * w2 * L2 * M2 * cd
    )

    N2 = 2.0 * sd * R

    dN2_t1 = (
        2.0 * (
            cd * R
            + sd * dR_t1
        )
    )

    dN2_t2 = (
        2.0 * (
            -cd * R
            + sd * dR_t2
        )
    )

    dN2_w1 = (
        2.0 * sd * dR_w1
    )

    dN2_w2 = (
        2.0 * sd * dR_w2
    )

    # --------------------------------------------------------
    # Quotient rule
    # --------------------------------------------------------

    D2 = D**2

    a1_t1 = (
        dN1_t1 * D
        - N1 * dD_t1
    ) / (L1 * D2)

    a1_t2 = (
        dN1_t2 * D
        - N1 * dD_t2
    ) / (L1 * D2)

    a1_w1 = dN1_w1 / (L1 * D)
    a1_w2 = dN1_w2 / (L1 * D)

    a2_t1 = (
        dN2_t1 * D
        - N2 * dD_t1
    ) / (L2 * D2)

    a2_t2 = (
        dN2_t2 * D
        - N2 * dD_t2
    ) / (L
