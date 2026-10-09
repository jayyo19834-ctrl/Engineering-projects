# ============================================================
# GATE 5B REVISION 2
# DOUBLE PENDULUM VARIATIONAL LYAPUNOV VALIDATION
# ============================================================
#
# Previous invalid/incomplete Gate 5B blob:
# 027a1352a277fe6d5a920f152969dfea5fd51d9f
#
# That blob was never scientifically executed.
#
# This revision preserves the intended Gate 5B method and
# all frozen scientific acceptance criteria.
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

COMPLEX_STEP = 1e-20


# ============================================================
# FROZEN CRITERIA
# ============================================================

JACOBIAN_MAX_ABS_ERROR = 1e-10

GATE3B_REFERENCE = 1.244525852944515

MIN_POSITIVE_EXPONENT = 0.50

MAX_RELATIVE_STD_PERCENT = 10.0

MAX_RELATIVE_DIFFERENCE_PERCENT = 25.0


# ============================================================
# INITIAL CONDITIONS
# ============================================================

INITIAL_STATE = np.array(
    [
        0.9 * np.pi,
        0.0,
        0.9 * np.pi,
        0.0,
    ],
    dtype=float,
)

INITIAL_TANGENT = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
    ],
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
        -G
        * (2.0 * M1 + M2)
        * np.sin(t1)

        - M2
        * G
        * np.sin(t1 - 2.0 * t2)

        - 2.0
        * M2
        * np.sin(d)
        * Q
    )

    R = (
        w1**2
        * L1
        * (M1 + M2)

        + G
        * (M1 + M2)
        * np.cos(t1)

        + w2**2
        * L2
        * M2
        * np.cos(d)
    )

    N2 = (
        2.0
        * np.sin(d)
        * R
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

def jacobian_complex_step(state):

    x = np.asarray(
        state,
        dtype=float,
    )

    J = np.zeros(
        (4, 4),
        dtype=float,
    )

    for j in range(4):

        z = x.astype(complex)

        z[j] += (
            1j * COMPLEX_STEP
        )

        J[:, j] = (
            np.imag(
                pendulum_deriv(z)
            )
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

    dD_t1 = (
        2.0
        * M2
        * s2d
    )

    dD_t2 = (
        -dD_t1
    )

    # --------------------------------------------------------
    # First acceleration numerator
    # --------------------------------------------------------

    Q = (
        w2**2 * L2
        + w1**2 * L1 * cd
    )

    dQ_t1 = (
        -w1**2
        * L1
        * sd
    )

    dQ_t2 = (
        w1**2
        * L1
        * sd
    )

    dQ_w1 = (
        2.0
        * w1
        * L1
        * cd
    )

    dQ_w2 = (
        2.0
        * w2
        * L2
    )

    N1 = (
        -G
        * (2.0 * M1 + M2)
        * np.sin(t1)

        - M2
        * G
        * np.sin(t1 - 2.0 * t2)

        - 2.0
        * M2
        * sd
        * Q
    )

    dN1_t1 = (
        -G
        * (2.0 * M1 + M2)
        * np.cos(t1)

        - M2
        * G
        * np.cos(t1 - 2.0 * t2)

        - 2.0
        * M2
        * (
            cd * Q
            + sd * dQ_t1
        )
    )

    dN1_t2 = (
        2.0
        * M2
        * G
        * np.cos(t1 - 2.0 * t2)

        - 2.0
        * M2
        * (
            -cd * Q
            + sd * dQ_t2
        )
    )

    dN1_w1 = (
        -2.0
        * M2
        * sd
        * dQ_w1
    )

    dN1_w2 = (
        -2.0
        * M2
        * sd
        * dQ_w2
    )

    # --------------------------------------------------------
    # Second acceleration numerator
    # --------------------------------------------------------

    R = (
        w1**2
        * L1
        * (M1 + M2)

        + G
        * (M1 + M2)
        * np.cos(t1)

        + w2**2
        * L2
        * M2
        * cd
    )

    dR_t1 = (
        -G
        * (M1 + M2)
        * np.sin(t1)

        - w2**2
        * L2
        * M2
        * sd
    )

    dR_t2 = (
        w2**2
        * L2
        * M2
        * sd
    )

    dR_w1 = (
        2.0
        * w1
        * L1
        * (M1 + M2)
    )

    dR_w2 = (
        2.0
        * w2
        * L2
        * M2
        * cd
    )

    N2 = (
        2.0
        * sd
        * R
    )

    dN2_t1 = (
        2.0
        * (
            cd * R
            + sd * dR_t1
        )
    )

    dN2_t2 = (
        2.0
        * (
            -cd * R
            + sd * dR_t2
        )
    )

    dN2_w1 = (
        2.0
        * sd
        * dR_w1
    )

    dN2_w2 = (
        2.0
        * sd
        * dR_w2
    )

    # --------------------------------------------------------
    # Quotient rule
    # --------------------------------------------------------

    D2 = D**2

    a1_t1 = (
        dN1_t1 * D
        - N1 * dD_t1
    ) / (
        L1 * D2
    )

    a1_t2 = (
        dN1_t2 * D
        - N1 * dD_t2
    ) / (
        L1 * D2
    )

    a1_w1 = (
        dN1_w1
        / (L1 * D)
    )

    a1_w2 = (
        dN1_w2
        / (L1 * D)
    )

    a2_t1 = (
        dN2_t1 * D
        - N2 * dD_t1
    ) / (
        L2 * D2
    )

    a2_t2 = (
        dN2_t2 * D
        - N2 * dD_t2
    ) / (
        L2 * D2
    )

    a2_w1 = (
        dN2_w1
        / (L2 * D)
    )

    a2_w2 = (
        dN2_w2
        / (L2 * D)
    )

    return np.array(
        [
            [
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            [
                a1_t1,
                a1_w1,
                a1_t2,
                a1_w2,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            [
                a2_t1,
                a2_w1,
                a2_t2,
                a2_w2,
            ],
        ],
        dtype=float,
    )


# ===== END PART 1 — PASTE PART 2 DIRECTLY BELOW =====
# ============================================================
# PREDEFINED JACOBIAN VALIDATION STATES
# ============================================================

JACOBIAN_TEST_STATES = [

    np.array(
        [
            0.9 * np.pi,
            0.0,
            0.9 * np.pi,
            0.0,
        ]
    ),

    np.array(
        [
            1.0,
            0.5,
            0.4,
            -0.3,
        ]
    ),

    np.array(
        [
            -0.8,
            1.2,
            0.7,
            -0.9,
        ]
    ),

    np.array(
        [
            2.4,
            -1.1,
            -1.6,
            0.8,
        ]
    ),

    np.array(
        [
            0.25,
            2.0,
            -0.45,
            -1.5,
        ]
    ),
]


# ============================================================
# JACOBIAN VALIDATION
# ============================================================

def validate_jacobian():

    print()
    print("=" * 72)
    print("ANALYTIC JACOBIAN VALIDATION")
    print("=" * 72)

    errors = []

    for i, state in enumerate(
        JACOBIAN_TEST_STATES,
        start=1,
    ):

        J_complex = (
            jacobian_complex_step(
                state
            )
        )

        J_analytic = (
            jacobian_analytic(
                state
            )
        )

        error = np.max(
            np.abs(
                J_complex
                - J_analytic
            )
        )

        errors.append(error)

        print(
            f"State {i}: "
            f"max abs error = "
            f"{error:.3e}"
        )

    worst = max(errors)

    passed = (
        worst
        <= JACOBIAN_MAX_ABS_ERROR
    )

    print()

    print(
        f"Worst error = "
        f"{worst:.3e}"
    )

    print(
        "Frozen tolerance = "
        f"{JACOBIAN_MAX_ABS_ERROR:.3e}"
    )

    print(
        f"Jacobian PASS = "
        f"{passed}"
    )

    return passed, worst


# ============================================================
# VARIATIONAL EQUATIONS
# ============================================================

def variational_deriv(
    t,
    combined,
):

    state = combined[:4]

    tangent = combined[4:]

    state_dot = (
        pendulum_deriv(
            state
        )
    )

    tangent_dot = (
        jacobian_analytic(
            state
        )
        @ tangent
    )

    return np.concatenate(
        [
            state_dot,
            tangent_dot,
        ]
    )


# ============================================================
# LYAPUNOV ESTIMATOR
# ============================================================

def variational_lyapunov(
    initial_state,
    initial_tangent,
    dt_renorm,
    total_time,
):

    state = np.array(
        initial_state,
        dtype=float,
    )

    tangent = np.array(
        initial_tangent,
        dtype=float,
    )

    norm = np.linalg.norm(
        tangent
    )

    if norm <= 0.0:

        raise RuntimeError(
            "Initial tangent "
            "has zero norm."
        )

    tangent /= norm

    log_growth = 0.0

    elapsed = 0.0

    while elapsed < total_time:

        interval = min(
            dt_renorm,
            total_time - elapsed,
        )

        initial = np.concatenate(
            [
                state,
                tangent,
            ]
        )

        sol = solve_ivp(
            variational_deriv,
            (
                0.0,
                interval,
            ),
            initial,
            method="DOP853",
            rtol=RTOL,
            atol=ATOL,
        )

        if not sol.success:

            raise RuntimeError(
                "Variational integration "
                "failed."
            )

        final = sol.y[:, -1]

        state = final[:4]

        tangent = final[4:]

        norm = np.linalg.norm(
            tangent
        )

        if (
            not np.isfinite(norm)
            or norm <= 0.0
        ):

            raise RuntimeError(
                "Invalid tangent norm."
            )

        log_growth += np.log(
            norm
        )

        tangent /= norm

        elapsed += interval

    return (
        log_growth
        / total_time
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 72)

print(
    "GATE 5B REVISION 2 — "
    "VARIATIONAL LYAPUNOV VALIDATION"
)

print("=" * 72)

print()

print(
    "Previous incomplete blob:"
)

print(
    "027a1352a277fe6d5a920f152969dfea5fd51d9f"
)

print()

print(
    "Gate 3B reference = "
    f"{GATE3B_REFERENCE:.12f} 1/s"
)

print()

print(
    "Frozen criteria:"
)

print(
    f"Mean > "
    f"{MIN_POSITIVE_EXPONENT:.2f} 1/s"
)

print(
    f"Relative std <= "
    f"{MAX_RELATIVE_STD_PERCENT:.1f}%"
)

print(
    "All 160-s estimates positive"
)

print(
    "Difference from Gate 3B <= "
    f"{MAX_RELATIVE_DIFFERENCE_PERCENT:.1f}%"
)


# ============================================================
# HARD JACOBIAN GATE
# ============================================================

jacobian_pass, jacobian_error = (
    validate_jacobian()
)

if not jacobian_pass:

    print()
    print("=" * 72)
    print("GATE 5B BLOCKED")
    print("=" * 72)

    print(
        "Analytic Jacobian failed "
        "the frozen complex-step "
        "cross-check."
    )

    print(
        "Lyapunov matrix was "
        "NOT executed."
    )

    raise SystemExit(1)


# ============================================================
# CONVERGENCE MATRIX
# ============================================================

results = []

print()
print("=" * 72)
print("CONVERGENCE MATRIX")
print("=" * 72)

for total_time in TOTAL_TIMES:

    print()

    print(
        f"TOTAL TIME = "
        f"{total_time:.0f} s"
    )

    for dt in (
        RENORM_INTERVALS
    ):

        start = perf_counter()

        exponent = (
            variational_lyapunov(
                INITIAL_STATE,
                INITIAL_TANGENT,
                dt,
                total_time,
            )
        )

        runtime = (
            perf_counter()
            - start
        )

        results.append(
            {
                "time":
                    total_time,

                "dt":
                    dt,

                "lambda":
                    exponent,

                "runtime":
                    runtime,
            }
        )

        print(
            f"T={total_time:6.1f} "
            f"dt={dt:5.3f} "
            f"lambda={exponent: .8f} "
            f"runtime={runtime:7.2f}s"
        )


# ============================================================
# CONVERGENCE SUMMARY
# ============================================================

print()
print("=" * 72)
print("CONVERGENCE SUMMARY")
print("=" * 72)

for total_time in TOTAL_TIMES:

    values = np.array(
        [
            r["lambda"]
            for r in results
            if r["time"]
            == total_time
        ]
    )

    mean = np.mean(
        values
    )

    std = np.std(
        values
    )

    relative_std = (
        100.0
        * std
        / abs(mean)
    )

    print()

    print(
        f"T = "
        f"{total_time:.0f} s"
    )

    print(
        "Mean exponent      = "
        f"{mean:.8f}"
    )

    print(
        "Standard deviation = "
        f"{std:.8f}"
    )

    print(
        "Minimum exponent   = "
        f"{np.min(values):.8f}"
    )

    print(
        "Maximum exponent   = "
        f"{np.max(values):.8f}"
    )

    print(
        "Relative std       = "
        f"{relative_std:.3f}%"
    )


# ============================================================
# LONGEST-RUN STATISTICS
# ============================================================

longest_time = max(
    TOTAL_TIMES
)

longest = np.array(
    [
        r["lambda"]
        for r in results
        if r["time"]
        == longest_time
    ]
)

long_mean = np.mean(
    longest
)

long_std = np.std(
    longest
)

long_relative_std = (
    100.0
    * long_std
    / abs(long_mean)
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
    longest > 0.0
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
        "PASS - Gate 5B independent "
        "variational validation criteria "
        "satisfied"
    )

elif (
    positive_mean_pass
    and all_positive_pass
):

    verdict = (
        "INCONCLUSIVE - positive "
        "exponent recovered, but "
        "convergence or Gate 3B "
        "agreement criteria were "
        "not fully satisfied"
    )

else:

    verdict = (
        "FAIL - required positive "
        "Lyapunov behavior "
        "not recovered"
    )


# ============================================================
# 160-SECOND DIAGNOSTIC
# ============================================================

print()
print("=" * 72)
print("160-SECOND DIAGNOSTIC")
print("=" * 72)

print(
    "Mean exponent = "
    f"{long_mean:.8f} 1/s"
)

print(
    f"Std = "
    f"{long_std:.8f}"
)

print(
    "Minimum = "
    f"{np.min(longest):.8f}"
)

print(
    "Maximum = "
    f"{np.max(longest):.8f}"
)

print(
    "Relative std = "
    f"{long_relative_std:.3f}%"
)

print(
    "Gate 3B reference = "
    f"{GATE3B_REFERENCE:.8f} 1/s"
)

print(
    "Difference from Gate 3B = "
    f"{relative_difference:.3f}%"
)

print(
    "Worst Jacobian error = "
    f"{jacobian_error:.3e}"
)


# ============================================================
# CRITERION CHECKS
# ============================================================

print()
print("=" * 72)
print("CRITERION CHECKS")
print("=" * 72)

print(
    "Jacobian validation = "
    f"{jacobian_pass}"
)

print(
    f"Mean > "
    f"{MIN_POSITIVE_EXPONENT:.2f} "
    f"= {positive_mean_pass}"
)

print(
    "Relative std <= "
    f"{MAX_RELATIVE_STD_PERCENT:.1f}% "
    f"= {spread_pass}"
)

print(
    "All 160-s estimates "
    "positive = "
    f"{all_positive_pass}"
)

print(
    "Difference from Gate 3B <= "
    f"{MAX_RELATIVE_DIFFERENCE_PERCENT:.1f}% "
    f"= {agreement_pass}"
)


# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)

print(verdict)


# ============================================================
# CLAIM BOUNDARY
# ============================================================

print()
print("=" * 72)
print("CLAIM BOUNDARY")
print("=" * 72)

print(
    "A PASS provides an independent "
    "tangent-space numerical cross-check "
    "of the Gate 3B positive Lyapunov "
    "result."
)

print(
    "It is not mathematical proof "
    "of chaos and is not physical "
    "experimental validation."
)

print(
    "Gate 5 and the previous incomplete "
    "Gate 5B blob remain preserved."
)

print()
print("=" * 72)

# ===== END PART 2 — COMPLETE FILE =====
