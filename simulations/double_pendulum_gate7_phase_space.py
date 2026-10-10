# ============================================================
# GATE 7 — PHASE-SPACE DISCRIMINATION
# ============================================================
#
# Implements the frozen Gate 7 preregistration.
# Do not change states, horizons, thresholds, or verdict
# logic after results exist.
#
# Machinery is the Gate 6 spectrum procedure:
#   same vector field, analytic Jacobian, DOP853,
#   rtol = 1e-10, atol = 1e-12, identity tangent basis,
#   QR reorthonormalization.
#   Exponents are computed first, then sorted descending.
#
# Matrix: 6 states x 2 horizons x 3 QR intervals = 36 runs,
# subject to the Jacobian hard gate.
#
# Classification and verdict use the three 320-s runs only.
# The 160-s results are diagnostic only.
#
# CHAOTIC and REGULAR are operational finite-time labels.
# REGULAR does not mean mathematically proven nonchaotic.
# An intermediate CHAOTIC label is not a statement about
# its surrounding phase-space region.
#
# I1-I4 are labeled and do not affect the verdict.
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
# FROZEN NUMERICAL SETTINGS
# ============================================================

RTOL = 1e-10
ATOL = 1e-12

TOTAL_TIMES = [
    160.0,
    320.0,
]

QR_INTERVALS = [
    0.025,
    0.050,
    0.100,
]

COMPLEX_STEP = 1e-20

JACOBIAN_MAX_ABS_ERROR = 1e-10


# ============================================================
# FROZEN CLASSIFICATION THRESHOLDS
# Applied only to the three 320-s runs.
# ============================================================

CHAOTIC_MEAN = 0.50
REGULAR_MEAN = 0.10
REGULAR_RUN_MAX = 0.15
MAX_RELATIVE_STD = 0.10

GATE6_REFERENCE = 1.11534062
MAX_GATE6_RELATIVE_DIFFERENCE = 0.25


# ============================================================
# FROZEN INITIAL STATES
# Chosen before execution. Do not add or drop any.
# ============================================================

STATES = [
    (
        "C+",
        "positive control",
        np.array([0.9 * np.pi, 0.0, 0.9 * np.pi, 0.0]),
    ),
    (
        "C0",
        "regular control",
        np.array([0.10, 0.0, 0.10, 0.0]),
    ),
    (
        "I1",
        "low intermediate",
        np.array([0.50, 0.0, 0.50, 0.0]),
    ),
    (
        "I2",
        "horizontal",
        np.array([np.pi / 2.0, 0.0, np.pi / 2.0, 0.0]),
    ),
    (
        "I3",
        "asymmetric",
        np.array([2.0, 0.0, 1.0, 0.0]),
    ),
    (
        "I4",
        "high, not the control",
        np.array([0.8 * np.pi, 0.0, 0.6 * np.pi, 0.0]),
    ),
]


# ============================================================
# PREDEFINED JACOBIAN VALIDATION STATES
# Same five states as Gate 5B / Gate 6.
# ============================================================

JACOBIAN_TEST_STATES = [

    np.array([
        0.9 * np.pi,
        0.0,
        0.9 * np.pi,
        0.0,
    ]),

    np.array([
        1.0,
        0.5,
        0.4,
        -0.3,
    ]),

    np.array([
        -0.8,
        1.2,
        0.7,
        -0.9,
    ]),

    np.array([
        2.4,
        -1.1,
        -1.6,
        0.8,
    ]),

    np.array([
        0.25,
        2.0,
        -0.45,
        -1.5,
    ]),
]


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
# FROZEN COMPLEX-STEP JACOBIAN
# Validation only. Not used in the spectrum runs.
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
# Same analytic Jacobian as frozen Gate 5B.
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


# ============================================================
# PREDEFINED JACOBIAN VALIDATION STATES
# Same five states as Gate 5B.
# ============================================================

JACOBIAN_TEST_STATES = [

    np.array([
        0.9 * np.pi,
        0.0,
        0.9 * np.pi,
        0.0,
    ]),

    np.array([
        1.0,
        0.5,
        0.4,
        -0.3,
    ]),

    np.array([
        -0.8,
        1.2,
        0.7,
        -0.9,
    ]),

    np.array([
        2.4,
        -1.1,
        -1.6,
        0.8,
    ]),

    np.array([
        0.25,
        2.0,
        -0.45,
        -1.5,
    ]),
]


# ============================================================
# JACOBIAN PRE-GATE
# Failure blocks every spectrum run.
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

        error = np.max(
            np.abs(
                jacobian_complex_step(state)
                - jacobian_analytic(state)
            )
        )

        errors.append(error)

        print(
            f"State {i}: "
            f"max abs error = {error:.3e}"
        )

    worst = max(errors)

    passed = (
        worst
        <= JACOBIAN_MAX_ABS_ERROR
    )

    print()
    print(
        f"Worst error = {worst:.3e}"
    )
    print(
        "Frozen tolerance = "
        f"{JACOBIAN_MAX_ABS_ERROR:.3e}"
    )
    print(
        f"Jacobian PASS = {passed}"
    )

    return passed, worst, errors


# ============================================================
# 20-STATE VARIATIONAL FIELD
# Columns of Phi are the tangent vectors.
# ============================================================

def variational_deriv(t, combined):

    state = combined[:4]

    phi = combined[4:].reshape(
        (4, 4),
        order="F",
    )

    state_dot = pendulum_deriv(state)

    phi_dot = (
        jacobian_analytic(state)
        @ phi
    )

    return np.concatenate(
        [
            state_dot,
            phi_dot.reshape(
                16,
                order="F",
            ),
        ]
    )


# ============================================================
# QR SPECTRUM
# Exponents are returned unsorted. The caller sorts.
# ============================================================

def lyapunov_spectrum(
    initial_state,
    dt_qr,
    total_time,
):

    state = np.array(
        initial_state,
        dtype=float,
    )

    phi = np.eye(4)

    log_sum = np.zeros(4)

    elapsed = 0.0

    while elapsed < total_time - 1e-12:

        interval = min(
            dt_qr,
            total_time - elapsed,
        )

        initial = np.concatenate(
            [
                state,
                phi.reshape(16, order="F"),
            ]
        )

        sol = solve_ivp(
            variational_deriv,
            (0.0, interval),
            initial,
            method="DOP853",
            rtol=RTOL,
            atol=ATOL,
        )

        if not sol.success:

            raise RuntimeError(
                sol.message
            )

        final = sol.y[:, -1]

        state = final[:4]

        phi = final[4:].reshape(
            (4, 4),
            order="F",
        )

        q, r = np.linalg.qr(phi)

        diagonal = np.abs(
            np.diag(r)
        )

        if (
            not np.all(np.isfinite(diagonal))
            or np.any(diagonal <= 0.0)
        ):

            raise RuntimeError(
                "Invalid QR diagonal."
            )

        log_sum += np.log(diagonal)

        phi = q

        elapsed += interval

    return log_sum / total_time


def sort_descending(exponents):

    return np.sort(
        np.asarray(exponents, dtype=float)
    )[::-1]


def classify_320(runs):
    """Operational label from the three 320-s largest exponents.

    CHAOTIC requires every locked inequality.
    REGULAR requires every locked inequality.
    Otherwise INDETERMINATE.
    """

    values = np.array(
        [row[3][0] for row in runs],
        dtype=float,
    )

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    relative = (
        std / abs(mean)
        if mean != 0.0
        else np.inf
    )

    chaotic = (
        mean > CHAOTIC_MEAN
        and np.all(values > 0.0)
        and relative <= MAX_RELATIVE_STD
    )

    regular = (
        mean <= REGULAR_MEAN
        and np.all(values <= REGULAR_RUN_MAX)
    )

    if chaotic:
        label = "CHAOTIC"
    elif regular:
        label = "REGULAR"
    else:
        label = "INDETERMINATE"

    return {
        "values": values,
        "mean": mean,
        "std": std,
        "relative": relative,
        "label": label,
        "chaotic": chaotic,
        "regular": regular,
    }


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 72)
print("GATE 7 — PHASE-SPACE DISCRIMINATION")
print("=" * 72)
print()
print(
    "Frozen Gate 6 reference = "
    f"{GATE6_REFERENCE:.8f} 1/s"
)
print("Horizons = 160 s diagnostic, 320 s classification")
print("QR intervals = 0.025, 0.050, 0.100")
print("Sorting = descending after each run")
print("Standard deviation = population")
print("I1-I4 do not affect the verdict")


# ============================================================
# HARD JACOBIAN GATE
# ============================================================

jacobian_passed, worst_error, jacobian_errors = (
    validate_jacobian()
)

if not jacobian_passed:

    print()
    print("=" * 72)
    print(
        "GATE 7 BLOCKED — "
        "JACOBIAN VALIDATION FAILED"
    )
    print("=" * 72)
    print("No spectrum run was executed.")
    raise SystemExit(0)


# ============================================================
# FROZEN 36-RUN MATRIX
# ============================================================

print()
print("=" * 72)
print("SPECTRUM MATRIX")
print("=" * 72)

results = []

for label, role, state in STATES:

    print()
    print(f"STATE {label} — {role}")
    print(
        "y0 = "
        + np.array2string(
            state,
            precision=8,
            separator=", ",
        )
    )

    for total_time in TOTAL_TIMES:

        for dt_qr in QR_INTERVALS:

            started = perf_counter()

            raw = lyapunov_spectrum(
                state,
                dt_qr,
                total_time,
            )

            ordered = sort_descending(raw)

            runtime = perf_counter() - started

            results.append(
                (
                    label,
                    role,
                    total_time,
                    dt_qr,
                    ordered,
                    runtime,
                )
            )

            print(
                f"T={total_time:6.1f} "
                f"dt={dt_qr:.3f} "
                f"l1={ordered[0]: .8f} "
                f"l2={ordered[1]: .8f} "
                f"l3={ordered[2]: .8f} "
                f"l4={ordered[3]: .8f} "
                f"runtime={runtime:7.2f}s"
            )


# ============================================================
# 160-SECOND DIAGNOSTIC
# Not used for classification or verdict.
# ============================================================

print()
print("=" * 72)
print("160-SECOND DIAGNOSTIC")
print("Not used for classification or verdict.")
print("=" * 72)

for label, role, state in STATES:

    runs = [
        row for row in results
        if row[0] == label and row[2] == 160.0
    ]

    values = np.array([row[4][0] for row in runs])
    mean = np.mean(values)
    std = np.std(values, ddof=0)

    print(
        f"{label}: "
        f"mean={mean:.8f} "
        f"std={std:.8f} "
        f"runs="
        + ", ".join(f"{v:.8f}" for v in values)
    )


# ============================================================
# 320-SECOND CLASSIFICATION
# ============================================================

print()
print("=" * 72)
print("320-SECOND CLASSIFICATION")
print("=" * 72)
print(
    "CHAOTIC and REGULAR are operational "
    "finite-time labels, not proofs."
)

classified = {}

for label, role, state in STATES:

    runs = [
        row for row in results
        if row[0] == label and row[2] == 320.0
    ]

    # adapt tuple layout for classify_320
    adapted = [
        (row[0], row[1], row[2], row[4], row[5])
        for row in runs
    ]

    info = classify_320(adapted)
    classified[label] = info

    print()
    print(f"{label} — {role}")
    print(
        "runs = "
        + ", ".join(f"{v:.8f}" for v in info["values"])
    )
    print(f"mean = {info['mean']:.8f}")
    print(f"std = {info['std']:.8f}")
    print(
        f"relative std = {100.0 * info['relative']:.3f}%"
    )
    print(f"label = {info['label']}")


# ============================================================
# CONTROL CHECKS
# ============================================================

c_plus = classified["C+"]
c_zero = classified["C0"]

gate6_difference = (
    abs(c_plus["mean"] - GATE6_REFERENCE)
    / abs(GATE6_REFERENCE)
)

c_plus_chaotic = c_plus["label"] == "CHAOTIC"
c_zero_regular = c_zero["label"] == "REGULAR"
gate6_ok = (
    gate6_difference
    <= MAX_GATE6_RELATIVE_DIFFERENCE
)

print()
print("=" * 72)
print("CONTROL CHECKS")
print("=" * 72)
print(f"C+ CHAOTIC = {c_plus_chaotic}")
print(f"C0 REGULAR = {c_zero_regular}")
print(
    "C+ difference from Gate 6 = "
    f"{100.0 * gate6_difference:.3f}%"
)
print(f"C+ agrees with Gate 6 = {gate6_ok}")
print(f"Worst Jacobian error = {worst_error:.3e}")
print("I1-I4 influence on verdict = none")


# ============================================================
# VERDICT
# ============================================================

if not jacobian_passed:

    verdict = (
        "BLOCKED - Jacobian validation failed"
    )

elif not c_plus_chaotic:

    verdict = (
        "FAIL - positive control did not satisfy "
        "the complete CHAOTIC classification"
    )

elif c_zero_regular and gate6_ok:

    verdict = (
        "PASS - controls separated under the "
        "frozen finite-time classification"
    )

else:

    verdict = (
        "INCONCLUSIVE - positive control was "
        "CHAOTIC, but regular-control and/or "
        "Gate 6 agreement criteria were not satisfied"
    )

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
    "A PASS means this procedure distinguished "
    "the preregistered controls."
)
print(
    "REGULAR means the finite-time largest "
    "exponent met the preregistered regular "
    "thresholds. It is not a proof that the "
    "trajectory is nonchaotic."
)
print(
    "An intermediate CHAOTIC label is a result "
    "under this finite-time procedure, not a "
    "statement about its surrounding region."
)
print(
    "This is not a claim that every trajectory "
    "is chaotic, and it is not a physical experiment."
)
print()
print("=" * 72)
