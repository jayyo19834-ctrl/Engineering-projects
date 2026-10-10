# ============================================================
# GATE 8 — PREREGISTERED SYMMETRIC-ANGLE TRANSECT
# ============================================================
#
# Implements the frozen Gate 8 preregistration.
# Do not change states, horizon, thresholds, or verdict
# logic after results exist.
#
# Machinery is the Gate 7 320-s classifier:
#   same vector field, analytic Jacobian, DOP853,
#   rtol = 1e-10, atol = 1e-12, identity tangent basis,
#   QR intervals 0.025, 0.050, 0.100.
#   Exponents are computed first, then sorted descending.
#
# Matrix: 11 states x 3 QR intervals = 33 runs, all at 320 s.
# There is no 160-s diagnostic matrix.
#
# theta_k = 0.50 + k/10 * (pi/2 - 0.50), k = 0..10
# state_k = [theta_k, 0, theta_k, 0]
#
# theta_0 and theta_10 are classification anchors only.
# They are not magnitude-reproduction tests.
# Interior points have zero influence on the verdict.
# INDETERMINATE is a legitimate interior result.
# Multiple label changes are legitimate.
# Monotonicity is not required.
# No interpolation and no inserted theta points.
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

TOTAL_TIME = 320.0

QR_INTERVALS = [
    0.025,
    0.050,
    0.100,
]

COMPLEX_STEP = 1e-20

JACOBIAN_MAX_ABS_ERROR = 1e-10


# ============================================================
# FROZEN CLASSIFICATION THRESHOLDS
# Same operational rules as Gate 7.
# ============================================================

CHAOTIC_MEAN = 0.50
REGULAR_MEAN = 0.10
REGULAR_RUN_MAX = 0.15
MAX_RELATIVE_STD = 0.10


# ============================================================
# FROZEN TRANSECT
# Formula is authoritative. No points may be added.
# ============================================================

THETA = np.array(
    [
        0.50
        + k / 10.0 * (np.pi / 2.0 - 0.50)
        for k in range(11)
    ],
    dtype=float,
)

STATES = [
    (
        k,
        float(theta),
        np.array([theta, 0.0, theta, 0.0]),
    )
    for k, theta in enumerate(THETA)
]


# ============================================================
# PREDEFINED JACOBIAN VALIDATION STATES
# Same five states as Gate 5B / Gate 6 / Gate 7.
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
print(
    "GATE 8 — PREREGISTERED "
    "SYMMETRIC-ANGLE TRANSECT"
)
print("=" * 72)
print()
print("Horizon = 320 s only")
print("QR intervals = 0.025, 0.050, 0.100")
print("States = 11")
print("Expected runs = 33")
print("Sorting = descending after each run")
print("Standard deviation = population")
print("Interior points do not affect the verdict")
print("No interpolation and no inserted points")


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
        "GATE 8 BLOCKED — "
        "JACOBIAN VALIDATION FAILED"
    )
    print("=" * 72)
    print("No spectrum run was executed.")
    raise SystemExit(0)


# ============================================================
# FROZEN 33-RUN MATRIX
# ============================================================

print()
print("=" * 72)
print("SPECTRUM MATRIX")
print("=" * 72)

results = []
completed = 0

for k, theta, state in STATES:

    print()
    print(
        f"STATE k={k:02d}  "
        f"theta={theta:.10f}"
    )
    print(
        "y0 = "
        + np.array2string(
            state,
            precision=10,
            separator=", ",
        )
    )

    for dt_qr in QR_INTERVALS:

        started = perf_counter()

        raw = lyapunov_spectrum(
            state,
            dt_qr,
            TOTAL_TIME,
        )

        ordered = sort_descending(raw)

        runtime = perf_counter() - started

        results.append(
            (
                k,
                theta,
                dt_qr,
                ordered,
                runtime,
            )
        )

        completed += 1

        print(
            f"T={TOTAL_TIME:6.1f} "
            f"dt={dt_qr:.3f} "
            f"l1={ordered[0]: .8f} "
            f"l2={ordered[1]: .8f} "
            f"l3={ordered[2]: .8f} "
            f"l4={ordered[3]: .8f} "
            f"runtime={runtime:7.2f}s"
        )

matrix_complete = completed == 33


# ============================================================
# 320-SECOND CLASSIFICATION
# ============================================================

print()
print("=" * 72)
print("320-SECOND CLASSIFICATION")
print("=" * 72)
print(
    "Labels are operational finite-time "
    "classifications, not proofs."
)
print(
    "No transition angle is estimated."
)

classified = {}

for k, theta, state in STATES:

    runs = [
        row for row in results
        if row[0] == k
    ]

    adapted = [
        (row[0], row[1], row[2], row[3], row[4])
        for row in runs
    ]

    info = classify_320(adapted)
    classified[k] = info

    print()
    print(
        f"k={k:02d}  theta={theta:.10f}"
    )
    print(
        "runs = "
        + ", ".join(
            f"{v:.8f}" for v in info["values"]
        )
    )
    print(f"mean = {info['mean']:.8f}")
    print(f"std = {info['std']:.8f}")
    print(
        f"relative std = "
        f"{100.0 * info['relative']:.3f}%"
    )
    print(f"label = {info['label']}")


# ============================================================
# ANCHOR CHECKS
# Classification only. Not magnitude agreement.
# ============================================================

anchor_low = classified[0]["label"] == "REGULAR"
anchor_high = classified[10]["label"] == "CHAOTIC"

print()
print("=" * 72)
print("ANCHOR CHECKS")
print("=" * 72)
print(f"Completed runs = {completed} / 33")
print(f"Matrix complete = {matrix_complete}")
print(
    "theta_0 REGULAR = "
    f"{anchor_low}"
)
print(
    "theta_10 CHAOTIC = "
    f"{anchor_high}"
)
print(
    f"Worst Jacobian error = {worst_error:.3e}"
)
print("Interior influence on verdict = none")


# ============================================================
# LABEL SEQUENCE
# Reported only. Does not affect the verdict.
# ============================================================

print()
print("=" * 72)
print("LABEL SEQUENCE")
print("Reported only. No interpolation.")
print("=" * 72)

sequence = [
    classified[k]["label"]
    for k in range(11)
]

print(" ".join(sequence))


# ============================================================
# VERDICT
# ============================================================

if not jacobian_passed:

    verdict = (
        "BLOCKED - Jacobian validation failed"
    )

elif not matrix_complete:

    verdict = (
        "FAIL - frozen 33-run matrix "
        "did not complete"
    )

elif anchor_low and anchor_high:

    verdict = (
        "PASS - frozen transect completed "
        "and both classification anchors held"
    )

else:

    verdict = (
        "INCONCLUSIVE - matrix completed, "
        "but one or both endpoint "
        "classifications did not hold"
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
    "A PASS means this procedure completed "
    "one frozen line and reproduced the "
    "two anchor labels."
)
print(
    "It does not locate a universal boundary."
)
print(
    "REGULAR means the finite-time largest "
    "exponent met the preregistered regular "
    "thresholds. It is not a proof that the "
    "trajectory is nonchaotic."
)
print(
    "Interior labels, including multiple "
    "changes and INDETERMINATE, are results "
    "under this procedure. They are not "
    "statements about surrounding regions."
)
print(
    "This is not a physical experiment."
)
print()
print("=" * 72)
