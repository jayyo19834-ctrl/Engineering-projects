# ============================================================
# GATE 6 — FULL LYAPUNOV SPECTRUM / CONSERVATIVE STRUCTURE
# ============================================================
#
# STATUS
#
# This file implements the frozen Gate 6 specification.
# Acceptance thresholds are fixed before execution.
# Do not change them after implementation or after results.
#
# If this implementation is later found defective, preserve
# it as invalid and write a new revision against the same
# Gate 6 specification.
#
# SAME PHYSICS AS GATE 5B
#   G = 9.81, M1 = M2 = 1, L1 = L2 = 1
#   initial state [0.9*pi, 0, 0.9*pi, 0]
#   DOP853, rtol = 1e-10, atol = 1e-12
#
# SPECTRUM PROCEDURE
#   Tangent matrix starts as the 4x4 identity.
#   Propagation is the variational equation dPhi/dt = J Phi.
#   Reorthonormalization is QR at the frozen intervals.
#   All four exponents are computed first, then sorted
#   descending. Criteria use that sorted order.
#
# CHECKS 2-4
#   A check passes only if every required inequality
#   passes on every 320-s run. A failed structural run
#   is not averaged away.
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
    40.0,
    80.0,
    160.0,
    320.0,
]

QR_INTERVALS = [
    0.025,
    0.050,
    0.100,
]

COMPLEX_STEP = 1e-20

INITIAL_STATE = np.array(
    [
        0.9 * np.pi,
        0.0,
        0.9 * np.pi,
        0.0,
    ],
    dtype=float,
)


# ============================================================
# FROZEN GATE 6 CRITERIA
# ============================================================

JACOBIAN_MAX_ABS_ERROR = 1e-10

MIN_LARGEST_EXPONENT = 0.50

MAX_VOLUME_RATIO = 0.05

MAX_EXTREME_PAIR_RATIO = 0.10

MAX_CENTRAL_ABS = 0.10

GATE5B_REFERENCE = 1.11193635

MAX_GATE5B_RELATIVE_DIFFERENCE = 0.25

MAX_RELATIVE_STD = 0.10


# ============================================================
# DOUBLE-PENDULUM VECTOR FIELD
# Same field as frozen Gate 5B.
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


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 72)
print(
    "GATE 6 — FULL LYAPUNOV SPECTRUM"
)
print("=" * 72)
print()
print(
    "Frozen Gate 5B reference = "
    f"{GATE5B_REFERENCE:.8f} 1/s"
)
print(
    "Tangent basis = 4x4 identity"
)
print(
    "Reorthonormalization = QR"
)
print(
    "Sorting = descending after each run"
)
print(
    "Standard deviation = population"
)


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
        "GATE 6 BLOCKED — "
        "JACOBIAN VALIDATION FAILED"
    )
    print("=" * 72)
    print(
        "No spectrum run was executed."
    )
    raise SystemExit(0)


# ============================================================
# FROZEN 12-RUN MATRIX
# ============================================================

print()
print("=" * 72)
print("SPECTRUM MATRIX")
print("=" * 72)

results = []

for total_time in TOTAL_TIMES:

    print()
    print(
        f"TOTAL TIME = {total_time:.0f} s"
    )

    for dt_qr in QR_INTERVALS:

        started = perf_counter()

        raw = lyapunov_spectrum(
            INITIAL_STATE,
            dt_qr,
            total_time,
        )

        ordered = sort_descending(raw)

        runtime = (
            perf_counter() - started
        )

        results.append(
            (
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
# SUMMARIES
# ============================================================

print()
print("=" * 72)
print("CONVERGENCE SUMMARY")
print("=" * 72)

for total_time in TOTAL_TIMES:

    group = [
        row[2][0]
        for row in results
        if row[0] == total_time
    ]

    values = np.array(group)

    mean = np.mean(values)
    std = np.std(values, ddof=0)
    relative = (
        std / abs(mean)
        if mean != 0.0
        else np.inf
    )

    print()
    print(
        f"T = {total_time:.0f} s"
    )
    print(
        f"Mean lambda1       = {mean:.8f}"
    )
    print(
        f"Standard deviation = {std:.8f}"
    )
    print(
        f"Minimum lambda1    = {np.min(values):.8f}"
    )
    print(
        f"Maximum lambda1    = {np.max(values):.8f}"
    )
    print(
        f"Relative std       = {100.0 * relative:.3f}%"
    )


# ============================================================
# 320-SECOND DIAGNOSTIC
# Checks 2-4 are per run. No averaging-away.
# ============================================================

long_runs = [
    row
    for row in results
    if row[0] == 320.0
]

lambda1 = np.array(
    [row[2][0] for row in long_runs]
)

mean_320 = np.mean(lambda1)
std_320 = np.std(lambda1, ddof=0)
relative_std_320 = (
    std_320 / abs(mean_320)
    if mean_320 != 0.0
    else np.inf
)

reference_difference = (
    abs(mean_320 - GATE5B_REFERENCE)
    / abs(GATE5B_REFERENCE)
)

print()
print("=" * 72)
print("320-SECOND DIAGNOSTIC")
print("=" * 72)
print(
    f"Mean lambda1 = {mean_320:.8f} 1/s"
)
print(
    f"Std = {std_320:.8f}"
)
print(
    f"Relative std = {100.0 * relative_std_320:.3f}%"
)
print(
    "Gate 5B reference = "
    f"{GATE5B_REFERENCE:.8f} 1/s"
)
print(
    "Difference from Gate 5B = "
    f"{100.0 * reference_difference:.3f}%"
)
print(
    f"Worst Jacobian error = {worst_error:.3e}"
)

print()
print(
    "Per-run structural checks"
)
print(
    "A failed run is not averaged away."
)

volume_ok = True
extreme_ok = True
central_ok = True

for total_time, dt_qr, ordered, runtime in long_runs:

    volume_ratio = (
        abs(np.sum(ordered))
        / abs(ordered[0])
    )

    extreme_ratio = (
        abs(ordered[0] + ordered[3])
        / abs(ordered[0])
    )

    this_volume = (
        volume_ratio <= MAX_VOLUME_RATIO
    )
    this_extreme = (
        extreme_ratio <= MAX_EXTREME_PAIR_RATIO
    )
    this_central = (
        abs(ordered[1]) <= MAX_CENTRAL_ABS
        and abs(ordered[2]) <= MAX_CENTRAL_ABS
    )

    volume_ok = volume_ok and this_volume
    extreme_ok = extreme_ok and this_extreme
    central_ok = central_ok and this_central

    print(
        f"dt={dt_qr:.3f} "
        f"sum={np.sum(ordered): .8f} "
        f"volume={100.0 * volume_ratio:.3f}% "
        f"pair={100.0 * extreme_ratio:.3f}% "
        f"l2={ordered[1]: .8f} "
        f"l3={ordered[2]: .8f} "
        f"volume_ok={this_volume} "
        f"pair_ok={this_extreme} "
        f"central_ok={this_central}"
    )


# ============================================================
# SEVEN FROZEN CHECKS
# ============================================================

check_1 = (
    mean_320 > MIN_LARGEST_EXPONENT
)
check_2 = volume_ok
check_3 = extreme_ok
check_4 = central_ok
check_5 = (
    reference_difference
    <= MAX_GATE5B_RELATIVE_DIFFERENCE
)
check_6 = (
    relative_std_320
    <= MAX_RELATIVE_STD
)
check_7 = jacobian_passed

print()
print("=" * 72)
print("CRITERION CHECKS")
print("=" * 72)
print(
    f"1 Positive largest exponent = {check_1}"
)
print(
    f"2 Volume conservation, every 320-s run = {check_2}"
)
print(
    f"3 Extreme-pair symmetry, every 320-s run = {check_3}"
)
print(
    f"4 Central pair near zero, every 320-s run = {check_4}"
)
print(
    f"5 Gate 5B agreement = {check_5}"
)
print(
    f"6 Renormalization robustness = {check_6}"
)
print(
    f"7 Jacobian pre-gate = {check_7}"
)


# ============================================================
# VERDICT
# ============================================================

if not check_7:

    verdict = (
        "BLOCKED - Jacobian validation failed"
    )

elif not check_1:

    verdict = (
        "FAIL - required positive largest "
        "exponent not recovered"
    )

elif (
    check_2
    and check_3
    and check_4
    and check_5
    and check_6
):

    verdict = (
        "PASS - Gate 6 spectrum and "
        "conservative-structure criteria satisfied"
    )

else:

    verdict = (
        "INCONCLUSIVE - positive largest exponent "
        "recovered, but one or more "
        "conservative-structure or convergence "
        "criteria were not satisfied"
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
    "A PASS is a finite-time numerical cross-check "
    "of a positive largest exponent and the expected "
    "conservative pairing for this trajectory."
)
print(
    "It is not a mathematical proof of chaos, "
    "not a claim about every trajectory, "
    "and not a physical experiment."
)
print()
print("=" * 72)
