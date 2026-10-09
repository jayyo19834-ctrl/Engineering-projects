import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Parameters ---
m = 1.0
k = 4.0
c = 0.5
F0 = 1.0
omega_d = 1.5
x0 = 0.0
v0 = 0.0

# --- Driven damped oscillator ---
def driven(t, y, w):
    x, v = y
    force = F0 * np.cos(w * t)
    acceleration = (force - c*v - k*x) / m
    return [v, acceleration]

# --- Main simulation ---
t_eval = np.linspace(0, 40, 4000)

sol = solve_ivp(
    lambda t, y: driven(t, y, omega_d),
    (0, 40),
    [x0, v0],
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

if not sol.success:
    raise RuntimeError(sol.message)

# --- Transient vs steady state ---
tau = 2*m/c
transient_end = 5*tau

idx = sol.t > transient_end
ss_amp = np.max(np.abs(sol.y[0, idx]))

# --- Analytical steady-state amplitude ---
omega_n = np.sqrt(k/m)

def theoretical_amplitude(w):
    denominator = np.sqrt(
        (k - m*w**2)**2 + (c*w)**2
    )
    return F0 / denominator

A_theory = theoretical_amplitude(omega_d)

print(f"Damping time constant: {tau:.2f} s")
print(f"Numerical amplitude: {ss_amp:.4f} m")
print(f"Analytical amplitude: {A_theory:.4f} m")
print(f"Ratio: {ss_amp/A_theory:.4f}")

# --- Frequency sweep ---
omegas = np.linspace(0.5, 3.5, 61)
amps = []

for w in omegas:
    s = solve_ivp(
        lambda t, y: driven(t, y, w),
        (0, 60),
        [0, 0],
        t_eval=np.linspace(0, 60, 3000),
        rtol=1e-9,
        atol=1e-11
    )

    if not s.success:
        raise RuntimeError(s.message)

    mask = s.t > 5*tau
    amps.append(np.max(np.abs(s.y[0, mask])))

amps = np.array(amps)

peak_idx = np.argmax(amps)
peak_w = omegas[peak_idx]
peak_amp = amps[peak_idx]

omega_peak_theory = np.sqrt(
    k/m - c**2/(2*m**2)
)

print(f"Numerical resonance: {peak_w:.3f} rad/s")
print(f"Theoretical resonance: {omega_peak_theory:.3f} rad/s")
print(f"Peak amplitude: {peak_amp:.4f} m")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(9, 7))

axes[0].plot(
    sol.t, sol.y[0],
    label='Displacement'
)
axes[0].axvspan(
    0, transient_end,
    alpha=0.15,
    color='red',
    label='Transient region'
)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Displacement (m)')
axes[0].set_title('Driven Damped Oscillator')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    omegas, amps,
    label='Numerical frequency sweep'
)
axes[1].plot(
    omegas,
    theoretical_amplitude(omegas),
    '--',
    label='Analytical response'
)
axes[1].axvline(
    omega_peak_theory,
    color='red',
    linestyle=':',
    label='Theoretical resonance'
)
axes[1].set_xlabel('Driving frequency (rad/s)')
axes[1].set_ylabel('Amplitude (m)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('driven_oscillator.png', dpi=120)
plt.close(fig)

print("Plot saved as driven_oscillator.png")
