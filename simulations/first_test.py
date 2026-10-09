
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical parameters
m = 1.0
k = 4.0
c = 0.5
x0 = 1.0
v0 = 0.0

# Equation of motion
def oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -(c*v + k*x) / m
    return [dxdt, dvdt]

# Numerical integration
sol = solve_ivp(
    oscillator,
    [0, 20],
    [x0, v0],
    t_eval=np.linspace(0, 20, 2000),
    rtol=1e-9,
    atol=1e-11
)

if not sol.success:
    raise RuntimeError(sol.message)

# Mechanical energy
E = 0.5*m*sol.y[1]**2 + 0.5*k*sol.y[0]**2

print(f"Initial energy: {E[0]:.4f} J")
print(f"Energy at t=20: {E[-1]:.4f} J")
print(f"Energy decay: {(1 - E[-1]/E[0])*100:.1f}%")

# Plot displacement and velocity
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sol.t, sol.y[0], label='Displacement x(t)')
ax.plot(sol.t, sol.y[1], label='Velocity v(t)', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m), Velocity (m/s)')
ax.set_title(f'Damped Harmonic Oscillator (m={m}, k={k}, c={c})')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('oscillator.png', dpi=120)
plt.close(fig)

print("Plot saved as oscillator.png")

# Frequency calculations
omega_n = np.sqrt(k/m)
omega_d = np.sqrt(k/m - (c/(2*m))**2)

print(f"Natural frequency: {omega_n:.3f} rad/s")
print(f"Damped frequency: {omega_d:.3f} rad/s")
print(f"Period: {2*np.pi/omega_d:.3f} s")
