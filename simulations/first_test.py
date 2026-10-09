import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

m = 1.0
k = 4.0
c = 0.5
x0 = 1.0
v0 = 0.0

def oscillator(t, y):
    x, v = y
    return [-k/m*x - c/m*v, v 0, 20], , t_eval=np.linspace(0, 20, 2000))

E = 0.5*m*sol.y[1 0]**2
print(f"Initial energy: {E[0 -1]:.4f} J")
print(f"Energy decay: {(1 - E[-1 0])*100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sol.t, sol.y[0 1], label='velocity v(t)', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('x (m), v (m/s)')
ax.set_title(f'Damped Harmonic Oscillator (m={m}, k={k}, c={c})')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('oscillator.png', dpi=120)
print("Plot saved as oscillator.png")

omega = np.sqrt(k/m - (c/(2*m))**2)
print(f"Natural frequency: {np.sqrt(k/m):.3f} rad/s")
print(f"Damped frequency: {omega:.3f} rad/s")
print(f"Period: {2*np.pi/omega:.3f} s")
