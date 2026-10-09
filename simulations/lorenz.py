import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Classic chaotic attractor. sigma, rho, beta are the standard values.
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz(t, state):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]

# Two starts, a millionth apart in x.
y0 = [1.0, 1.0, 1.0]
y0_b = [1.0 + 1e-6, 1.0, 1.0]
t = np.linspace(0, 30, 6000)

sol = solve_ivp(lorenz, [0, 30], y0, t_eval=t, rtol=1e-9, atol=1e-11)
sol_b = solve_ivp(lorenz, [0, 30], y0_b, t_eval=t, rtol=1e-9, atol=1e-11)
if not sol.success or not sol_b.success:
    raise RuntimeError("integration failed")

sep = np.linalg.norm(sol.y - sol_b.y, axis=0)

print(f"x range: {sol.y[0].min():.2f} to {sol.y[0].max():.2f}")
print(f"z range: {sol.y[2].min():.2f} to {sol.y[2].max():.2f}")
print(f"Separation at t=0: {sep[0]:.2e}")
print(f"Separation at t=20: {sep[np.argmin(np.abs(t - 20))]:.4f}")
print(f"Separation at t=30: {sep[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(sol.y[0], sol.y[2], lw=0.4)
axes[0].set_xlabel('x')
axes[0].set_ylabel('z')
axes[0].set_title('Lorenz attractor')
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(sol.t, np.maximum(sep, 1e-16))
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Separation')
axes[1].set_title('Two starts, 1e-6 apart')
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('lorenz.png', dpi=120)
plt.close(fig)
print("Plot saved as lorenz.png")
