import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Two linked pendulums. Angles are measured from hanging straight down.
g = 9.81
m1, m2 = 1.0, 1.0
L1, L2 = 1.0, 1.0

def deriv(t, y):
    th1, w1, th2, w2 = y
    delta = th1 - th2
    den = 2*m1 + m2 - m2*np.cos(2*th1 - 2*th2)
    dw1 = (
        -g*(2*m1 + m2)*np.sin(th1)
        - m2*g*np.sin(th1 - 2*th2)
        - 2*np.sin(delta)*m2*(w2**2*L2 + w1**2*L1*np.cos(delta))
    ) / (L1 * den)
    dw2 = (
        2*np.sin(delta)*(
            w1**2*L1*(m1 + m2)
            + g*(m1 + m2)*np.cos(th1)
            + w2**2*L2*m2*np.cos(delta)
        )
    ) / (L2 * den)
    return [w1, dw1, w2, dw2]

def energy(y):
    th1, w1, th2, w2 = y
    V = -(m1 + m2)*g*L1*np.cos(th1) - m2*g*L2*np.cos(th2)
    T = 0.5*m1*(L1*w1)**2 + 0.5*m2*(
        (L1*w1)**2 + (L2*w2)**2 + 2*L1*L2*w1*w2*np.cos(th1 - th2)
    )
    return T + V

# Nearly upright. A second run starts 0.0001 rad off, to show chaos.
y0 = [0.9*np.pi, 0.0, 0.9*np.pi, 0.0]
y0_b = [0.9*np.pi + 1e-4, 0.0, 0.9*np.pi, 0.0]
t = np.linspace(0, 10, 4000)

sol = solve_ivp(deriv, [0, 10], y0, t_eval=t, rtol=1e-9, atol=1e-11)
sol_b = solve_ivp(deriv, [0, 10], y0_b, t_eval=t, rtol=1e-9, atol=1e-11)
if not sol.success or not sol_b.success:
    raise RuntimeError("integration failed")

E = np.array([energy(sol.y[:, i]) for i in range(sol.y.shape[1])])
sep = np.abs(sol.y[0] - sol_b.y[0])

print(f"Initial energy: {E[0]:.4f} J")
print(f"Energy at t=10: {E[-1]:.4f} J")
print(f"Energy drift: {(E.max() - E.min()):.6f} J")
print(f"Angle separation at t=0: {sep[0]:.2e} rad")
print(f"Angle separation at t=10: {sep[-1]:.4f} rad")

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(sol.t, sol.y[0], label='theta1')
axes[0].plot(sol.t, sol.y[2], label='theta2', alpha=0.8)
axes[0].set_ylabel('Angle (rad)')
axes[0].set_title('Double pendulum')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(sol.t, np.maximum(sep, 1e-16), label='|theta1 - theta1_nearby|')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Separation (rad)')
axes[1].set_title('Two starts, 0.0001 rad apart')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('double_pendulum.png', dpi=120)
plt.close(fig)
print("Plot saved as double_pendulum.png")
