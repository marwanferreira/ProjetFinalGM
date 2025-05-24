import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

g = 9.81
l = 1.0
T = 1.0
N = 200
dt = T / N

theta0 = 1.0
omega0 = 0.0

def simulate_pendulum(h):
    theta = np.zeros(N+1)
    omega = np.zeros(N+1)
    theta[0] = theta0
    omega[0] = omega0

    for i in range(N):
        theta[i+1] = theta[i] + dt * omega[i]
        omega[i+1] = omega[i] + dt * ( - (g/l) * np.sin(theta[i]) + h[i] )

    return theta, omega

def penalty_cost(h, mu=1e6):
    theta, omega = simulate_pendulum(h)

    energy = dt * np.sum(h**2)

    dyn_err_theta = np.sum((theta[1:] - theta[:-1] - dt * omega[:-1])**2)
    dyn_err_omega = np.sum((omega[1:] - omega[:-1] - dt * (-(g/l) * np.sin(theta[:-1]) + h))**2)

    final_err = theta[-1]**2 + omega[-1]**2

    cost = energy + mu * (dyn_err_theta + dyn_err_omega + final_err)
    return cost

h0 = np.zeros(N)

result = minimize(penalty_cost, h0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': True})

h_opt = result.x
theta_opt, omega_opt = simulate_pendulum(h_opt)

t = np.linspace(0, T, N+1)

plt.figure(figsize=(12, 6))
plt.plot(t, theta_opt, label=r'$\theta(t)$ (angle)')
plt.plot(t, omega_opt, label=r'$\omega(t)$ (vitesse angulaire)')
plt.plot(t[:-1], h_opt, label=r'$h(t)$ (contrôle optimal)')
plt.xlabel("Temps (s)")
plt.title("Simulation pendule avec contrôle optimal (pénalisation contraintes)")
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimisation réussie : {result.success}")
print(f"Message : {result.message}")
print(f"Valeur finale de la fonction coût : {result.fun}")
