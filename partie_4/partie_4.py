##Partie 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Paramètres physiques ---
g = 9.81         # Gravité (m/s²)
l = 1.0          # Longueur du pendule (m)
kf = 0.5         # Coefficient de frottement

# --- Paramètres de simulation ---
N = 200                      # Nombre de points
t_start, t_end = 0.0, 10.0    # Intervalle de temps [s]
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta_init = 1.0             # Angle initial (rad)
omega_init = 0.0             # Vitesse angulaire initiale (rad/s)

# --- Simulation avec frottement ---
def simulate_pendulum_friction(h):
    theta = np.zeros(N + 1)
    omega = np.zeros(N + 1)
    theta[0] = theta_init
    omega[0] = omega_init
    for i in range(N):
        theta[i+1] = theta[i] + dt * omega[i]
        omega[i+1] = omega[i] + dt * ( - (g / l) * np.sin(theta[i]) - kf * omega[i] + h[i] )
    return theta, omega

# --- Fonction coût pénalisée ---
def penalty_cost_friction(h, mu=1e6):
    theta, omega = simulate_pendulum_friction(h)
    energy = dt * np.sum(h**2)
    dyn_err_theta = np.sum((theta[1:] - theta[:-1] - dt * omega[:-1])**2)
    dyn_err_omega = np.sum((omega[1:] - omega[:-1] - dt * ( - (g / l) * np.sin(theta[:-1]) - kf * omega[:-1] + h ))**2)
    final_err = theta[-1]**2 + omega[-1]**2
    return energy + mu * (dyn_err_theta + dyn_err_omega + final_err)

# --- Optimisation ---
h0 = np.zeros(N)
result = minimize(penalty_cost_friction, h0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': True})
h_opt = result.x
theta_opt, omega_opt = simulate_pendulum_friction(h_opt)

# --- Affichage des résultats ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axes[0].plot(t_grid, np.degrees(theta_opt), linestyle='--', color='navy')
axes[0].set_ylabel("Angle θ(t) [°]")
axes[0].set_title("Évolution de l’angle du pendule")

axes[1].plot(t_grid, omega_opt, linestyle='-.', color='darkred')
axes[1].set_ylabel("Vitesse ω(t) [rad/s]")
axes[1].set_title("Évolution de la vitesse angulaire")

axes[2].plot(t_grid[:-1], h_opt, linestyle='-', color='green')
axes[2].set_ylabel("Contrôle h(t)")
axes[2].set_xlabel("Temps [s]")
axes[2].set_title("Évolution du contrôle optimal")

plt.suptitle("Optimisation par pénalisation avec frottement (Euler explicite)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Logs de l’optimisation ---
print(f"Optimisation réussie : {result.success}")
print(f"Message : {result.message}")
print(f"Valeur finale de la fonction coût : {result.fun}")