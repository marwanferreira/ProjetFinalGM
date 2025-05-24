"""
Optimisation par pénalisation : grandes oscillations non linéaires
-------------------------------------------------------------------
Méthode : minimisation de l’énergie de contrôle sous contraintes dynamiques
Forme du coût :
    J(h) = ∫ h(t)² dt + μ · (erreurs dynamiques + erreur finale)

Les équations dynamiques sont conservées non-linéaires (sin(θ) gardé).


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Paramètres physiques ---
g = 9.81
l = 1.0

# --- Paramètres de simulation ---
N = 2000
t_start, t_end = 0.0, 10.0
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta0 = 1.0
omega0 = 0.0

# --- Simulation complète du pendule (non-linéaire) ---
def simulate_pendulum(h):
    theta = np.zeros(N + 1)
    omega = np.zeros(N + 1)
    theta[0] = theta0
    omega[0] = omega0
    for i in range(N):
        theta[i + 1] = theta[i] + dt * omega[i]
        omega[i + 1] = omega[i] + dt * (- (g / l) * np.sin(theta[i]) + h[i])
    return theta, omega

# --- Fonction coût pénalisée ---
def penalty_cost(h, mu=1e6):
    theta, omega = simulate_pendulum(h)
    energy = dt * np.sum(h**2)
    dyn_err_theta = np.sum((theta[1:] - theta[:-1] - dt * omega[:-1])**2)
    dyn_err_omega = np.sum((omega[1:] - omega[:-1] - dt * (-(g/l) * np.sin(theta[:-1]) + h))**2)
    final_err = theta[-1]**2 + omega[-1]**2
    return energy + mu * (dyn_err_theta + dyn_err_omega + final_err)

# --- Optimisation numérique (L-BFGS-B) ---
h0 = np.zeros(N)
result = minimize(penalty_cost, h0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': True})
h_opt = result.x
theta_opt, omega_opt = simulate_pendulum(h_opt)

# --- Affichage graphique ---
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

plt.suptitle("Optimisation par pénalisation (non-linéaire, grandes oscillations)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Résumé ---
print(f"Optimisation réussie : {result.success}")
print(f"Message : {result.message}")
print(f"Valeur finale de la fonction coût : {result.fun:.4f}")
