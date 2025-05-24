"""
Optimisation du contrôle h(t) via gradient adjoint (cas linéarisé)
------------------------------------------------------------------
Hypothèse : petites oscillations ⇒ sin(θ) ≈ θ
Formulation : Lagrangien + méthode d’Euler explicite + gradient à pas fixe

Objectif : atteindre θ(tf) = 0 et ω(tf) = 0 tout en minimisant J(h) = ∫ h(t)² dt

"""

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques du pendule ---
g = 9.81         # gravité (m/s²)
l = 1.0          # longueur du pendule (m)

# --- Paramètres de simulation ---
N = 10000
t_start, t_end = 0.0, 10.0
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta_init = 1.0     # rad
omega_init = 0.0     # rad/s

# --- Initialisation du contrôle ---
h = np.zeros(N)

# --- Hyperparamètres de la descente de gradient ---
eta = 5e-5               # pas d’apprentissage
n_iterations = 100       # nombre d'itérations

for _ in range(n_iterations):

    # --- Simulation directe (forward) ---
    theta = np.zeros(N + 1)
    omega = np.zeros(N + 1)
    theta[0] = theta_init
    omega[0] = omega_init

    for i in range(N):
        theta[i + 1] = theta[i] + dt * omega[i]
        omega[i + 1] = omega[i] + dt * (- (g / l) * theta[i] + h[i])  # linéarisé : sin(theta) ≈ theta

    # --- Équations adjointes (backward) ---
    lambda_theta = np.zeros(N + 1)
    lambda_omega = np.zeros(N + 1)
    lambda_theta[N] = 0.0
    lambda_omega[N] = 0.0

    for i in reversed(range(N)):
        lambda_theta[i] = lambda_theta[i + 1] + dt * lambda_omega[i + 1] * (g / l)  # cos(theta) ≈ 1
        lambda_omega[i] = lambda_omega[i + 1] + dt * (lambda_theta[i + 1] - omega[i])

    # --- Mise à jour du contrôle optimal ---
    grad = h - 0.5 * lambda_omega[:-1]          # ∇J(h)
    h = h - eta * grad                          # descente à pas fixe
    h = np.clip(h, -2.0, 2.0)                   # projection : bornes physiques

# --- Affichage graphique (structure académique) ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Angle θ(t)
axes[0].plot(t_grid, np.degrees(theta), linestyle='--', color='navy')
axes[0].set_ylabel("Angle θ(t) [°]")
axes[0].set_title("Évolution de l’angle du pendule")

# Vitesse ω(t)
axes[1].plot(t_grid, omega, linestyle='-.', color='darkred')
axes[1].set_ylabel("Vitesse ω(t) [rad/s]")
axes[1].set_title("Évolution de la vitesse angulaire")

# Contrôle h(t)
axes[2].plot(t_grid[:-1], h, linestyle='-', color='green')
axes[2].set_ylabel("Contrôle h(t)")
axes[2].set_xlabel("Temps [s]")
axes[2].set_title("Évolution du contrôle optimal")

plt.suptitle("Optimisation du contrôle par gradient (cas linéarisé)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
