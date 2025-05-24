"""
Optimisation du contrôle h(t) pour ramener un pendule à l’équilibre
-------------------------------------------------------------------
Formulation : Lagrangien + méthode d’Euler explicite
Objectif : atteindre θ(tf) = 0 et ω(tf) = 0 en minimisant l'énergie J(h) = ∫ h(t)² dt


"""

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques du système ---
g = 9.81         # Accélération gravitationnelle (m/s²)
l = 1.0          # Longueur du pendule (m)

# --- Paramètres de simulation ---
N = 10000
t_start, t_end = 0.0, 10.0
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta_init = 1.0   # Angle initial en radian
omega_init = 0.0   # Vitesse angulaire initiale

# --- Initialisation du contrôle h(t) ---
h = np.zeros(N)

# --- Hyperparamètres de l’optimisation ---
n_iterations = 100
eta = 5e-5   # pas d’apprentissage

# --- Boucle d'optimisation ---
for _ in range(n_iterations):

    # ----- Forward pass (simulation directe) -----
    theta = np.zeros(N + 1)
    omega = np.zeros(N + 1)
    theta[0] = theta_init
    omega[0] = omega_init

    for i in range(N):
        theta[i + 1] = theta[i] + dt * omega[i]
        omega[i + 1] = omega[i] + dt * (- (g / l) * np.sin(theta[i]) + h[i])

    # ----- Backward pass (multiplicateurs de Lagrange) -----
    lambda_theta = np.zeros(N + 1)
    lambda_omega = np.zeros(N + 1)

    for i in reversed(range(N)):
        lambda_theta[i] = lambda_theta[i + 1] + dt * lambda_omega[i + 1] * (g / l) * np.cos(theta[i])
        lambda_omega[i] = lambda_omega[i + 1] + dt * (lambda_theta[i + 1] - omega[i])

    # ----- Mise à jour stabilisée de h(t) -----
    grad = h - 0.5 * lambda_omega[:-1]
    h = h - eta * (grad + 1e-4 * h)
    h = np.clip(h, -2.0, 2.0)  # limite physique du contrôle

# --- Affichage graphique (même logique que Partie 1) ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

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

plt.suptitle("Optimisation du contrôle pour ramener le pendule à l’équilibre", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
