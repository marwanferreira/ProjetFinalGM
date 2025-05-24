"""
Simulation du pendule simple sans contrôle
------------------------------------------
Résolution de l'équation du pendule : θ̈ + (g/l)·sin(θ) = 0
Méthode numérique : Euler explicite
Cas étudié : sans frottement ni force de contrôle

"""

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques du système ---
g = 9.81         # Accélération gravitationnelle (m/s²)
l = 1.0          # Longueur du pendule (m)

# --- Paramètres de simulation ---
N = 10000                     # Nombre de points de discrétisation
t_start, t_end = 0.0, 10.0    # Intervalle de temps [s]
dt = (t_end - t_start) / N    # Pas de temps
t_grid = np.linspace(t_start, t_end, N)

# --- Conditions initiales aléatoires mais fixées pour reproductibilité ---
np.random.seed(42)
theta_init = np.pi * (2 * np.random.rand() - 1)  # Angle initial entre -π et π
omega_init = np.random.rand()                    # Vitesse angulaire initiale

# --- Initialisation des tableaux ---
theta = np.zeros(N)
omega = np.zeros(N)
theta[0] = theta_init
omega[0] = omega_init

# --- Intégration par la méthode d'Euler explicite ---
for i in range(N - 1):
    theta[i + 1] = theta[i] + dt * omega[i]
    omega[i + 1] = omega[i] - dt * (g / l) * np.sin(theta[i])  # Pas de h(t)

# --- Affichage graphique (esthétique académique) ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Courbe de l'angle θ(t)
axes[0].plot(t_grid, np.degrees(theta), linestyle='--', color='darkblue')
axes[0].set_ylabel("Angle θ(t) [°]")
axes[0].set_title("Évolution de l’angle du pendule")

# Courbe de la vitesse angulaire ω(t)
axes[1].plot(t_grid, omega, linestyle='-.', color='darkred')
axes[1].set_ylabel("Vitesse ω(t) [rad/s]")
axes[1].set_xlabel("Temps [s]")
axes[1].set_title("Évolution de la vitesse angulaire")

plt.suptitle("Simulation du pendule simple sans contrôle (Euler explicite)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
