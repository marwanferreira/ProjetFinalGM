"""
Pendule simple avec longueur variable : l(t) = l0 + α·sin(ν·t)
--------------------------------------------------------------
Ce programme simule un pendule dont la longueur varie légèrement
au cours du temps. Cette variation influence la période du pendule
et introduit des modulations dans son comportement oscillatoire.

La méthode utilisée est l'intégration d'Euler explicite.

"""


import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques ---
g = 9.81
l0 = 1.0
alpha = 0.1
nu = 2 * np.pi

# --- Simulation ---
N = 10000
t0, tf = 0.0, 20.0
dt = (tf - t0) / N
t = np.linspace(t0, tf, N)

np.random.seed(2025)
theta0 = np.pi * (2 * np.random.rand() - 1)
omega0 = np.random.rand()

theta = np.zeros(N)
omega = np.zeros(N)
theta[0] = theta0
omega[0] = omega0

for i in range(N - 1):
    l_t = l0 + alpha * np.sin(nu * t[i])
    theta[i+1] = theta[i] + dt * omega[i]
    omega[i+1] = omega[i] - dt * (g / l_t) * np.sin(theta[i])

# --- Affichage amélioré ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(t, np.degrees(theta), color='teal', label="θ(t) [°]")
axs[0].set_ylabel("Angle θ(t) [°]")
axs[0].set_title("Évolution de l’angle du pendule")

axs[1].plot(t, omega, color='crimson', linestyle='--', label="ω(t) [rad/s]")
axs[1].set_ylabel("Vitesse ω(t)")
axs[1].set_xlabel("Temps [s]")
axs[1].set_title("Évolution de la vitesse angulaire")

for ax in axs:
    ax.legend()
    ax.grid(True)

plt.suptitle("Pendule avec longueur variable : l(t) = l0 + α·sin(ν·t)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
