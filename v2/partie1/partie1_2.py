"""
Pendule simple avec force non linéaire supplémentaire : sin(2θ)
---------------------------------------------------------------
Ce programme simule le mouvement d’un pendule simple soumis à deux forces :
- La force gravitationnelle classique : - (g / l) · sin(θ)
- Une force supplémentaire non linéaire : - (kr / l) · sin(2θ)

Ce type de configuration permet d'explorer une dynamique plus riche que celle
du pendule simple. Le terme sin(2θ) modifie la symétrie du système et peut
entraîner des effets complexes dans le mouvement.

L’intégration est réalisée par la méthode d’Euler explicite sur un intervalle de 10 secondes.


"""

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques ---
g = 9.81        # gravité (m/s²)
l = 1.0         # longueur du pendule (m)
kr = 2.0        # intensité de la force sin(2θ)

# --- Paramètres de simulation ---
N = 10000
t0, tf = 0.0, 10.0
dt = (tf - t0) / N
t = np.linspace(t0, tf, N)

# --- Conditions initiales ---
np.random.seed(123)
theta0 = np.pi * (2 * np.random.rand() - 1)  # angle initial entre -π et π
omega0 = np.random.rand()                    # vitesse initiale

theta = np.zeros(N)
omega = np.zeros(N)
theta[0] = theta0
omega[0] = omega0

# --- Simulation par Euler explicite ---
for i in range(N - 1):
    theta[i+1] = theta[i] + dt * omega[i]
    omega[i+1] = omega[i] + dt * (
        - (g / l) * np.sin(theta[i]) - (kr / l) * np.sin(2 * theta[i])
    )

# --- Affichage graphique ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, np.degrees(theta), label="θ(t) [°]", color='darkblue', linestyle='-')
ax.plot(t, omega, label="ω(t) [rad/s]", color='crimson', linestyle='--')
ax.set_title("Pendule simple avec force sin(2θ) (Euler explicite)")
ax.set_xlabel("Temps [s]")
ax.set_ylabel("Valeurs")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
