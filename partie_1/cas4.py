## Cas 4: On ajuste une fonction h non nulle

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
N = 1000
g = 9.81
l = 1.0
T = 10
dt = T / N
t = np.linspace(0, T, N+1)

# Initialisation
theta = np.zeros(N+1)
omega = np.zeros(N+1)

# Conditions initiales
theta[0] = 1.0
omega[0] = 0.0

# Fonction de contrôle h(t, theta, omega)
def h(t, theta, omega):
    return -2 * theta - 1.5 * omega  # contrôle PD

# Méthode d’Euler explicite
for i in range(N):
    theta[i+1] = theta[i] + dt * omega[i]
    omega[i+1] = omega[i] + dt * ( - (g / l) * np.sin(theta[i]) + h(t[i], theta[i], omega[i]))

# --- Ajout esthétique ---
plt.style.use('seaborn-v0_8-darkgrid')          # style sobre et pro
plt.figure(figsize=(10, 5))                     # taille graphique

plt.plot(t, theta, linestyle='--', color='midnightblue', label="Angle θ(t) [rad]")    # angle en pointillé
plt.plot(t, omega, linestyle='-.', color='darkred', label="Vitesse ω(t) [rad/s]")  # vitesse en tiret-point

plt.xlabel("Temps (s)")
plt.ylabel("Valeurs physiques")
plt.title("Pendule avec contrôle h(t) (Euler explicite)", fontsize=13)
plt.legend()
plt.tight_layout()                               # alignement propre
plt.show()
