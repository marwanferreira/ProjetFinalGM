"""
pendule_test_gradient_v1.py

Ce script correspond à la partie 1 du projet GMF - Optimisation du retour à l’équilibre.

- Objectif :
    Simuler le comportement d’un pendule simple soumis à une commande non nulle fixée h(t),
    sans chercher à optimiser cette commande.

- Cette étape permet de :
    - Valider le modèle dynamique du pendule,
    - Implémenter correctement la méthode d’Euler explicite,
    - Tester l’effet d’une commande arbitraire (ici, cosinus) sur la trajectoire.

- Ce n’est pas encore une commande optimale, il s’agit simplement d’une exploration du système avant l’optimisation (Partie 2).
"""

import numpy as np
import matplotlib.pyplot as plt

# === Paramètres physiques ===
g = 9.81     # gravité (m/s²)
l = 1.0      # longueur du pendule (m)
T = 1.0      # durée totale (s)
N = 200      # nombre de pas
dt = T / N   # pas de temps
time = np.linspace(0, T, N+1)

# === Conditions initiales ===
theta0 = 1.0         # angle initial (rad)
theta_dot0 = 0.0     # vitesse initiale (rad/s)

# === Commande imposée (non optimisée) ===
def h(t):
    return 0.1 * np.cos(2 * np.pi * t)

# === Initialisation des états ===
theta = np.zeros(N+1)
theta_dot = np.zeros(N+1)
theta[0] = theta0
theta_dot[0] = theta_dot0

# === Intégration par Euler explicite ===
for n in range(N):
    theta[n+1] = theta[n] + dt * theta_dot[n]
    theta_dot[n+1] = theta_dot[n] + dt * (-g/l * np.sin(theta[n]) + h(time[n]))

# === Affichage des résultats ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, theta, label="θ(t)", color='blue')
plt.title("Angle θ(t) du pendule (Euler explicite)")
plt.xlabel("Temps (s)")
plt.ylabel("θ(t)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, theta_dot, label="θ̇(t)", color='orange')
plt.title("Vitesse angulaire θ̇(t) du pendule")
plt.xlabel("Temps (s)")
plt.ylabel("θ̇(t)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
