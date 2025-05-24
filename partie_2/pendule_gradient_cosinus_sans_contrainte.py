"""
pendule_gradient_cosinus_sans_contrainte.py

Ce script Python correspond à la **partie 2 du projet GMF - Optimisation du retour à l’équilibre** :
    - Formulation discrète du problème d'optimisation dynamique avec contraintes d'égalité,
    - Utilisation de la méthode du gradient pour approximer la commande optimale h(t).

Objectif :
    On cherche à contrôler un pendule non-linéaire fortement perturbé pour le ramener vers l’équilibre (θ, θ̇) = (0, 0),
    en minimisant une fonction de coût énergétique J(h) = ∫ h(t)^2 dt.

Dans cette version :
    ✅ Le système est simulé avec la méthode d'Euler explicite,
    ✅ La commande h(t) est initialisée par une fonction cosinus non-nulle,
    ❌ Aucune contrainte n'est imposée sur l'état final (le pendule ne revient pas nécessairement à l'équilibre).

Ce script sert principalement à **valider la descente de gradient** sans contrainte terminale.
"""

import numpy as np
import matplotlib.pyplot as plt

# === Paramètres physiques ===
g = 9.81     # gravité (m/s^2)
l = 1.0      # longueur du pendule (m)
T = 1.0      # temps final (s)
N = 200      # nombre de pas de discrétisation
dt = T / N   # pas de temps
time = np.linspace(0, T, N+1)

# === Conditions initiales ===
theta0 = 1.0          # angle initial (rad)
theta_dot0 = 0.0      # vitesse angulaire initiale (rad/s)

# === Paramètres de la descente de gradient ===
alpha = 0.1           # pas d’apprentissage
max_iter = 200        # nombre maximal d’itérations

# === Commande initiale non nulle (cosinus) ===
h = 0.1 * np.cos(2 * np.pi * np.linspace(0, T, N))  # tableau de taille N

def forward(h):
    """
    Simulation avant du pendule via la méthode d’Euler explicite.
    Retourne les trajectoires θ(t) et θ̇(t).
    """
    theta = np.zeros(N+1)
    theta_dot = np.zeros(N+1)
    theta[0] = theta0
    theta_dot[0] = theta_dot0

    for n in range(N):
        theta[n+1] = theta[n] + dt * theta_dot[n]
        theta_dot[n+1] = theta_dot[n] + dt * (-g/l * np.sin(theta[n]) + h[n])

    return theta, theta_dot

def backward(theta, theta_dot):
    """
    Système adjoint résolu en arrière (équations de Lagrange).
    Retourne les multiplicateurs λ1 et λ2.
    """
    lam1 = np.zeros(N+1)
    lam2 = np.zeros(N+1)

    for n in reversed(range(N)):
        lam1[n] = lam1[n+1] + dt * lam2[n+1] * (g/l * np.cos(theta[n]))
        lam2[n] = lam2[n+1] + dt * lam1[n+1]

    return lam1, lam2

def compute_gradient(lam2, h):
    """
    Calcule le gradient de la fonction de coût J(h) = ∫ h(t)^2 dt.
    """
    return lam2[:N] + 2 * h * dt

# === Boucle de descente de gradient ===
for iteration in range(max_iter):
    theta, theta_dot = forward(h)
    lam1, lam2 = backward(theta, theta_dot)
    grad = compute_gradient(lam2, h)
    h -= alpha * grad

# === Résultat final après optimisation ===
theta_opt, theta_dot_opt = forward(h)

# === Affichage des résultats ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, T, N), h, label="Commande optimale $h(t)$", color='purple')
plt.title("Commande optimale $h(t)$")
plt.xlabel("Temps (s)")
plt.ylabel("h(t)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, theta_opt, label="θ(t)", color='blue')
plt.plot(time, theta_dot_opt, label="θ̇(t)", color='orange')
plt.title("État du pendule sous commande optimale")
plt.xlabel("Temps (s)")
plt.ylabel("État")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
