"""
pendule_gradient_contrainte_finale.py

Ce script correspond à la **partie 2 du projet GMF - Optimisation du retour à l’équilibre**,
dans sa version **avec contrainte finale** sur l'état du pendule.

Objectif :
    Trouver une commande optimale h(t) qui ramène le pendule en position d’équilibre (θ, θ̇) = (0, 0)
    au temps final T = 1, tout en minimisant le coût énergétique :
        J(h) = ∫₀¹ h(t)² dt

Méthode :
    - Simulation numérique par Euler explicite,
    - Optimisation par descente de gradient sur h(t),
    - Prise en compte des contraintes finales via des multiplicateurs de Lagrange (mu₁, mu₂),
        introduits comme conditions terminales dans le système adjoint.

Cette version permet au pendule de revenir à l’équilibre de manière plus contrôlée,
contrairement à la version précédente sans contrainte.
"""

import numpy as np
import matplotlib.pyplot as plt

# === Paramètres physiques ===
g = 9.81           # gravité (m/s²)
l = 1.0            # longueur du pendule (m)
T = 1.0            # temps final (s)
N = 200            # nombre de pas de discrétisation
dt = T / N         # pas de temps
time = np.linspace(0, T, N+1)

# === Conditions initiales ===
theta0 = 1.0       # angle initial (rad)
theta_dot0 = 0.0   # vitesse angulaire initiale (rad/s)

# === Paramètres de la descente de gradient ===
alpha = 0.01        # pas d’apprentissage
max_iter = 200     # nombre d’itérations

# === Multiplicateurs de Lagrange (contrainte finale) ===
mu1 = 5.0        # pénalisation sur θ(T)
mu2 = 2.0        # pénalisation sur θ̇(T)

# === Commande initiale (cosinus) ===
h = 0.1 * np.cos(2 * np.pi * np.linspace(0, T, N))

def forward(h):
    """
    Simulation avant du pendule via Euler explicite.
    Retourne les trajectoires θ(t) et θ̇(t) sur [0, T].
    """
    theta = np.zeros(N+1)
    theta_dot = np.zeros(N+1)
    theta[0] = theta0
    theta_dot[0] = theta_dot0

    for n in range(N):
        theta[n+1] = theta[n] + dt * theta_dot[n]
        theta_dot[n+1] = theta_dot[n] + dt * (-g/l * np.sin(theta[n]) + h[n])

    return theta, theta_dot

def backward(theta):
    """
    Résolution du système adjoint à rebours.
    Utilise les multiplicateurs mu₁ et mu₂ en conditions finales.
    """
    lam1 = np.zeros(N+1)
    lam2 = np.zeros(N+1)

    lam1[N] = mu1
    lam2[N] = mu2

    for n in reversed(range(N)):
        lam1[n] = lam1[n+1] + dt * lam2[n+1] * (g/l * np.cos(theta[n]))
        lam2[n] = lam2[n+1] + dt * lam1[n+1]

    return lam1, lam2

def compute_gradient(lam2, h):
    """
    Calcul du gradient de la fonction coût :
        J(h) = ∫ h(t)² dt
    """
    return lam2[:N] + 2 * h * dt

# === Boucle de descente de gradient ===
for iteration in range(max_iter):
    theta, theta_dot = forward(h)
    lam1, lam2 = backward(theta)
    grad = compute_gradient(lam2, h)
    h -= alpha * grad

# === Simulation finale avec h optimal ===
theta_opt, theta_dot_opt = forward(h)

# === Affichage ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, T, N), h, label="Commande optimale $h(t)$", color='purple')
plt.title("Commande optimale avec contrainte finale")
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
plt.show(block=False)
input("Appuie sur Entrée pour fermer la figure...")
