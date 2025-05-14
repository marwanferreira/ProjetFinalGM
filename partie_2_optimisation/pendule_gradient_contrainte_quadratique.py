"""
pendule_gradient_contrainte_quadratique.py

!!! REFERENCE POUR LA PARTE 2: !!!


Ce script correspond à la partie 2 du projet GMF - Optimisation du retour à l’équilibre,
version améliorée avec **pénalisation quadratique** de l’état final.

Objectif :
    Trouver une commande optimale h(t) qui ramène le pendule en position d’équilibre (θ, θ̇) = (0, 0)
    au temps T = 1, tout en minimisant :
        J(h) = ∫₀¹ h(t)² dt + λ₁·θ(T)² + λ₂·θ̇(T)²

Méthode :
    - Simulation par Euler explicite,
    - Optimisation par descente de gradient,
    - Pénalisation de l’état final dans la fonction coût (pas via multiplicateurs).
"""

import numpy as np
import matplotlib.pyplot as plt

# === Paramètres physiques ===
g = 9.81
l = 1.0
T = 1.0
N = 200
dt = T / N
time = np.linspace(0, T, N+1)

# === Conditions initiales ===
theta0 = 1.0
theta_dot0 = 0.0

# === Paramètres de l'optimisation ===
alpha = 0.01
max_iter = 200

# === Poids des pénalités sur l’état final ===
lambda1 = 20.0   # pondération sur θ(T)^2
lambda2 = 5.0    # pondération sur θ̇(T)^2

# === Commande initiale ===
h = 0.1 * np.cos(2 * np.pi * np.linspace(0, T, N))

def forward(h):
    theta = np.zeros(N+1)
    theta_dot = np.zeros(N+1)
    theta[0] = theta0
    theta_dot[0] = theta_dot0

    for n in range(N):
        theta[n+1] = theta[n] + dt * theta_dot[n]
        theta_dot[n+1] = theta_dot[n] + dt * (-g/l * np.sin(theta[n]) + h[n])

    return theta, theta_dot

def backward(theta, theta_dot):
    lam1 = np.zeros(N+1)
    lam2 = np.zeros(N+1)

    # Conditions terminales basées sur la pénalisation quadratique
    lam1[N] = 2 * lambda1 * theta[N]
    lam2[N] = 2 * lambda2 * theta_dot[N]

    for n in reversed(range(N)):
        lam1[n] = lam1[n+1] + dt * lam2[n+1] * (g/l * np.cos(theta[n]))
        lam2[n] = lam2[n+1] + dt * lam1[n+1]

    return lam1, lam2

def compute_gradient(lam2, h):
    return lam2[:N] + 2 * h * dt

# === Boucle de descente de gradient ===
for iteration in range(max_iter):
    theta, theta_dot = forward(h)
    lam1, lam2 = backward(theta, theta_dot)
    grad = compute_gradient(lam2, h)
    h -= alpha * grad

# === Simulation finale ===
theta_opt, theta_dot_opt = forward(h)

# === Affichage ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, T, N), h, label="Commande optimale $h(t)$", color='purple')
plt.title("Commande optimale avec pénalisation quadratique de l'état final")
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
