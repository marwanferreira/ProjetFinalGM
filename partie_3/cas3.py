### Cas 3 Partie 3 – Déplacement vers un autre angle

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres du problème ---
g = 9.81
l = 1.0
T = 1.0
N = 1000
dt = T / N

theta0 = -0.5
omega0 = 0.5
theta_target = 0.3
omega_target = 0.0

# --- Initialisation ---
theta = np.zeros(N + 1)
omega = np.zeros(N + 1)
lambda_theta = np.zeros(N + 1)
lambda_omega = np.zeros(N + 1)
h = np.zeros(N)

theta[0] = theta0
omega[0] = omega0

# --- Algorithme de gradient projeté ---
num_iterations = 50

for iteration in range(num_iterations):
    for k in range(N):
        theta[k + 1] = theta[k] + dt * omega[k]
        omega[k + 1] = omega[k] + dt * (-(g / l) * theta[k] + h[k])

    lambda_theta[N] = 0.0
    lambda_omega[N] = 0.0

    for k in reversed(range(N)):
        lambda_theta[k] = lambda_theta[k + 1] + dt * lambda_omega[k + 1] * (g / l)
        lambda_omega[k] = lambda_omega[k + 1] + dt * (lambda_theta[k + 1] - omega[k])
        h[k] = 0.5 * lambda_omega[k]

# --- Coût de la commande ---
J = np.sum(h**2) * dt
print(f"Cas 3 - Coût quadratique J(h) = {J:.4f}")

# --- Tracé graphique stylisé ---
t = np.linspace(0, T, N + 1)
t_h = t[:-1]

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(10, 5))

plt.plot(t, theta, linestyle='--', color='darkblue', label="Angle θ(t) [rad]")
plt.plot(t, omega, linestyle='-.', color='darkred', label="Vitesse ω(t) [rad/s]")
plt.plot(t_h, h, linestyle=':', color='darkgreen', label="Commande h(t)")

plt.xlabel("Temps (s)")
plt.ylabel("Valeurs physiques")
plt.title("Cas 3 – Déplacement vers un autre angle (objectif θ = 0.3 rad)")
plt.legend()
plt.tight_layout()
plt.show()
