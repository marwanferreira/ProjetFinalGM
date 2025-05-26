### Cas 5 Partie 3 – Oscillation initiale avec vitesse opposée

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques et numériques ---
g = 9.81
l = 1.0
T = 1.0
N = 1000
dt = T / N

theta0 = 1.0
omega0 = -1.0
theta_target = 0.0
omega_target = 0.0

# --- Initialisation des vecteurs ---
theta = np.zeros(N + 1)
omega = np.zeros(N + 1)
lambda_theta = np.zeros(N + 1)
lambda_omega = np.zeros(N + 1)
h = np.zeros(N)

theta[0] = theta0
omega[0] = omega0

# --- Résolution par descente de gradient ---
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

# --- Calcul du coût ---
J = np.sum(h**2) * dt
print(f"Cas 5 - Coût quadratique J(h) = {J:.4f}")

# --- Tracé graphique (respect charte graphique) ---
t = np.linspace(0, T, N + 1)
t_h = t[:-1]

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(12, 6))

plt.plot(t, theta, linestyle='--', color='darkblue', label="Angle θ(t) [rad]")
plt.plot(t, omega, linestyle='-.', color='darkred', label="Vitesse ω(t) [rad/s]")
plt.plot(t_h, h, linestyle=':', color='darkgreen', label="Commande h(t)")

plt.xlabel("Temps (s)")
plt.ylabel("Valeurs physiques")
plt.title("Cas 5 – Oscillation initiale avec vitesse opposée")
plt.legend()
plt.tight_layout()
plt.show()
