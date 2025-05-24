import numpy as np
import matplotlib.pyplot as plt

# Constantes
g = 9.81
l = 1.0
T = 1.0
N = 100
dt = T / N

# Conditions initiales
theta0 = 1.0
omega0 = 0.0

# Initialisation
theta = np.zeros(N+1)
omega = np.zeros(N+1)
lambda_theta = np.zeros(N+1)
lambda_omega = np.zeros(N+1)
h = np.zeros(N)

# Conditions initiales
theta[0] = theta0
omega[0] = omega0

# Nombre d'itérations
num_iterations = 50

for iteration in range(num_iterations):
    # 1. Forward pass : simulation de theta et omega
    for k in range(N):
        theta[k+1] = theta[k] + dt * omega[k]
        omega[k+1] = omega[k] + dt * (- (g / l) * theta[k] + h[k])  # linéarisé

    # 2. Conditions finales des multiplicateurs
    lambda_theta[N] = 0.0
    lambda_omega[N] = 0.0

    # 3. Backward pass : calcul des multiplicateurs de Lagrange
    for k in reversed(range(N)):
        lambda_theta[k] = lambda_theta[k+1] + dt * lambda_omega[k+1] * (g / l)
        lambda_omega[k] = lambda_omega[k+1] + dt * (lambda_theta[k+1] - omega[k])

        # 4. Mise à jour du contrôle
        h[k] = 0.5 * lambda_omega[k]

# Tracé
t = np.linspace(0, T, N+1)
plt.figure(figsize=(12, 6))
plt.plot(t, theta, label=r'$\theta(t)$')
plt.plot(t, omega, label=r'$\omega(t)$')
plt.plot(t[:-1], h, label=r'$h(t)$')
plt.xlabel("Temps (s)")
plt.title("Pendule - Méthode du gradient à pas fixe")
plt.legend()
plt.grid(True)
plt.show()
