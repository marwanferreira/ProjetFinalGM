###☻cas 1 Partie 3 (retour à l'équilibre classic)
import numpy as np
import matplotlib.pyplot as plt

g = 9.81
l = 1.0
T = 1.0
N = 1000
dt = T / N

theta0 = 1.0
omega0 = 0.0
theta_target = 0.0
omega_target = 0.0

theta = np.zeros(N + 1)
omega = np.zeros(N + 1)
lambda_theta = np.zeros(N + 1)
lambda_omega = np.zeros(N + 1)
h = np.zeros(N)

theta[0] = theta0
omega[0] = omega0

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

J = np.sum(h**2) * dt
print(f"Cas 1 - Coût quadratique J(h) = {J:.4f}")

t = np.linspace(0, T, N + 1)
plt.figure(figsize=(12, 6))
plt.plot(t, theta, label=r'$\theta(t)$')
plt.plot(t, omega, label=r'$\omega(t)$')
plt.plot(t[:-1], h, label=r'$h(t)$')
plt.xlabel("Temps (s)")
plt.title("Cas 1 - Retour à l'équilibre classique")
plt.legend()
plt.grid(True)
plt.show()