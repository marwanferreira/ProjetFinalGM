##code partie 3

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
N = 500
T = 1
dt = T / N
g = 9.81
l = 1.0
eta = 0.1
max_iter = 100

# Conditions initiales
theta0 = -0.5
omega0 = 0.2

def gradient_descent_quadratic(theta0, omega0):
    h = np.zeros(N)
    cost_history = []

    def simulate(h):
        theta = np.zeros(N + 1)
        omega = np.zeros(N + 1)
        theta[0] = theta0
        omega[0] = omega0
        for i in range(N):
            theta[i + 1] = theta[i] + dt * omega[i]
            omega[i + 1] = omega[i] + dt * (- g / l * theta[i] + h[i])
        return theta, omega

    def project(h):
        theta, omega = simulate(h)
        err_theta = theta[-1]
        err_omega = omega[-1]
        correction = np.linspace(1, 0, N)
        h -= eta * (err_theta + err_omega) * correction
        return h

    for _ in range(max_iter):
        grad = 2 * h * dt
        h -= eta * grad
        h = project(h)
        cost = np.sum(h**2) * dt
        cost_history.append(cost)

    theta_final, omega_final = simulate(h)
    return h, theta_final, omega_final, cost_history

# Exécution
h_opt, theta, omega, cost_history = gradient_descent_quadratic(theta0, omega0)

# Vecteurs de temps
t_vals = np.linspace(0, T, N + 1)
t_vals_h = np.linspace(0, T, N)

# Tracé des courbes
plt.figure(figsize=(16, 8))

# θ(t)
plt.subplot(2, 2, 1)
plt.plot(t_vals, theta, label="θ(t)")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Angle θ(t)")
plt.xlabel("Temps (s)")
plt.ylabel("θ (radian)")
plt.grid()
plt.legend()

# ω(t)
plt.subplot(2, 2, 2)
plt.plot(t_vals, omega, label="ω(t)", color="orange")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Vitesse angulaire ω(t)")
plt.xlabel("Temps (s)")
plt.ylabel("ω (rad/s)")
plt.grid()
plt.legend()

# h(t)
plt.subplot(2, 2, 3)
plt.plot(t_vals_h, h_opt, label="h(t)", color="green")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Commande optimale h(t)")
plt.xlabel("Temps (s)")
plt.ylabel("h")
plt.grid()
plt.legend()

# Coût J(h)
plt.subplot(2, 2, 4)
plt.plot(range(len(cost_history)), cost_history, label="J(h)", color="red")
plt.title("Évolution du coût J(h)")
plt.xlabel("Itération")
plt.ylabel("J(h)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()