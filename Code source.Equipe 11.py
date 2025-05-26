







##etape 3
import numpy as np
import matplotlib.pyplot as plt

# Constantes
g = 9.81
l = 1.0
T = 1
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


###test méthode non linéaire

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques ---
g = 9.81        # gravité (m/s²)
l = 1.0         # longueur du pendule (m)
kr = 2.0        # intensité de la force sin(2θ)

# --- Paramètres de simulation ---
N = 10000
t0, tf = 0.0, 10.0
dt = (tf - t0) / N
t = np.linspace(t0, tf, N)

# --- Conditions initiales ---
np.random.seed(123)
theta0 = np.pi * (2 * np.random.rand() - 1)  # angle initial entre -π et π
omega0 = np.random.rand()                    # vitesse initiale

theta = np.zeros(N)
omega = np.zeros(N)
theta[0] = theta0
omega[0] = omega0

# --- Simulation par Euler explicite ---
for i in range(N - 1):
    theta[i+1] = theta[i] + dt * omega[i]
    omega[i+1] = omega[i] + dt * (
        - (g / l) * np.sin(theta[i]) - (kr / l) * np.sin(2 * theta[i])
    )

# --- Affichage graphique ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, np.degrees(theta), label="θ(t) [°]", color='darkblue', linestyle='-')
ax.plot(t, omega, label="ω(t) [rad/s]", color='crimson', linestyle='--')
ax.set_title("Pendule simple avec force sin(2θ) (Euler explicite)")
ax.set_xlabel("Temps [s]")
ax.set_ylabel("Valeurs")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()




##test partie 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp



# --- Paramètres physiques du système ---
g = 9.81         # Gravité (m/s²)
l = 1.0          # Longueur du pendule (m)

# --- Paramètres de simulation ---
N = 2000
t_start, t_end = 0.0, 10.0
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta_init = 1.0     # Angle initial (rad)
omega_init = 0.0     # Vitesse initiale (rad/s)

# --- Paramètres de l’approximation ---
degree = 10  # degré des monômes pour approximer h(t)

def h_basis(t, coeffs):
    return sum(c * t**i for i, c in enumerate(coeffs))

# --- Équation différentielle complète avec h(t) continu ---
def pendulum_ode(t, y, coeffs):
    theta, omega = y
    h_t = h_basis(t, coeffs)
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta) + h_t
    return [dtheta_dt, domega_dt]

# --- Fonction objectif (coût + pénalisation finale) ---
def objective(coeffs):
    sol = solve_ivp(pendulum_ode, [t_start, t_end], [theta_init, omega_init],
                    args=(coeffs,), t_eval=np.linspace(t_start, t_end, 100))

    h_vals = np.array([h_basis(t, coeffs) for t in sol.t])
    energy = np.sum(h_vals**2) * (t_end - t_start) / len(h_vals)

    final_theta = sol.y[0, -1]
    final_omega = sol.y[1, -1]
    penalty = final_theta**2 + final_omega**2

    mu = 1e3
    return energy + mu * penalty

# --- Optimisation des coefficients ---
initial_coeffs = np.zeros(degree + 1)
res = minimize(objective, initial_coeffs, method='L-BFGS-B', options={'maxiter': 500, 'disp': True})
coeffs_opt = res.x

# --- Résolution finale du pendule ---
t_vals = t_grid
def deriv(t, y): return pendulum_ode(t, y, coeffs_opt)
sol = solve_ivp(deriv, [t_start, t_end], [theta_init, omega_init], t_eval=t_vals)
theta_vals = sol.y[0]
omega_vals = sol.y[1]
h_vals = [h_basis(t, coeffs_opt) for t in t_vals]

# --- Affichage graphique (même structure que Partie 1 à 4.2) ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axes[0].plot(t_vals, np.degrees(theta_vals), linestyle='--', color='navy')
axes[0].set_ylabel("Angle θ(t) [°]")
axes[0].set_title("Évolution de l’angle du pendule")

axes[1].plot(t_vals, omega_vals, linestyle='-.', color='darkred')
axes[1].set_ylabel("Vitesse ω(t) [rad/s]")
axes[1].set_title("Évolution de la vitesse angulaire")

axes[2].plot(t_vals, h_vals, linestyle='-', color='green')
axes[2].set_ylabel("Contrôle h(t)")
axes[2].set_xlabel("Temps [s]")
axes[2].set_title("Contrôle optimal approximé par base polynomiale")

plt.suptitle("Optimisation du contrôle continu par base polynomiale", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Résumé terminal ---
print(f"Optimisation réussie : {res.success}")
print(f"Message : {res.message}")
print(f"Valeur finale de la fonction objectif : {res.fun:.4f}")






## partie 3

"""
Optimisation du contrôle h(t) via gradient adjoint (cas linéarisé)
------------------------------------------------------------------
Hypothèse : petites oscillations ⇒ sin(θ) ≈ θ
Formulation : Lagrangien + méthode d’Euler explicite + gradient à pas fixe

Objectif : atteindre θ(tf) = 0 et ω(tf) = 0 tout en minimisant J(h) = ∫ h(t)² dt

"""

import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques du pendule ---
g = 9.81         # gravité (m/s²)
l = 1.0          # longueur du pendule (m)

# --- Paramètres de simulation ---
N = 10000
t_start, t_end = 0.0, 10.0
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta_init = 1.0     # rad
omega_init = 0.0     # rad/s

# --- Initialisation du contrôle ---
h = np.zeros(N)

# --- Hyperparamètres de la descente de gradient ---
eta = 5e-5               # pas d’apprentissage
n_iterations = 100       # nombre d'itérations

for _ in range(n_iterations):

    # --- Simulation directe (forward) ---
    theta = np.zeros(N + 1)
    omega = np.zeros(N + 1)
    theta[0] = theta_init
    omega[0] = omega_init

    for i in range(N):
        theta[i + 1] = theta[i] + dt * omega[i]
        omega[i + 1] = omega[i] + dt * (- (g / l) * theta[i] + h[i])  # linéarisé : sin(theta) ≈ theta

    # --- Équations adjointes (backward) ---
    lambda_theta = np.zeros(N + 1)
    lambda_omega = np.zeros(N + 1)
    lambda_theta[N] = 0.0
    lambda_omega[N] = 0.0

    for i in reversed(range(N)):
        lambda_theta[i] = lambda_theta[i + 1] + dt * lambda_omega[i + 1] * (g / l)  # cos(theta) ≈ 1
        lambda_omega[i] = lambda_omega[i + 1] + dt * (lambda_theta[i + 1] - omega[i])

    # --- Mise à jour du contrôle optimal ---
    grad = h - 0.5 * lambda_omega[:-1]          # ∇J(h)
    h = h - eta * grad                          # descente à pas fixe
    h = np.clip(h, -2.0, 2.0)                   # projection : bornes physiques

# --- Affichage graphique (structure académique) ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

# Angle θ(t)
axes[0].plot(t_grid, np.degrees(theta), linestyle='--', color='navy')
axes[0].set_ylabel("Angle θ(t) [°]")
axes[0].set_title("Évolution de l’angle du pendule")

# Vitesse ω(t)
axes[1].plot(t_grid, omega, linestyle='-.', color='darkred')
axes[1].set_ylabel("Vitesse ω(t) [rad/s]")
axes[1].set_title("Évolution de la vitesse angulaire")

# Contrôle h(t)
axes[2].plot(t_grid[:-1], h, linestyle='-', color='green')
axes[2].set_ylabel("Contrôle h(t)")
axes[2].set_xlabel("Temps [s]")
axes[2].set_title("Évolution du contrôle optimal")

plt.suptitle("Optimisation du contrôle par gradient (cas linéarisé)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


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


##Partie 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Paramètres physiques ---
g = 9.81         # Gravité (m/s²)
l = 1.0          # Longueur du pendule (m)
kf = 0.5         # Coefficient de frottement

# --- Paramètres de simulation ---
N = 200                      # Nombre de points
t_start, t_end = 0.0, 10.0    # Intervalle de temps [s]
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta_init = 1.0             # Angle initial (rad)
omega_init = 0.0             # Vitesse angulaire initiale (rad/s)

# --- Simulation avec frottement ---
def simulate_pendulum_friction(h):
    theta = np.zeros(N + 1)
    omega = np.zeros(N + 1)
    theta[0] = theta_init
    omega[0] = omega_init
    for i in range(N):
        theta[i+1] = theta[i] + dt * omega[i]
        omega[i+1] = omega[i] + dt * ( - (g / l) * np.sin(theta[i]) - kf * omega[i] + h[i] )
    return theta, omega

# --- Fonction coût pénalisée ---
def penalty_cost_friction(h, mu=1e6):
    theta, omega = simulate_pendulum_friction(h)
    energy = dt * np.sum(h**2)
    dyn_err_theta = np.sum((theta[1:] - theta[:-1] - dt * omega[:-1])**2)
    dyn_err_omega = np.sum((omega[1:] - omega[:-1] - dt * ( - (g / l) * np.sin(theta[:-1]) - kf * omega[:-1] + h ))**2)
    final_err = theta[-1]**2 + omega[-1]**2
    return energy + mu * (dyn_err_theta + dyn_err_omega + final_err)

# --- Optimisation ---
h0 = np.zeros(N)
result = minimize(penalty_cost_friction, h0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': True})
h_opt = result.x
theta_opt, omega_opt = simulate_pendulum_friction(h_opt)

# --- Affichage des résultats ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axes[0].plot(t_grid, np.degrees(theta_opt), linestyle='--', color='navy')
axes[0].set_ylabel("Angle θ(t) [°]")
axes[0].set_title("Évolution de l’angle du pendule")

axes[1].plot(t_grid, omega_opt, linestyle='-.', color='darkred')
axes[1].set_ylabel("Vitesse ω(t) [rad/s]")
axes[1].set_title("Évolution de la vitesse angulaire")

axes[2].plot(t_grid[:-1], h_opt, linestyle='-', color='green')
axes[2].set_ylabel("Contrôle h(t)")
axes[2].set_xlabel("Temps [s]")
axes[2].set_title("Évolution du contrôle optimal")

plt.suptitle("Optimisation par pénalisation avec frottement (Euler explicite)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Logs de l’optimisation ---
print(f"Optimisation réussie : {result.success}")
print(f"Message : {result.message}")
print(f"Valeur finale de la fonction coût : {result.fun}")

###Partie 4 sous partie 3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp



# --- Paramètres physiques du système ---
g = 9.81         # Gravité (m/s²)
l = 1.0          # Longueur du pendule (m)

# --- Paramètres de simulation ---
N = 2000
t_start, t_end = 0.0, 10.0
dt = (t_end - t_start) / N
t_grid = np.linspace(t_start, t_end, N + 1)

# --- Conditions initiales ---
theta_init = 1.0     # Angle initial (rad)
omega_init = 0.0     # Vitesse initiale (rad/s)

# --- Paramètres de l’approximation ---
degree = 10  # degré des monômes pour approximer h(t)

def h_basis(t, coeffs):
    return sum(c * t**i for i, c in enumerate(coeffs))

# --- Équation différentielle complète avec h(t) continu ---
def pendulum_ode(t, y, coeffs):
    theta, omega = y
    h_t = h_basis(t, coeffs)
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta) + h_t
    return [dtheta_dt, domega_dt]

# --- Fonction objectif (coût + pénalisation finale) ---
def objective(coeffs):
    sol = solve_ivp(pendulum_ode, [t_start, t_end], [theta_init, omega_init],
                    args=(coeffs,), t_eval=np.linspace(t_start, t_end, 100))

    h_vals = np.array([h_basis(t, coeffs) for t in sol.t])
    energy = np.sum(h_vals**2) * (t_end - t_start) / len(h_vals)

    final_theta = sol.y[0, -1]
    final_omega = sol.y[1, -1]
    penalty = final_theta**2 + final_omega**2

    mu = 1e3
    return energy + mu * penalty

# --- Optimisation des coefficients ---
initial_coeffs = np.zeros(degree + 1)
res = minimize(objective, initial_coeffs, method='L-BFGS-B', options={'maxiter': 500, 'disp': True})
coeffs_opt = res.x

# --- Résolution finale du pendule ---
t_vals = t_grid
def deriv(t, y): return pendulum_ode(t, y, coeffs_opt)
sol = solve_ivp(deriv, [t_start, t_end], [theta_init, omega_init], t_eval=t_vals)
theta_vals = sol.y[0]
omega_vals = sol.y[1]
h_vals = [h_basis(t, coeffs_opt) for t in t_vals]

# --- Affichage graphique (même structure que Partie 1 à 4.2) ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axes[0].plot(t_vals, np.degrees(theta_vals), linestyle='--', color='navy')
axes[0].set_ylabel("Angle θ(t) [°]")
axes[0].set_title("Évolution de l’angle du pendule")

axes[1].plot(t_vals, omega_vals, linestyle='-.', color='darkred')
axes[1].set_ylabel("Vitesse ω(t) [rad/s]")
axes[1].set_title("Évolution de la vitesse angulaire")

axes[2].plot(t_vals, h_vals, linestyle='-', color='green')
axes[2].set_ylabel("Contrôle h(t)")
axes[2].set_xlabel("Temps [s]")
axes[2].set_title("Contrôle optimal approximé par base polynomiale")

plt.suptitle("Optimisation du contrôle continu par base polynomiale", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Résumé terminal ---
print(f"Optimisation réussie : {res.success}")
print(f"Message : {res.message}")
print(f"Valeur finale de la fonction objectif : {res.fun:.4f}")