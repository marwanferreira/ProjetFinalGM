"""
Optimisation du contrôle h(t) par base polynomiale (formulation continue)
-------------------------------------------------------------------------
Méthode : approximer h(t) ≈ ∑ cᵢ·tⁱ, puis résoudre par optimisation continue
Avantage : évite la discrétisation directe → formulation fluide et naturelle

"""

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
