## Cas 2: Avec frottements et pas de fonction h

import numpy as np
import matplotlib.pyplot as plt

# Conditions initiales
g = 9.81
N = 10000
l = 1
t0 = 0
tf = 15
pas = (tf - t0) / N
theta = []
omega = []
t = [t0 + (i * pas) for i in range(N)]

omega0 = np.random.rand()
theta0 = 2 * np.pi * np.random.rand()
theta.append(theta0)
omega.append(omega0)

# Coefficient de frottement lambda
lambd = 0.5

# Intégration d’Euler avec frottement
for i in range(N - 1):
    theta.append(theta[i] + pas * omega[i])
    omega.append(omega[i] + pas * (-lambd * omega[i] - (g / l) * np.sin(theta[i])))

# --- Ajouts esthétiques du graphique ---
plt.style.use('seaborn-v0_8-darkgrid')      # style pro
plt.figure(figsize=(10, 5))                 # taille

plt.plot(t, np.degrees(theta[:N]), linestyle='--', color='navy', label="Angle (°)")            # style angle
plt.plot(t, omega, linestyle='-.', color='darkred', label="Vitesse angulaire (rad/s)")         # style vitesse

plt.xlabel("Temps (s)")
plt.ylabel("Valeurs physiques")
plt.title("Pendule avec frottement (λ = {:.2f})".format(lambd), fontsize=13)
plt.legend()
plt.tight_layout()                          # alignement marges propre
plt.show()
