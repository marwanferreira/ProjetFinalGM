## Question 1 avec différents cas
import numpy as np
import matplotlib.pyplot as plt

## Cas 1: Pas de frottements et pas de fonction h

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
theta0 = 2 * np.pi * (np.random.rand()-0.5)
theta.append(theta0)
omega.append(omega0)

for i in range(N - 1):
    theta.append(theta[i] + (pas * omega[i]))
    omega.append(omega[i] + pas * (-(g / l) * np.sin(theta[i])))

# --- Ajouts esthétiques du tracé ---
plt.style.use('seaborn-v0_8-darkgrid')  # ajout esthétique : style clean
plt.figure(figsize=(10, 5))             # ajout esthétique : taille personnalisée

plt.plot(t, np.degrees(theta[:N]), linestyle='--', color='darkblue', label="Angle (°)")   # ajout couleur/linestyle
plt.plot(t, omega, linestyle='-.', color='darkred', label="Vitesse angulaire (rad/s)")    # ajout couleur/linestyle

plt.xlabel("Temps (s)")
plt.ylabel("Valeurs physiques")         # ajout esthétique : axe Y général
plt.title("Pendule libre (sans frottements et sans fonction de contrainte)", fontsize=13) # ajout fontsize
plt.legend()
plt.tight_layout()                      # ajout esthétique : ajustement marges
plt.show()
