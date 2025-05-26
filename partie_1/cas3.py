## Cas 3: Avec ressort et frottements

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

k = 1.0       # constante du ressort
m = 1.0       # masse
omega0 = np.sqrt(k / m)  # pulsation propre
lambd = 0.5   # coefficient d'amortissement

x0 = 1.0      # position initiale
v0 = 0.0      # vitesse initiale

x = [x0]
v = [v0]

# Méthode d’Euler pour x'' + λx' + ω₀²x = 0
for i in range(N - 1):
    x.append(x[i] + pas * v[i])
    a = -lambd * v[i] - omega0**2 * x[i]
    v.append(v[i] + pas * a)

# --- Ajouts esthétiques ---
plt.style.use('seaborn-v0_8-darkgrid')       # style académique
plt.figure(figsize=(10, 5))                  # taille plus large

plt.plot(t, x, linestyle='--', color='darkgreen', label="Position $x(t)$")         # position → vert pointillé
plt.plot(t, v, linestyle='-.', color='darkred', label="Vitesse $v(t)$")         # vitesse → orange tiret-point

plt.xlabel("Temps (s)")
plt.ylabel("Valeurs physiques")
plt.title(f"Système masse-ressort avec frottement (λ = {lambd})", fontsize=13)
plt.legend()
plt.tight_layout()                           # alignement propre
plt.show()
