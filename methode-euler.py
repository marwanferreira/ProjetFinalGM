import numpy as np

import matplotlib.pyplot as plt



# Paramètres physiques

g = 9.81 #gravité

l = 1.0 #longueur du pendule



# Discrétisation

N = 1000

T = 1.0

dt = T / N

t = np.linspace(0, T, N+1) #tableau de N+ valeurs prises entre 0 et T



# Conditions initiales

x1 = np.zeros(N+1) # theta

x2 = np.zeros(N+1) # theta_dot

x1[0] = 1.0 # angle initial

x2[0] = 0.0 # vitesse initiale



# Fonction de contrôle h(t) choisie

def h(t):

return np.cos(2 * np.pi * t) # exemple de fonction non nulle



# Résolution avec Euler explicite

for n in range(N):

x1[n+1] = x1[n] + dt * x2[n]

x2[n+1] = x2[n] + dt * (-g/l * np.cos(x1[n]) + h(t[n]))



# Affichage

plt.plot(t, x1, label="θ(t)")

plt.plot(t, x2, label="θ̇(t)")

plt.title("Résolution du pendule simple avec Euler explicite")

plt.xlabel("Temps (s)")

plt.ylabel("Valeurs")

plt.legend()

plt.grid()

plt.show()





########code 2



import numpy as np

import matplotlib.pyplot as plt



# Paramètres

g = 9.81

l = 1.0

T = 1.0

N = 200

dt = T / N



# Initialisation

x1 = np.zeros(N+1) # theta

x2 = np.zeros(N+1) # theta_dot

theta0 = 1.0 # condition initiale

theta_dot0 = 0.0

x1[0] = theta0

x2[0] = theta_dot0



h = np.zeros(N) # contrôle initial

alpha = 0.1 # pas du gradient

max_iter = 200



def forward(h):

x1 = np.zeros(N+1)

x2 = np.zeros(N+1)

x1[0] = theta0

x2[0] = theta_dot0

for n in range(N):

x1[n+1] = x1[n] + dt * x2[n]

x2[n+1] = x2[n] + dt * (-g/l * np.sin(x1[n]) + h[n])

return x1, x2



def backward(x1, x2):

# équations adjointes, résolues à rebours

lambda1 = np.zeros(N+1)

lambda2 = np.zeros(N+1)

for n in reversed(range(N)):

lambda2[n] = lambda2[n+1] + dt * lambda1[n+1]

lambda1[n] = lambda1[n+1] + dt * lambda2[n+1] * (g/l * np.cos(x1[n]))

return lambda1, lambda2



def compute_gradient(lambda2):

return lambda2[:N] + 2 * h * dt # dérivée de J(h) = sum(h² dt)



for k in range(max_iter):

x1, x2 = forward(h)

lambda1, lambda2 = backward(x1, x2)

grad = compute_gradient(lambda2)

h -= alpha * grad # descente du gradient



# Résultat final

x1_final, x2_final = forward(h)



# Affichage

plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)

plt.plot(np.linspace(0, T, N), h, label="Commande h(t)")

plt.title("Commande optimale")

plt.grid()

plt.legend()



plt.subplot(2,1,2)

plt.plot(np.linspace(0, T, N+1), x1_final, label="θ(t)")

plt.plot(np.linspace(0, T, N+1), x2_final, label="θ̇(t)")

plt.title("État du pendule après optimisation")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()
