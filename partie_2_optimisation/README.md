# Partie 2 – Optimisation du Retour à l’Équilibre

## Objectif

Dans cette partie, l’objectif est de **ramener un pendule en position d’équilibre stable** \((\theta, \dot{\theta}) = (0, 0)\) à un instant final donné \( T = 1 \), en déterminant une commande optimale \( h(t) \).

La commande doit permettre ce retour **tout en minimisant une fonction de coût** représentant l’énergie ou l'effort de contrôle.

---

## Méthodologie

Le problème est formulé comme une **optimisation dynamique** discrétisée à l’aide de la méthode d’Euler explicite.  
L’optimisation se fait par **descente de gradient** sur le vecteur de commande \( h = (h_0, \dots, h_{N-1}) \).

### Modèle utilisé :

\[
\ddot{\theta}(t) + \frac{g}{l} \sin(\theta(t)) = h(t)
\]

Réécrit comme un système du premier ordre et résolu numériquement.

---

## Expériences réalisées

Trois variantes d’optimisation ont été comparées :

### 1. `pendule_gradient_cosinus_sans_contrainte.py`
- Commande initiale en cosinus, sans contrainte finale.
- Permet de tester la descente de gradient sur \( h \) avec une fonction coût simple :  
  \[
  J(h) = \int_0^1 h(t)^2 \,dt
  \]
- ✅ Utilisée pour valider l'algorithme, mais **ne permet pas d’atteindre l’équilibre**.

### 2. `pendule_gradient_contrainte_finale.py`
- Version avec **contraintes terminales fortes**, imposées via des multiplicateurs de Lagrange \((\mu_1, \mu_2)\).
- Plus efficace, mais nécessite un **réglage manuel** des multiplicateurs, ce qui peut déstabiliser l’optimisation.

### 3. `pendule_gradient_contrainte_quadratique.py` ✅ **(version retenue)**
- On pénalise directement \( \theta(T)^2 \) et \( \dot{\theta}(T)^2 \) dans la fonction coût :  
  \[
  J(h) = \int_0^1 h(t)^2 \,dt + \lambda_1 \cdot \theta(T)^2 + \lambda_2 \cdot \dot{\theta}(T)^2
  \]
- Formulation **plus souple et mathématiquement propre**.
- ✅ Fournit une commande lisse, efficace, et un retour progressif à l'équilibre.

---

## Résultats obtenus

- La **descente de gradient fonctionne bien** dans les trois cas.
- Seule la **formulation avec contrainte finale** permet d’assurer un retour précis à \((0, 0)\).
- La **formulation quadratique** est la plus simple à régler, stable et élégante.

---

## Conclusion

> Cette deuxième partie a permis de comparer plusieurs stratégies de commande optimale pour ramener un pendule à l’équilibre.
>
> La version finale utilisant une **pénalisation quadratique de l’état final** s’est révélée être la meilleure approche. Elle permet :
> - de **revenir précisément à l’équilibre**,
> - avec une **commande de faible amplitude**,
> - et une **implémentation robuste**.
>
> Elle constituera la base idéale pour étudier des extensions plus complexes dans la suite du projet (non-linéarités fortes, amortissement, etc.).

---

## Dossiers utiles

- `/figures_partie2/` : contient les visualisations générées pour chaque variante.
- `/pendule_gradient_*.py` : fichiers Python correspondant à chaque test.

---

Tu peux lancer chaque fichier directement dans un environnement Python 3 avec `matplotlib` installé.

