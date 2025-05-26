# Projet GMF – Optimisation du retour à l’état d’équilibre

**Auteurs** : Yvann Landuré, Julien Ruiz, Samuel Bourhis, Marwan Ferreira da Silva  
**Date** : 18 mai 2025  

---

## Introduction

Ce projet s’inscrit dans le cadre du module GMF (Génie Mathématiques et Finance) à CY Tech. Il a pour objectif l’étude et le contrôle du mouvement d’un pendule simple soumis à une perturbation initiale quelconque.

Le système est modélisé par l’équation suivante :

> $$\ddot{\theta}(t) + \frac{g}{l} \sin(\theta(t)) = h(t)$$

où $h(t)$ représente une commande externe agissant sur le pendule. L’objectif est de ramener le système à l’état d’équilibre stable $(\theta, \dot{\theta}) = (0, 0)$ en un temps fini, ici normalisé à l’intervalle $[0, 1]$, tout en minimisant l’énergie fournie via $h(t)$.

---

## Objectifs

- Implémenter des schémas numériques classiques (Euler explicite, méthode du gradient).
- Formuler un problème d’optimisation dynamique sous contraintes d’égalité à l’aide d’un lagrangien.
- Résoudre numériquement le problème pour différentes conditions initiales et représenter graphiquement les résultats.

---

## Livrables

- Un rapport rédigé détaillant les méthodes, résultats et interprétations.
- L’ensemble du code source Python utilisé pour les simulations.
- Une présentation orale synthétique.

---

## Membres du groupe

- **Samuel Bourhis**
- **Marwan Ferreira da Silva**
- **Yvann Landuré**
- **Julien Ruiz**
