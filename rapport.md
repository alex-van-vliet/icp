---
title: Who's the fastest ?
author:
- Alex Van Vliet
- Théophane Vié
- Paul Khuat-Duy
...

\newpage

# Introduction

## ICP

L'algorithme de l'ICP (Iterative closest point) permet de minimiser la distance entre le nuage de point d'un objet dans deux position différentes.

Il existe de nombreuses variantes de l'ICP toutes rencontrant plus ou moins d'étape en fonction du résultat rechérché. Voici les étapes de l'algorithme que nous avons choisi :

- détection des points les plus proches dans les deux nuages de points
- calcul de la matrice de transformation pour passer des points les plus proches aux autres
- application de la matrice au premier nuage

Ces étapes sont reproduites tant que l'erreur n'est pas suffisament faible, c'est à dire tant que les deux nuages de points ne sont pas superposés.


# Implémentation CPU

Source : http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICP09.pdf

Parties parallélisée :

- recherche du point le plus proche
- recentrer le nuage de points


# Implementation GPU

Perfs: Plus lent que le CPU
Utilisation de ManagedMemory partout pour ne pas avoir a se préoccuper des accès entre CPU et GPU

# Indicateurs de performance

## Google Test

Source: https://github.com/google/googletest


## Google Benchmark

Source: https://github.com/google/benchmark


## Flame graph

Source: https://github.com/jonhoo/inferno

## nvprof

Source: https://docs.nvidia.com/cuda/profiler-users-guide/index.html

# Bottlenecks

## Matching the closest point

# GPU improvement

## v2: Ne plus utiliser de ManagedMemory

## v3: Parallélisation  de la moyenne

## v4: Parallélisation du kernel de covariance

## v5: Parallélisation du kernel apply alignment

## v6: Parallélisation du kernel compute error

## v7: Matrices en column-major order

## v8: Ajout d'un VP Tree

## v9:

### v9.1: VP Tree recherche en itératif

### v9.2: Ne plus inclure le centre dans la liste de noeuds

## v10: Ne plus inclure le centre (version récursive)

## V11:

### v11.1: Séparation de la covariance en produit et somme

### v11.2: Somme par block pour le kernel de covariance

## v12: block add mean sum

## v13: Utilisation des warps pour réduire les sommes de blocks (moyenne & covariance)

## v14: Somme par blocks sur les éléments deux à deux (moyenne & covariance)

## v15: Déroulement de boucle pour les sommes de blocks (moyenne & covariance)

# Summary

Pair programming durant tout le projet.
