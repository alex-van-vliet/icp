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


# CPU implementation

Quick description

Source : http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICP09.pdf

Parallelized closest point finding and centering the cloud of points


# GPU implementation

Quick description of the GPU baseline

Perfs: slower than CPU
Everything in ManagedMemory


# Performance indicators

## Google Test

Source: https://github.com/google/googletest


## Google Benchmark

Source: https://github.com/google/benchmark


## Flame graph

Source: https://github.com/jonhoo/inferno

## nvprof

Source: https://docs.nvidia.com/cuda/profiler-users-guide/index.html

# Performance bottlenecks

## Matching the closest point

# GPU improvement

## v2: stop using ManagedMemory

## v3: parallelize mean

## v4: parallelize covariance kernel

## v5: parallelize apply alignment kernel

## v6: parallelize compute error kernel

## v7: use column ordering

## v8: add vp trees

## v9:

### v9.1: search vtree in iterative

### v9.2: remove center from points in iterative

## v10: remove center from points in recursive

## V11:

### v11.1: split covariance into product and sum

### v11.2: block add covariance kernel

## v12: block add mean sum

## v13: use all warps to reduce block sums (mean & covariance)

## v14: load two elements directly in block sums (mean & covariance)

## v15: unroll loop in block sums (mean & covariance)

# Summary

summary avec le benchmark

Pair programming during the whole project development.
