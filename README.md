# Capsule Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leopoldmaillard/CapsNet_kmnist/blob/main/capsnet_kmnist.ipynb)

By [Lucie Clair](https://github.com/LucieClair) & [Léopold Maillard](https://github.com/leopoldmaillard), as part of the INSA Rouen's Deep Learning course project, 2021.

This repository provides understanding, implementation and full training process for NeurIPS 2017 paper [*"Dynamic Routing Between Capsules"*](https://arxiv.org/abs/1710.09829) by Geoffrey E. Hinton, Sara Sabour & Nicholas Frosst.

## Misc

Explication d'Aurélien Géron : https://www.youtube.com/watch?v=pPN8d0E3900

Article Medium : https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8

## Entraînement du 11/04/2021

- Plus de 8 millions de paramètres entraînables, 13 min / epoch sur mon CPU.
- 80s / epoch sur un GPU Google Colab
- Sur 50 epochs, 3 rooting steps : Train Accuracy : 99.45%, Val Accuracy : 93.58
- What to do next :
  - Data augmentation (va aider pour l'overfit)
  - Dropout ?
  - Hyper-parameters tuning : n_routing, batch_size
  - Loss sur une trend descendante : entraîner sur + d'epoch
- Faire test sur le eval model
- Afficher les reconstructions

