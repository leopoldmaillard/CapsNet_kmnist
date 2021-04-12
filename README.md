# Capsule Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leopoldmaillard/CapsNet_kmnist/blob/main/capsnet_kmnist.ipynb)

Capsule Neural Network paper understanding & implementation.

Paper : https://arxiv.org/abs/1710.09829

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

