# Capsule Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leopoldmaillard/CapsNet_kmnist/blob/main/capsnet_kmnist.ipynb)

By [Lucie Clair](https://github.com/LucieClair) & [Léopold Maillard](https://github.com/leopoldmaillard), as part of the INSA Rouen's Deep Learning course project, 2021.

This repository provides understanding, implementation and full training process for NeurIPS 2017 paper [*"Dynamic Routing Between Capsules"*](https://arxiv.org/abs/1710.09829) by Geoffrey E. Hinton, Sara Sabour & Nicholas Frosst.

## About this work

### Motivation

### Dataset

The dataset chosen to illustrate our work is [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist), which proposes a more challenging alternative to MNIST. Indeed, most of the recent Deep Learning models achieve more than 99.5% accuracy on MNIST, so it can be interesting to evaluate our model on more challenging datasets. K-MNIST consists of 70,000 images (28x28, grayscale) in 10 classes, one for each row of the Japanese Hiragana. 

Unlike other datasets like Fashion-MNIST, we haven't found any other implementations involving capsules on K-MNIST, and we will be thus able to compare our CapsNet results with benchmarked models. Finally, given the nature of the dataset, capsules seem **instinctively** particularly suitable for the task.

<p align="center">
  <img src="https://github.com/rois-codh/kmnist/raw/master/images/kmnist_examples.png">
   Kuzushiji-MNIST 10 classes
</p>

### Content

## Implementation details

## Training Process

## Results

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

Explication d'Aurélien Géron : https://www.youtube.com/watch?v=pPN8d0E3900

Article Medium : https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8

