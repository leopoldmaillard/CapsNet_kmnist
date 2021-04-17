# Capsule Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leopoldmaillard/CapsNet_kmnist/blob/main/capsnet_kmnist.ipynb)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-brightgreen)

By [Lucie Clair](https://github.com/LucieClair) & [Léopold Maillard](https://github.com/leopoldmaillard), as part of the INSA Rouen's Deep Learning course project, 2021.

This repository provides understanding, implementation and full training process for NeurIPS 2017 paper [*"Dynamic Routing Between Capsules"*](https://arxiv.org/abs/1710.09829) by Geoffrey E. Hinton, Sara Sabour & Nicholas Frosst.

## About this work

### Motivation

### Dataset

The dataset chosen to illustrate our work is [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist), which proposes a more challenging alternative to MNIST. Indeed, most of the recent Deep Learning models achieve more than 99.5% accuracy on MNIST, so it can be interesting to evaluate our model on more challenging datasets. K-MNIST consists of 70,000 images (28x28, grayscale) in 10 classes, one for each row of the Japanese Hiragana. 

Unlike other datasets like Fashion-MNIST, we haven't found any other implementations involving capsules on K-MNIST, and we will be thus able to compare our CapsNet results with benchmarked models. Finally, given the nature of the dataset, capsules seem **instinctively** particularly suitable for the task.

<div align="center">
  <img src="https://github.com/rois-codh/kmnist/raw/master/images/kmnist_examples.png">
</div>
<div align="center">Kuzushiji-MNIST 10 classes</div>


### Content 

This repositorty contains :
- A Capsule's original paper explanation in French.
- Kuzushiji-MNIST data (~20mb only).
- Capsule Layer and CapsNet TF2 implementation.
- An IPython Notebook for training the model (this can be executed in Google Colab).

## Implementation details

Unlike many Deep Learning models, there is no built-in functions in libraries like TensorFlow or Pytorch to sequentially build a CapsNet architecture. Thus, our model implementation relies on [Xifeng Guo's repository](https://github.com/XifengGuo/CapsNet-Keras) which provides the lower-level TensorFlow 2 code needed to build the model.

In particular, ```capsulelayers.py```provides the CapsuleLayer class. A Capsule Layer is similar to a Dense Layer, except that it outputs a **vector** instead of a scalar. This is also where the inner-loop for **routing** mechanism between capsules takes place. It basically ensures that a capsule sends its output vector to higher level capsule, taking into account how big the scalar product between the two vectors is. Finally, since the length of the output vector should represent the probability that the feature represented by the capsule is included in the input, Capsule Layer uses **squashing** activation so that short vector tends to 0-length and long vectors tends to 1-length.

As in the paper, we will use the **Adam Optimizer** with its TensorFlow default parameters.

## Training Process

After preparing the data and building the model, we started training it. Training a CapsNet is a challenge in several ways :
- It involves new **tunable hyper-parameters** (number of routing iterations, number of capsules, importance of the reconstruction loss used for regularization) in addition to the *traditionnal* hyper-parameters (learning rate, batch size, number of training epochs).
- CapsNet's numerous trainable parameters and routing mechanism (that adds a for-loop in the process) make it a model that **takes time** to train, even on low resolution images and training on GPU doesn't seem to be an option.

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

