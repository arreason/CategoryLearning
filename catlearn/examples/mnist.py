#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gather all code pertaining dataset generation for
classification on MNIST dataset

MNIST consists in recognizing digit associated to a 28x28 grayscale image
Link: http://yann.lecun.com/exdb/mnist/
"""

from typing import Callable, Optional
import torch
from torchvision import datasets, transforms

MNIST_PIXEL_MEAN = 0.1307
MNIST_PIXEL_STD = 0.3081


class AugmentedMNIST(torch.utils.data.Dataset):
    """
    Data augmentation on MNIST dataset

    Augmentation is performed by adding an extra
    dimension to the desired batch size.

    Each original input is a 1x28x28 images. For a batch size D
    and @nb_augment L this will result in a Dx(L+1)x1x28x28 tensor

    Params:
    - train (bool) -> training or test set
    - path: (str) -> folder to store downloaded (auto download)
                     original MNIST data
    - nb_augment: (int) -> number of random data augmentation to perform
                           on this input
    - augmentation_transform: (None or Callable) ->
                              what randomized transformations to apply
    """

    MNIST_PIXEL_MEAN = 0.1307
    MNIST_PIXEL_STD = 0.3081

    def __init__(
            self,
            train: bool,
            path: str,
            nb_augment: int = 0,
            augmentation_transform: Optional[Callable] = None) -> None:
        self.mnist_data = datasets.MNIST(path, download=True, train=train)
        self.nb_augment = nb_augment
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),  # convert image to PyTorch tensor
            transforms.Normalize((MNIST_PIXEL_MEAN,),
                                 (MNIST_PIXEL_STD,))  # normalize inputs
            ])

        if augmentation_transform is None:
            augmentation_transform = lambda x: x
        self.apply_transform = transforms.Compose([
            augmentation_transform, self.normalize_transform])

    def __len__(self):
        """
        Length of the augmented dataset
        """
        return len(self.mnist_data)

    def __getitem__(self, idx):
        """
        Retrieve a specific items sequence from the dataset
        """
        unmodified_img, target = self.mnist_data[idx]

        return (torch.stack(
            [self.normalize_transform(unmodified_img)] +
            [self.apply_transform(unmodified_img)
             for _ in range(self.nb_augment)]),
                target)


class AffineAugmentedMNIST(AugmentedMNIST):
    """
    Apply a random affine transformation to MNIST input images

    For more information on the parameters, refer to AugmentedMNIST
    or torchvision.transforms.RandomAffine.
    """
    def __init__(
            self,
            train: bool,
            path: str,
            nb_augment: int,
            degrees=0,
            translate=None,
            scale=None,
            shear=None) -> None:
        random_affine = transforms.RandomAffine(
            degrees, translate, scale, shear)
        AugmentedMNIST.__init__(self, train, path, nb_augment, random_affine)
