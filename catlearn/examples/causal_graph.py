#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name

""" Toy exmaple tailored for decision model

Sample a small graph exposing different kinds
of causal dependencies.

The goal is given iid graph samples to infer
causal relationship between the Xs aka features.
"""

from typing import Sequence, Tuple, Generator
from functools import reduce
from operator import mul
import numpy
import torch


def _draw_batch_tuple_indices(
        num_examples: int,
        transition_mat: numpy.ndarray,
        tuple_size: int) -> torch.Tensor:
    """
    draw a batch of positive examples given an adjacency matrix
    inputs:
        num_examples: int, size of the batch
        adj_mat: numpy.array, square matrix,
                transition matrix of the causal graph
        tuple_size: int, the size of the tuples to draw
    outputs:
        a tensor of integers of shape (num_examples, tuple_size)
    """
    assert tuple_size >= 1, "Tuple size should be at least 1"

    # to get subsequent indices, given previous indices
    def choice_gen(indices: numpy.ndarray) -> Generator[int, None, None]:
        """
        Given an array of integer indices, returns for each one a randomly
        drawn index from the possibilities left by the transition matrix
        inputs:
            indices: numpy.ndarray, size (nb_examples,)
        outputs:
            numpy.ndarray; size (nb_examples,)
        """
        for i in indices:
            yield numpy.random.choice(transition_mat.shape[0], 1,
                                      p=transition_mat[i].numpy())[0]

    def rec_choice(
            indices: Tuple[torch.Tensor, ...],
            order: int) -> Tuple[torch.Tensor, ...]:
        """
        Given the first choices, get the next choices compatible with the
        transition matrix until reaching order
        inputs:
            indices: Sequence[numpy.ndarray], the choices which were already
                    made
            order: int, the number of choices which remain to be made - 1
        outputs:
            Sequence[numpy.ndarray], a sequence of arrays containing all the
            random choices at each order
        """
        if order == 0:
            return indices
        return (rec_choice(indices, order-1)
                + (torch.IntTensor(
                    list(choice_gen(indices[-1]))),))

    # first indices
    firsts = torch.randint(low=0, high=transition_mat.shape[0],
                           size=(num_examples,),
                           dtype=torch.int)

    # get next indices
    all_indices = rec_choice((firsts,), tuple_size-1)

    return torch.stack(all_indices, -1)


class CausalGraphDataset:
    """
    Template class for causal graph datasets. Should contain:
        - a property for returning an adjacency matrix
        - function to generate a dataset
    """

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """ Graph adjacency matrix """
        raise NotImplementedError(
            "You need to define the dataset adjacency matrix property"
            "when sublcassing CausalGraphDataset")

    @property
    def transition_matrix(self) -> torch.FloatTensor:
        """
        Return a row-normalized version of the adjacency matrix
            (cast to float)
        Using a cast to numpy arrays because of torch apparent bug on some
        tensor shape broadcastings.
        """

        adj_mat = self.adjacency_matrix.to(torch.float).numpy()
        return torch.from_numpy(
            adj_mat / adj_mat.sum(axis=1)[:, numpy.newaxis])

    @property
    def data(self) -> torch.Tensor:
        """ Get underlying data """
        raise NotImplementedError(
            "You need to define the data property when subclassing"
            "CausalGraphDataset")

    @property
    def shape(self) -> Tuple[int]:
        """
        The shape of the dataset
        """
        return self.data.shape

    @property
    def dim(self) -> int:
        """
        The dimension of datapoints of the dataset
        """
        return self.adjacency_matrix.shape[0]

    @property
    def batch_shape(self) -> Tuple[int]:
        """
        The shape of the batch
        """
        return self.data.shape[:-1]

    @property
    def ndim(self) -> int:
        """
        the number of dimensions of the dataset
        """
        return len(self.shape)

    @property
    def batch_ndim(self) -> int:
        """
        the number of dimensions of the batch shape
        """
        return len(self.batch_shape)

    @property
    def numel(self) -> int:
        """
        the number of elements in the batch
        """
        return reduce(mul, self.shape, 1)

    @property
    def numexamples(self) -> int:
        """
        The number of examples in the batch
        """
        return reduce(mul, self.batch_shape, 1)

    def __getitem__(self, *args, **kwargs) -> torch.Tensor:
        """
        Item access: accesses the underlying items of the data attribute
        """
        return self.data.__getitem__(self, *args, **kwargs)

    def draw_positive_batch(self, tuple_size: int) -> torch.Tensor:
        """
        Draw a batch of matching relation tuples of given size.
        inputs:
            tuple_size: int, the size of the positive tuples
        oututs:
            torch.Tensor, of shape self.batch_shape + (tuple_size,)
        """
        indices = _draw_batch_tuple_indices(
            self.numexamples, self.transition_matrix, tuple_size)

        data = self.data.view(self.numexamples, self.dim)

        batch = torch.FloatTensor(
            [[data[i, p] for p in indices[i]]
             for i in range(self.numexamples)])

        return batch.view(self.batch_shape + (tuple_size,))

    def draw_random_batch(self, tuple_size: int) -> torch.Tensor:
        """
        Draw a batch of tuples at random, without regards for matching
        relations.
        inputs:
            tuple_size: int, the size of the positive tuples
        ouputs
            torch.Tensor, of shape self.batch_shape + (tuple_size,)
        """

        indices = torch.randint(low=0, high=self.dim,
                                size=(self.numexamples, tuple_size),
                                dtype=torch.int)

        data = self.data.view(self.numexamples, self.dim)

        batch = torch.FloatTensor(
            [[data[i, p] for p in indices[i]]
             for i in range(self.numexamples)])
        return batch.view(self.batch_shape + (tuple_size,))

    def draw_balanced_batch(self, tuple_size: int) -> torch.Tensor:
        """
        Draw a batch of tuples, half positive, half at random
        inputs:
            tuple_size: int, the size of the positive tuples
        outputs:
            torch.Tensor, of shape (self.batch_shape) +  (2, tuple_size)
        """
        # draw positive and random examples
        positive_batch = self.draw_positive_batch(tuple_size)
        negative_batch = self.draw_random_batch(tuple_size)

        # concatenate
        batch = torch.stack((positive_batch, negative_batch), dim=-1)

        nb_examples = positive_batch.shape[:-1]
        labels = torch.stack((torch.ones(nb_examples),
                              - torch.ones(nb_examples)), dim=-1)

        return batch, labels


class Example1(CausalGraphDataset):
    """
    Our first example. Generates iid samples of causal graph with given
    batch shape.
    Current graph:

    X0 ~ N(0, 1)
    X1 ~ U(0, 1)
    X2 ~ U(0, 1)
    X3 ~ Bernoulli(X1 * X2)
    X4 ~ Bernoulli(X2)
    X5 = 1 - X4

    X0       X1    X2
             |    / |
             |   /  |
             v  /   v
             X3<    X4
                    ^
                    |
                    v
                     X5

    Input:
        bath_shape: sequence of desired axis shape
    Output:
        tensor of samples of shape (batch_shape) + (dim=6,)
    """

    # the adjacency matrix of the dataset; constant
    _ADJACENCY_MATRIX = numpy.array(
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 1]])

    @property
    def adjacency_matrix(self):
        """
        Get the adjacency matrix of the dataset
        """
        return torch.from_numpy(Example1._ADJACENCY_MATRIX)

    @property
    def data(self):
        return self.__data

    def __init__(self, batch_shape: Sequence[int]) -> None:
        """
        Create new instance of this example
        """
        shape = tuple(batch_shape) + (6,)

        # define the data contained in the dataset
        data = torch.empty(shape)
        data[..., 0].normal_(0., 1.)
        data[..., 1].uniform_(0., 1.)
        data[..., 2].uniform_(0., 1.)
        data[..., 3].bernoulli_(data[..., 1] * data[..., 2])
        data[..., 4].bernoulli_(data[..., 2])
        data[..., 5] = 1. - data[..., 4]

        # register data attribute
        self.__data = data


if __name__ == '__main__':
    # command line tool to generate and persist dataset
    # Output CSV style.

    import pandas
    import argparse

    parser = argparse.ArgumentParser(description="Generate iid samples from a "
                                     "small causal graph")
    parser.add_argument('n_samples', type=int,
                        help="number of iid samples to draw")
    parser.add_argument('output_csv', type=str,
                        help="CSV output filename")
    args = parser.parse_args()

    dataset = Example1((args.n_samples,))
    df = pandas.DataFrame(dataset.data.numpy())
    df.index.name = "sample"
    df.to_csv(args.output_csv)
