#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:55:23 2018
@author: christophe_c
Factories to define the units and compositions of different R-algebras. These
operators should accept batched inputs, and only work on the last dimension
by default
"""
from functools import reduce
from operator import mul
import torch

from catlearn.tensor_utils import (
    repeat_tensor, zeros_like, ones_like, DEFAULT_EPSILON)


class Algebra:
    """
        This is a template with the minimum an algebra should implement.
    """
    @property
    def flatdim(self) -> int:
        """
        dimension of a representing vector
        """

        raise NotImplementedError("You need to define flatdim method when"
                                  "defining a new algebra model")

    @property
    def dim(self) -> int:
        """
        semantic dimension
        """

        raise NotImplementedError("You need to define dim method when"
                                  "defining a new algebra model")

    def unit(self, data_tuple: torch.Tensor) -> torch.Tensor:
        """
        generator of units in batch; deduce size of batch as
        data_tuple.shape[:-1]
        """
        assert data_tuple.ndimension() >= 1

        raise NotImplementedError("You need to define unit method when"
                                  "defining a new algebra model")

    def comp(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        composition of two algebra elements. return a third such element
        """
        assert data1.shape == data2.shape
        assert data1.ndimension() >= 1
        assert data1.shape[-1] == self.flatdim

        raise NotImplementedError("You need to define comp method when"
                                  "defining a new algebra model")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dim})"

    def equals(
            self,
            left: torch.Tensor,
            right: torch.Tensor,
            epsilon: float = DEFAULT_EPSILON) -> bool:
        """
        Default equality operator (expects float-castable tensors)
        """
        return ((left - right) < epsilon).all()


class VectAlgebra(Algebra):
    """
        A class to encapsulate a finite dimensional (possibly dependant on
        a vector input)
        R-vector space algebra:
            unit is 0-vector
            composition is vector sum
    """

    def __init__(self, dim: int) -> None:
        """
        Creates new instance of finite dimensional vector space algebra of a
        given dimension
        """
        assert dim >= 1, "Dimension should be strictly positive"

        self._dim: int = dim

    @property
    def dim(self) -> int:
        """
        returns the actual dimension of an element of the model
        """
        return self._dim

    @property
    def flatdim(self) -> int:
        """
        returns the actual dimension of an element of the model, once flattened
        """
        return self._dim

    def unit(self, data_tuple: torch.Tensor) -> torch.Tensor:
        """
        vector unit. Takes a batch as inputs, returns 0 vectors
        for each vector in the batch
        """
        assert data_tuple.ndimension() >= 1, "input should have ndim >= 1"

        return zeros_like(data_tuple, data_tuple.shape[:-1] + (self.dim,))

    def comp(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        vector sum. Takes two batches of same shape as inputs, returns their
        sum
        """
        assert data1.shape == data2.shape, "inputs should have the same shape"
        assert data1.ndimension() >= 1, "input should have ndim >= 1"
        assert data1.shape[-1] == self.dim, "inputs should match dim of model"

        return data1 + data2


class VectMultAlgebra(Algebra):
    """
        A class to encapsulate a finite dimensional (possibly dependant on
        a vector input)
        R-vector space algebra:
            unit is 1-vector
            composition is elementwise multiplication
    """

    def __init__(self, dim: int) -> None:
        """
        Creates new instance of finite dimensional vector space algebra of a
        given dimension
        """
        assert dim >= 1, "Dimension should be strictly positive"

        self._dim: int = dim

    @property
    def dim(self) -> int:
        """
        returns the actual dimension of an element of the model
        """
        return self._dim

    @property
    def flatdim(self) -> int:
        """
        returns the actual dimension of an element of the model, once flattened
        """
        return self._dim

    def unit(self, data_tuple: torch.Tensor) -> torch.Tensor:
        """
        vector unit. Takes a batch as inputs, returns 0 vectors
        for each vector in the batch
        """
        assert data_tuple.ndimension() >= 1, "input should have ndim >= 1"

        return ones_like(data_tuple, data_tuple.shape[:-1] + (self.dim,))

    def comp(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        vector sum. Takes two batches of same shape as inputs, returns their
        sum
        """
        assert data1.shape == data2.shape, "inputs should have the same shape"
        assert data1.ndimension() >= 1, "input should have ndim >= 1"
        assert data1.shape[-1] == self.dim, "inputs should match dim of model"

        return data1 * data2


class MatrixAlgebra(Algebra):
    """
    A class to encapsulate unit and composition of square matrix algebra in a
    given dimension.
    Constructor inputs:
        dim: int; the dimension of the square matrices
    """

    def __init__(self, dim: int) -> None:
        """
        new instance of matrix algebra, from wanted dimension
        """
        assert dim >= 0, "Dimension should be strictly positive"

        self._dim: int = dim

    @property
    def dim(self) -> int:
        """
        returns the actual dimension of an element of the model
        """
        return self._dim

    @property
    def flatdim(self) -> int:
        """
        returns the actual dimension of an element of the model, once flattened
        """
        return self._dim * self._dim

    def viewmat(self, vect: torch.Tensor) -> torch.Tensor:
        """
        returns a view of a batch of flattened matrix as an actual batch of
        matrices of the right shape
        input: torch.Tensor of shape batch_shape + (dim * dim)
        output: torch.Tensor of shape batch_shape + (dim, dim)
        """
        assert vect.ndimension() >= 1, "input should have ndim >= 1"
        assert vect.shape[-1] == self.flatdim, "input should match dim of model"
        batch_shape = vect.shape[:-1]

        return vect.view(batch_shape + (self.dim, self.dim))

    def unit(self, data_tuple: torch.Tensor) -> torch.Tensor:
        """
        matrix unit. Takes a batch of points as input, returns a batch
        of unit matrices (reshaped in vectors) of the same shape
        inputs:
            data_tuple: a batch of vectors of size: batch_shape + (last_dim,)
        outputs:
            a batch of matrices with size batch_shape + (dim * dim,)
        """
        assert data_tuple.ndimension() >= 1, "input should have ndim >= 1"
        batch_shape = data_tuple.shape[:-1]
        n_points = reduce(mul, batch_shape, 1)

        # create a tensor n_points unit matrices
        flat_id = repeat_tensor(torch.eye(self.dim,
                                          dtype=data_tuple.dtype,
                                          device=data_tuple.device),
                                n_points,
                                axis=0)

        # reshape and return
        return flat_id.view(batch_shape + (self.flatdim,))

    def comp(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        definition of matrix composition, depending on dimension. Takes two
        batches of flattened square matrices of same shape,
        returns their batch product (flattened again)
        inputs size: batch_shape + (dim * dim,)
        output size: batch_shape + (dim * dim,)
        """
        assert data1.shape == data2.shape, "inputs should have the same shape"
        assert data1.ndimension() >= 1, "inputs should have ndim >= 1"
        assert data1.shape[-1] == self.flatdim, "inputs should match dim of model"

        # shape of the batch
        batch_shape = data1.shape[:-1]

        # view as flat batches of square matrices
        matrix1 = data1.view([-1, self.dim, self.dim])
        matrix2 = data2.view([-1, self.dim, self.dim])

        # batch product
        prod_matrix = torch.bmm(matrix1, matrix2)

        # reshape for return
        return prod_matrix.view(batch_shape + (self.flatdim,))


class AffineAlgebra(Algebra):
    """
    A class to encapsulate unit and composition of the real affine monoid of a
    given dimension (semi-direct product of square matrices of given dim acting
    on vectors of the same dim).
    The layout is matrix first (size: dim * dim), then vector (size: dim)
    Constructor inputs:
        dim: int; the dimension of the square matrices and the vectors
    """

    def __init__(self, dim: int) -> None:
        """
        creates a new instance of real affine monoid model,
        of a given dimension
        """
        assert dim >= 1, "Dimension should be strictly positive"

        self._dim: int = dim

    @property
    def dim(self) -> int:
        """
        returns the actual dimension of an element of the model
        """
        return self._dim

    @property
    def flatdim(self) -> int:
        """
        returns the actual dimension required to store one element of the model
        in its is flattened state: (1 + dim) * dim
        """
        return (1 + self._dim) * self._dim

    def viewmat(self, aff: torch.Tensor) -> torch.Tensor:
        """
        taking a batch of flattened (mat, vector) couples as inputs, returns
        a view of the corresponding batch of matrices
        """
        assert aff.ndimension() >= 1, "input should have ndim >= 1"
        assert aff.shape[-1] == self.flatdim, "input should match dim of model"

        # get batch schape
        batch_shape = aff.shape[:-1]

        # flatt view of matrices
        flat_mat = aff[..., :(self.dim * self.dim)]

        return flat_mat.view(batch_shape + (self.dim, self.dim))

    def viewvect(self, aff: torch.Tensor) -> torch.Tensor:
        """
        taking a batch of flattened (mat, vector) couples as inputs, returns
        a view of the corresponding batch of vectors as column vectors (last
        dim is 1)
        """
        assert aff.ndimension() >= 1, "input should have ndim >= 1"
        assert aff.shape[-1] == self.flatdim, "input should match dim of model"

        return aff[..., (self.dim * self.dim):, None]

    def unit(self, data_tuple: torch.Tensor) -> torch.Tensor:
        """
        affine unit (id matrix, 0 vector). Takes a batch of points as input,
        returns a batch of unit matrices (reshaped in vectors)
        of the same shape
        inputs:
            data_tuple: a batch of vectors of any size
        outputs:
            a batch of matrix-vector couples with the same size
            (except last dim), flattened
        """
        assert data_tuple.ndimension() >= 1, "Input should have ndim >= 1"

        # shape and size of batch
        batch_shape = data_tuple.shape[:-1]
        n_points = reduce(mul, batch_shape, 1)

        # create a single unit, then repeat it
        single_unit = torch.cat((torch.eye(self.dim,
                                           dtype=data_tuple.dtype,
                                           device=data_tuple.device).view([-1]),
                                 zeros_like(data_tuple, (self.dim,))))
        flat_unit = repeat_tensor(single_unit, n_points, axis=0)

        # reshape for return
        return flat_unit.view(batch_shape + (self.flatdim,))

    def comp(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        returns composition of two affine elements, that is matrix-vector
        couples (M1, v1), (M2, v2), given by: (M1@M2, M1@v2 + v1)
        """
        assert data1.shape == data2.shape, "inputs should have the same shape"
        assert data1.ndimension() >= 1, "inputs should have ndim >= 1"
        assert data1.shape[-1] == self.flatdim, "inputs should match model dim"

        # get batch shape and size
        n_points = reduce(mul, data1.shape, 1) // self.flatdim

        # flatten batches
        flat1 = data1.view([n_points, self.flatdim])
        flat2 = data2.view([n_points, self.flatdim])

        # get a view of all corresponding batches of matrices and vectors
        mat1 = self.viewmat(flat1)
        vect1 = self.viewvect(flat1)

        mat2 = self.viewmat(flat2)
        vect2 = self.viewvect(flat2)

        # compute result matrix and vector parts
        result_mat = torch.bmm(mat1, mat2)
        result_vect = torch.bmm(mat1, vect2) + vect1

        # flatten and concatenate both parts to get the output
        result = torch.cat(
            (result_mat.view([n_points, self.dim * self.dim]),
             result_vect.view([n_points, self.dim])),
            -1
        )

        # reshape and return
        return result.view(data1.shape)
