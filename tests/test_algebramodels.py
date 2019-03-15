#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""
tests for algebra models
"""

import inspect
from typing import Any, Tuple
import pytest
import torch

from catlearn import algebra_models
from tests.test_tools import random_int_list

# List of algebras to verify
# Automagic, adding an algebra will get tested right away
CLASSES_TO_TEST = list(
    filter(
        lambda x: (inspect.isclass(x) and
                   x != algebra_models.Algebra and
                   issubclass(x, algebra_models.Algebra)),
        map(
            lambda x: x[1],
            inspect.getmembers(algebra_models)
        )
    )
)


@pytest.fixture(params=[1, 3])
def ndim(request: Any) -> int:
    """ Albegra dimension """
    return request.param


@pytest.fixture(params=CLASSES_TO_TEST)
def algebra(request: Any, ndim: int) -> algebra_models.Algebra:
    """ Instanciate an algebra """
    return request.param(ndim)  # Instanciate a model


@pytest.fixture(params=[1, 4])
def batch_ndim(request: Any) -> int:
    """ Number of batch dimensions """
    return request.param


@pytest.fixture(params=[(1, 20)])
def batch_shape(request: Any, batch_ndim: int) -> Tuple[int, ...]:
    """ Innput batch shape """
    return tuple(random_int_list(batch_ndim,
                                 batch_ndim,
                                 *request.param)())


@pytest.fixture(params=[1, 4])
def data_point(request: Any, batch_shape: Tuple[int, ...]) -> torch.Tensor:
    """ Sample data point from feature space """
    return torch.rand(batch_shape + (request.param,))


def sample_elem(algebra: algebra_models.Algebra,
                batch_shape: Tuple[int, ...]) -> torch.Tensor:
    """ Sample Algebra element """
    return torch.rand(batch_shape + (algebra.flatdim,))


class TestAlgebra:
    """ Regroup all tests pertaining to an algebra structure

    """

    @staticmethod
    def test_unit_shape(
            algebra: algebra_models.Algebra,
            data_point: torch.Tensor,
            batch_shape: torch.Tensor) -> None:
        """
        test that unit has the appropriate size, when generated from random
        feature vector
        """
        units = algebra.unit(data_point)
        expected = batch_shape + (algebra.flatdim,)
        assert units.shape == expected

    @staticmethod
    def test_comp_shape(
            algebra: algebra_models.Algebra,
            batch_shape: torch.Tensor) -> None:
        """
        test that composition has the appropriate size, for two randomly
        generated batches
        """
        first_elem = sample_elem(algebra, batch_shape)
        last_elem = sample_elem(algebra, batch_shape)
        composite = algebra.comp(first_elem, last_elem)
        expected = batch_shape + (algebra.flatdim,)
        assert composite.shape == expected

    @staticmethod
    def test_unit_left(
            algebra: algebra_models.Algebra,
            data_point: torch.Tensor,
            batch_shape: torch.Tensor) -> None:
        """
        test that multiplication of a random batch by a batch of units
        on the left does not change the batch value
        """
        elems = sample_elem(algebra, batch_shape)
        unit = algebra.unit(data_point)
        composite = algebra.comp(unit, elems)
        assert algebra.equals(composite, elems)

    @staticmethod
    def test_unit_right(
            algebra: algebra_models.Algebra,
            data_point: torch.Tensor,
            batch_shape: torch.Tensor) -> None:
        """
        test that multiplication of a random batch by a batch of units
        on the left does not change the batch value
        """
        elems = sample_elem(algebra, batch_shape)
        unit = algebra.unit(data_point)
        composite = algebra.comp(elems, unit)
        assert algebra.equals(composite, elems)

    @staticmethod
    def test_associativity(
            algebra: algebra_models.Algebra) -> None:
        """
        test algebra implementations are actually associative
        """
        batch_shape = (3, 100)
        elems = sample_elem(algebra, batch_shape)
        left = algebra.comp(
            algebra.comp(elems[0], elems[1]),
            elems[2])
        right = algebra.comp(
            elems[0],
            algebra.comp(elems[1], elems[2]))
        assert algebra.equals(left, right)
