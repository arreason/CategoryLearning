#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""
Created on Fri May 11 12:49:40 2018

@author: christophe_c

tests for full layer models
"""

import random
from typing import List, Tuple
import pytest
import torch
from torch import nn

from tests.test_tools import random_int_list
from catlearn.full_layer_models import (
    ConstantModel,
    LayeredModel,
    FullPerceptron,
    FullPerceptronWithRandomState,
    couple_layered_models,
    reduce_model)

TensorShape = Tuple[int, ...]


@pytest.fixture(params=[(1, 3, 1, 5)])
def batch_shape(request) -> TensorShape:
    """ Input batch shape """
    return tuple(random_int_list(*request.param)())


@pytest.fixture(params=[(1, 3, 8, 16)])
def input_shape(request) -> TensorShape:
    """ Shape of a single input """
    return tuple(random_int_list(*request.param)())


@pytest.fixture(params=[(1, 3, 8, 16)])
def output_shape(request) -> TensorShape:
    """ Shape of a single output """
    return tuple(random_int_list(*request.param)())


@pytest.fixture(params=[(1, 4, 2, 32)])
def num_units(request) -> List[int]:
    """ Number of hidden units per layer """
    return list(random_int_list(*request.param)())


@pytest.fixture(params=[(1, 32)])
def num_random(request) -> int:
    """ Random state extension """
    return random.randint(*request.param)


def test_constant_model() -> None:
    """ Test ConstantModel creation and usage """
    cons = ConstantModel(lambda x: x**2, "square")
    assert str(cons) == "square"
    assert cons.forward(2.0) == 4.0
    assert list(cons.parameters()) == []


class TestLayeredModel:
    """ Test creation and manipulation of LayereModel

    """
    @staticmethod
    def get_layered_model(nlayer):
        """ Create a leyered model with given number of
        identical ConstantMoel layers
        """
        consts = [ConstantModel(lambda x: 2 * x, "double")
                  for _ in range(nlayer)]
        return LayeredModel((1,), (1,), consts)

    def test_creation(self) -> None:
        """ Test creation of a layered model """
        nlayer = 4
        layered = self.get_layered_model(nlayer)
        assert all(isinstance(c, ConstantModel)
                   for c in layered.children())
        assert len(list(layered.children())) == nlayer
        # Self, Sequential and 4 "double"
        assert len(list(layered.modules())) == 2 + nlayer
        assert list(layered.parameters()) == []
        # pylint: disable=not-callable
        expected = torch.tensor((2**nlayer,), dtype=torch.float32)
        assert layered.forward(torch.ones(1)) == expected

    def test_reduce_model(self) -> None:
        """ Test reduce_model helper """
        nlayer = 5
        layered = self.get_layered_model(nlayer)
        reduced = reduce_model(layered)

        assert len(reduced.layers) == 1
        assert len(list(layered.children())) == nlayer
        assert layered.forward(torch.ones(1)) \
            == reduced.forward(torch.ones(1))

    def test_couple_models(self) -> None:
        """ Test couple_layered_models helper """
        l1 = self.get_layered_model(4)
        l2 = self.get_layered_model(3)
        l3 = self.get_layered_model(5)
        coupled = couple_layered_models(l1, l2, l3)
        expected = self.get_layered_model(4 + 3 + 5)

        assert len(coupled.layers) == len(expected.layers)
        assert len(list(coupled.children())) \
            == len(list(expected.children()))
        assert coupled.forward(torch.ones(1)) \
            == expected.forward(torch.ones(1))


class TestFullPerceptron:
    """
    Test facility for the base FullPerceptron class
    """

    @staticmethod
    def get_model(
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int]) -> FullPerceptron:
        """ Construct Torch module """
        return FullPerceptron(input_shape, output_shape, num_units, nn.Sigmoid)

    def test_output_shape(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int],
            batch_shape: TensorShape) -> None:
        """
        draw random input and verify output has the right shape
        """
        model = self.get_model(input_shape, output_shape, num_units)
        input_batch = torch.rand(batch_shape + input_shape)
        output_batch = model.forward(input_batch)

        expected_shape = batch_shape + output_shape
        assert output_batch.shape == expected_shape

    def test_params_list(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int]) -> None:
        """
        test that the parameters' list has the right size
        """
        model = self.get_model(input_shape, output_shape, num_units)
        parameters = list(model.parameters())
        # Input reshape, (Linear, Sigmoid), output reshape
        assert len(parameters) == 1 + 2 * len(num_units) + 1

    def test_children_list(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int]) -> None:
        """
        test that the parameters' list has the right size
        """
        model = self.get_model(input_shape, output_shape, num_units)
        children = list(model.children())
        # Initial and final reshape
        assert isinstance(children[0], ConstantModel)
        assert isinstance(children[-1], ConstantModel)
        # Alternating Linear -> Sigmoid motif
        for i, child in enumerate(children[1:-1]):
            if i % 2 == 0:
                expected_type = nn.Linear
            else:
                expected_type = nn.Sigmoid
            assert isinstance(child, expected_type), \
                f"Type mismatch on layer {1+i}: {expected_type}"


class TestFullPerceptronWithRandomState:
    """
    Test facility for perceptron model with random state
    """
    RNG_CHECK_ORDER = 4

    @staticmethod
    def get_model(
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int],
            num_random: int) -> FullPerceptronWithRandomState:
        """ Construct Torch module """
        return FullPerceptronWithRandomState(input_shape,
                                             output_shape,
                                             num_units,
                                             num_random,
                                             nn.Sigmoid)

    def test_draw_state(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int],
            num_random: int,
            batch_shape: TensorShape) -> None:
        """
        unit test random drawing
        """
        model = self.get_model(input_shape, output_shape,
                               num_units, num_random)
        base = model.draw_state(batch_shape)
        for i in range(1, self.RNG_CHECK_ORDER):
            assert (base != model.draw_state(batch_shape)).all(), i

    def test_input_augment(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int],
            num_random: int,
            batch_shape: TensorShape) -> None:
        """
        draw random input, verify that augmented input is random
        """
        model = self.get_model(input_shape, output_shape,
                               num_units, num_random)
        batch_input = torch.rand(batch_shape + input_shape)
        base_augmented_input = model.augment_input(batch_input)

        # verify that subsequent draws are different
        for i in range(1, self.RNG_CHECK_ORDER):
            new_draw = model.augment_input(batch_input)
            # verify deterministic part is unchanged
            assert (base_augmented_input == new_draw).any(), i
            # verify random part is actually changing
            assert not (base_augmented_input == new_draw).all(), i

    def test_output_shape(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int],
            num_random: int,
            batch_shape: TensorShape) -> None:
        """
        draw random input and verify output has the right shape
        """
        model = self.get_model(input_shape, output_shape,
                               num_units, num_random)
        input_batch = torch.rand(batch_shape + input_shape)
        output_batch = model.forward(input_batch)

        expected_shape = batch_shape + output_shape
        assert output_batch.shape == expected_shape

    def test_params_list(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int],
            num_random: int) -> None:
        """
        check that the parameters' list has the right shape
        """
        model = self.get_model(input_shape, output_shape,
                               num_units, num_random)
        parameters = list(model.parameters())
        assert len(parameters) == 2 * (len(num_units) + 1)

    def test_children_list(
            self,
            input_shape: TensorShape,
            output_shape: TensorShape,
            num_units: List[int],
            num_random: int) -> None:
        """
        test that the parameters' list has the right size
        """
        model = self.get_model(input_shape, output_shape,
                               num_units, num_random)
        children = list(model.children())
        # Input augmentation and flattening
        assert isinstance(children[0], ConstantModel)
        assert str(children[0]) == "RandomAugment"
        # FullPerceptron Initial reshape
        # redundant since augment_input flattens the input
        assert isinstance(children[1], ConstantModel)
        # Alternating Linear -> Sigmoid motif
        for i, child in enumerate(children[2:-1]):
            if i % 2 == 0:
                expected_type = nn.Linear
            else:
                expected_type = nn.Sigmoid
            assert isinstance(child, expected_type), \
                f"Type mismatch on layer {1+i}: {expected_type}"
        # Final reshape
        assert isinstance(children[-1], ConstantModel)
