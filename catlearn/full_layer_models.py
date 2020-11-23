#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file introduces factories to define full perceptron models
"""

from __future__ import annotations
from functools import reduce
from itertools import chain
from operator import mul
from typing import Any, Callable, Sequence, Tuple, Iterable, Union, IO
import torch
import torch.nn as nn

from catlearn.tensor_utils import Tsor, AbstractModel


class ConstantModel(nn.Module, AbstractModel):
    """ Encapsulate a non-parametrized transform """

    def __init__(self, func: Callable, name: str = 'ConstantModel') -> None:
        nn.Module.__init__(self)
        self._forward = func
        self.name = name

    def __repr__(self) -> str:
        """
        Readable representation

        We must do so because func might not be readable,
        and if it part of nn modules or sub package, can get tricky.
        """
        return self.name

    def forward(self, *inputs: Tsor) -> Tsor:
        """ Forward override """
        assert len(inputs) == 1
        return self._forward(inputs[0])

    def named_parameters(self) -> Iterable[Tuple[str, Any]]:
        return iter(())

    def freeze(self) -> Tsor:
        pass

    def unfreeze(self) -> Tsor:
        pass


class LayeredModel(nn.Module):
    """
    A generic class to represent layered models.
    Takes batch of tensors as inputs and outputs
    """
    def __init__(
            self,
            input_shape: Sequence[int],
            output_shape: Sequence[int],
            layers: Iterable[Callable]) -> None:

        # parent init
        nn.Module.__init__(self)

        # register inputs as parameters
        self._input_shape = torch.Size(input_shape)
        self._output_shape = torch.Size(output_shape)

        # Uses Sequential here for easy human readable
        # representation, reuse of forward definition.
        self.layers = nn.Sequential(*list(layers))

    @property
    def input_shape(self) -> torch.Size:
        """ Get shape of expected input tensor

        """
        return self._input_shape

    @property
    def input_ndim(self) -> int:
        """ Get number of dimensions of expected input tensor

        """
        return len(self.input_shape)

    @property
    def input_numel(self) -> int:
        """ Get number of elements of expected input tensor

        """
        return reduce(mul, self.input_shape, 1)

    @property
    def output_shape(self) -> torch.Size:
        """ Get expected output tnsor shape

        """
        return self._output_shape

    @property
    def output_ndim(self) -> int:
        """
        Get expected number of dimensions of output tensor
        """
        return len(self.output_shape)

    @property
    def output_numel(self) -> int:
        """ Get expected number of elements of output tensor

        """
        return reduce(mul, self.output_shape, 1)

    @property
    def children(self) -> Callable[[], Iterable[nn.Module]]:
        """ Gather children models of parent

        """
        return lambda: self.layers.children()

    @property
    def named_parameters(self) -> Callable[[], Iterable[Any]]:
        """ Function to gather all parameters of the model: go through layer and
        check if each layer has an attribute "parameters"
        """
        return lambda: self.layers.named_parameters()

    def forward(self, *inputs: Tsor) -> Tsor:
        """ Forward pass of Layered Model """
        assert len(inputs) == 1
        input_batch = inputs[0]
        assert input_batch.ndimension() >= self.input_ndim
        assert input_batch.shape[-self.input_ndim:] == self.input_shape
        return self.layers.forward(input_batch)

    def _set_requires_grad_flag(self, value: bool) -> None:
        """ Set requires_grad flag on all parameters of the model

        """
        for layer in self.parameters():
            layer.requires_grad_(value)

    def freeze(self) -> None:
        """ Freeze parameters of the model

        After this call, parameters will not be
        affected by torch.backward pass.
        Parameters are NOT frozen by default.
        """
        return self._set_requires_grad_flag(False)

    def unfreeze(self) -> None:
        """ Inverse of model freeze

        After this call, parameters can
        be optimized again.
        """
        return self._set_requires_grad_flag(True)


def couple_layered_models(
        *models: LayeredModel) -> LayeredModel:
    """
    Concatenates a sequence of Layered models into a single,
    unified one, resulting in a LayeredModel with all the layers.

    Model order is respected, and they must all be compatible,
    ie outshape shape of previous module is equal to input shape
    of the next one.
    """
    assert len(models) >= 2
    assert all(m1.output_shape == m2.input_shape
               for m1, m2 in zip(models, models[1:]))

    # list of layers is concatenation of lists of layers of the two models
    layers = chain(* map(lambda m: m.layers, models))

    # input shape inherited from first model, output shape from second model
    input_shape = models[0].input_shape
    output_shape = models[-1].output_shape

    return LayeredModel(input_shape, output_shape, layers)


def reduce_model(model: LayeredModel) -> LayeredModel:
    """
    Takes a layered model as input and compose all its layers to a single one
    """
    return LayeredModel(model.input_shape, model.output_shape, [model])


class FullPerceptron(LayeredModel):
    """
    Creates a full perceptron model with Tensor shaped input and output
    Constructor takes as arguments:
        input_shape: shape of 1 input (Tuple)
        output_shape: shape of 1 output (Tuple)
        num_units: lists of numbers of units of intermediary layers
        activation_factory: create new activation function,
        separate for each layer
    """
    def __init__(
            self,
            input_shape: Sequence[int],
            output_shape: Sequence[int],
            num_units: Sequence[int],
            activation_factory: Callable[[], nn.Module]) -> None:
        """
        Create a new multilayer perceptron, taking arbitrary size inputs
        and returning arbitrary size outputs
        """

        # register input and output shapes as attributes
        self._input_shape = torch.Size(input_shape)
        self._output_shape = torch.Size(output_shape)

        # get list of units for a flat perceptron model and create this model
        internal_modules = self._get_layer_sequence(
            num_units, activation_factory)
        modules = chain(
            [ConstantModel(
                self._view_input,
                f"Flatten {tuple(self._input_shape)}")],
            internal_modules,
            [ConstantModel(
                self._view_output,
                f"Reshape {tuple(self._output_shape)}")])
        LayeredModel.__init__(self, input_shape, output_shape, modules)

    def _get_layer_sequence(
            self,
            num_units: Sequence[int],
            activation_factory: Callable[[], nn.Module]
    ) -> Iterable[nn.Module]:
        """ Generate the sequence of modules to use """
        # total number of inputs and outputs
        input_numel = reduce(mul, self._input_shape, 1)
        output_numel = reduce(mul, self._output_shape, 1)
        all_units = [input_numel] + list(num_units) + [output_numel]
        internals = (
            (nn.Linear(in_shape, out_shape, bias=True),
             activation_factory())
            for in_shape, out_shape in zip(all_units, all_units[1:-1]))
        last_layer = nn.Linear(all_units[-2], all_units[-1], bias=True)
        return chain(*internals, [last_layer])

    def _view_input(self, batch_input: Tsor) -> Tsor:
        """
        A function to reshape batch for input in the underlying
        perceptron model, by flattening the input shape
        inputs:
            batch_input: Tsor, last dims are self.input_shape
        outputs:
            Tsor, view of input with last self.input_ndim
            flattened to a singledimension
        """
        assert batch_input.ndimension() >= self.input_ndim
        assert batch_input.shape[-self.input_ndim:] == self.input_shape

        batch_shape = batch_input.shape[:-self.input_ndim]
        return batch_input.view(batch_shape + (-1,))

    def _view_output(self, batch_output: Tsor) -> Tsor:
        """
        A function to reshape batch out of the underlying perceptron model
        for output, respecting the instance's output_shape
        inputs:
            batch_input: Tsor, last dim is self.output_numel
        outputs:
            Tsor, last dim is recast as self.output_shape
        """
        assert batch_output.ndimension() >= 1
        assert batch_output.shape[-1] == self.output_numel

        batch_shape = batch_output.shape[:-1]
        return batch_output.view(batch_shape + self.output_shape)

    def save(self, flike: Union[str, IO]):
        """
        Save the model to a given location (path or file-like object)
        """
        torch.save(self, flike)

    @staticmethod
    def load(flike: Union[str, IO]) -> FullPerceptron:
        """
        Load a model from a given location (path or file-like object)
        """
        return torch.load(flike)


class FullPerceptronWithRandomState(LayeredModel):
    """
    create a full layer perceptron which augments its input with a random state
    Constructor takes as input:
        input_shape: shape of 1 input (Tuple)
        output_shape: shape of output (Tuple)
        num_units: lists of numbers of units of intermediary layers
        activation_factory: create new activation function,
                            separate for each layer
        num_random: the number of random states
        random generator: random generator to use; default as torch.rand
    """
    def __init__(self,
                 input_shape: Sequence[int],
                 output_shape: Sequence[int],
                 num_units: Sequence[int],
                 num_random: int,
                 activation_factory: Callable[[], nn.Module],
                 random_generator: Callable = torch.rand) -> None:
        """
        Creates a new instance of perceptron model with random state
        """
        assert num_random >= 1, "if no randomization, use FullPerceptron class"

        # register parameters for random draws
        self.num_random = num_random
        self.random = random_generator

        # register input and output shapes as attributes
        self._input_shape = torch.Size(input_shape)
        self._output_shape = torch.Size(output_shape)

        # get number of inputs, including random state
        num_inputs = num_random + reduce(mul, input_shape, 1)

        # create a perceptron model without random state
        deterministic_model = FullPerceptron(
            (num_inputs,), output_shape, num_units, activation_factory)

        # add input augmentation at beginning of ops list and call parent init
        op_list = chain([ConstantModel(self.augment_input, f"RandomAugment")],
                        deterministic_model.children())
        LayeredModel.__init__(self, input_shape, output_shape, op_list)

    def draw_state(self, batch_shape: Tuple[int, ...]) -> Tsor:
        """
        draw a batch of state vectors for input in the underlying perceptron
        model
        inputs:
            batch_shape:  Tuple of integers, shape of input
        """
        return self.random(batch_shape + (self.num_random,))

    def augment_input(self, batch_input: Tsor) -> Tsor:
        """
        flatten inputs in the batch and concatenate it with a randomly drawn
        state
        """
        assert batch_input.shape[-self.input_ndim:] == self.input_shape

        batch_shape = batch_input.shape[:-self.input_ndim]
        flat_input = batch_input.view(batch_shape + (-1,))
        random_state = self.draw_state(batch_shape)

        return torch.cat((flat_input, random_state), -1)

    def save(self, flike: Union[str, IO]):
        """
        Save the model to a given location (path or file-like object)
        """
        torch.save(self, flike)

    @staticmethod
    def load(flike: Union[str, IO]) -> FullPerceptronWithRandomState:
        """
        Load a model from a given location (path or file-like object)
        """
        return torch.load(flike)
