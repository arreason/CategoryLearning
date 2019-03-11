#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:05:38 2018

@author: christophe_c

specialization of categorical models with full layer perceptrons
"""

from typing import Callable, Sequence
import torch
from torch import nn, optim

from catlearn.categorical_model import (
    RelationModel, ScoringModel,
    TrainableDecisionCatModel, DEFAULT_EPSILON)
from catlearn.algebra_models import Algebra
from catlearn.full_layer_models import FullPerceptron

ActivationFactory = Callable[[], nn.Module]


class ScoringPerceptron(FullPerceptron, ScoringModel):
    """
    A class for scoring model using full perceptron. Constructor takes as
    inputs:
        nb_features: number of features of a data point
        dim_rel: flat dimension of relations
        num_units: the number of units of each hidden layer
        nb_outputs: number of scores to output
        activation_factory: create new activation function,
                            separate foreach layer
    """

    def __init__(
            self,
            nb_features: int,
            dim_rel: int,
            num_units: Sequence[int],
            nb_outputs: int = 1,
            activation_factory: ActivationFactory = nn.Sigmoid
    ) -> None:
        """
        create new perceptron based scoring model
        """

        self._nb_features = nb_features
        self._relation_dim = dim_rel
        self._nb_outputs = nb_outputs

        # call parent init
        input_dim = 2 * nb_features + dim_rel
        FullPerceptron.__init__(self, [input_dim], [nb_outputs + 1],
                                num_units, activation_factory)

        self.layers.add_module('Softmax', nn.Softmax(dim=-1))

    @property
    def nb_features(self) -> int:
        """
        returns number of input features
        """
        return self._nb_features

    @property
    def relation_dim(self) -> int:
        """
        returns the flattened relation dimension
        """
        return self._relation_dim

    def forward(self,
                *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, expect 3 input tensors
        source, destination, relation.
        """
        assert len(inputs) == 3
        # concatenate inputs
        cat_input = torch.cat(inputs, -1)

        # call parent forward on cat input
        raw_output = FullPerceptron.forward(self, cat_input)

        # Final loose normalization
        return raw_output[..., :-1]


class RelationPerceptron(FullPerceptron, RelationModel):
    """
    A class for relation models using a full perceptron. Constructor takes as
    inputs:
        nb_features: the number of features of a data point
        dim_rel: the flat dimension of relations
        num_units: the number of units of each hidden layer
        activation_factory: create new activation function,
                            separate for each layer
    """

    def __init__(
            self,
            nb_features: int,
            dim_rel: int,
            num_units: Sequence[int],
            activation_factory: ActivationFactory = nn.Sigmoid
    ) -> None:
        """
        create new perceptron based relation model
        """

        self._nb_features = nb_features
        self._relation_dim = dim_rel

        # call parent init
        input_dim = 2 * nb_features
        FullPerceptron.__init__(self, [input_dim], [dim_rel], num_units,
                                activation_factory)

    @property
    def nb_features(self) -> int:
        """
        returns number of input features
        """
        return self._nb_features

    @property
    def relation_dim(self) -> int:
        """
        returns the flattened relation dimension
        """
        return self._relation_dim

    def forward(
            self,
            *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, expect 2 input tensors
        source, destination.
        """
        # concatenate inputs and call parent forward
        assert len(inputs) == 2
        cat_input = torch.cat(inputs, -1)
        return FullPerceptron.forward(self, cat_input)


class FullPerceptronDecisionModel(TrainableDecisionCatModel):
    """
    Specialization of DecisionCatModel using perceptrons for relations
    and scoring, and an algebra model from category.algebra_models for relation
    composition and unit
    inputs:
        nb_features: int, the dimension of the feature space
        nb_rels: int, the number of relation models
        algebra_model: an instance of a class built on the template
        category.algebra_models.Algebra (must implement unit and comp methods,
        as well as a flatdim property returning the dimension required for
        storing a relation)
        weights_prior: prior weights for matching function
        num_units_rel: number of units of hidden layers of relation models
        num_units_score: number of units of hidden layers of scoring model
        activation_factory: create new activation function,
                            separate for each layer
        optimizer: Gradient-descent update scheme
    """

    def __init__(
            self,
            nb_features: int,
            nb_rels: int,
            algebra_model: Algebra,
            weight_priors: torch.Tensor,
            num_units_rel: Sequence[int],
            num_units_score: Sequence[int],
            scores: Sequence[bool],
            activation_factory: ActivationFactory = nn.Sigmoid,
            optimizer: Callable = optim.Adam,
            epsilon: float = DEFAULT_EPSILON,
            **kwargs) -> None:
        """
        instanciate a new model
        """

        # register number of features as attributes
        self._nb_features = nb_features

        # create perecptrons for relation models
        relation_models = [
            RelationPerceptron(nb_features, algebra_model.flatdim,
                               num_units_rel, activation_factory)
            for _ in range(nb_rels)]

        # create the perceptron for scoring
        scoring_model = ScoringPerceptron(
            nb_features, algebra_model.flatdim,
            num_units_score, len(scores), activation_factory)

        # call parent init
        TrainableDecisionCatModel.__init__(
            self, relation_models, scoring_model,
            weight_priors=weight_priors,
            algebra_model=algebra_model,
            scores=scores,
            optimizer=optimizer,
            epsilon=epsilon, **kwargs)

    @property
    def nb_features(self) -> int:
        """
        returns number of input features
        """
        return self._nb_features

    @property
    def nb_scores(self) -> int:
        """
        returns dimensionality of scores
        """
        return self.score_dim
