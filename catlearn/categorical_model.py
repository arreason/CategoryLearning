#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:12:01 2018
@author: christophe_c
This file introduces the factories for creating the necessary cost functions
for the categorical model
"""

from functools import reduce
from itertools import chain
from typing import (
    Any, Callable, Iterable, NamedTuple, Sequence, Tuple, Hashable, Mapping)

import torch

from catlearn.tensor_utils import (
    DEFAULT_EPSILON,
    repeat_tensor,
    zeros_like,
    subproba_kl_div,
    remap_subproba)
from catlearn.algebra_models import Algebra


# Abstract type definitions
# NB: docstring temporarily disabled
# pylint: disable=missing-docstring
class AbstractModel:
    @property
    def parameters(self) -> Callable[[], Iterable[Any]]:
        raise NotImplementedError()

    def freeze(self) -> None:
        raise NotImplementedError()

    def unfreeze(self) -> None:
        raise NotImplementedError()


class RelationModel(AbstractModel):
    def __call__(self,
                 src: torch.Tensor,
                 dst: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class ScoringModel(AbstractModel):
    def __call__(self,
                 src: torch.Tensor,
                 dst: torch.Tensor,
                 rel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
# pylint: enable=missing-docstring


class DecisionCatModelCosts(NamedTuple):
    """ Tracks the different costs returned when
    by training DecisionCatModel
    """
    unit: torch.Tensor
    associativity: torch.Tensor
    matching: torch.Tensor
    lambda_unit: float
    lambda_associativity: float
    matching_weights: torch.Tensor


class DecisionCatModel:
    """
    A class to abstract a decision categorical model
    Feed a collection of relation models and a scoring model to it in order to
    get the necessary cost functions. Note that features are expected to be
    vectorlike, as well as relations, thus you need to cast things accordingly
    (for ex if datapoints are images, or relations are matrices).
    Note that algebra_model.comp should work directly on batches of arbitrary
    shapes, thus beware when designing your problem
    (tip: use view and einsum for matrix multiplication on the 2 last indices
    if you want to model your relations as matrices)
    !!!
    algebra_model.comp should support array broadcasting
    on batch dimensions for everything to work fine
    !!!
    Constructor takes as input:
        relation_models: a collection of relation models (Sequence[Callable]),
                taking a couple of 2 datapoints and returning a relation vector
        scoring_model: a single scoring model (Callable), taking a couple of
                datapoints and a relation between them, and returning
                confidence scores. Each entry in output tensor should be in
                [0, 1] and the sum over scores must be <= 1
        weight_priors: the weights used for sorted sum reductions
                This is a rating of the relationships according to their
                respective scores, and serves as a bayesian selection prior
                of the models
        algebra_model: the underlying algebra model of relation.
                See algebra_models module for several predefined algebras.
        scores: sequence of booleans which len indicates score output
                dimension. True if correspondig score index should be treated
                as an equivalance.
    """

    def __init__(
            self,
            relation_models: Mapping[Hashable, RelationModel],
            scoring_model: ScoringModel,
            weight_priors: torch.Tensor,
            algebra_model: Algebra,
            scores: Sequence[bool],
            epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Create a new instance of Decision categorical model
        """
        assert epsilon > 0., "epsilon should be strictly positive"
        assert weight_priors.ndimension() == 1, (
            "weight_priors should be a 1-dimensional array")
        assert (weight_priors >= 0).all(), (
            "weight_priors should be positive values only")
        total_weight = torch.sum(weight_priors)
        assert total_weight > 0, "Weight_priors should have non zero values"

        self._relation_models = tuple(relation_models)
        self._scoring_model = scoring_model
        self.weight_priors = weight_priors / total_weight
        self.algebra_model = algebra_model
        self.score_dim = len(scores)
        self.equivalence_indices = [i for i, eq in enumerate(scores) if eq]
        self.nb_equivalences = len(self.equivalence_indices)
        self.epsilon = epsilon

    def score(
            self, src: torch.Tensor, tar: torch.Tensor,
            rel: torch.Tensor) -> torch.Tensor:
        """
        apply scoring to a batch of relations.
        arguments:
            src: sources of the relations of the batch
            tar: targets of the relations of the batch
            rel: relation values in algebra of the batch
        """
        assert src.shape[:-1] == tar.shape[:-1]
        assert rel.shape[:-1] == src.shape[:-1]

        batch_shape = src.shape[:-1]

        return remap_subproba(
                self._scoring_model(src, tar, rel),
                self._scoring_model(
                    src, src, self.algebra_model.unit(batch_shape)))

    def relation_model(self, name: Hashable) -> RelationModel:
        """
        access the relation model with the given name.
        """
        return self._relation_model[name]

    @property
    def nb_relation_models(self) -> int:
        """
        The number of relation models of this instance of DecisionCatModel
        """
        return len(self._relation_models)

    def generate_relations(
            self, data_points: torch.Tensor,
            relations: Sequence[Hashable]) -> torch.Tensor:
        """
        generate the array of 1-order relations corresponding to the given
        relation models
        """
        assert data_points.ndimension() >= 2
        assert data_points.shape[-2] == len(relations)

        relations = (
            self.relation_model(relations[idx])(
                data_points[..., idx, :], data_points[..., 1 + idx, :])
            for idx in range(len(relations)))
        return torch.stack(relations, dim=-2)

    def get_composites(
            self, data_points: torch.Tensor,
            relations: Sequence[Hashable]) -> Tuple[torch.Tensor]:
        """
        Get 2 lists:
            - the first consists of successive composites starting from
            first point
            - the second consists of successive composites starting from
            last point (reversed)
        """
        # get relations
        relations = self.generate_relations(data_points, relations)

        # get subsequent composites, direct and opposite direction
        direct_composites = zeros_like(relations, relations.shape)
        opposite_composites = zeros_like(relations, relations.shape)

        direct_composites[..., 0, :] = relations[..., 0, :]
        opposite_composites[..., -1, :] = relations[..., -1, :]

        for idx in range(1, len(relations)):
            direct_composites[idx] = self.algebra_model.comp(
                direct_composites[..., idx-1, :], relations[..., idx, :])
            opposite_composites[-idx - 1] = self.algebra_model.comp(
                relations[..., -idx - 1, :], opposite_composites[..., -idx, :])

        return direct_composites, opposite_composites

    def compute_causality(
            self, data_points: torch.Tensor,
            relations: Sequence[Hashable]) -> torch.Tensor:
        """
        returns the causality cost corresponding to relations
        """
        # get composites
        direct_composites, opposite_composites = self.get_composites(
            data_points, relations)

        # compute causality scores
        direct_scores = direct_composites[..., :-1, :].sum(dim=-1)
        opposite_scores = opposite_composites[..., :-1, :].sum(dim=-1)
        composite_scores = direct_composites[..., -1, :].sum(dim=-1)

        return torch.relu(
            torch.log(composite_scores[..., None]/(
                direct_scores * opposite_scores + self.epsilon))).sum(dim=-1)

    def get_relation(
            self,
            data_points: torch.Tensor,
            relations: Sequence[Hashable]) -> torch.Tensor:
        """
        generate the value of the relation from source to tar
        """
        first_orders = self.generate_relations(data_points, relations)
        return reduce(self.algebra_model.comp, first_orders)

    def matching_cost(
            self,
            data_points: torch.Tensor,
            relations: Sequence[Hashable],
            labels: torch.Tensor) -> torch.Tensor:
        """
        Generates the matching cost function on a relation, given labels
        inputs:
             data_points: a sequence of datapoints
             relations: a sequence of corresponding relation model names
        outputs:
            the matching score for the labels (
                    from first to last of sequence of poitns)
        """
        assert data_points.shape[:-2] == labels.shape[:-1]
        assert len(relations) + 1 == data_points.shape[-2]

        relation = self.get_relation(data_points, relations)

        # get individual scores of relations
        scores = self.scoring_model(
            data_points[..., 0, :], data_points[..., 1, :], relation)

        # KL_div(labels || scores), including implicit negative class
        kl_divs = subproba_kl_div(scores, labels, epsilon=self.epsilon)

        # return the corresponding matching score
        return kl_divs


class TrainableDecisionCatModel(DecisionCatModel):
    """
    Specialization of DecisionCatModel working with trainable
    torch modules.

    Constructor takes as input:
        relation_models: a collection of relation models (Sequence[Callable]),
                taking a couple of 2 datapoints and returning a relation vector
        scoring_model: a single scoring model (Callable), taking a couple of
                datapoints and a relation between them, and returning
                confidence scores. Each entry in output tensor should be in
                [0, 1] and the sum over scores must be <= 1
        weight_priors: the weights used for sorted sum reductions
                This is a rating of the relationships according to their
                respective scores, and serves as a bayesian selection prior
                of the models
        algebra_model: the underlying algebra model of relation.
                       See algebra_models
                module for several predefined algebras.
        scores: sequence of booleans which len indicates score output
                dimension. True if correspondig score index should be treated
                as an equivalance.
        optimizer: Gradient-descent update scheme
        epsilon: division offset, to avoid divison by 0
    """

    def __init__(
            self,
            relation_models: Sequence[RelationModel],
            scoring_model: ScoringModel,
            weight_priors: torch.Tensor,
            algebra_model: Algebra,
            scores: Sequence[bool],
            optimizer: Callable,
            epsilon: float = DEFAULT_EPSILON,
            **kwargs) -> None:
        """
        instanciate a new model
        """
        DecisionCatModel.__init__(
            self, relation_models, scoring_model,
            weight_priors=weight_priors,
            algebra_model=algebra_model,
            scores=scores,
            epsilon=epsilon)

        # register optimizer with remaining arguments to the constructor
        self._optimizer = optimizer(self.parameters(), **kwargs)

    @property
    def relation_dim(self) -> int:
        """
        returns the dimension required to store one relation using
        the underlying algebra model
        """
        return self.algebra_model.flatdim

    @property
    def parameters(self) -> Callable[[], Iterable[Any]]:
        """
        returns an iterator over parameters of the model
        """
        return lambda: chain(*(rel.parameters()
                               for rel in self.relation_models),
                             self.scoring_model.parameters())

    def freeze(self) -> None:
        """
        Freeze all adjustable weights (score and relations)
        """
        self.scoring_model.freeze()
        for rel in self.relation_models:
            rel.freeze()

    def unfreeze(self) -> None:
        """
        Inverse of freeze method
        """
        self.scoring_model.unfreeze()
        for rel in self.relation_models:
            rel.unfreeze()

    def train(
            self,
            tuple_batch: torch.Tensor,
            order: int,
            matching_weights: torch.Tensor = torch.ones([]),
            lambda_associativity: float = 1.,
            lambda_unit: float = 1.
    ) -> DecisionCatModelCosts:
        """
        perform one training step on a batch of tuples
        """
        # reset gradient
        self._optimizer.zero_grad()

        # get all costs
        costs = self.cost(
            tuple_batch,
            order,
            lambda_associativity,
            lambda_unit,
            matching_weights)

        total_cost = DecisionCatModel.get_total_cost(costs)

        # backprop on mean of total costs
        total_cost.mean().backward()

        # step optimizer
        self._optimizer.step()

        return costs

    def evaluate(
            self,
            batch_to_eval: torch.Tensor,
            order: int) -> torch.Tensor:
        """
        Evaluate potential labelling of an input
        batch of data tuples.

        Params:
        - batch_to_eval: a [...] x N_tuple x N_features tensor
         where N_tuple is the length of a given data points tuple.
        - order: maximum composition order to use for matching

        Returns:
        Eval: [...] x (N_scores+1) tensor where the last dimension
        is the matching cost associated with the following
        tentative labelling:
        - Eval[..., 0] -> No match ie labels(j) = 0
        - Eval[..., i] -> labels(j) = (i == j)
        It is a cost, so lower is better, and each entry is positive.
        """
        assert order >= 1
        assert batch_to_eval.ndimension() >= 2

        # Enhance data input tensor
        # Each data tuple should appear once
        # for each potential labelling
        # From D: [...] x N_tuple x N_features data tensor
        # to D': [...] x (N_scores+1) x N_tuple x N_features tensor
        broadcasted_eval_batch = repeat_tensor(
            batch_to_eval, self.score_dim + 1, axis=-3)

        # Creating a L: [...] x (N_scores+1) x N_scores label tensor
        one_point_labels = torch.cat([
            torch.zeros((1, self.score_dim)),
            torch.eye(self.score_dim)])
        labels_shape = (1, 1, self.score_dim + 1, self.score_dim)
        tentative_labels = (torch.ones(batch_to_eval.shape[:-2] + (1, 1))
                            * one_point_labels.reshape(labels_shape))

        # Needs reshape as it may introduce a leading
        # 1-sized dimension
        return self.matching_cost(
            broadcasted_eval_batch, tentative_labels, order).reshape(
                batch_to_eval.shape[:-2] + (self.score_dim + 1,))
