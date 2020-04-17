"""
This file introduces the factories for creating the necessary cost functions
for the categorical model
"""

from __future__ import annotations
from itertools import chain
from typing import Any, Callable, IO, Iterable, Mapping, Tuple, Union

import torch
from torch.optim import Optimizer

from catlearn.tensor_utils import (
    DEFAULT_EPSILON, Tsor, remap_subproba)
from catlearn.graph_utils import DirectedGraph, NodeType
from catlearn.composition_graph import CompositeArrow, ArrowType
from catlearn.relation_cache import RelationCache
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
                 src: Tsor,
                 dst: Tsor,
                 rel: ArrowType) -> Tsor:
        raise NotImplementedError()

class ScoringModel(AbstractModel):
    def __call__(self,
                 src: Tsor,
                 dst: Tsor,
                 rel: Tsor) -> Tsor:
        raise NotImplementedError()

# pylint: enable=missing-docstring

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
        relation_model: a model taking 2 datapoints and an encoded relation label,
                returning a relation vector
        label_universe: mapping from the set of possible relation label to
                a suitable form for relation_model
        scoring_model: a single scoring model (Callable), taking a couple of
                datapoints and a relation between them, and returning
                confidence scores. Each entry in output tensor should be in
                [0, 1] and the sum over scores must be <= 1
        algebra_model: the underlying algebra model of relation.
                See algebra_models module for several predefined algebras.
        scores: sequence of booleans which len indicates score output
                dimension. True if correspondig score index should be treated
                as an equivalance.
    """

    def __init__(
            self,
            relation_model: RelationModel,
            label_universe: Mapping[ArrowType, Tsor],
            scoring_model: ScoringModel,
            algebra_model: Algebra,
            epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Create a new instance of Decision categorical model
        """
        assert epsilon > 0., "epsilon should be strictly positive"

        self._relation_model = relation_model
        self._label_universe = label_universe
        self._scoring_model = scoring_model
        self._algebra_model = algebra_model
        self.epsilon = epsilon

    @property
    def algebra(self) -> Algebra:
        """
        access the algebra of the decision model
        """
        return self._algebra_model

    def score(
            self, source: Tsor, target: Tsor,
            relation: Tsor) -> Tsor:
        """
        compute score of a relation batch, making sure that the score of
        an identity sums to 1
        """
        assert source.shape == target.shape
        assert source.shape[:-1] == relation.shape[:-1]

        reference = self._scoring_model(
            source, source, self.algebra.unit(source))
        main_score = self._scoring_model(
            source, target, relation)
        return remap_subproba(main_score, reference)

    def generate_cache(
            self, data_points: Mapping[NodeType, Tsor],
            relations: Iterable[CompositeArrow[NodeType, ArrowType]]
        ) -> Tsor:
        """
        generate a batch from a list of relations and datas
        """
        return RelationCache(
            self._relation_model, self._label_universe,
            self.score, self.algebra.comp, data_points, relations)

    def cost(
            self,
            data_points: Mapping[NodeType, Tsor],
            relations: Iterable[CompositeArrow[NodeType, ArrowType]],
            labels: DirectedGraph[NodeType]) -> Tuple[
                Tsor,
                RelationCache[NodeType, ArrowType],
                DirectedGraph[NodeType]]:
        """
        Generates the matching cost function on a relation, given labels
        inputs.
        Returns:
            - the total matching cost
            - the relation cache that was computed
            - the label mathching
        """
        # generate the relation cache and match against labels
        cache = self.generate_cache(data_points, relations)
        matched = cache.match(labels)

        total = sum(
            sum(elem for _, elem in costs.values())
            for costs in matched.edges.values())
        nb_labels = sum(len(costs) for costs in matched.edges.values())
        return (
            total/max(nb_labels, 1) + cache.causality_cost,
            cache, matched)

    def save(self, flike: Union[str, IO]):
        """
        Save the model to a given location (path or file-like object)
        """
        torch.save(self, flike)

    @staticmethod
    def load(flike: Union[str, IO]) -> DecisionCatModel:
        """
        Load a model from a given location (path or file-like object)
        """
        return torch.load(flike)





class TrainableDecisionCatModel(DecisionCatModel):
    """
    Specialization of DecisionCatModel working with trainable
    torch modules.

    Constructor takes as input:
        relation_model: a model taking 2 datapoints and an encoded relation label,
                returning a relation vector
        label_universe: mapping from the set of possible relation label to
               a suitable form for relation_model
        scoring_model: a single scoring model (Callable), taking a couple of
                datapoints and a relation between them, and returning
                confidence scores. Each entry in output tensor should be in
                [0, 1] and the sum over scores must be <= 1
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
            relation_model: RelationModel,
            label_universe: Mapping[ArrowType, Tsor],
            scoring_model: ScoringModel,
            algebra_model: Algebra,
            optimizer: Optimizer,
            epsilon: float = DEFAULT_EPSILON,
            **kwargs) -> None:
        """
        instanciate a new model
        """
        DecisionCatModel.__init__(
            self, relation_model, label_universe,
            scoring_model, algebra_model=algebra_model,
            epsilon=epsilon)

        self._cost = torch.zeros(())

        # register optimizer with remaining arguments to the constructor
        self._optimizer = optimizer(self.parameters(), **kwargs)

    @property
    def relation_dim(self) -> int:
        """
        returns the dimension required to store one relation using
        the underlying algebra model
        """
        return self.algebra.flatdim

    @property
    def parameters(self) -> Callable[[], Iterable[Any]]:
        """
        returns an iterator over parameters of the model
        """
        return lambda: chain(self._relation_model.parameters(),
                             self._scoring_model.parameters())

    def freeze(self) -> None:
        """
        Freeze all adjustable weights (score and relations)
        """
        self._scoring_model.freeze()
        self._relation_model.freeze()

    def unfreeze(self) -> None:
        """
        Inverse of freeze method
        """
        self._scoring_model.unfreeze()
        self._relation_model.unfreeze()

    @property
    def total_cost(self) -> Tsor:
        """
        Get total cost currently stored
        """
        return self._cost

    def reset(self) -> None:
        """
        reset gradient values and total cost stored
        """
        self._optimizer.zero_grad()
        self._cost = torch.zeros(())

    def train(
            self,
            data_points: Mapping[NodeType, Tsor],
            relations: Iterable[CompositeArrow[NodeType, ArrowType]],
            labels: DirectedGraph[NodeType],
            step: bool = True) -> Tuple[
                RelationCache[NodeType, ArrowType], DirectedGraph[NodeType]]:
        """
        perform one training step on a batch of tuples
        """
        # backprop on the batch
        cost, cache, matched = self.cost(data_points, relations, labels)
        self._cost = self._cost + cost

        if step:
            # backprop on cost
            self._cost.backward()

            # step optimizer
            self._optimizer.step()

            # reset gradient and total costs
            self.reset()

        return cache, matched

    def save(self, flike: Union[str, IO]):
        """
        Save the model to a given location (path or file-like object)
        """
        torch.save(self, flike)

    @staticmethod
    def load(flike: Union[str, IO]) -> TrainableDecisionCatModel:
        """
        Load a model from a given location (path or file-like object)
        """
        return torch.load(flike)
