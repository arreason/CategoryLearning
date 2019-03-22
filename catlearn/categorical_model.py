#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:12:01 2018
@author: christophe_c
This file introduces the factories for creating the necessary cost functions
for the categorical model
"""

from itertools import chain
from types import MappingProxyType
from typing import (
    Any, Callable, Iterable, Mapping, Generic, TypeVar)

import torch

from catlearn.tensor_utils import (
    DEFAULT_EPSILON,
    subproba_kl_div,
    remap_subproba)
from catlearn.graph_utils import (
    CompositeArrow, CompositionGraph, DirectedGraph)
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


NodeType = TypeVar("NodeType")
ArrowType = TypeVar("ArrowType")


class RelationCache(
        Generic[NodeType, ArrowType],  # pylint: disable=unsubscriptable-object
        CompositionGraph[NodeType, ArrowType, torch.Tensor]):
    """
    A cache to keep the values of all relations
    """
    def __init__(
            self,
            generators: Mapping[
                ArrowType, Callable[
                    [torch.tensor, torch.tensor], torch.Tensor]],
            scorer: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
            comp: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            datas: Mapping[NodeType, torch.Tensor],
            arrows: Iterable[CompositeArrow[NodeType, ArrowType]],
            epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Initialize a new cache of relations, from:
           generators:  mapping whose:
               - keys are possible arrow names
               - values are functions taking a pair of source, target data
                   points as input and returning a relation value
           scorer:  a score function returning a score vector from 2 points
               and 1 relation value,
           comp: a composition operation returning a relation value from
               2 relation values (supposed associative)
        """
        self.epsilon = epsilon
        # base assumption: all arrows node supports should be in

        # save a tensor containing the causality cost and the number
        # of compositions being taken into account in it
        self._causality_cost = torch.zeros(())
        self._nb_compositions = 0

        # keep in memory the datas dictionary
        self._datas = dict(datas)

        # memorize existing arrow model names
        self.generators = generators

        def rel_comp(
                cache, arrow: CompositeArrow[NodeType, ArrowType]
            ) -> torch.Tensor:
            """
            compute the composite of all order 1 arrows, and account for
            causality cost
            """
            # case of length 1
            if len(arrow) == 1:
                rel_value = self.generators[arrow.arrows[0]](
                    cache.data(arrow[0:0]), cache.data(arrow[1:1]))
                rel_score = scorer(
                    cache.data(arrow[0:0]), cache.data(arrow[1:1]),
                    rel_value)
                return rel_score, rel_value

            # now take care of case of length >= 2
            # compute the value of the arrow by composition
            comp_value = comp(cache.data(arrow[:1]), cache.data(arrow[1:]))
            comp_scores = scorer(
                cache.data(arrow[0:0]), cache.data(arrow.op[0:0]), comp_value)
            comp_validity = comp_scores.sum()

            # recursively access all the scores of the subarrow splits
            # and check the composition score
            for idx in range(1, len(arrow)):

                # compute update to causality score
                fst_scores = cache[arrow[:idx]].sum()
                scd_scores = cache[arrow[idx:]].sum()
                causal_score = torch.relu(
                    torch.log(
                        (self.epsilon + comp_validity) /
                        (self.epsilon + fst_scores * scd_scores)))

                cache._causality_cost += causal_score
                cache._nb_compositions += 1

                # if causality update > 0., relation is not valid: 0. scores
                if causal_score > 0.:
                    comp_final_scores = torch.zeros(comp_scores.shape)
                else:
                    comp_final_scores = comp_scores

            return comp_final_scores, comp_value

        super().__init__(rel_comp, arrows)


    @property
    def causality_cost(self):
        """
        current value of causality cost on the whole cache
        """
        return self._causality_cost/max(self._nb_compositions, 1)

    def data(
            self, arrow: CompositeArrow[NodeType, ArrowType]
        ) -> torch.Tensor:
        """
        Takes as argument a composite arrow.
        Returns:
            - if length of the arrow is 0, the corresponding data point
            - if length of arrow >=1, the relation value
        """
        if arrow:
            return super().__getitem__(arrow)[1]

        # if arrow has length 0, return data corresponding to its node
        return self._datas[arrow[0]]  # type: ignore


    def __getitem__(
            self, arrow: CompositeArrow[NodeType, ArrowType]
        ) -> torch.Tensor:
        """
        Returns the value associated to an arrow.
        If the arrow has length 0 (contains only a node):
            returns 0.
        If the arrow has length >= 1:
            returns the score vector of the relation
        """
        if not arrow:
            raise ValueError("Cannot get the score of an arrow of length 0")
        else:
            return super().__getitem__(arrow)[0]

    def __setitem__(self, node: NodeType, data_point: torch.Tensor) -> None:
        """
        set value of a new data point.
        Note: cannot set the value of an arrow, as these are computed using
        relation generation and compositions.
        """
        self._datas[node] = data_point

    def add(self, arrow: CompositeArrow[NodeType, ArrowType]) -> None:
        """
        add composite arrow to the cache, starting by checking that data
        for all the points is available.
        """
        if not set(arrow.nodes) <= set(self._datas):
            raise ValueError(
                "Cannot add a composite whose support is not included"
                "in the base dataset")

        super().add(arrow)

    def flush(self) -> None:
        """
        Flush all relations and datas content
        """
        super().flush()
        self._datas = {}
        self._causality_cost = 0.
        self._nb_compositions = 0

    def match(self, labels: DirectedGraph[NodeType]) -> torch.Tensor:
        """
        Match the composition graph with a graph of labels. For each label
        vector, get the best match in the graph. if No match is found, set to
        + infinity.
        """
        result_graph = DirectedGraph[NodeType]()
        for src, tar in labels.edges:
            # add edge if necessary
            if not result_graph.has_edge(src, tar):
                result_graph.add_edge(src, tar)

            # go through labels and match them. Keep only the best
            for name, label in labels[src][tar].items():
                # check if arrows exist to match label
                try:
                    scores = {
                        arr.derive(): self[arr]
                        for arr in self.arrows(src, tar)}
                except KeyError:
                    scores = {}

                # if no score is available, evaluate all models
                if not scores:
                    scores = {
                        CompositeArrow(arr): self._comp(
                            CompositeArrow([src, tar], [arr]))[0]
                        for arr in self.generators}

                # evaluate candidate relationships
                candidates = {
                    arr: subproba_kl_div(score, label)
                    for arr, score in scores.items()}

                # save the best match in result graph
                best_match = min(candidates, key=candidates.get)
                result_graph[src][tar][name] = (
                    best_match, candidates[best_match])

        return result_graph


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
        algebra_model: the underlying algebra model of relation.
                See algebra_models module for several predefined algebras.
        scores: sequence of booleans which len indicates score output
                dimension. True if correspondig score index should be treated
                as an equivalance.
    """

    def __init__(
            self,
            relation_models: Mapping[ArrowType, RelationModel],
            scoring_model: ScoringModel,
            algebra_model: Algebra,
            epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Create a new instance of Decision categorical model
        """
        assert epsilon > 0., "epsilon should be strictly positive"

        self._relation_models = relation_models
        self._scoring_model = scoring_model
        self._algebra_model = algebra_model
        self.epsilon = epsilon

        # to store the number of inputs taken into account for both scores
        self._cost = torch.zeros(())

    @property
    def algebra(self) -> Algebra:
        """
        access the algebra of the decision model
        """
        return self._algebra_model

    @property
    def relations(self) -> Mapping[ArrowType, RelationModel]:
        """
        access the mapping to relation models
        """
        return MappingProxyType(self._relation_models)

    def score(
            self, source: torch.Tensor, target: torch.Tensor,
            relation: torch.Tensor) -> torch.Tensor:
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
            self, data_points: Mapping[NodeType, torch.Tensor],
            relations: Iterable[CompositeArrow[NodeType, ArrowType]]
        ) -> torch.Tensor:
        """
        generate a batch from a list of relations and datas
        """
        return RelationCache(
            self.relations, self.score,
            self.algebra.comp, data_points, relations)

    def cost(
            self,
            data_points: Mapping[NodeType, torch.Tensor],
            relations: Iterable[CompositeArrow[NodeType, ArrowType]],
            labels: DirectedGraph[NodeType]) -> torch.Tensor:
        """
        Generates the matching cost function on a relation, given labels
        inputs
        """
        # generate the relation cache and match against labels
        cache = self.generate_cache(data_points, relations)
        matched = cache.match(labels)

        total = sum(
            sum(elem for _, elem in costs.values())
            for costs in matched.edges.values())
        nb_labels = sum(len(costs) for costs in matched.edges.values())
        return total/max(nb_labels, 1) + cache.causality_cost


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
            relation_models: Mapping[ArrowType, RelationModel],
            scoring_model: ScoringModel,
            algebra_model: Algebra,
            optimizer: Callable,
            epsilon: float = DEFAULT_EPSILON,
            **kwargs) -> None:
        """
        instanciate a new model
        """
        DecisionCatModel.__init__(
            self, relation_models, scoring_model,
            algebra_model=algebra_model,
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
        return lambda: chain(*(rel.parameters()
                               for rel in self.relations.values()),
                             self._scoring_model.parameters())

    def freeze(self) -> None:
        """
        Freeze all adjustable weights (score and relations)
        """
        self._scoring_model.freeze()
        for rel in self.relations.values():
            rel.freeze()

    def unfreeze(self) -> None:
        """
        Inverse of freeze method
        """
        self._scoring_model.unfreeze()
        for rel in self.relations.values():
            rel.unfreeze()

    @property
    def total_cost(self) -> torch.tensor:
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
            data_points: Mapping[NodeType, torch.Tensor],
            relations: Iterable[CompositeArrow[NodeType, ArrowType]],
            labels: DirectedGraph[NodeType],
            step: bool = True) -> None:
        """
        perform one training step on a batch of tuples
        """
        # backprop on the batch
        self._cost += self.cost(data_points, relations, labels)

        if step:
            # backprop on cost
            self._cost.backward()

            # step optimizer
            self._optimizer.step()

            # reset gradient and total costs
            self.reset()
