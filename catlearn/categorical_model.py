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
    Any, Callable, Iterable, Union, Mapping,
    NamedTuple, Sequence, Tuple, Hashable, Mapping, TypeVar)

import torch

from catlearn.tensor_utils import (
    DEFAULT_EPSILON,
    zeros_like,
    subproba_kl_div,
    remap_subproba)
from catlearn.graph_utils import CompositeArrow, CompositionGraph
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
AlgebraType = TypeVar("AlgebraType")
DataType = TypeVar("DataType")


class RelationCache(
        Generic[DataType],
        CompositionGraph[NodeType, ArrowType, AlgebraType]):
    """
    A cache to keep the values of all relations
    """
    def __init__(
            self,
            generators: Mapping[
                ArrowType,
                Callable[[DataType, Datatype], AlgebraType]],
            scorer: Callable[[DataType, DataType, AlgebraType], torch.Tensor],
            comp: Callable[[AlgebraType, AlgebraType], AlgebraType],
            datas: Mapping[NodeType, torch.Tensor]) -> None:
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

        # base assumption: all arrows node supports should be in

        # save a tensor containing the causality cost and the number
        # of compositions being taken into account in it
        self._causality_cost = torch.zeros(())
        self._nb_compositions = 0

        # keep in memory the datas dictionary
        self._datas = datas

        def rel_gen(
                self, src: NodeType, tar: NodeType, arr: ArrowType
                ) -> AlgebraType:
            """
            Generate a new first order relation from 2 data points
            """
            # compute relation, then its score and return both as
            # (score, rel) tuple
            rel_value = generator[arr](
                self._datas[fst_node], self._datas[scd_node])
            rel_score = scorer(
                self._datas[fst_node], self._datas[scd_node], rel_value)
            return rel_score, rel_value

        def rel_comp(
                self, arrow: CompositeArrow[NodeType, ArrowType]
                ) -> AlgebraType:
            """
            compute the composite of all order 1 arrows, and account for
            causality cost
            """
            # compute the value of the arrow
            comp_value = comp(self[arrow[:1]], self[arrow[1:]])
            comp_scores = scorer(self[arrow[0]], self[arrow[-1]], comp_value)
            comp_validity = comp_scores.sum()

            # recursively access all the scores of the subarrow splits
            # and check the composition score
            comp_final_score = comp_scores.clone()
            for idx in range(1, len(arrow)):

                # compute update to causality score
                fst_scores = self[arrow[:idx]][0]
                scd_scores = self[arrow[idx:]][0]
                causal_score = torch.relu(
                    torch.log(
                        comp_scores.sum()/(fst_scores * scd_scores).sum()))
                self._causality_cost += causal_score

                # if causality update > 0., relation is not valid: 0. scores
                if causal_score > 0.:
                    comp_final_scores[:] = 0.

            return comp_final_scores, comp_value

        super().__init__(rel_gen, rel_comp)

    def __getitem__(
            self, arrow: CompositeArrow[ArrowType, NodeType]
            ) -> Union[DataType, Tuple[torch.Tensor, AlgebraType]]:
        """
        Returns the value associated to an arrow.
        If the arrow has length 0 (contains only a node):
            returns the data associated to the point
        If the arrow has length >= 1:
            returns the tuple (score, relation value)
        """
        if len(arrow) == 0:
            return self._datas[arrow[0]]
        else:
            return super().__getitem__(arrow)

    def __setitem__(self, node: NodeType, data_point: DataType) -> None:
        """
        set value of a new data point.
        Note: cannot set the value of an arrow, as these are computed using
        relation generation and compositions.
        """
        self._data[node] = data_point

    def add(self, arrow: Composite[NodeType, ArrowType]) -> None:
        """
        add composite arrow to the cache, starting by checking that data
        for all the points is available.
        """
        if not set(arrow.nodes) <= set(self._dict):
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
            algebra_model: Algebra,
            epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Create a new instance of Decision categorical model
        """
        assert epsilon > 0., "epsilon should be strictly positive"

        self._relation_models = relation_models
        self._scoring_model = scoring_model
        self.algebra_model = algebra_model
        self.epsilon = epsilon

        # tensors to store the causality and matching total scores
        self._total_causal_cost = torch.Tensor(0.)
        self._total_matching_cost = torch.Tensor(0.)

        # for storing relation cache
        self._relation_cache = {}

        # to store the number of inputs taken into account for both scores
        self._nb_causal = 0
        self._nb_matching = 0

    def reset_causal(self) -> None:
        """
        reset causality score counting
        """
        self._total_causal_cost = 0.
        self._nb_causal = 0

    def reset_matching(self) -> None:
        """
        reset matching score counting
        """
        self._total_matching_cost = 0.
        self._nb_matching = 0

    def total_cost(self) -> None:
        """
        get total cost currently recorded
        """
        return (
            self._total_causal_cost/self._nb_causal
            + self._total_matching_cost/self._nb_matching)

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

    def add_relations(
            self,
            datas: Mapping[Hashable, torch.Tensor],
            relations: Sequence[CompositeArrow[Hashable, Hashable]]
            ) -> Tuple[torch.Tensor]:
        """
        Get 2 lists:
            - the first consists of successive composites starting from
            first point
            - the second consists of successive composites starting from
            last point (reversed)
        """
        # add datas to cache
        self._relation_cache.update(datas)

        # get relations
        relations = for self.

        return direct_composites, opposite_composites

    def causal_cost(
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

        causal = torch.relu(
            torch.log(composite_scores[..., None]/(
                direct_scores * opposite_scores + self.epsilon))).mean(dim=-1)

        self._total_causal_cost += causal.sum()
        self._nb_causal += causal.numel()

        return causal

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
        matching = subproba_kl_div(scores, labels, epsilon=self.epsilon)

        self._total_matching_cost += matching.sum()
        self._nb_matching += matching.numel()


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
            relation_models: Mapping[Hashable, RelationModel],
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
                               for rel in self._relation_models.values()),
                             self.scoring_model.parameters())

    def freeze(self) -> None:
        """
        Freeze all adjustable weights (score and relations)
        """
        self.scoring_model.freeze()
        for rel in self._relation_models.values():
            rel.freeze()

    def unfreeze(self) -> None:
        """
        Inverse of freeze method
        """
        self.scoring_model.unfreeze()
        for rel in self._relation_models.values():
            rel.unfreeze()

    def train(
            self,
            tuple_batch: torch.Tensor,
            order: int,
            matching_weights: torch.Tensor = torch.ones([]),
            lambda_associativity: float = 1.,
            lambda_unit: float = 1.):
        """
        perform one training step on a batch of tuples
        """
        # backprop on mean of total costs
        total_cost = self.total_cost()
        total_cost.backward()

        # step optimizer
        self._optimizer.step()

        # reset gradient and total costs
        self._optimizer.zero_grad()
        self.reset_causal()
        self.reset_matching()

        return total_cost
