#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name, invalid-name, abstract-method, no-self-use
"""
Tests for the categorical_model file.
"""
from typing import Any, Dict, Iterable, List, Callable
from functools import reduce
from itertools import combinations, product
from math import inf
import random

import torch
from torch import nn

import pytest

from catlearn.tensor_utils import subproba_kl_div, DEFAULT_EPSILON, Tsor
from catlearn.graph_utils import (
    DirectedGraph, CompositeArrow)
from catlearn.categorical_model import (
    RelationModel, ScoringModel, RelationCache,
    DecisionCatModel, TrainableDecisionCatModel)
from catlearn.algebra_models import (
    Algebra, VectAlgebra, MatrixAlgebra, AffineAlgebra)

from tests.test_tools import pytest_generate_tests
# List of algebras to verify
# Automagic, adding an algebra will get tested right away
CLASSES_TO_TEST = {VectAlgebra, MatrixAlgebra, AffineAlgebra}


# path to the data of the generators used for synthetic datasets
DATA_DIR = "./tests/test_categorical_model/"


FLAKY_TEST_RETRY = 3


@pytest.fixture(params=[1, 8])
def nb_features(request: Any) -> int:
    """ Feature space dimension """
    return request.param


@pytest.fixture(params=[1, 3])
def nb_scores(request: Any) -> int:
    """ Score vector dimension"""
    return request.param


@pytest.fixture(params=[1, 3])
def nb_relations(request: Any) -> int:
    """ number of relations in the model"""
    return request.param


@pytest.fixture(params=[3])
def nb_labels(request: Any) -> int:
    """ number of randomly chosen labels under which to match"""
    return request.param


@pytest.fixture(params=[(1, 3)])
def dim_rels(request: Any) -> int:
    """ Number of relation models """
    return random.randint(*request.param)


@pytest.fixture(params=CLASSES_TO_TEST)
def algebra(request: Any, dim_rels: int) -> Algebra:
    """ Algebra to use """
    return request.param(dim_rels)


@pytest.fixture
def relations(
        request: Any, nb_relations: int, nb_features: int,
        algebra: Algebra) -> Dict[int, RelationModel]:
    """ relation models """

    class CustomRelation(RelationModel):
        """ Fake relations """
        def __init__(self, nb_features: int, algebra: Algebra) -> None:
            self.linear = nn.Linear(2 * nb_features, algebra.flatdim)

        @property
        def parameters(self) -> Callable[[], Iterable[Any]]:
            return self.linear.parameters

        def __call__(self, x: Tsor, y: Tsor) -> Tsor:
            """ Compute x R y """
            return self.linear(torch.cat((x, y), -1))

    return {
        idx: CustomRelation(nb_features, algebra)
        for idx in range(nb_relations)}


@pytest.fixture
def scoring(
        nb_features: int, nb_scores: int, algebra: Algebra) -> ScoringModel:
    """ scoring model """
    class CustomScore(ScoringModel):
        """ Fake score """
        def __init__(self, nb_features: int,
                     nb_scores: int, algebra: Algebra) -> None:
            self.linear = nn.Linear(
                2 * nb_features + algebra.flatdim, nb_scores + 1)
            self.softmax = torch.nn.Softmax(dim=-1)

        @property
        def parameters(self) -> Callable[[], Iterable[Any]]:
            return self.linear.parameters


        def __call__(self,
                     src: Tsor,
                     dst: Tsor,
                     rel: Tsor) -> Tsor:
            """ Compute S(src, dst, rel) """
            cat_input = torch.cat((src, dst, rel), -1)
            return self.softmax(self.linear(cat_input))[..., :-1]

    return CustomScore(nb_features, nb_scores, algebra)


@pytest.fixture(params=[1, 5])
def arrow(request: Any, nb_relations: int) -> CompositeArrow[int, int]:
    """Composite arrow for tests"""
    return CompositeArrow(
        range(1 + request.param),
        random.choices(range(nb_relations), k=request.param))


def get_labels(
        nodes: Iterable[int],
        nb_scores: int, nb_labels: int) -> DirectedGraph[int]:
    """
    Generate a random label graph
    """
    labels = DirectedGraph[int]()
    sources: List[int] = random.choices(list(nodes), k=nb_labels)
    targets: List[int] = random.choices(list(nodes), k=nb_labels)

    for idx, (src, tar) in enumerate(product(sources, targets)):
        labels.add_edge(src, tar)
        scores = torch.softmax(
            torch.rand(1 + nb_scores), dim=-1)[..., :-1]
        labels[src][tar][idx] = scores

    return labels


def get_model(
        relations: Dict[int, RelationModel],
        scoring: ScoringModel, algebra: Algebra,
        trainable: bool = False) -> DecisionCatModel:
    """
    Get a categorical model using test relation models and scoring
    If trainable is set to True, returns a trainable decision model.
    """
    if trainable:
        return TrainableDecisionCatModel(
            relations, scoring, algebra, torch.optim.SGD, lr=0.001)
    return DecisionCatModel(relations, scoring, algebra)


class TestRelationCache:
    """
    Tests for RelationCache class
    """
    @staticmethod
    def get_cache(
            relations: Dict[int, RelationModel],
            scoring: ScoringModel, algebra: Algebra,
            data_dim: int,
            *arr: CompositeArrow[int, int]) -> RelationCache:
        """
        get a relation cache, initialized with random data and one composite
        arrow if given (otherwise it is empty)
        """
        if arr:
            datas = {idx: torch.rand(data_dim) for idx in set().union(*arr)}  # type: ignore
            return RelationCache[int, int](
                relations, scoring, algebra.comp, datas, arr)

        return RelationCache[int, int](
            relations, scoring, algebra.comp, {}, ())

    @staticmethod
    def test_relation(
            nb_features: int,
            relations: Dict[int, RelationModel],
            scoring: ScoringModel, algebra: Algebra,
            arrow: CompositeArrow[int, int]) -> None:
        """
        Test stored value of relation
        """
        # create cache with one composite arrow
        cache = TestRelationCache.get_cache(
            relations, scoring, algebra, nb_features, arrow)

        # compute expected value of corresponding relation
        expected_result = reduce(
            algebra.comp,
            (
                cache.data(arrow[idx:(idx + 1)])  # type: ignore
                for idx in range(len(arrow))))

        assert (
            (cache.data(arrow) - expected_result).norm()
            <= len(arrow) * algebra.flatdim * DEFAULT_EPSILON)

    @staticmethod
    def test_score(
            nb_features: int,
            relations: Dict[int, RelationModel],
            scoring: ScoringModel, algebra: Algebra,
            arrow: CompositeArrow[int, int]) -> None:
        """
        Test stored value of score
        """
        # create cache with one composite arrow
        cache = TestRelationCache.get_cache(
            relations, scoring, algebra, nb_features, arrow)

        # collect minimum of all total scores of parts of arrow
        # note that actual score of arrow is also taken in the loop
        # but it does not matter here
        min_score = inf
        for fst, last in combinations(range(len(arrow)), 2):
            min_score = min(min_score, cache[arrow[fst:last]].sum())  # type: ignore

        # recompute raw score of arrow
        raw_score = scoring(
            cache.data(arrow[0:0]), cache.data(arrow.op[0:0]),  # type: ignore
            cache.data(arrow))

        # deduce actual score from raw score and minimum score of parts
        if raw_score.sum() > min_score:
            expected_result = torch.zeros(raw_score.shape)
        else:
            expected_result = raw_score

        assert subproba_kl_div(
            cache[arrow], expected_result) <= DEFAULT_EPSILON

    @staticmethod
    def test_matching(
            nb_features: int,
            nb_labels: int,
            nb_scores: int,
            relations: Dict[int, RelationModel],
            scoring: ScoringModel, algebra: Algebra,
            arrow: CompositeArrow[int, int]) -> None:
        """
        Test for graph matching
        """
        # create cache with one composite arrow
        cache = TestRelationCache.get_cache(
            relations, scoring, algebra, nb_features, arrow)

        # create label graph
        labels = get_labels(arrow, nb_scores, nb_labels)

        # match label graph against cache
        matches = cache.match(labels)

        # check that matches graph contains everything necessary
        for (src, tar), edge_labels in labels.edges().items():  # pylint: disable=no-member
            for label, value in edge_labels.items():
                # check label has been matched
                assert label in matches[src][tar]

                # collect available scores for matching
                available = {
                    idx.derive(): cache[idx] for idx in cache.arrows(src, tar)}

                # evaluate all models if none available
                if not available:
                    available = {
                        CompositeArrow(arr): scoring(
                            cache.data(CompositeArrow(src)),
                            cache.data(CompositeArrow(tar)),
                            relations[arr](
                                cache.data(CompositeArrow(src)),
                                cache.data(CompositeArrow(tar))))
                        for arr in cache.generators}

                kldivs = {
                    idx: subproba_kl_div(score, value)
                    for idx, score in available.items()}

                expected_match = min(kldivs, key=kldivs.get)
                expected_cost = kldivs[expected_match]

                assert matches[src][tar][label][0] == expected_match
                assert (
                    (matches[src][tar][label][1] - expected_cost).abs()
                    <= 2. * DEFAULT_EPSILON)


class TestDecisionCatModel:
    """ Tests for DecisionCatModel class"""
    @staticmethod
    def test_cost(
            arrow: CompositeArrow[int, int],
            relations: Dict[int, RelationModel], scoring: ScoringModel,
            algebra: Algebra,
            nb_features: int, nb_labels: int, nb_scores: int) -> None:
        """
        Test cost function of decision cat model
        """
        # create model
        model = get_model(relations, scoring, algebra, trainable=False)

        # generate datapoints for arrow
        datas = {node: torch.rand(nb_features) for node in arrow}

        # generate labels
        labels = get_labels(arrow, nb_scores, nb_labels)

        # compute cost. we want to verify all goes smoothly and
        # the result is a positive finite real number as intended
        cost = model.cost(datas, [arrow], labels)
        assert torch.isfinite(cost)
        assert cost >= 0


class TestTrainableDecisionCatModel:
    """ Tests for TrainableDecisionCatModel class"""
    params: Dict[str, List[Any]] = {
        "test_train": [dict(nb_steps=50)]}

    @staticmethod
    @pytest.mark.flaky(reruns=FLAKY_TEST_RETRY)
    def test_train(
            arrow: CompositeArrow[int, int],
            relations: Dict[int, RelationModel], scoring: ScoringModel,
            algebra: Algebra,
            nb_features: int, nb_labels: int,
            nb_steps: int, nb_scores: int) -> None:
        """
        Test training, verifying that overerall cost on a batch lowers
        during training when using same data
        """
        # get model
        model = get_model(relations, scoring, algebra, trainable=True)

        # generate datapoints for arrow
        datas = {node: torch.rand(nb_features) for node in arrow}

        # generate labels
        labels = get_labels(arrow, nb_scores, nb_labels)

        # compute cost
        initial_cost = model.cost(datas, [arrow], labels).item()

        # let's train for severam steps
        for _ in range(nb_steps):
            model.train(datas, [arrow], labels, step=True)  # type: ignore

        final_cost = model.cost(datas, [arrow], labels).item()

        assert final_cost <= initial_cost
