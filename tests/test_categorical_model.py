#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the categorical_model file.
"""
from typing import Any, Dict, Optional
from functools import reduce
from itertools import combinations, product
from math import inf
import random

import torch
from torch import nn

import pytest

from catlearn.tensor_utils import subproba_kl_div
from catlearn.categorical_model import (
    DEFAULT_EPSILON, RelationModel, ScoringModel, CompositeArrow,
    DirectedGraph, RelationCache, DecisionCatModel, TrainableDecisionCatModel)
from catlearn.algebra_models import (
    Algebra, VectAlgebra, MatrixAlgebra, AffineAlgebra)
from catlearn.causal_generation_utils import (
    CausalGenerator, CausalGraphBatch,
    CausalDatasetFromGraph, generate_dataset)

from tests.test_tools import pytest_generate_tests
# List of algebras to verify
# Automagic, adding an algebra will get tested right away
CLASSES_TO_TEST = {VectAlgebra, MatrixAlgebra, AffineAlgebra}


@pytest.fixture(params=[2, 8])
def nb_features(request: Any) -> int:
    """ Feature space dimension """
    return request.param


@pytest.fixture(params=[1, 3, 9])
def nb_scores(request: Any) -> int:
    """ Score vector dimension"""
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_relations(request: Any) -> int:
    """ number of relations in the model"""
    return request.param


@pytest.fixture(params=[2, 5])
def nb_labels(request: Any) -> int:
    """ number of randomly chosen labels under which to match"""
    return request.param

@pytest.fixture(params=[(2, 8)])
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

        def __call__(self, x: torch.tensor, y: torch.tensor) -> torch.Tensor:
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
            self.nb_scores = nb_scores

        def __call__(self,
                     src: torch.Tensor,
                     dst: torch.Tensor,
                     rel: torch.Tensor) -> torch.Tensor:
            """ Compute S(src, dst, rel) """
            cat_input = torch.cat((src, dst, rel), -1)
            return self.softmax(self.linear(cat_input))[..., :-1]

    return CustomScore(nb_features, nb_scores, algebra)


@pytest.fixture(params=[1, 3, 5, 11])
def arrow(request: Any, nb_relations: int) -> CompositeArrow[int, int]:
    """Composite arrow for tests"""
    return CompositeArrow(
        range(1 + request.param),
        random.choices(range(nb_relations), k=request.param))


class TestRelationCache:
    """
    Tests for RelationCache class
    """
    @staticmethod
    def get_cache(
            relations: Dict[int, RelationModel],
            scoring: ScoringModel, algebra: Algebra,
            data_dim: Optional[int] = None,
            *arr: CompositeArrow[int, int]) -> RelationCache:
        """
        get a relation cache, initialized with random data and one composite
        arrow if given (otherwise it is empty)
        """
        if arr:
            datas = {idx: torch.rand(data_dim) for idx in set().union(*arr)}
            return RelationCache[int, int](
                relations, scoring, algebra.comp, datas, *arr)

        return RelationCache[int, int](relations, scoring, algebra.comp, {})

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
        labels = DirectedGraph[int]()
        sources = random.choices(arrow.nodes, k=nb_labels)
        targets = random.choices(arrow.nodes, k=nb_labels)

        for idx, (src, tar) in enumerate(product(sources, targets)):
            labels.add_edge(src, tar)
            scores = torch.softmax(
                torch.rand(1 + scoring.nb_scores), dim=-1)[..., :-1]
            labels[src][tar][idx] = scores

        # match label graph against cache
        matches = cache.match(labels)

        # check that matches graph contains everything necessary
        for (src, tar), edge_labels in labels.edges().items():
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
    pass


class TestTrainableDecisionCatModel:
    """ Tests for TrainableDecisionCatModel class"""
    pass
