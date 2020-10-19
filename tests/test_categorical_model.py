#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name, invalid-name, abstract-method, no-self-use
# pylint: disable=too-few-public-methods,too-many-locals
"""
Tests for the categorical_model file.
"""
from typing import Any, Dict, Mapping, Iterable, List, Callable
from functools import reduce
from itertools import combinations, product
from tempfile import NamedTemporaryFile
from math import inf
import random

import torch
from torch import nn

import pytest
from tests.test_tools import pytest_generate_tests

from catlearn.tensor_utils import subproba_kl_div, Tsor
from catlearn.graph_utils import DirectedGraph
from catlearn.composition_graph import CompositeArrow
from catlearn.relation_cache import RelationCache, NegativeMatch
from catlearn.categorical_model import (
    RelationModel, ScoringModel,
    DecisionCatModel, TrainableDecisionCatModel)
from catlearn.algebra_models import (
    Algebra, VectAlgebra, VectMultAlgebra, MatrixAlgebra, AffineAlgebra)


# List of algebras to verify
# Automagic, adding an algebra will get tested right away
CLASSES_TO_TEST = {VectAlgebra, VectMultAlgebra, MatrixAlgebra, AffineAlgebra}


# path to the data of the generators used for synthetic datasets
DATA_DIR = "./tests/test_categorical_model/"


FLAKY_TEST_RETRY = 3

# Numerical precision for tests. Since some issues can occur based on precision
# of batch matrix multiplication, can't take it too low
TEST_EPSILON = 1e-4


@pytest.fixture(params=[1, 8])
def nb_features(request: Any) -> int:
    """ Feature space dimension """
    return request.param


@pytest.fixture(params=[1, 3])
def nb_scores(request: Any) -> int:
    """ Score vector dimension """
    return request.param

@pytest.fixture(params=[3])
def nb_labels(request: Any) -> int:
    """ number of randomly chosen labels under which to match """
    return request.param

@pytest.fixture(params=[(1, 3)])
def dim_rels(request: Any) -> int:
    """ Number of relation models """
    return random.randint(*request.param)


@pytest.fixture(params=CLASSES_TO_TEST)
def algebra(request: Any, dim_rels: int) -> Algebra:
    """ Algebra to use """
    return request.param(dim_rels)


@pytest.fixture(params=[False, True])
def reload(request) -> bool:
    """ Save and reload model before running the test """
    return request.param


class CustomRelation(RelationModel):
    """ Fake relations """
    def __init__(self, nb_features: int, nb_labels: int, algebra: Algebra) -> None:
        self.linear = nn.Linear(2 * nb_features + nb_labels, algebra.flatdim)

    @property
    def parameters(self) -> Callable[[], Iterable[Any]]:
        return self.linear.parameters

    def __call__(self, x: Tsor, y: Tsor, l: Tsor) -> Tsor:
        """ Compute x R y """
        return self.linear(torch.cat((x, y, l), -1))


@pytest.fixture
def relation(
        nb_labels: int, nb_features: int,
        algebra: Algebra) -> RelationModel:
    """ relation model """
    return CustomRelation(nb_features, nb_labels, algebra)

@pytest.fixture
def label_universe(nb_labels: int) -> Mapping[int, Tsor]:
    """ Universe of all possible label """
    def one_hot(label: int):
        enc = torch.zeros(nb_labels)
        enc[label] = 1.0
        return enc
    return {i:one_hot(i) for i in range(nb_labels)}

class CustomScore(ScoringModel):
    """ Fake score """
    def __init__(
            self, nb_features: int,
            nb_scores: int, algebra: Algebra) -> None:
        self.linear = nn.Linear(
            2 * nb_features + algebra.flatdim, nb_scores + 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    @property
    def parameters(self) -> Callable[[], Iterable[Any]]:
        return self.linear.parameters

    def __call__(self, src: Tsor, dst: Tsor, rel: Tsor) -> Tsor:
        """ Compute S(src, dst, rel) """
        cat_input = torch.cat((src, dst, rel), -1)
        return self.softmax(self.linear(cat_input))[..., :-1]


@pytest.fixture
def scoring(
        nb_features: int, nb_scores: int, algebra: Algebra) -> ScoringModel:
    """ scoring model """
    return CustomScore(nb_features, nb_scores, algebra)


@pytest.fixture(params=[1, 5])
def arrow(request: Any, label_universe: Mapping[int, Tsor]) -> CompositeArrow[int, Tsor]:
    """Composite arrow for tests"""
    return CompositeArrow(
        range(1 + request.param),
        random.choices(list(label_universe), k=request.param))

@pytest.fixture(params=[True, False])
def match_negatives(request: Any):
    """Wether negative matches should be counted"""
    return request.param

def get_labels(
        nodes: Iterable[int],
        nb_scores: int,
        nb_labels: int) -> DirectedGraph[int]:
    """
    Generate a random label graph
    """
    graph = DirectedGraph[int]()
    sources: List[int] = random.choices(list(nodes), k=nb_labels)
    targets: List[int] = random.choices(list(nodes), k=nb_labels)
    labels: List[int] = random.choices(list(range(nb_labels)), k=nb_labels**2)

    for idx, (src, tar) in enumerate(product(sources, targets)):
        graph.add_edge(src, tar)
        scores = torch.softmax(
            torch.rand(1 + nb_scores), dim=-1)[..., :-1]
        graph[src][tar][labels[idx]] = scores
    return graph


def get_trainable_model(
        relation: RelationModel,
        label_universe: Mapping[int, Tsor],
        scoring: ScoringModel,
        algebra: Algebra) -> TrainableDecisionCatModel:
    """
    Prepare a trainable decision model
    """
    return TrainableDecisionCatModel(
        relation, label_universe,
        scoring, algebra, torch.optim.SGD, lr=0.001)

class TestRelationCache:
    """
    Tests for RelationCache class
    """
    params: Dict[str, List[Any]] = {
        "test_prune": [
            dict(nb_to_prune=0),
            dict(nb_to_prune=1),
            dict(nb_to_prune=2),
            dict(nb_to_prune=-1),
            dict(nb_to_prune=-2),
        ],
        "test_build_composites": [
            dict(max_arrow_length=1, max_arrow_number=100),
            dict(max_arrow_length=2, max_arrow_number=100),
            dict(max_arrow_length=10, max_arrow_number=100),
        ]
    }

    @staticmethod
    def get_cache(
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel,
            algebra: Algebra,
            data_dim: int,
            *arr: CompositeArrow[int, Tsor]) -> RelationCache:
        """
        get a relation cache, initialized with random data and one composite
        arrow if given (otherwise it is empty)
        """
        if arr:
            datas = {idx: torch.rand(data_dim) for idx in set().union(*arr)}  # type: ignore
        else:
            datas = {}
            arr = ()

        return RelationCache[int, Tsor](
            relation, label_universe, scoring, algebra.comp, datas, arr)

    @staticmethod
    def test_relation(
            nb_features: int,
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel, algebra: Algebra,
            arrow: CompositeArrow[int, Tsor]) -> None:
        """
        Test stored value of relation
        """
        # create cache with one composite arrow
        cache = TestRelationCache.get_cache(
            relation, label_universe, scoring, algebra, nb_features, arrow)

        # compute expected value of corresponding relation
        expected_result = reduce(
            algebra.comp,
            (
                cache.data(arrow[idx:(idx + 1)])  # type: ignore
                for idx in range(len(arrow))))

        assert (
            (cache.data(arrow) - expected_result).norm()
            <= len(arrow) * algebra.flatdim * TEST_EPSILON)

    @staticmethod
    def test_score(
            nb_features: int,
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel, algebra: Algebra,
            arrow: CompositeArrow[int, Tsor]) -> None:
        """
        Test stored value of score
        """
        # create cache with one composite arrow
        cache = TestRelationCache.get_cache(
            relation, label_universe, scoring, algebra, nb_features, arrow)

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
            cache[arrow], expected_result) <= TEST_EPSILON

    @staticmethod
    def test_matching(
            nb_features: int,
            nb_labels: int,
            nb_scores: int,
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel, algebra: Algebra,
            arrow: CompositeArrow[int, int],
            match_negatives: bool) -> None:
        """
        Test for graph matching
        """
        # create cache with one composite arrow
        cache = TestRelationCache.get_cache(
            relation, label_universe,
            scoring, algebra, nb_features, arrow)

        # create label graph
        labels = get_labels(arrow, nb_scores, nb_labels)

        # match label graph against cache
        matches = cache.match(labels, match_negatives=match_negatives)

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
                            relation(
                                cache.data(CompositeArrow(src)),
                                cache.data(CompositeArrow(tar)),
                                arr))
                        for arr in cache.label_universe}

                kldivs = {
                    idx: (
                        subproba_kl_div(score, value)
                        - float(match_negatives) * subproba_kl_div(
                            score, torch.zeros(score.shape)
                            ))
                    for idx, score in available.items()}

                expected_match = min(kldivs, key=kldivs.get)
                expected_score = available[expected_match]
                expected_cost = subproba_kl_div(
                    expected_score, value)

                assert matches[src][tar][label][0] == expected_match
                assert (
                    (matches[src][tar][label][1] - expected_cost).abs()
                    <= TEST_EPSILON)

            # check that negatives are matched if match_negatives is true
            if match_negatives:
                positives = set(
                    matches[src][tar][label][0] for label in edge_labels)
                expected_negatives = set(
                    arr.derive() for arr in cache.arrows(src, tar)) - positives
                for negative in expected_negatives:
                    expected_label = NegativeMatch(negative)
                    assert expected_label in matches[src][tar]

                    score = cache[negative.suspend(src, tar)]
                    expected_cost = subproba_kl_div(
                        score, torch.zeros(score.shape))
                    assert (
                        (matches[src][tar][expected_label][1] - expected_cost)
                        <= TEST_EPSILON)

    @staticmethod
    def test_prune(
            nb_features: int,
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel, algebra: Algebra,
            arrow: CompositeArrow[int, Tsor],
            nb_to_prune: int) -> None:
        """
            Test pruning operation.
            If nb_to_prune is a positive integer, tries to prune nb_to_prune
            relations
            If nb_to_prune is a negative integer, tries to keep at most
            nb_to_prune relations + 1
        """
        # create cache with one composite arrow
        cache = TestRelationCache.get_cache(
            relation, label_universe,
            scoring, algebra, nb_features, arrow)

        # target number of relations to keep
        nb_to_keep = (
            len(cache) - nb_to_prune if nb_to_prune >= 0
            else -nb_to_prune + 1)

        initial_content = frozenset(cache)

        pruned = cache.prune_relations(nb_to_keep)

        final_content = frozenset(cache)

        assert initial_content == (final_content ^ pruned)

        assert(
            len(cache) <= nb_to_keep
            or all(
                len(list(cache.arrows(relation[0], relation[-1]))) == 1
                for relation in cache.arrows() if len(relation) == 1))


class TestDecisionCatModel:
    """ Tests for DecisionCatModel class"""
    @staticmethod
    def test_cost(
            arrow: CompositeArrow[int, Tsor],
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel,
            algebra: Algebra,
            nb_features: int,
            nb_labels: int,
            nb_scores: int,
            match_negatives: bool) -> None:
        """
        Test cost function of decision cat model
        """
        # create model
        model = DecisionCatModel(relation, label_universe, scoring, algebra)

        # generate datapoints for arrow
        datas = {node: torch.rand(nb_features) for node in arrow}

        # generate labels
        labels = get_labels(arrow, nb_scores, nb_labels)

        # compute cost. we want to verify all goes smoothly
        cost, cache, matched = model.cost(
            datas, [arrow], labels, match_negatives=match_negatives)

        # the resulting cost should be a positive finite real number
        assert torch.isfinite(cost)
        assert cost >= 0

        # compute cache and matching for comparison
        expected_cache = RelationCache[int, Tsor](
            relation, label_universe, model.score, model.algebra.comp, datas, [arrow])
        expected_matched = expected_cache.match(
            labels, match_negatives=match_negatives)

        # verify obtained cache and matching are identical to expected
        assert set(cache.keys()) == set(expected_cache.keys())
        for arr, scores in cache.items():
            assert subproba_kl_div(
                scores, expected_cache[arr]) <= TEST_EPSILON

        assert set(matched.edges()) == set(expected_matched.edges())
        for (src, tar), labels in matched.edges().items():  # pylint: disable=no-member
            for name, label in labels.items():
                assert torch.abs(
                    label[1] - expected_matched[src][tar][name][1]
                    ) <= TEST_EPSILON


class TestTrainableDecisionCatModel:
    """ Tests for TrainableDecisionCatModel class"""
    params: Dict[str, List[Any]] = {
        "test_train_with_update": [dict(nb_steps=50)]}

    @staticmethod
    def test_train_without_update(
            arrow: CompositeArrow[int, Tsor],
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel,
            algebra: Algebra,
            nb_features: int,
            nb_labels: int,
            nb_scores: int,
            reload: bool,
            match_negatives: bool) -> None:
        """
        Test training, verifying that not activating updates does:
            - conserve gradients
            - lead to same results of evaluation
        """
        # get model and its parameters
        model = get_trainable_model(relation, label_universe, scoring, algebra)

        # generate datapoints for arrow
        datas = {node: torch.rand(nb_features) for node in arrow}

        # generate labels
        labels = get_labels(arrow, nb_scores, nb_labels)

        # go through one train stage
        expected_cache, expected_matched = model.train(
            datas, [arrow], labels, step=False, match_negatives=match_negatives)

        # reset model, extract everything again
        if reload:
            with NamedTemporaryFile() as tmpfile:
                model.save(tmpfile)
                model = TrainableDecisionCatModel.load(tmpfile.name)
        else:
            model.reset()
        cache, matched = model.train(
            datas, [arrow], labels, step=False, match_negatives=match_negatives)

        # verify obtained cache and matching are identical to expected
        assert set(cache.keys()) == set(expected_cache.keys())
        for arr, scores in cache.items():
            assert subproba_kl_div(
                scores, expected_cache[arr]) <= TEST_EPSILON

        assert set(matched.edges()) == set(expected_matched.edges())
        for (src, tar), labels in matched.edges().items():  # pylint: disable=no-member
            for name, label in labels.items():
                assert torch.abs(
                    label[1] - expected_matched[src][tar][name][1]
                    ) <= TEST_EPSILON

    @staticmethod
    @pytest.mark.flaky(reruns=FLAKY_TEST_RETRY)
    def test_train_with_update(
            arrow: CompositeArrow[int, Tsor],
            relation: RelationModel,
            label_universe: Mapping[int, Tsor],
            scoring: ScoringModel,
            algebra: Algebra,
            nb_features: int,
            nb_labels: int,
            nb_steps: int,
            nb_scores: int,
            match_negatives: bool) -> None:
        """
        Test training, verifying that overerall cost on a batch lowers
        during training when using same data
        """
        # get model
        model = get_trainable_model(relation, label_universe, scoring, algebra)

        # generate datapoints for arrow
        datas = {node: torch.rand(nb_features) for node in arrow}

        # generate labels
        labels = get_labels(arrow, nb_scores, nb_labels)

        # compute cost
        initial_cost, _, _ = model.cost(
            datas, [arrow], labels, match_negatives=match_negatives)

        # let's train for several steps
        for _ in range(nb_steps):
            model.train(
                datas, [arrow], labels, step=True,
                match_negatives=match_negatives)  # type: ignore

        final_cost, _, _ = model.cost(
            datas, [arrow], labels, match_negatives=match_negatives)

        assert final_cost <= initial_cost
