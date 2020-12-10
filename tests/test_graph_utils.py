#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name, invalid-name, abstract-method,no-self-use,too-few-public-methods
"""
tests for random graph generation
"""

from typing import (
    Any, Callable, Hashable, Dict, Iterable, Tuple,
    Set, FrozenSet, Optional, List)
from itertools import repeat
from collections import Counter
from copy import copy
from operator import and_, or_, add, mul, matmul
from string import ascii_letters, digits
import random

import pytest
#pylint: disable=unused-import
from tests.test_tools import pytest_generate_tests

from catlearn.graph_utils import (
    DirectedGraph, DirectedAcyclicGraph, GraphRandomFactory,
    sample_vertices, sample_edges,
    pagerank_sample, hubs_sample, authorities_sample,
    uniform_vertex_sample, uniform_edge_sample,
    random_walk_vertex_sample, random_walk_edge_sample,
    n_hop_sample, generate_random_graph)


@pytest.fixture(params=[0, 432358, 98765, 326710, 54092])
def seed(request: Any) -> int:
    """
    seed for random operations
    """
    return request.param


@pytest.fixture(params=[0., 0.15, 0.45])
def pruning_factor(request: Any) -> float:
    """
    pruning factor for random graph pruning operations
    """
    return request.param


class TestDirectedGraph:
    """
    Unit tests for DirectedGraph class
    """

    # allowed chars for stringification
    ALLOWED_CHARS = (frozenset(digits) | frozenset(ascii_letters))

    # parameters for tests
    params: Dict[str, List[Any]] = {
        "test_init": [
            dict(initializer_dict={0: [0, "1"], 2.: [()]},
                 expected_dict={
                     0: frozenset({0, "1"}),
                     "1": frozenset(), 2: frozenset({()}),
                     (): frozenset()}
                 ),
            dict(initializer_dict={0: [1], 1: [[]]},
                 expected_dict=None)
            ],
        "test_op": [
            dict(initializer_dict={0: [1], 1: []},
                 expected_op=DirectedGraph({0: [], 1: [0]})),
            dict(initializer_dict={0: [1], 1: [0]},
                 expected_op=DirectedGraph({0: [1], 1: [0]})),
            dict(initializer_dict={0: [], 1: []},
                 expected_op=DirectedGraph({0: [], 1: []}))
            ],
        "test_delitem": [
            dict(graph=DirectedGraph({0: [], 1: [0, 1], 2: [1]}),
                 node_to_remove=1,
                 expected_graph=DirectedGraph({0: [], 2: []})
                 ),
            dict(graph=DirectedGraph({}),
                 node_to_remove=1,
                 expected_graph=DirectedGraph({})
                 )
            ],
        "test_setitem": [
            dict(graph=DirectedGraph({0: [], 1: [], 2: []}),
                 node_to_add=0,
                 children=[1, 2],
                 expected_graph=DirectedGraph(
                     {0: [1, 2], 1: [], 2: []})
                 ),
            dict(graph=DirectedGraph({1: [], 2: []}),
                 node_to_add=0,
                 children=[1, 2],
                 expected_graph=DirectedGraph(
                     {0: [1, 2], 1: [], 2: []})
                 ),
            dict(graph=DirectedGraph({}),
                 node_to_add=0,
                 children=[1, 2],
                 expected_graph=DirectedGraph(
                     {0: [1, 2], 1: [], 2: []})
                 )
            ],
        "test_len": [
            dict(graph=DirectedGraph({}),
                 expected_length=0),
            dict(graph=DirectedGraph({0: [], 1: []}),
                 expected_length=2),
            dict(graph=DirectedGraph({0: [1], 1: []}),
                 expected_length=2)
            ],
        "test_iter": [
            dict(graph=DirectedGraph({0: [], 1: [1], 2: []}),
                 nodes_set={0, 1, 2}),
            dict(graph=DirectedGraph({}),
                 nodes_set={})
            ],
        "test_under": [
            dict(graph=DirectedGraph({0: [1, 2], 1: [], 2: []}),
                 node=0,
                 nodes_set={1, 2}),
            dict(
                graph=DirectedGraph(
                    {0: [1, 2], 1: [], 2: [3, 1], 3: [4], 4: []}),
                node=0,
                nodes_set={1, 2, 3, 4}
                )
            ],
        "test_over": [
            dict(graph=DirectedGraph({0: [], 1: [0], 2: [0]}),
                 node=0,
                 nodes_set={1, 2}),
            dict(
                graph=DirectedGraph(
                    {0: [], 1: [0, 2], 2: [0], 3: [2], 4: [3]}),
                node=0,
                nodes_set={1, 2, 3, 4})
            ],
        "test_subgraph": [
            dict(graph=DirectedGraph({0: [], 1: [], 2: []}),
                 nodes_set={0},
                 expected_graph=DirectedGraph({0: []})),
            dict(graph=DirectedGraph({0: [0, 1], 1: [], 2: [0, 1]}),
                 nodes_set={0, 1},
                 expected_graph=DirectedGraph({0: [0, 1], 1: []}))
            ],
        "test_binary_operator": [
            dict(binary_op=or_,
                 first_graph=DirectedGraph({0: [1], 1: []}),
                 second_graph=DirectedGraph({0: [], 1: [1]}),
                 expected_graph=DirectedGraph({
                     (0, 0): [(0, 1)], (0, 1): [],
                     (1, 0): [], (1, 1): [(1, 1)]})
                 ),
            dict(binary_op=and_,
                 first_graph=DirectedGraph({0: [1], 1: []}),
                 second_graph=DirectedGraph({0: [], 1: [1]}),
                 expected_graph=DirectedGraph({
                     (0, 0): [], (0, 1): [(1, 1)],
                     (1, 0): [], (1, 1): []})),
            dict(binary_op=add,
                 first_graph=DirectedGraph({0: [1], 1: []}),
                 second_graph=DirectedGraph({0: [], 1: [1]}),
                 expected_graph=DirectedGraph({
                     (0, 0): [(0, 1), (1, 0), (1, 1)],
                     (0, 1): [(1, 0), (1, 1)],
                     (1, 0): [], (1, 1): [(1, 1)]})
                 ),
            dict(binary_op=matmul,
                 first_graph=DirectedGraph({0: [1], 1: []}),
                 second_graph=DirectedGraph({0: [], 1: [1]}),
                 expected_graph=DirectedGraph({
                     (0, 0): [(1, 0)], (1, 0): [],
                     (0, 1): [(1, 1), (0, 1)], (1, 1): [(1, 1)]})
                 ),
            dict(binary_op=mul,
                 first_graph=DirectedGraph({0: [1], 1: []}),
                 second_graph=DirectedGraph({0: [], 1: [1]}),
                 expected_graph=DirectedGraph({
                     (0, 0): [(1, 0)],
                     (1, 0): [],
                     (0, 1): [(0, 1), (0, 1), (1, 1)],
                     (1, 1): [(0, 1), (1, 1)]})
                 )
            ],
        "test_prune": [
            dict(
                graph=DirectedGraph(
                    {0: [1], 1: [2, 3], 2: [], 3: [], 4: [0, 1]}),
                node_to_prune=1,
                expected_graph=DirectedGraph(
                    {0: [2, 3], 2: [], 3: [], 4: [0, 2, 3]})
                )
            ],
        "test_integerify": [
            dict(graph=DirectedGraph({(0, 0): [], "1": []}),
                 expected_graph=DirectedGraph({0: [], 1: []})),
            dict(
                graph=DirectedGraph({0.5: ["a"], "a": [], 2: [0.5, "a"]}),
                expected_graph=DirectedGraph({2: [1], 1: [], 0: [2, 1]})
                )
            ],
        "test_stringify": [
            dict(graph=DirectedGraph({(0, 0): [], "1": []}),
                 expected_graph=DirectedGraph({"0x0": [], "0x1": []})),
            dict(graph=DirectedGraph({0.5: ["a"], "a": [], 2: [0.5, "a"]}),
                 expected_graph=DirectedGraph(
                     {"0x2": ["0x1"], "0x1": [], "0x0": ["0x2", "0x1"]})
                 )
            ],
        "test_rand_prune": [
            dict(graph=DirectedGraph({
                0: [1, 2, 3, 4], 2: [3, 4, 6], 6: [5, 9, 11]})),
            dict(graph=DirectedGraph({0: []})),
            dict(graph=DirectedGraph({}))
            ],
        "test_dualize_relations": [
            dict(graph=DirectedGraph({0: [1], 1: []}),
                 expected_graph=DirectedGraph({0: [1], 1: []})),
            dict(graph=DirectedGraph({0: {1: {"label": 1}}, 1: []}),
                 expected_graph=DirectedGraph({
                     0: {1: {("label", False): 1, ("label", True): 1}},
                     1: []}))
            ]
        }

    @staticmethod
    def test_init(
            initializer_dict: Dict[Hashable, Iterable[Hashable]],
            expected_dict: Optional[Dict[Hashable, FrozenSet[Hashable]]]):
        """
        check that initialization builds the correct graphs by adding missing
        nodes as keys.
        If the expected_dict is None, awaiting failure with a TypeError raised,
        because the initnializer_dict is invalid (some nodes are not hashable)
        """
        if expected_dict is None:
            with pytest.raises(TypeError):
                graph: DirectedGraph = DirectedGraph[Hashable](
                    initializer_dict)
        else:
            graph = DirectedGraph[Hashable](initializer_dict)
            dict_of_graph = {
                node: frozenset(children)
                for node, children in graph.items()}
            assert dict_of_graph == expected_dict

    @staticmethod
    def test_op(
            initializer_dict: Dict[Hashable, Iterable[Hashable]],
            expected_op: DirectedGraph) -> None:
        """
        test that the opposite of the graph is correctly computed at init
        """
        graph = DirectedGraph[Hashable](initializer_dict)
        graph_op = graph.op
        assert graph_op == expected_op

    @staticmethod
    def test_delitem(
            graph: DirectedGraph,
            node_to_remove: Hashable,
            expected_graph: DirectedGraph):
        """
        test that node deletion works as intended
        """
        # copy graph
        graph_copy = copy(graph)

        # remove node from copy
        if node_to_remove in graph:
            del graph_copy[node_to_remove]

            # check the new keys of the graph are well defined
            assert graph_copy == expected_graph
        else:
            with pytest.raises(KeyError):
                # deleting non existing key should raise a key error
                del graph_copy[node_to_remove]

    @staticmethod
    def test_setitem(
            graph: DirectedGraph,
            node_to_add: Hashable, children: Iterable[Hashable],
            expected_graph: DirectedGraph) -> None:
        """
        test that resetting the list of children of a node works as intended
        """
        graph_copy = copy(graph)

        # reset list associated to node
        graph_copy[node_to_add] = children

        # verify that graph and its oppposite have expected value
        assert graph_copy == expected_graph
        assert graph_copy.op == expected_graph.op

    @staticmethod
    def test_len(graph: DirectedGraph, expected_length: int) -> None:
        """
        Check that the length of the graph and its opposite are right
        """
        assert len(graph) == expected_length
        assert len(graph.op) == expected_length

    @staticmethod
    def test_iter(graph: DirectedGraph, nodes_set: Set) -> None:
        """
        Test that itarating over the nodes of the graph goes though all of the
        necessary nodes, once only
        """
        assert Counter(iter(graph)) == Counter(nodes_set)

    @staticmethod
    def test_under(graph: DirectedGraph, node: Hashable,
                   nodes_set: Set[Hashable]) -> None:
        """
        tests that the under method returns the required set of nodes
        """
        assert graph.under(node) == frozenset(nodes_set)

    @staticmethod
    def test_over(graph: DirectedGraph, node: Hashable,
                  nodes_set: Set[Hashable]) -> None:
        """
        tests that the over method returns the required set of nodes
        """
        assert graph.over(node) == frozenset(nodes_set)

    @staticmethod
    def test_subgraph(graph: DirectedGraph, nodes_set: Set[Hashable],
                      expected_graph: DirectedGraph):
        """
        Test that subgraph extraction gives the right graph
        """
        assert graph.subgraph(nodes_set) == expected_graph

    @staticmethod
    def test_binary_operator(
            binary_op: Callable[[Any, Any], Any],
            first_graph: DirectedGraph, second_graph: DirectedGraph,
            expected_graph: DirectedGraph) -> None:
        """
        Verify that the given binary operator applied to first_graph and
        second_graph yields expected_graph.
        """
        result_graph = binary_op(first_graph, second_graph)
        result_type = type(result_graph)
        expected_type = type(expected_graph)
        assert result_type == expected_type
        assert result_graph == expected_graph

    @staticmethod
    def test_prune(
            graph: DirectedGraph,
            node_to_prune: Hashable,
            expected_graph: DirectedGraph) -> None:
        """
        Verify that pruning the given node from graph yields the required graph
        """
        result_graph = graph.prune(node_to_prune)
        assert result_graph == expected_graph

    @staticmethod
    def test_integerify(
            graph: DirectedGraph, expected_graph: DirectedGraph):
        """
        Verify that integerification yields a consistent graph.
        Ideally we would want a test of isomorphism,
        but it is complicated, so it tests if
        the integerification yields the correct integers for each node.
        Hence this is a reggression test.
        """
        integerified_graph = graph.integerify()
        assert integerified_graph == expected_graph

    @staticmethod
    def test_stringify(
            graph: DirectedGraph, expected_graph: DirectedGraph):
        """
        Verify that stringification yields a consistent graph. Ideally we would
        want a test of isomorphism, but it is complicated, so it tests if
        the stringification yields the correct strings for each node.
        Hence this is a reggression test.
        """
        stringified_graph = graph.stringify()
        assert all(
            set(node) <= __class__.ALLOWED_CHARS for node in stringified_graph)  # type: ignore # pylint: disable=undefined-variable
        assert stringified_graph == expected_graph

    @staticmethod
    def test_rand_prune(
            graph: DirectedGraph, pruning_factor: float, seed: int) -> None:
        """
        test that random pruning gives the same result with same seeding
        """
        # initialize two random generator with same seed
        first_random_generator = random.Random()
        first_random_generator.seed(seed)

        second_random_generator = random.Random()
        second_random_generator.seed(seed)

        # prune graph using the two different random generators
        first_pruned_graph = graph.rand_prune(
            pruning_factor, random_generator=first_random_generator)
        second_pruned_graph = graph.rand_prune(
            pruning_factor, random_generator=second_random_generator)

        assert first_pruned_graph == second_pruned_graph

    @staticmethod
    def test_dualize_relations(
            graph: DirectedGraph, expected_graph: DirectedGraph) -> None:
        """
        test dual property
        """
        result = graph.dualize_relations()
        assert result == expected_graph


class TestAcyclicDirectedGraph:
    """
    Unit tests for DirectedAcyclicGraph class
    """
    params: Dict[str, List[Any]] = {
        "test_init": [
            dict(initializer_dict={0: [1], 1: []}, has_loops=False),
            dict(initializer_dict={0: [0], 1: []}, has_loops=True),
            dict(
                initializer_dict={0: [1], 1: [2, 3], 2: [0]}, has_loops=True)
            ]
        }

    @staticmethod
    def test_init(
            initializer_dict: Dict[Hashable, Iterable[Hashable]],
            has_loops: bool) -> None:
        """
        Test initialization of acyclic directed graph, in particular that
        pasting an initializer dictionary with a loop leads to an appropriate
        error.
        """
        if has_loops:
            with pytest.raises(AssertionError):
                DirectedAcyclicGraph[Hashable](initializer_dict)
        else:
            result = DirectedAcyclicGraph[Hashable](initializer_dict)
            directed_graph_result = DirectedGraph[Hashable](initializer_dict)
            assert dict(result) == dict(directed_graph_result)


@pytest.fixture(params=[1, 3, 5, 7])
def nb_steps(request: Any) -> int:
    """
    number of steps for the random graph generation
    """
    return request.param


@pytest.fixture(params=[
    (0.75, 0., 0., 0., 0.), (0., 0.75, 0., 0., 0.), (0., 0., 0.75, 0., 0.),
    (0., 0., 0., 0.75, 0.), (0., 0., 0., 0., 0.75),
    (0.15, 0.15, 0.15, 0.15, 0.15)])
def weights(request: Any) -> Tuple[float, float, float, float, float]:
    """
    weights of operations for random graph factory
    """
    return request.param


class TestGraphRandomFactory:
    """
    Unit tests for random graph factory
    """
    params: Dict[str, List[Any]] = {
        "test_result_equality": [
            dict(nb_graphs=3, initial_graphs=[DirectedAcyclicGraph({})]),
            dict(
                nb_graphs=3,
                initial_graphs=[
                    DirectedAcyclicGraph({0: [], 1: []}),
                    DirectedAcyclicGraph({0: [1], 1: []})])
            ]
        }

    @staticmethod
    def test_result_equality(
            nb_graphs: int, initial_graphs: List[DirectedAcyclicGraph],
            nb_steps: int, weights: Tuple[float, float, float, float, float],
            pruning_factor: float, seed: int) -> None:
        """
        Tests that results generated from the same seeds with the same
        parameters, but 2 different factories, are equal
        """
        # instantiate 2 random generators and initialize them with the
        # same seed
        first_random_generator = random.Random()
        first_random_generator.seed(seed)

        second_random_generator = random.Random()
        second_random_generator.seed(seed)

        # instantiate two graph factories with the same parameters, using the
        # two random generators
        first_factory = GraphRandomFactory(
            weights, nb_graphs, pruning_factor, first_random_generator,
            *initial_graphs)
        second_factory = GraphRandomFactory(
            weights, nb_graphs, pruning_factor, second_random_generator,
            *initial_graphs)

        assert all(
            repeat(next(first_factory) == next(second_factory), nb_steps))  # type: ignore


class TestSubgraphSampling:
    """
    Subgraph sampling test suite
    """

    @staticmethod
    @pytest.fixture
    def rng():
        """ Return PRNG to use in the test """
        return random.Random()

    @staticmethod
    @pytest.fixture(params=[3, 5])
    def nb_steps(request: Any) -> int:
        """
        number of steps for the random graph generation
        """
        return request.param

    @staticmethod
    @pytest.fixture
    def graph(nb_steps, rng):
        """
        Randomly create somewhat complex graphs, then return a complex one
        """
        return generate_random_graph(nb_steps, rng)

    @staticmethod
    @pytest.fixture(params=[1, 3, 5])
    def n_seeds(request: Any) -> int:
        """ Number of seeds for the random walk """
        return request.param

    @staticmethod
    @pytest.fixture(params=[2, 10, 20])
    def n_iter(request: Any) -> int:
        """ Random walk loop count """
        return request.param

    @staticmethod
    @pytest.fixture(params=[0, 1, 4])
    def max_degree(request: Any) -> int:
        """ Max degree allowed in sampled graph """
        return request.param

    @staticmethod
    def test_sample_vertices(graph, rng):
        """ Test sample respect a simple probability distribution: only first k nodes """
        firsts = frozenset(list(graph)[:5])
        ranking = lambda g: {v: 1. if v in firsts else 0. for v in g}
        sg = sample_vertices(graph, len(graph), ranking, rng)
        selected = frozenset(list(sg))
        assert selected <= firsts

    @staticmethod
    def test_sample_edges(graph, rng):
        """Test sample of edges with respect to a simple proba dist: only
        first k nodes"""
        edges = graph.edges
        firsts = frozenset(list(edges)[:5])
        ranking = lambda g: {e: 1. if e in firsts else 0. for e in edges}
        sg = sample_edges(graph, len(list(edges)), ranking, rng)
        selected = frozenset(list(sg.edges))
        assert selected <= firsts

    @staticmethod
    def test_uniform_vertex(graph, rng):
        """ Sanity checks on uniform vertex sampler """
        n_vertices = max(1, len(graph) - 4)
        sg = uniform_vertex_sample(graph, n_vertices, rng)
        assert 1 <= len(sg) <= n_vertices
        assert all(v in graph for v in sg)

    @staticmethod
    def test_uniform_edge(graph, rng):
        """Sanity checks on uniform edge sampler"""
        n_edges = max(0, len(list(graph.edges)) - 4)
        sg = uniform_edge_sample(graph, n_edges, rng)
        se = sg.edges
        assert 0 <= len(se) <= n_edges
        assert all(v in graph for v in sg)
        assert all(e in graph.edges for e in se)

    # Sometimes the algorithm does not converge
    @staticmethod
    @pytest.mark.flaky(reruns=3)
    def test_pagerank(graph, rng):
        """ Sanity checks on pagerank sampler """
        n_vertices = max(1, len(graph) - 4)
        sg = pagerank_sample(graph, n_vertices, rng)
        assert 1 <= len(sg) <= n_vertices

    # Sometimes the algorithm does not converge
    @staticmethod
    @pytest.mark.flaky(reruns=3)
    def test_hubs(graph, rng):
        """ Sanity checks on hubs sampler """
        n_vertices = max(1, len(graph) - 4)
        sg = hubs_sample(graph, n_vertices, rng)
        assert 1 <= len(sg) <= n_vertices

    # Sometimes the algorithm does not converge
    @staticmethod
    @pytest.mark.flaky(reruns=3)
    def test_authorities(graph, rng):
        """ Sanity checks on authorities sampler """
        n_vertices = max(1, len(graph) - 4)
        sg = authorities_sample(graph, n_vertices, rng)
        assert 1 <= len(sg) <= n_vertices

    @staticmethod
    def test_random_walk_vertex(graph, rng, n_iter, n_seeds):
        """ Sanity check on random walk sampler """
        sg = random_walk_vertex_sample(graph, rng, n_iter, n_seeds=n_seeds)
        assert 1 <= len(sg) <= n_iter + n_seeds
        # Assert we have at most n_seeds roots
        isRoot = lambda v: frozenset() <= sg.over(v) <= frozenset(v)
        assert 1 <= sum(1 for v in sg if isRoot(v)) <= n_seeds

    @staticmethod
    def test_random_walk_vertex_specified_root(graph, rng, n_iter):
        """ Sanity checks on random walk sampler """
        seed = rng.choice(list(graph))
        sg = random_walk_vertex_sample(graph, rng, n_iter, seeds=[seed])
        assert 1 <= len(sg) <= n_iter + 1 # 1 seed
        assert frozenset() <= sg.over(seed) <= frozenset(seed) # empty, or self-reference

    @staticmethod
    def test_random_walk_vertex_with_dual(graph, rng, n_iter):
        """ Sanity checks on random walk sampler """
        sg = random_walk_vertex_sample(graph, rng, n_iter, use_opposite=True)
        assert 1 <= len(sg) <= n_iter + 1 # 1 seed

    @staticmethod
    def test_random_walk_edge(graph, rng, n_iter, n_seeds, max_degree):
        """ Sanity checks on random walk over edges sampler """
        sg = random_walk_edge_sample(
            graph, rng, n_iter, n_seeds=n_seeds,
            max_in_degree=max_degree, max_out_degree=max_degree)
        assert 0 <= len(sg.edges) <= n_iter + n_seeds
        # Assert we have at most n_seeds roots
        isRoot = lambda v: frozenset() <= sg.over(v) <= frozenset(v)
        assert 0 <= sum(1 for v in sg if isRoot(v)) <= n_seeds

    @staticmethod
    def test_random_walk_edge_use_all(graph, rng, n_iter, n_seeds, max_degree):
        """ Sanity checks on random walk over edges sampler """
        sg = random_walk_edge_sample(
            graph, rng, n_iter, n_seeds=n_seeds,
            use_opposite=True, use_both_ends=True,
            max_in_degree=max_degree, max_out_degree=max_degree)
        assert 0 <= len(sg.edges) <= n_iter + n_seeds

    @staticmethod
    @pytest.fixture(params=[0, 1, 2])
    def n_hops(request: Any) -> int:
        """ Number of hops """
        return request.param

    @staticmethod
    def test_n_hop_sampler(graph, rng, n_hops, n_seeds):
        """ Sanity checks on n_hop sampler """
        sg = n_hop_sample(graph, n_hops, n_seeds=n_seeds, rng=rng)
        assert all(v in graph for v in sg)

    @staticmethod
    def test_random_walk_edge_star_pattern(rng, max_degree):
        """ Test subsampling of {1, k} -> 0 and then 0 -> {1, k} """
        n_iter = max(100, max_degree * 100)
        if max_degree <= 0:
            max_degree = 10
        graph = DirectedGraph()
        for v in range(1, max_degree * 2):
            graph.add_edge(v, 0)

        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, n_seeds=1,
            use_opposite=False, use_both_ends=False)
        assert len(subgraph.edges) == 1  # only the seed

        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, n_seeds=1,
            use_opposite=False, use_both_ends=True)
        assert len(subgraph.edges) == 1  # only the seed

        # Here we activate matching of edges *->0
        # => we can sample almost all the graph
        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, n_seeds=1,
            use_opposite=True, use_both_ends=False,
            max_out_degree=1,
            max_in_degree=max_degree)
        assert len(subgraph.edges) == min(max_degree, n_iter)

    @staticmethod
    @pytest.fixture(params=[2, 10])
    def chain_length(request: Any) -> int:
        """ Length of the chain to generate """
        return request.param

    @staticmethod
    def test_random_walk_edge_chain_pattern(rng, chain_length):
        """ Test subsampling a chain sub-graph """
        n_iter = max(20, 100 * chain_length)
        multiplier = 4
        graph = DirectedGraph()
        for v in range(chain_length):
            for i in range(multiplier):
                for j in range(multiplier):
                    graph.add_edge((v, i), (v+1, j))

        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, n_seeds=1,  # any starting point in the chain
            use_opposite=True, use_both_ends=True,
            max_in_degree=1, max_out_degree=1)
        # Assert graph is a chain of expected length
        assert len(subgraph.edges) == chain_length
        assert(0 <= d <= 1 for _, d in subgraph.out_degree())
        assert(0 <= d <= 1 for _, d in subgraph.in_degree())

        # Find vertices - sorting here should retrieve the chaining
        vertices = sorted(list(subgraph.nodes))
        assert all(v in set((i, m) for m in range(multiplier))
                   for i, v in enumerate(vertices))

        # Inspect linkage
        assert all(e in subgraph.edges for e in zip(vertices, vertices[1:]))

    @staticmethod
    def test_random_walk_edge_functional_pattern(rng):
        """
        Minimal test case for the different sampling head and direction

        Graph to use should enable precise testing of the options.

        4 -> 0 -> 1 -> 2
        5 <- 0    1 <- 3
        """
        n_iter = 100  # override fixture to stabilize the tests

        graph = DirectedGraph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(3, 1)
        graph.add_edge(4, 0)
        graph.add_edge(0, 5)

        # Step 1 - use_opposite=False, use_both_ends=False
        # From (0, 1), we should sample only edges starting from 1
        expected_subgraph = DirectedGraph()
        expected_subgraph.add_edge(0, 1)
        expected_subgraph.add_edge(1, 2)
        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, seeds=[(0,1)],
            use_opposite=False, use_both_ends=False)
        assert subgraph == expected_subgraph

        # Step 2 - use_opposite=False, use_both_ends=True
        # From (0, 1), we should sample only edges starting from 0 or 1
        expected_subgraph = DirectedGraph()
        expected_subgraph.add_edge(0, 1)
        expected_subgraph.add_edge(1, 2)
        expected_subgraph.add_edge(0, 5)
        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, seeds=[(0,1)],
            use_opposite=False, use_both_ends=True)
        assert subgraph == expected_subgraph

        # Step 3 - use_opposite=True, use_both_ends=False
        expected_subgraph = DirectedGraph()
        expected_subgraph.add_edge(0, 1)
        expected_subgraph.add_edge(1, 2)
        expected_subgraph.add_edge(3, 1)
        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, seeds=[(0,1)],
            use_opposite=True, use_both_ends=False)
        assert subgraph == expected_subgraph

        # Step 4 - use_opposite=True, use_both_ends=True => catch all
        subgraph = random_walk_edge_sample(
            graph, rng, n_iter, seeds=[(0,1)],
            use_opposite=True, use_both_ends=True)
        assert subgraph == graph
