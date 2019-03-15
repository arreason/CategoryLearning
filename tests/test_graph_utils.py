#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""
tests for random graph generation
"""

from typing import (
    Any, Callable, Hashable, Dict, Iterable, Tuple,
    Set, FrozenSet, Optional, List, Union)
from itertools import repeat
from collections import Counter
from copy import copy
from operator import and_, or_, add, mul, matmul
from string import ascii_letters, digits
import random

import pytest

from catlearn.graph_utils import (
    DirectedGraph, DirectedAcyclicGraph, GraphRandomFactory,
    CompositeArrow, CompositionGraph)


@pytest.fixture(params=[0, 432358, 98765, 326710, 54092])
def seed(request: Any) -> int:
    """
    seed for random operations
    """
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_nodes(request: Any) -> int:
    """
    number of nodes to retrieve from a stored dataset
    """
    return request.param


@pytest.fixture(params=[(1,), (5, 10), (100,), (2, 2, 2)])
def batch_shape(request: Any) -> int:
    """
    batch shape for dataset and databatch objects
    """
    return request.param


@pytest.fixture(params=[0., 0.15, 0.45])
def pruning_factor(request: Any) -> float:
    """
    pruning factor for random graph pruning operations
    """
    return request.param


def pytest_generate_tests(metafunc):
    """
    called once for each test function. Decorates the test runs with the right
    parametrization. The parametrization of each function is to be found in a
    params dictionary in the same scope as the function, in a list associated
    to the key being the name of the function.
    """

    # collect parameters list for test
    func_name = metafunc.function.__name__
    if hasattr(metafunc.cls, "params") and func_name in metafunc.cls.params:
        funcarglist = metafunc.cls.params[func_name]
    else:
        # if no parameters are declared, do as if the params list is empty
        funcarglist = []

    # if specific arguments are declared, execute tests using those
    if funcarglist:
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames, [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist]
            )
    else:
        # no specific parameters. Any existing fixture will still be used.
        print(
            f"No specific parametrization found"
            f" for {metafunc.cls.__name__}.{func_name}")
        metafunc.parametrize([], [])


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
                expected_graph=DirectedGraph({0: [2], 2: [], 1: [0, 2]})
                )
            ],
        "test_stringify": [
            dict(graph=DirectedGraph({(0, 0): [], "1": []}),
                 expected_graph=DirectedGraph({"0x0": [], "0x1": []})),
            dict(graph=DirectedGraph({0.5: ["a"], "a": [], 2: [0.5, "a"]}),
                 expected_graph=DirectedGraph(
                     {"0x0": ["0x2"], "0x2": [], "0x1": ["0x0", "0x2"]})
                 )
            ],
        "test_rand_prune": [
            dict(graph=DirectedGraph({
                0: [1, 2, 3, 4], 2: [3, 4, 6], 6: [5, 9, 11]})),
            dict(graph=DirectedGraph({0: []})),
            dict(graph=DirectedGraph({}))
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
            set(node) <= __class__.ALLOWED_CHARS for node in stringified_graph)  # type: ignore
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


class TestCompositeArrow:
    """
    Test the composite arrow class
    """
    params: Dict[str, List[Any]] = {
        "test_getitem": [
            dict(
                arrow=CompositeArrow[str, int](["a", "b", "c"], [1, 2]),
                index=1, expected_result="b"),
            dict(
                arrow=CompositeArrow[int, int](range(11), range(10)),
                index=slice(None, None, 2),
                expected_result=tuple(range(0, 11, 2))),
            dict(
                arrow=CompositeArrow[int, int](range(11), range(10)),
                index=slice(1, 8, 3),
                expected_result=tuple(range(1, 8, 3))),
            dict(
                arrow=CompositeArrow[str, int](["a", "b", "c"], [1, 2]),
                index=slice(1, None),
                expected_result=CompositeArrow[str, int](["b", "c"], [2])),
            dict(
                arrow=CompositeArrow[str, int](["a", "b", "c"], [1, 2]),
                index=slice(1),
                expected_result=CompositeArrow[str, int](["a", "b"], [1])),
            dict(
                arrow=CompositeArrow[str, int](
                    ["a", "b", "c", "d"], [1, 2, 3]),
                index=slice(1, 2),
                expected_result=CompositeArrow[str, int](
                    ["b", "c"], [2]))
            ],
        "test_derive": [
            dict(
                arrow=CompositeArrow[str, int](["a", "b", "c"], [3, 7]),
                expected_result=CompositeArrow[int, str]([3, 7], ["b"])),
            dict(
                arrow=CompositeArrow[str, int](["a", "b"], [9]),
                expected_result=CompositeArrow(9))
            ],
        "test_suspend": [
            dict(arrow=CompositeArrow[str, int](["a", "b", "c"], [3, 7])),
            dict(arrow=CompositeArrow[str, int](["a", "b"], [9]))
            ],
        "test_binary_op": [
            dict(
                binary_op=add,
                first_operand=CompositeArrow[str, int](
                    ["a", "b", "c"], [0, 1]),
                second_operand=CompositeArrow[str, int](
                    ["c", "d", "e"], [9, -1]),
                expected_result=CompositeArrow[str, int](
                    ["a", "b", "c", "d", "e"], [0, 1, 9, -1])),
            dict(
                binary_op=add,
                first_operand=CompositeArrow[str, int](["a", "b"], [0]),
                second_operand=CompositeArrow[str, int]("b"),
                expected_result=CompositeArrow[str, int](["a", "b"], [0])),
            dict(
                binary_op=add,
                first_operand=CompositeArrow[str, int](["a", "b"], [0]),
                second_operand=CompositeArrow[str, int](["c", "d"], [1]),
                expected_result=None),
            dict(
                binary_op=matmul,
                first_operand=CompositeArrow[str, int](
                    ["a", "b", "c", "d"], [0, 1, 3]),
                second_operand=CompositeArrow[str, int](
                    ["b", "c", "d", "e"], [1, 3, -1]),
                expected_result=CompositeArrow[str, int](
                    ["a", "b", "c", "d", "e"], [0, 1, 3, -1])),
            dict(
                binary_op=matmul,
                first_operand=CompositeArrow[str, int](["a", "b"], [0]),
                second_operand=CompositeArrow[str, int](["b", "c"], [9]),
                expected_result=CompositeArrow[str, int](
                    ["a", "b", "c"], [0, 9])),
            dict(
                binary_op=lambda x, y: x.comp(y, -1),
                first_operand=CompositeArrow[str, int](
                    ["a", "b", "c"], [0, 1]),
                second_operand=CompositeArrow[str, int](
                    ["b", "d", "c"], [-1, 9]),
                expected_result=None),
            dict(
                binary_op=lambda x, y: x.comp(y, -1),
                first_operand=CompositeArrow[str, int](
                    ["a", "b", "c", "d"], [0, 1, 3]),
                second_operand=CompositeArrow[str, int](
                    ["c", "d", "e", "f"], [3, 9, -1]),
                expected_result=CompositeArrow[str, int](
                    ["a", "b", "c", "d", "e", "f"], [0, 1, 3, 9, -1])),
            dict(
                binary_op=lambda x, y: x.comp(y, -1),
                first_operand=CompositeArrow[str, int](["a", "b"], [0]),
                second_operand=CompositeArrow[str, int](["a", "b"], [0]),
                expected_result=CompositeArrow[str, int](
                    ["a", "b"], [0])),
            dict(
                binary_op=matmul,
                first_operand=CompositeArrow[str, int](
                    ["a", "b", "c"], [0, 1]),
                second_operand=CompositeArrow[str, int](
                    ["b", "d", "c"], [-1, 9]),
                expected_result=None),
            ]
        }

    @staticmethod
    def test_getitem(
            arrow: CompositeArrow[Hashable, Hashable],
            index: Union[int, slice],
            expected_result: Union[
                Hashable, Tuple[Hashable, ...],
                CompositeArrow[Hashable, Hashable]]
        ) -> None:
        """
        test CompositeArrow item getting
        """
        result = arrow[index]
        assert result == expected_result

    @staticmethod
    def test_derive(
            arrow: CompositeArrow[Hashable, Hashable],
            expected_result: CompositeArrow[Hashable, Hashable]) -> None:
        """
        Test arrow derivation
        """
        result = arrow.derive()
        assert result == expected_result

    @staticmethod
    def test_suspend(
            arrow: CompositeArrow[Hashable, Hashable]) -> None:
        """
        Test arrow suspension, by taking the suspension of an arrow derivation
        on its source and target (which should realize the identity)
        """
        source = arrow[0]
        target = arrow[-1]
        derived = arrow.derive()
        result = derived.suspend(source, target)
        assert result == arrow

    @staticmethod
    def test_binary_op(
            binary_op: Callable[[Any, Any], Any],
            first_operand: CompositeArrow[Hashable, Hashable],
            second_operand: CompositeArrow[Hashable, Hashable],
            expected_result: Optional[CompositeArrow[Hashable, Hashable]]
        ) -> None:
        """
        test a binary operation on composite arrows
        """
        if expected_result is None:
            with pytest.raises(ValueError):
                result = binary_op(first_operand, second_operand)
        else:
            result = binary_op(first_operand, second_operand)
            assert result == expected_result
