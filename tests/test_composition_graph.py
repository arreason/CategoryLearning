#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name, too-few-public-methods
"""
Created on Tue Mar 26 11:50:31 2019

@author: christophe_c
"""
from typing import Any, List, Dict, Union, Hashable, Optional, Callable, Tuple
from operator import add, matmul
from itertools import combinations

import pytest

from catlearn.composition_graph import CompositeArrow, CompositionGraph

#pylint: disable=unused-import
from tests.test_tools import pytest_generate_tests


@pytest.fixture(params=[3, 5, 7])
def to_add(request: Any) -> CompositeArrow[str, str]:
    """
    return an arrow to add to CompositionGraph object
    """
    return hex_arrow(request.param)


def repr_comp(
        graph: CompositionGraph[str, str, str],
        arrow: CompositeArrow[str, str]) -> str:
    """
    composition function for a composition graph
    in which stored values are the string
    representations of CompositionArrow[str, str] type.
    """
    for idx in range(1, len(arrow)):
        assert arrow[idx:] in graph and arrow[:idx] in graph

    return repr(arrow)


def hex_arrow(length: int) -> CompositeArrow[str, str]:
    """
    Define a composite arrow from integer hex codes of a given length
    """
    nodes = [hex(idx) for idx in range(1 + length)]
    arrows = ["_".join(hexes) for hexes in zip(nodes[:-1], nodes[1:])]
    return CompositeArrow(nodes, arrows)


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
        "test_op": [
            dict(
                arrow=CompositeArrow[str, int](["a", "b", "c"], [3, 7]),
                expected_result=CompositeArrow[str, int](
                    ["c", "b", "a"], [7, 3])),
            dict(
                arrow=CompositeArrow[str, int]("a"),
                expected_result=CompositeArrow[str, int]("a"))
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
    def test_op(
            arrow: CompositeArrow[Hashable, Hashable],
            expected_result: CompositeArrow[Hashable, Hashable]) -> None:
        """
        Test arrow reversal
        """
        result = arrow.op
        assert result == expected_result

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


class TestCompositionGraph:
    """
    Class for all tests pertaining to the class CompositionGraph
    """
    @staticmethod
    def test_getitem(to_add: CompositeArrow[str, str]) -> None:
        """
        Test item getting works properly
        """
        graph = CompositionGraph(repr_comp)
        graph.add(to_add)

        if len(to_add) >= 1:
            assert graph[to_add] == repr_comp(graph, to_add)
        else:
            assert pytest.raises(IndexError)

    @staticmethod
    def test_add(to_add: CompositeArrow[str, str]) -> None:
        """
        Verify that arrow adding works.
        """
        graph = CompositionGraph(repr_comp)
        graph.add(to_add)

        for fst, last in combinations(range(len(to_add)), 2):
            assert (
                graph[to_add[fst:last]] == repr_comp(graph, to_add[fst:last]))  # type: ignore

    @staticmethod
    def test_del(to_add: CompositeArrow[str, str]) -> None:
        """
        Test deletion of an arrow from graph
        """
        for fst, last in combinations(range(len(to_add)), 2):
            graph = CompositionGraph(repr_comp)
            graph.add(to_add)
            del graph[to_add[fst:last]]  # type: ignore
            assert to_add not in graph
