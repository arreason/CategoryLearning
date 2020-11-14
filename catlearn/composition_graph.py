#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:29:23 2019

@author: christophe_c
""""""

Utilities for graphs with composite arrows
"""
from __future__ import annotations
from typing import (
    TypeVar, Generic, Optional, Union, Iterable, Tuple, Iterator, Callable)
from collections import abc
from itertools import chain
from math import inf

from catlearn.graph_utils import DirectedGraph, NodeType

ArrowType = TypeVar("ArrowType")
AlgebraType = TypeVar("AlgebraType")


class CompositeArrow(Generic[NodeType, ArrowType], abc.Sequence):  # pylint: disable=unsubscriptable-object
    """
    A class for composite arrows in a graph. To initialize such a composite,
    one should provide:
        - nodes: Iterable[NodeType], the nodes of the composite in order
        - arrows: Iterable[ArrowType], the arrow labels of the composite in
            order
            Number of nodes should be 1 more than number of arrows
    """

    def __init__(
            self, nodes: Union[NodeType, Iterable[NodeType]],
            arrows: Optional[Iterable[ArrowType]] = None) -> None:
        """
        Initialize a new composite arrow from nodes and arrow labels data
        """
        super().__init__()

        # if no arrows are given, the first argument is assumed
        # to be a unique node
        if arrows is None:
            self._nodes = (nodes,)
            self._arrows = ()
        else:
            self._nodes = tuple(nodes)  # type: ignore
            self._arrows = tuple(arrows)  # type: ignore

        if not len(self.nodes) == len(self.arrows) + 1:
            raise ValueError("nodes and arrows length don't match")

    def derive(self) -> CompositeArrow[ArrowType, NodeType]:
        """
        Return the arrow's inside, forfeiting first and last node.
        """
        return CompositeArrow[ArrowType, NodeType](  # type: ignore
            self.arrows, self[1:-1:1])

    def suspend(
            self, source: ArrowType, target: ArrowType
        ) -> CompositeArrow[ArrowType, NodeType]:
        """
        Return an arrow whose derived is given arrow, with provided source
        and target
        """
        return CompositeArrow[ArrowType, NodeType](
            (source,) + self.arrows + (target,), self.nodes)

    @property
    def nodes(self) -> Tuple[NodeType, ...]:
        """
        Get the nodes supporting the composite as a tuple
        """
        return self._nodes  # type: ignore

    @property
    def arrows(self) -> Tuple[ArrowType, ...]:
        """
        Get the arrow labels of the composite as a tuple
        """
        return self._arrows

    def __getitem__(  # type: ignore
            self, index: Union[int, slice]
        ) -> Union[
            NodeType, Tuple[NodeType, ...],
            CompositeArrow[NodeType, ArrowType]]:
        """
        Access nodes and subcomposites.
            - Accessing an integer index will access the node at the given
                index
            - Accessing a slice without step ([i:j]) will give the subarrow
                containing the elementary arrows from i-th included to
                j-th excluded
        """
        if isinstance(index, int):
            return self.nodes[index]

        if isinstance(index, slice):
            # if step is not None, return nodes only
            if index.step:
                return self.nodes[index]

            # compute start and stop of nodes slice
            length = len(self)
            if index.start is None:
                start = 0
            elif index.start < 0:
                start = index.start + length
            else:
                start = index.start

            if index.stop is None:
                stop = 1 + length
            elif index.stop < 0:
                stop = index.stop + length + 1
            else:
                stop = index.stop + 1

            # get nodes and arrows to extract
            nodes_slice = slice(start, stop)
            nodes = self.nodes[nodes_slice]

            arrows_slice = slice(start, stop - 1)
            arrows = self.arrows[arrows_slice]

            return CompositeArrow(nodes, arrows)

        raise TypeError("Argument should be integer or slice.")

    def __len__(self) -> int:
        """
        Length of the composites: number of elementary arrows of which it
        is a composite
        """
        return len(self.arrows)

    def __eq__(self, arrow: CompositeArrow[NodeType, ArrowType]) -> bool:  # type: ignore
        """
        Test wether two composite arrows are equals, i.e:
            - they have the same nodes in the same order
            - they have the same arrows in the same order
        """
        return self.nodes == arrow.nodes and self.arrows == arrow.arrows

    @property
    def op(self) -> CompositeArrow[NodeType, ArrowType]:
        """
        Return arrow in reverse direction
        """
        return CompositeArrow(self.nodes[::-1], self.arrows[::-1])

    def comp(
            self, arrow: CompositeArrow[NodeType, ArrowType],
            overlap: int) -> CompositeArrow[NodeType, ArrowType]:
        """
        Return composite with n overlapping arrows in the middle
        if n negative, or all overlapping from n-th position of first arrow
        if n is positive
        """
        if overlap == 0:
            return self + arrow
        if overlap < 0:
            if not self[overlap:] == arrow[:-overlap]:  # type: ignore
                raise ValueError("Trying to compose non-matching arrows")
            nodes = self.nodes + arrow.nodes[-overlap + 1:]
            arrows = self.arrows + arrow.arrows[-overlap:]
        else:
            if not self[overlap:] == arrow[:len(self) - overlap]:  #  type: ignore
                raise ValueError("Trying to compose non-matching arrows")
            nodes = self.nodes + arrow.nodes[len(self) - overlap + 1:]
            arrows = self.arrows + arrow.arrows[len(self) - overlap:]

        return CompositeArrow(nodes, arrows)

    def __add__(
            self, arrow: CompositeArrow[NodeType, ArrowType]
        ) -> CompositeArrow[NodeType, ArrowType]:
        """
        Compose 2 arrows together. The last node of the first must be the same
        as the first node of the second.
        """
        if not self[-1] == arrow[0]:  # type: ignore
            raise ValueError(
                "Last node of first arrow different"
                " from first node of second arrow")

        nodes = self.nodes + arrow.nodes[1:]
        arrows = self.arrows + arrow.arrows
        return CompositeArrow(nodes, arrows)

    def __matmul__(
            self, arrow: CompositeArrow[NodeType, ArrowType]
        ) -> CompositeArrow[NodeType, ArrowType]:
        """
        Extend 2 composites which match on:
            - the first composite with its first arrow removed
            - the second composite with its last arrow removed
        """
        return self.comp(arrow, overlap=1)

    def __hash__(self) -> int:
        """
        Hash code of a composite comp is computed from the hash code of
        (comp.nodes, comp.arrows)
        """
        return hash((self.nodes, self.arrows))

    def __repr__(self) -> str:
        """
        String representation of a composite arrow
        """
        return (
            f"{type(self).__name__}({self[0]}"
            + "".join(
                f">{arrow}->{node}"
                for (arrow, node) in zip(self.arrows, self.nodes[1:]))) + ")"


class CompositionGraph(Generic[NodeType, ArrowType, AlgebraType], abc.Mapping):  # pylint: disable=unsubscriptable-object
    """
    Make a composition graph out of a list of Composite arrows
    Constructor Arguments:
        - comp: a callable which computes the composite of 2 arrow values
        - arrows: any number of composite arrows
    """
    def __init__(
            self,
            comp: Callable[
                [
                    CompositionGraph[NodeType, ArrowType, AlgebraType],
                    CompositeArrow[NodeType, ArrowType]],
                AlgebraType],
            arrows: Iterable[
                CompositeArrow[NodeType, ArrowType]] = iter(())) -> None:
        """
        Initialize a new composition graph.
        """
        super().__init__()
        self._graph = DirectedGraph[NodeType]()

        def _comp(
                arrow: CompositeArrow[NodeType, ArrowType]) -> AlgebraType:
            """
            Composition method attached to current composite graph
            """
            return comp(self, arrow)
        self._comp = _comp

        for arrow in arrows:
            self.add(arrow)

    @property
    def graph(self):
        """
        Access graph on which arrows are stored
        """
        return self._graph

    def add(self, arrow: CompositeArrow[NodeType, ArrowType]) -> None:
        """
        Add the given composite arrow to the composition graph
        NOTE: manage empty arrows
        """
        if not arrow:
            raise ValueError("Can only add arrows of length at least 1")

        # if the arrow is already in the structure, we can stop there
        if arrow in self:
            return

        # add edge to the graph if necessary
        if not self.graph.has_edge(arrow[0], arrow[-1]):
            self.graph.add_edge(arrow[0], arrow[-1])

        # the case of length 1 arrows is simple: generate the value
        # and put it in the graph

        if len(arrow) >= 2:
            # the case of higher order arrows is recursively defined:
            # all subcomposites of the arrow should be in the graph, so we have
            # to make the computation for all of them
            for idx in range(1, len(arrow)):

                # add the subarrows in the structure if needed
                fst_arrow = arrow[:idx]
                scd_arrow = arrow[idx:]
                if fst_arrow not in self:
                    self.add(fst_arrow)  # type: ignore
                if scd_arrow not in self:
                    self.add(scd_arrow)  # type: ignore

        # compute the value of the total arrow and register it
        value = self._comp(arrow)
        self._graph[arrow[0]][arrow[-1]][arrow.derive()] = value

    def flush(self) -> None:
        """
        Reset the structure, removing all composite arrows
        """
        self._graph = DirectedGraph()

    def arrows(
            self, src: Optional[NodeType] = None,
            tar: Optional[NodeType] = None,
            arrow_length_range: Tuple[int, Optional[int]] = (0, None),
        ) -> Iterator[CompositeArrow[NodeType, ArrowType]]:
        """
        Get an iterator over all arrows starting at src and ending at tar.
        If source or tar is None, will loop through all possible sources
        and arrows.
        If no existing arrows (or src/tar not in the underlying graph),
        returns an empty iterator.
        An arrow length range can also be specified. In this case,
        only arrows with a length in the specified range are returned
        """
        min_length = arrow_length_range[0]
        max_length = arrow_length_range[1] if arrow_length_range[1] else inf

        if (
                src is not None and tar is not None
                and self.graph.has_edge(src, tar)):
            # iterate over all edges from src to tar
            # careful to length: the derived arrow has length diminished by 1
            return (
                arr.suspend(src, tar) for arr in self.graph[src][tar]
                if (
                    len(arr) >= min_length - 1
                    and len(arr) < max_length - 1))
        if src is not None and tar is None and src in self.graph:
            # iterate over all edges starting at src
            return chain(*(
                self.arrows(src, node, arrow_length_range=arrow_length_range)
                for node in self.graph[src]))
        if src is None and tar is not None and tar in self.graph:
            # iterate over all edges ending at tar
            return chain(*(
                self.arrows(node, tar, arrow_length_range=arrow_length_range)
                for node in self.graph.op[tar]))
        if src is None and tar is None:
            # iterate over all edges of graph in this case
            return chain(*(
                self.arrows(node, None, arrow_length_range=arrow_length_range)
                for node in self.graph.nodes
            ))
        return iter(())

    def __iter__(self) -> Iterator[CompositeArrow[NodeType, ArrowType]]:
        """
        Return an iterator over all composite arrows of the structure
        """
        return chain(*(self.arrows(*edge) for edge in self.graph.edges))

    def __len__(self) -> int:
        """
        Return the number of composite arrows in the structure
        """
        return sum(1 for _ in iter(self))

    def __getitem__(
            self, arrow: CompositeArrow[NodeType, ArrowType]) -> AlgebraType:
        """
        Get value associated to given arrow
        """
        return self.graph[arrow[0]][arrow[-1]][arrow.derive()]

    def __delitem__(
            self, arrow: CompositeArrow[NodeType, ArrowType]) -> None:
        """
        Remove a composite arrow from the composite graph
        """
        # remove composite from graph
        # if it was the only composite linking its source and target
        # remove the edge from the graph
        del self.graph[arrow[0]][arrow[-1]][arrow.derive()]
        if not self.graph[arrow[0]][arrow[-1]]:
            self.graph.remove_edge(arrow[0], arrow[-1])

        # get all arrows to source and from target of the graph
        fst = list(self.arrows(tar=arrow[0]))  # type: ignore
        scd = list(self.arrows(src=arrow[-1]))  # type: ignore

        # if source or target is not linked to other points of graph,
        # remove it
        if not fst and not self.graph[arrow[0]]:
            del self.graph[arrow[0]]
        if not scd and not self.graph.op[arrow[-1]]:
            del self.graph[arrow[-1]]

        # remove all extensions of arrows existing in the composite graph
        for arr in fst:
            if arr + arrow in self:
                del self[arr + arrow]
        for arr in scd:
            if arrow + arr in self:
                del self[arrow + arr]

    def __repr__(self) -> str:
        """
        Get string representation of the composition graph
        """
        return (
            f"{type(self).__name__}("
            + ", ".join(f"{key}: {value}" for (key, value) in self.items())
            + ")")
