#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:32:59 2019

@author: christophe_c
"""
from typing import Mapping, Callable, Iterable, Generic, Tuple

import torch

from catlearn.tensor_utils import Tsor, DEFAULT_EPSILON, subproba_kl_div
from catlearn.graph_utils import DirectedGraph, NodeType
from catlearn.composition_graph import (
    ArrowType, CompositeArrow, CompositionGraph)

# Some convenient type aliases
GeneratorMapping = Mapping[ArrowType, Callable[[Tsor, Tsor], Tsor]]
Scorer = Callable[[Tsor, Tsor, Tsor], Tsor]
BinaryOp = Callable[[Tsor, Tsor], Tsor]


class RelationCache(
        Generic[NodeType, ArrowType],  # pylint: disable=unsubscriptable-object
        CompositionGraph[NodeType, ArrowType, Tsor]):
    """
    A cache to keep the values of all relations
    """
    def __init__(
            self,
            generators: GeneratorMapping,
            scorer: Scorer,
            comp: BinaryOp,
            datas: Mapping[NodeType, Tsor],
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
            ) -> Tuple[Tsor, Tsor]:
            """
            compute the composite of all the order 1 arrow making up the given
            arrow, and account for causality cost
            """
            # case of length 1
            if len(arrow) == 1:
                rel_value = self.generators[arrow.arrows[0]](
                    cache.data(arrow[0:0]), cache.data(arrow.op[0:0]))
                rel_score = scorer(
                    cache.data(arrow[0:0]), cache.data(arrow.op[0:0]),
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
        ) -> Tsor:
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
        ) -> Tsor:
        """
        Returns the value associated to an arrow.
        If the arrow has length 0, raises a ValueError
        If the arrow has length >= 1:
            returns the score vector of the relation.
        """
        if not arrow:
            raise ValueError("Cannot get the score of an arrow of length 0")
        else:
            return super().__getitem__(arrow)[0]

    def __setitem__(self, node: NodeType, data_point: Tsor) -> None:
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
        self._causality_cost = torch.zeros(())
        self._nb_compositions = 0

    def match(self, labels: DirectedGraph[NodeType]) -> Tsor:
        """
        Match the composition graph with a graph of labels. For each label
        vector, get the best match in the graph. if No match is found, set to
        + infinity.
        If no arrow is found in the cache, order 1 relations are generated
        for matching. These relations
        are added to the cache.
        """
        result_graph = DirectedGraph[NodeType]()
        for src, tar in labels.edges:
            # add edge if necessary
            if not result_graph.has_edge(src, tar):
                result_graph.add_edge(src, tar)

            # check if arrows exist to match label, add order 1 arrows
            # if they don't
            if (
                    not self.graph.has_edge(src, tar)
                    or not self.graph[src][tar]):
                for arr in self.generators:
                    self.add(CompositeArrow([src, tar], [arr]))

            # get scores of existing arrows from src to tar
            scores = {
                arr.derive(): self[arr]
                for arr in self.arrows(src, tar)}

            # go through labels and match them. Keep only the best
            for name, label in labels[src][tar].items():
                # evaluate candidate relationships
                candidates = {
                    arr: subproba_kl_div(score, label)
                    for arr, score in scores.items()}

                # save the best match in result graph
                best_match = min(candidates, key=candidates.get)
                result_graph[src][tar][name] = (
                    best_match, candidates[best_match])

        return result_graph
