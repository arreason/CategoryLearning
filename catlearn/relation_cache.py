from types import MappingProxyType
from typing import Mapping, Callable, Iterable, Iterator, Generic, Tuple
from collections import abc

import torch

from catlearn.tensor_utils import Tsor, DEFAULT_EPSILON, subproba_kl_div
from catlearn.graph_utils import DirectedGraph, NodeType
from catlearn.composition_graph import (
    ArrowType, CompositeArrow, CompositionGraph)

# Some convenient type aliases
RelationEmbedding = Callable[[Tsor, Tsor, Tsor], Tsor]
Scorer = Callable[[Tsor, Tsor, Tsor], Tsor]
BinaryOp = Callable[[Tsor, Tsor], Tsor]


class RelationCache(
        Generic[NodeType, ArrowType],  # pylint: disable=unsubscriptable-object
        abc.Mapping):
    """
    A cache to keep the values of all relations
    """

    def _graph_comp(self) -> Callable[
            [
                CompositionGraph[NodeType, ArrowType, Tsor],
                CompositeArrow[NodeType, ArrowType]
                ], Tuple[Tsor, Tsor]]:
        """
        get the composition method to use with the underlying graph. This
        function should be called only once during __init__
        """
        def graph_comp(
                graph: CompositionGraph[NodeType, ArrowType, Tsor],
                arrow: CompositeArrow[NodeType, ArrowType]
            ) -> Tuple[Tsor, Tsor]:
            """
            compute value associated to given arrow, knowing using values
            of its subparts
            """
            # case of length 1
            if len(arrow) == 1:
                rel_value = self.relation_embed(
                    self._datas[arrow[0]], self._datas[arrow[-1]],
                    self.label_universe[arrow.arrows[0]])  # type: ignore
                rel_score = self._scorer(
                    self._datas[arrow[0]], self._datas[arrow[-1]],  # type: ignore
                    rel_value)
                return rel_score, rel_value

            # now take care of case of length >= 2
            # compute the value of the arrow by composition
            comp_value = self._comp(graph[arrow[:1]][1], graph[arrow[1:]][1])  # type: ignore
            comp_scores = self._scorer(
                self._datas[arrow[0]], self._datas[arrow[-1]], comp_value)  # type: ignore
            comp_validity = comp_scores.sum()

            # recursively access all the scores of the subarrow splits
            # and check the composition score
            for idx in range(1, len(arrow)):

                # compute update to causality score
                fst_scores = graph[arrow[:idx]][0].sum()  # type: ignore
                scd_scores = graph[arrow[idx:]][0].sum()  # type: ignore
                causal_score = torch.relu(
                    torch.log(
                        (self.epsilon + comp_validity) /
                        (self.epsilon + fst_scores * scd_scores)))

                self._causality_cost += causal_score
                self._nb_compositions += 1

                # if causality update > 0., relation is not valid: 0. scores
                if causal_score > 0.:
                    comp_final_scores = torch.zeros(comp_scores.shape)
                else:
                    comp_final_scores = comp_scores

            return comp_final_scores, comp_value
        return graph_comp  # type: ignore

    def __init__(
            self,
            rel_embed: RelationEmbedding,
            label_universe: Mapping[ArrowType, Tsor],
            scorer: Scorer,
            comp: BinaryOp,
            datas: Mapping[NodeType, Tsor],
            arrows: Iterable[CompositeArrow[NodeType, ArrowType]],
            epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Initialize a new cache of relations, from:
           rel_embed: a relation embedding function returning a tensor from
               from 2 points and a tensor-valued label
           label_universe: mapping from the set of possible relation label to
               a suitable form for rel_embed
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
        self.relation_embed = rel_embed
        self._label_universe = label_universe

        # register scorer and composition operation
        self._scorer = scorer
        self._comp = comp

        # initialize underlying graph
        self._graph = CompositionGraph[NodeType, ArrowType, Tsor](
            self._graph_comp(), ())

        # register wraping attributes
        self.add = self._graph.add
        self.graph = self._graph.graph
        self.arrows = self._graph.arrows

        # fill the cache with provided arrows
        for arrow in arrows:
            self.add(arrow)


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
            return self._graph[arrow][1]

        # if arrow has length 0, return data corresponding to its node
        return self._datas[arrow[0]]  # type: ignore

    @property
    def label_universe(self) -> Mapping[ArrowType, Tsor]:
        """
        access the label encoding
        """
        return MappingProxyType(self._label_universe)

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
        return self._graph[arrow][0]

    def __iter__(self) -> Iterator[CompositeArrow[NodeType, ArrowType]]:
        """
        Iterate through arrows contained in the cache
        """
        return iter(self._graph)

    def __len__(self) -> int:
        """
        Return the number of composite arrows in the structure
        """
        return len(self._graph)

    def __setitem__(self, node: NodeType, data_point: Tsor) -> None:
        """
        set value of a new data point.
        Note: cannot set the value of an arrow, as these are computed using
        relation generation and compositions.
        """
        self._datas[node] = data_point

    def __delitem__(self, arrow: CompositeArrow[NodeType, ArrowType]) -> None:
        """
        Remove a composite arrow from the cache
        """
        del self._graph[arrow]

    def __repr__(self) -> str:
        """
        Returns a string representation of the cache
        """
        return (
            f"{type(self).__name__}"
            + f"(datas: {repr(self._datas)}"
            + ", arrows: {"
            + ", ".join(f"{key}: {value}" for (key, value) in self.items())
            + "})")

    def flush(self) -> None:
        """
        Flush all relations and datas content
        """
        self._graph.flush()
        self._datas = {}
        self._causality_cost = torch.zeros(())
        self._nb_compositions = 0

    def match(
            self, labels: DirectedGraph[NodeType],
            match_negatives: bool = True) -> Tsor:
        """
        Match the composition graph with a graph of labels. For each label
        vector, get the best match in the graph. if No match is found, set to
        + infinity.
        If no arrow is found in the cache, order 1 relations are generated
        for matching. These relations
        are added to the cache.
        """
        def matcher(score, label):
            """
            Used to compare a relation's score vector with a label
            """
            kl_div = subproba_kl_div(score, label)
            if match_negatives:
                return kl_div - subproba_kl_div(score, torch.zeros(score.shape))
            return kl_div

        result_graph = DirectedGraph[NodeType]()

        if match_negatives:
            for arr in self.arrows():
                if not result_graph.has_edge(arr[0], arr[-1]):
                    result_graph.add_edge(arr[0], arr[-1])
                score = self[arr]
                match = subproba_kl_div(score, torch.zeros(score.shape))
                result_graph[arr[0]][arr[-1]][
                    arr.derive()
                ] = arr.derive(), match

        for src, tar in labels.edges:
            # add edge if necessary
            if not result_graph.has_edge(src, tar):
                result_graph.add_edge(src, tar)

            # check if arrows exist to match label, add order 1 arrows
            # if they don't
            if (
                    not self.graph.has_edge(src, tar)
                    or not self.graph[src][tar]):
                for arr in self.label_universe:
                    self.add(CompositeArrow([src, tar], [arr]))

            # get scores of existing arrows from src to tar
            scores = {
                arr.derive(): self[arr]
                for arr in self.arrows(src, tar)}

            # go through labels and match them. Keep only the best
            for name, label in labels[src][tar].items():
                # evaluate candidate relationships
                candidates = {
                    arr: matcher(score, label)
                    for arr, score in scores.items()}

                # save the best match in result graph
                best_match = min(candidates, key=candidates.get)

                negative_match = 0
                if best_match in result_graph[src][tar]:
                    # get negative match score and remove it from graph
                    negative_match = result_graph[src][tar][best_match][1]
                    del result_graph[src][tar][best_match]

                result_graph[src][tar][name] = (
                    best_match, candidates[best_match] + negative_match)
        return result_graph
