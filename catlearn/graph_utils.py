"""
Created on Wed Mar 13 13:47:33 2019

@author: christophe_c
A library for graph utilities used in our project.
"""
from __future__ import annotations
from typing import (
    Iterable, FrozenSet, Sequence, TypeVar, Generic,
    Optional, Callable, Tuple, Iterator, Type, Any, Mapping)
from itertools import chain, product
from collections import abc
import random
import pickle

import numpy as np
from networkx import DiGraph, NetworkXError

NodeType = TypeVar("NodeType")


def mapping_product(mapping0: Mapping, mapping1: Mapping) -> Iterator:
    """
    from two mapping {k0: d0, ...}, {k1: d1, ...},
    returns an iterator emulating items of a dictionary {(k0, k1): (d0, d1)}
    """
    return (
        tuple(zip(pair0, pair1))
        for (pair0, pair1) in product(mapping0.items(), mapping1.items()))


class DirectedGraph(Generic[NodeType], DiGraph, abc.MutableMapping):  # pylint: disable=unsubscriptable-object
    """
    A class to encapsulate directed graphs. Initialized from a dictionary:
    Values are iterable which must range over keys of the dictionary.
    For a key k,  v in self[k] means there is a vertex k -> v.
    Different graph composition functions are provided in order to facilitate
    the generation of examples.
    """
    def __delitem__(self, node: NodeType) -> None:
        """
        delete a node from the graph
        """
        try:
            self.remove_node(node)
        except NetworkXError:
            raise KeyError(f"{node} could not be found in graph")

    def __setitem__(
            self, node: NodeType, children: Iterable[NodeType]) -> None:
        """
        add given edge to the graph
        """
        # wether input is a mapping
        is_mapping = isinstance(children, Mapping)

        self.add_node(node)
        for child in children:
            self.add_edge(node, child)
            if is_mapping:
                labels = children[child]  # type: ignore
                if isinstance(labels, Mapping):
                    self[node][child].update(labels)

    @property
    def op(self) -> DirectedGraph[NodeType]:
        """
        returns the opposite graph
        """
        return self.reverse(copy=False)

    def under(self, root: NodeType) -> FrozenSet[NodeType]:
        """
        Returns the set of all nodes acessible from given root,
        (not included root itself unless there is a cycle going back to it)
        """
        # get the set of first order children
        first_children = self[root]

        # use _under function to get subsequent children
        return self._under(frozenset(first_children))

    def _under(self, roots: FrozenSet[NodeType]) -> FrozenSet[NodeType]:
        """
        Returns the sef of all nodes accessible from given roots
        (including roots themselves)
        """
        # get the set of first order children which are not roots
        first_children = frozenset(chain(*(self[root] for root in roots)))

        # if first children are among roots, return roots
        if first_children <= roots:
            return roots

        # otherwise make recursive call on roots
        return self._under(roots | first_children)

    def over(self, root: NodeType) -> FrozenSet[NodeType]:
        """
        Returns the set of all nodes from which given root can be accessed,
        (not included root itself unless there is a cycle going back to it)
        """
        return self.op.under(root)

    def subgraph(self, nodes: Iterable[NodeType]) -> DirectedGraph[NodeType]:
        """
        Extract a subgraph of current graph constituted of given nodes
        """
        nodes_set = frozenset(nodes)
        return type(self)(
            {node: set(self[node]) & nodes_set for node in nodes_set})

    def prune(self, node: NodeType) -> DirectedGraph[NodeType]:
        """
        Prune a node out of the graph, but making sure that for any subgraph
        i -> pruned_node -> j, there is a i -> j vertex added
        """
        # get parents and children of node
        parents = self.op[node].copy()
        children = self[node].copy()

        # delete node
        del self[node]

        # add children to all parents' children sets
        for (parent, child) in product(parents, children):
            self.add_edge(parent, child)
            self[parent][child].update(mapping_product(
                parents[parent], children[child]))

        return self

    def __repr__(self) -> str:
        """
        return string representation of object
        """
        return (
            f"{type(self).__name__}("
            + ", ".join(
                f"{node} > {children}"
                for node, children in self.items())
            + ")")

    def __or__(
            self, other_graph: DirectedGraph[NodeType]
    ) -> DirectedGraph[Tuple[bool, NodeType]]:
        """
        Generates a new directed graph by making the disjoint sum
        of both graphs. Changes the nodes to:
            (False, node) for nodes from the first graph
            (True, node) for nodes from the second
        """
        remap_self = {
            (False, key): {
                (False, value):  labels for value, labels in values.items()}
            for key, values in self.items()}
        remap_other = {
            (True, key): {
                (True, value): labels for value, labels in values.items()}
            for key, values in other_graph.items()}
        return type(self)({**remap_self, **remap_other})

    def __and__(
            self, other_graph: DirectedGraph[NodeType]
    ) -> DirectedGraph[Tuple[NodeType, NodeType]]:
        """
        Generates a new directed graph by making the cartesian product
        of both graphs, in the category theoretic sense of the term.
        """
        return type(self)({
            (key0, key1): {
                product_node: mapping_product(*labels)
                for product_node, labels in mapping_product(value0, value1)}
            for (key0, value0), (key1, value1)
            in product(self.items(), other_graph.items())})

    def __add__(
            self, other_graph: DirectedGraph[NodeType]
    ) -> DirectedGraph[Tuple[bool, NodeType]]:
        """
        Generates a new directed graph by making the directed join of both
        graphs:
            nodes are:
                (False, node) for nodes in the first graph
                (True, node) for nodes in the second graph
            edges are:
                (x, n0) -> (x, n1) for n0, n1 nodes in the same base graph
                (False, n0) -> (True, n1) for any nodes n0 and n1 in the
                    first and second graph respectively?
        """
        remap_self = {
            (False, key): {
                **{(False, value): labels for value, labels in values.items()},
                **{(True, other_key): {} for other_key in other_graph}}
            for key, values in self.items()}
        remap_other = {
            (True, key): {
                (True, value): labels for value, labels in values.items()}
            for key, values in other_graph.items()}
        return type(self)({**remap_self, **remap_other})

    def __matmul__(
            self,
            other_graph: DirectedGraph[NodeType]
    ) -> DirectedGraph[Tuple[NodeType, NodeType]]:
        """
        Generates a new directed graph by making the product of the graph in
        that:
            - nodes are pairs (node0, node1) of nodes of each graph
            - (s0, s1) and (t0, t1) are linked by an edge if there is an edge
            (s0, t0), or (s1, t1)
        """
        return type(self)({
            (key0, key1): {
                **{
                    (value0, key1): {
                        (False, label_name): label_value
                        for (label_name, label_value) in labels0.items()}
                    for (value0, labels0) in self[key0].items()},
                **{
                    (key0, value1): {
                        (True, label_name): label_value
                        for (label_name, label_value) in labels1.items()}
                    for (value1, labels1) in other_graph[key1].items()}
                }
            for key0, key1 in product(self, other_graph)})

    def __mul__(
            self, other_graph: DirectedGraph[NodeType]
    ) -> DirectedGraph[Tuple[NodeType, NodeType]]:
        """
        Generates a new directed graph by making the directed product of both
        graphs (lexicographic order)
        """
        return type(self)({
            (key0, key1): {
                **{
                    (value0, key1): labels0
                    for value0, labels0 in values0.items()},
                **{
                    (key, value1): labels1
                    for (key, (value1, labels1))
                    in product(self, values1.items())}}
            for (key0, values0), (key1, values1)
            in product(self.items(), other_graph.items())})

    def remap_names(
            self, key_func: Callable[[Any], Any]
    ) -> DirectedGraph[Any]:
        """
        Remap the graph nodes to new names using a function key_func to
        generate the new names.
        Be careful that the function is 1-to-1 if you want to make sure you
        do not lose certain properties of the graph, such as being acyclic
        """
        return type(self)({
            key_func(key): {
                key_func(value): labels
                for value, labels in values.items()}
            for key, values in self.items()})

    def integerify(self) -> DirectedGraph[int]:
        """
        Remap the graph nodes to integers, starting from 0 in the order of
        access in the underlying dictionary
        """
        # create a dictionary which will be used to create the mapping
        sorted_nodes = enumerate(sorted(self, key=pickle.dumps))
        remapping = {node: index for index, node in sorted_nodes}
        return self.remap_names(remapping.get)

    def stringify(self) -> DirectedGraph[str]:
        """
        Remap the graph nodes to strings generated from the
        hash code of each key
        """
        return self.integerify().remap_names(hex)

    def rand_prune(
            self, pruning_factor: float,
            random_generator: Optional[random.Random] = None
    ) -> DirectedGraph[NodeType]:
        """
        Randomly prune nodes out of the graph. The number of nodes pruned out
        of the graph is floor(pruning_factor * len(self)),
        chosen without replacement
        """
        assert 0 <= pruning_factor <= 1, (
            "pruning_factor should be a number between 0. and 1.")
        nb_to_prune = int(np.floor(pruning_factor * len(self)))

        # draw nodes to be pruned
        if random_generator is None:
            to_prune = random.sample(
                list(self), k=nb_to_prune)  # type: ignore
        else:
            to_prune = random_generator.sample(
                list(self), k=nb_to_prune)  # type: ignore

        # prune nodes
        for node in to_prune:
            self.prune(node)

        return self


class DirectedAcyclicGraph(DirectedGraph[NodeType]):
    """
        A class to model general directed acyclic graphs
    """
    def __init__(self, *args, **kwargs) -> None:
        """
            Create an acyclic directed graph from a dictinary.
        """

        # init self as directed graph
        super().__init__(*args, **kwargs)

        # check that it is indeed a directed acyclic graph
        assert all(node not in self.under(node) for node in self), (
            "A directed acyclic graph cannot contain cycles")

    @property
    def ins(self) -> FrozenSet[NodeType]:
        """
        roots of the graph
        """
        return frozenset(node for node in self if not self.op[node])

    @property
    def outs(self) -> FrozenSet[NodeType]:
        """
        roots of the opposite graph
        """
        return frozenset(node for node in self if not self[node])

    def __setitem__(self, node: NodeType, children: Iterable[NodeType]):
        """
        item setting is forbidden in a DirectedAcyclicGraph. Go through
        a DirectedGraph representation if you need to do this, then recreate
        a DirectedAcyclicGraph out of it
        """
        raise NotImplementedError(
            "item setting is not supported by DirectedAcyclicGraph objects")


class GraphRandomFactory:
    """
    A class for factories for random graph generation.
    Arguments to the constructor are:
        weights: a sequence of 5 floating point numbers, whose sum
            is less than 1, for the weights of choosing respectively the
            or, and, add, mul, matmul
        nb_graphs: the number of graphs kept into memory by the generator
        pruning_factor: the max pruning factor for the eroding operation.
            The eroding operation is chosen whenever no other operation is
            chosen, and randomly prunes a number of points of one of the graphs
        random_generator: random.Random, the random generator to use
        *initial_graphs: DirectedGraph objects, seeds to initialize the random
            generation (at most nb_graphs). If not provided, a default seed
            point graph will be used.
    """
    OPS: Tuple[Callable[..., DirectedGraph], ...] = (
        DirectedGraph.__or__,
        DirectedGraph.__and__,
        DirectedGraph.__add__,
        DirectedGraph.__mul__,
        DirectedGraph.__matmul__)
    OPS_NARGS = (2, 2, 2, 2, 2)

    DEFAULT_SEED_GRAPH = DirectedGraph[str]({"0x0": []}).stringify()

    @classmethod
    def __init_subclass__(cls: Type) -> None:
        """
        Verify that the defined OPS and OPS_NARGS are tuples of same length
        """
        assert isinstance(cls.OPS, tuple), "OPS should be a tuple"
        assert isinstance(cls.OPS_NARGS, tuple), "OPS_NARGS should be a tuple"
        assert len(cls.OPS) == len(cls.OPS_NARGS), (
            "OPS and OPS_NARGS should have the same length")

    def __init__(
            self, weights: Sequence[float], nb_graphs: int,
            pruning_factor: float,
            random_generator: random.Random,
            *initial_graphs: DirectedGraph[Any]) -> None:
        """
        create a new factory
        """
        assert 0. <= pruning_factor <= 1., (
            "pruning factor should be between 0. and 1.")
        assert len(weights) == len(__class__.OPS), (  # type: ignore
            "Weights sequence should be of length 5, for"
            " or, and, add, mul, matmul operations respectively")
        assert 0 <= len(initial_graphs) <= nb_graphs, (
            "number of initializer "
            "graphs should be at most nb_graphs: {nb_graphs}")
        assert all(weight >= 0. for weight in weights), (
            "weights should be positive")
        assert sum(weights) <= 1, "Weights should sum to less to 1."

        self._pruning_factor = pruning_factor
        self._weights = np.array(tuple(weights) + (1. - sum(weights),))
        self._nb_graphs = nb_graphs
        self._random_generator = random_generator
        self._ops = type(self).OPS + (self.erode,)
        self._ops_nargs = type(self).OPS_NARGS + (1,)

        self._graphs: Tuple[DirectedGraph, ...]
        if initial_graphs:
            self._graphs = tuple(graph.stringify() for graph in initial_graphs)
        else:
            self._graphs = (type(self).DEFAULT_SEED_GRAPH,)

    def erode(self, graph: DirectedGraph) -> DirectedGraph:
        """
        Erode the given graph by pruning it of a factor of at most the
        pruning_factor of the factory.
        """
        # get a random pruning factor between 0. and actual pruning_factor
        pruning_factor = self._random_generator.uniform(
            0., self._pruning_factor)
        return graph.rand_prune(pruning_factor)

    @property
    def nb_graphs(self) -> int:
        """
        Number of graphs held in memory by the factory
        """
        return self._nb_graphs

    @property
    def graphs(self) -> Tuple[DirectedGraph, ...]:
        """
        tuple containing the graphs held in the memory of the factory
        """
        return self._graphs

    @property
    def weights(self) -> np.ndarray:
        """
        weights of or, and, add, mul operations for the factory
        """
        return self._weights

    @property
    def ops(self) -> Tuple[Callable[..., DirectedGraph], ...]:
        """
        return list of operations of the factory
        """
        return self._ops

    @property
    def ops_nargs(self) -> Tuple[int, ...]:
        """
        number of arguments of each operand
        """
        return self._ops_nargs

    def __next__(self) -> Tuple[DirectedGraph, ...]:
        """
        generate next step graphs from current ones
        """
        # draw operations to apply for each new graph
        ops = self._random_generator.choices(
            list(enumerate(self.ops)), weights=self.weights,
            k=self.nb_graphs)

        # number of operands for each operation
        ops_nargs = [self.ops_nargs[index] for index, _ in ops]

        # choose wether the operands are taken as direct or opposite
        operands_variance = (
            self._random_generator.choices([False, True], k=op_nargs)
            for op_nargs in ops_nargs)

        # get operands for each operation, with their right variance
        operands = (
            map(
                lambda graph, var: graph if var else graph.op,
                self._random_generator.choices(self.graphs, k=op_nargs),
                ops_vars)
            for op_nargs, ops_vars in zip(ops_nargs, operands_variance))

        # generate new graphs
        self._graphs = tuple(
            op(*args).stringify()
            for (_, op), args in zip(ops, operands))

        return self._graphs


def generate_random_graph(
        nb_steps: int, random_generator: random.Random,
        *args, **kwargs):
    """
    Generate a random graph using a random factory.
    Arguments:
        - nb_steps: the number of steps used for the generation of the factory
        - random_generator: the pseudo-random generator to be used
        - *args, **kwargs: arguments passed to the factory. The following
            default values will be used:
                pruning_factor: 0.15
                weights: 0.15, 0.15, 0.15, 0.15, 0.15
                nb_graphs: 5
    """
    # create factory
    factory = GraphRandomFactory(
        [0.15, 0.15, 0.15, 0.15, 0.14], 5, 0.15, random_generator,
        *args, **kwargs)

    # go through generation steps
    for _ in range(nb_steps):
        next(factory)  # type: ignore

    # return a randomly chosen graph in the factory's memory
    idx = random_generator.randint(0, factory.nb_graphs - 1)
    return factory.graphs[idx]
