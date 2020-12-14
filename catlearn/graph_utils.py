"""
Created on Wed Mar 13 13:47:33 2019

@author: christophe_c
A library for graph utilities used in our project.
"""
# from __future__ import annotations
from typing import (
    Iterable, FrozenSet, Sequence, TypeVar, Generic,
    Optional, Callable, Tuple, Iterator, Type, Any, Mapping)
from itertools import chain, product
import warnings
from collections import abc
import random
import pickle

import numpy as np
import networkx as nx
from networkx import DiGraph, NetworkXError, pagerank, hits
from catlearn.utils import (str_color, one_hot)
from catlearn.utils import (init_logger, str_color)


NodeType = TypeVar("NodeType")
ArrowType = TypeVar("ArrowType")



def mapping_product(mapping0: Mapping, mapping1: Mapping) -> Iterator:
    """
    From two mapping {k0: d0, ...}, {k1: d1, ...},
    returns an iterator emulating items of a dictionary {(k0, k1): (d0, d1)}
    """
    return (
        tuple(zip(pair0, pair1))
        for (pair0, pair1) in product(mapping0.items(), mapping1.items()))


class DirectedGraph(Generic[NodeType], DiGraph, abc.MutableMapping):  # pylint: disable=unsubscriptable-object
    """
    A class to encapsulate directed graphs. Can be initialized from a dictionary:
    Values are iterable which must range over keys of the dictionary.
    For a key k,  v in self[k] means there is a vertex k -> v.
    Complete list of available input formats:
    https://networkx.github.io/documentation/stable/reference/classes/digraph.html
    Different graph composition functions are provided in order to facilitate
    the generation of examples.
    """
    def __delitem__(self, node: NodeType) -> None:
        """
        delete a node from the graph
        """
        try:
            self.remove_node(node)
        except NetworkXError as cannot_remove_node:
            raise KeyError(f"{node} could not be found in graph") from cannot_remove_node

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
    def op(self) -> 'DirectedGraph[NodeType]':
        """
        returns the opposite graph
        """
        return self.reverse(copy=False)

    def dualize_relations(self) -> 'DirectedGraph[NodeType]':
        """
        returns the dually augmented graph

        No-op if graph has no labels.
        For each labelled edge (src, dst, label), the resulting
        graph will have 2 edges:
            * (src, dst, (label, False)) : the original edge
            * (src, dst, (label, True)): the dual label
        Comment: the label value is copied between an edge and its dual
        """
        dualg = DirectedGraph()
        for src, dst in self.edges():
            dualg.add_edge(src, dst)
            for label, value in self[src][dst].items():
                dualg[src][dst][(label, False)] = value
                dualg[src][dst][(label, True)] = value  # Dual
        return dualg


    def under(self, root: NodeType) -> FrozenSet[NodeType]:
        """
        Returns the set of all nodes accessible from given root,
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

    def subgraph_nodes(self, nodes: Iterable[NodeType]) -> 'DirectedGraph[NodeType]':
        """
        Extract a subgraph of current graph constituted of given nodes
        """
        nodes_set = frozenset(nodes)
        return type(self)(
            {node: set(self[node]) & nodes_set for node in nodes_set})

    def prune(self, node: NodeType) -> 'DirectedGraph[NodeType]':
        """
        Prune a node out of the graph, but making sure that for any subgraph
        i -> pruned_node -> j, there is a i -> j vertex added
        """
        # get parents and children of node
        parents = self.op[node]
        children = self[node]

        # delete node - will handle self (x->x) and back reference (x->y->x)
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
            self, other_graph: 'DirectedGraph[NodeType]'
    ) -> 'DirectedGraph[Tuple[bool, NodeType]]':
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
            self, other_graph: 'DirectedGraph[NodeType]'
    ) -> 'DirectedGraph[Tuple[NodeType, NodeType]]':
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
            self, other_graph: 'DirectedGraph[NodeType]'
    ) -> 'DirectedGraph[Tuple[bool, NodeType]]':
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
            other_graph: 'DirectedGraph[NodeType]'
    ) -> 'DirectedGraph[Tuple[NodeType, NodeType]]':
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
            self, other_graph: 'DirectedGraph[NodeType]'
    ) -> 'DirectedGraph[Tuple[NodeType, NodeType]]':
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
    ) -> 'DirectedGraph[Any]':
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

    def integerify(self) -> 'DirectedGraph[int]':
        """
        Remap the graph nodes to integers, starting from 0 in the order of
        access in the underlying dictionary
        """
        # create a dictionary which will be used to create the mapping
        sorted_nodes = enumerate(sorted(self, key=pickle.dumps))
        remapping = {node: index for index, node in sorted_nodes}
        return self.remap_names(remapping.get)

    def stringify(self) -> 'DirectedGraph[str]':
        """
        Remap the graph nodes to strings generated from the
        hash code of each key
        """
        return self.integerify().remap_names(hex)

    def rand_prune(
            self, pruning_factor: float,
            random_generator: Optional[random.Random] = None
    ) -> 'DirectedGraph[NodeType]':
        """
        Randomly prune nodes out of the graph. The number of nodes pruned out
        of the graph is floor(pruning_factor * len(self)),
        chosen without replacement
        """
        assert 0 <= pruning_factor <= 1, (
            "pruning_factor should be a number between 0. and 1.")
        nb_to_prune = int(np.floor(pruning_factor * len(self)))

        # Node sampler
        #pylint: disable=unnecessary-lambda
        if random_generator is None:
            choice = lambda g: random.choice(g)
        else:
            choice = lambda g: random_generator.choice(g)

        # prune nodes
        for _ in range(nb_to_prune):
            if len(self) == 0: # Choice function requires non-empty collections
                break
            # Draw node to prune
            node = choice(list(self)) # type: ignore
            self.prune(node)

        return self


class DirectedAcyclicGraph(DirectedGraph[NodeType]):
    """
        A class to model general directed acyclic graphs
    """
    def __init__(self, *args, **kwargs) -> None:
        """
            Create an acyclic directed graph from a dictionary.
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
        assert len(weights) == len(__class__.OPS), (  # type: ignore # pylint: disable=undefined-variable
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
        # Use reverse with copy instead of op, otherwise we get a 'frozen'
        # view into the graph and we cannot prune it
        operands = (
            map(
                lambda graph, var: graph if var else graph.reverse(copy=True),
                self._random_generator.choices(self.graphs, k=op_nargs),
                ops_vars)
            for op_nargs, ops_vars in zip(ops_nargs, operands_variance))

        # generate new graphs
        self._graphs = tuple(
            op(*args).stringify()
            for (_, op), args in zip(ops, operands))

        return self._graphs


def generate_random_graph(
        nb_steps: int, random_generator: random.Random, *args):
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
        [0.15, 0.15, 0.15, 0.15, 0.14],
        5, 0.15, random_generator, *args)

    # go through generation steps
    for _ in range(nb_steps):
        next(factory)  # type: ignore

    # return a randomly chosen graph in the factory's memory
    idx = random_generator.randint(0, factory.nb_graphs - 1)
    return factory.graphs[idx]


def pagerank_sample(
        graph: DirectedGraph[NodeType],
        sample_vertices_size: int,
        rng: random.Random,
        **kwargs) -> DirectedGraph[NodeType]:
    """
    Sample a random subgraph of `graph` with respect to vertices PageRank scores.

    Details of other parameters: see `sample`
    Pagerank calculation: see `networkx.pagerank`
    NB: `kwargs are all passed to `networkx.pagerank`
    """
    #pylint: disable=unnecessary-lambda
    return sample(
        graph, sample_vertices_size, lambda g: pagerank(g, **kwargs), rng)


def hubs_sample(
        graph: DirectedGraph[NodeType],
        sample_vertices_size: int,
        rng: random.Random,
        **kwargs) -> DirectedGraph[NodeType]:
    """
    Sample a random subgraph of `graph` with respect to vertices HITS hubs scores.

    Details of other parameters: see `sample`
    Pagerank calculation: see `networkx.hits`
    NB: `kwargs are all passed to `networkx.hits`
    """
    return sample(
        graph, sample_vertices_size, lambda g: hits(g, **kwargs)[0], rng)


def authorities_sample(
        graph: DirectedGraph[NodeType],
        sample_vertices_size: int,
        rng: random.Random,
        **kwargs) -> DirectedGraph[NodeType]:
    """
    Sample a random subgraph of `graph` with respect to vertices HITS authorities scores.

    Details of other parameters: see `sample`
    Pagerank calculation: see `networkx.hits`
    NB: `kwargs are all passed to `networkx.hits`
    """
    return sample(
        graph, sample_vertices_size, lambda g: hits(g, **kwargs)[1], rng)


def sample(
        graph: DirectedGraph[NodeType],
        sample_vertices_size: int,
        ranking: Callable[[DirectedGraph[NodeType]], Mapping[NodeType, float]],
        rng: random.Random,
        with_edges=False) -> DirectedGraph[NodeType]:
    """
    Sample a random subgraph of `graph` with respect to a probability
    distribution `ranking` over the vertices.

    Params:
    - graph: Input graph
    - sample_vertices_size: number of vertices to sample
        (with replacement so actual number might be lower)
    - ranking: probability distribution over the input graph vertices
    - rng: Random generator
    - with_edges: Ensure that every returned vertex has an edge
    NOTE: if sample_vertices_size is odd and with_edges=True,
    number of returned edges will be sample_vertices_size-1
    NOTE: warning print to be replaced with appropriate logging

    Returns:
    A valid random subgraph

    """
    if not with_edges:
        ranks = list(ranking(graph).items())
    else:
        ranks = list(ranking(graph.edges).items())
        # Check if sample_vertices_size is odd
        if sample_vertices_size & 1:
            warning_str = (f'sample_vertices_size {sample_vertices_size} is odd '
            f'and will be rounded to floor even number {sample_vertices_size - 1}'
            'when with_edges=True')
            warnings.warn(str_color('W', warning_str), UserWarning)
        sample_vertices_size //= 2

    # NOTE: all vertices are returned every call. Not optimal. Can be kept in memory
    # and sampled several times. Proposal to integrate into dataloader.
    vertices, weights = zip(*ranks)

    if not with_edges:
        sampled_vertices = set(rng.choices(vertices, weights, k=int(sample_vertices_size)))
    else:
        sampled_vertices = rng.choices(vertices, weights, k=int(sample_vertices_size))
        sampled_vertices = set([v for tpl in sampled_vertices for v in tpl])

    return graph.subgraph(sampled_vertices)


def uniform_sample(
        graph: DirectedGraph[NodeType],
        sample_vertices_size: int,
        rng: random.Random,
        with_edges: bool=False) -> DirectedGraph[NodeType]:
    """
    Sample a random subgraph of `graph` with a uniform probability over vertices
    Edges are returned.
    Details of parameters: see `sample`
    NOTE: uniform sampler implies equal weights.
    """
    weight = 1./len(graph)
    return sample(graph, sample_vertices_size,
                  lambda G: {v: weight for v in G}, rng, with_edges)


def random_walk_vertex_sample(
        graph: DirectedGraph[NodeType],
        rng: random.Random,
        n_iter: int,
        seeds: Optional[Iterable[NodeType]] = None,
        n_seeds: int = 1,
        use_opposite: bool = False) -> DirectedGraph[NodeType]:
    """
    Random Walk graph vertex subsampling
    Graph must be inverse-completed

    Params:
    - graph: the input graph
    - rng: random number generator to use
    - n_iter: number of iterations
    - seeds: to root the walk on specific vertices
    - n_seeds: to start random walk on specified number of uniformly selected vertices
    - use_opposite: if True, randomly walk either direct or dual graph fairly

    Number of returner nodes is always inferior to the number of iterations.
    Returns: a valid subgraph, where all edges existing between sampled vertices are kept
    """
    if len(graph) == 0 or n_seeds == 0:
        return DiGraph()
    if n_seeds * 2 < n_iter:
        warning_str = (f'n_iter {n_iter} must be at least > 2 * n_seeds'
            f' {2 * n_seeds} for consistent sampling results.')
        warnings.warn(str_color('W', warning_str), UserWarning)
    if seeds is None:
        sampled_vertices = list(rng.choices(list(graph), k=n_seeds))
    else:
        sampled_vertices = list(seeds)
    i = 0
    while i < n_iter:
        i += 1
        v = rng.choice(sampled_vertices)
        use_op = use_opposite and rng.randint(0, 1) # Flip a coin
        connected_vertices = list(graph.op[v] if use_op else graph[v])
        # print(connected_vertices)
        if connected_vertices:
            sampled_vertices.append(rng.choice(connected_vertices))
    sampled_subg = graph.subgraph(sampled_vertices)
    # Heuristic: subgraph edges must be at x2 n_seeds; if not, double n_iter and repeat
    if len(sampled_subg.edges) > n_seeds * 2 or n_iter > 10^6:
        return sampled_subg
    random_walk_vertex_sample(graph, rng, n_iter * 2, seeds, n_seeds, use_opposite)


def random_walk_edge_sample(
        graph: DirectedGraph[NodeType],
        rng: random.Random,
        n_iter: int,
        seeds: Optional[Iterable[ArrowType]] = None,
        n_seeds: int = 1,
        use_opposite: bool = False,
        use_both_ends: bool = False) -> DirectedGraph[NodeType]:
    """
    Random Walk graph edge subsampling
    NOTE: for large n_seeds equivalent for breath-first graph traversal
    with maximal depth of 1 and maximal degree of connection 3.
    NOTE: maximal degree of connection is not equivalent to graph diameter,
    but can be assumed as diameter for simplicity.

    Params:
    - graph: the input graph
    - rng: random number generator to use
    - n_iter: number of iterations
    - seeds: to root the walk on specific vertices
    - n_seeds: to start random walk on specified number of uniformly selected vertices
    - use_opposite: if True, randomly walk either direct or dual graph fairly
    - use_both_ends: if True, randomly select either end of sampled edges for growing the walk

    Returns: a valid subgraph

    Given an edge 1->2 in the graph 0->1->2->3 + 1->4 + 5->2:
    * use_opposite=False, use_both_ends=False:
      -> only candidate is 2->3
    * use_opposite=True, use_both_ends=False:
      -> candidates are 2->3 and 5->2
    * use_opposite=False, use_both_ends=True:
      -> candidates are 2->3 and 1->4
    * use_opposite=True, use_both_ends=True:
      -> candidates are 2->3, 5->2, 1->4 and 0->1
    """
    if len(graph.edges) == 0:
        return DirectedGraph()
    if seeds is None:
        sampled_edges = list(rng.choices(list(graph.edges), k=n_seeds))
    else:
        sampled_edges = list(seeds)
    i = 0
    sampled_graph = DirectedGraph(sampled_edges)
    while i < n_iter:
        i += 1
        e = rng.choice(sampled_edges)
        use_op = use_opposite and rng.randint(0, 1) # Flip a coin
        use_src = use_both_ends and rng.randint(0, 1) # Flip a coin
        src = e[0] if use_src else e[1]
        gview = graph.op if use_op else graph
        compatible_edges = [
            (dst, {}) for dst in gview[src] if not gview[src][dst]
        ] + [
            (dst, {k: v}) for dst in gview[src] for k,v in gview[src][dst].items()
        ]
        if compatible_edges:
            dst, labels = rng.choice(compatible_edges)
            if use_op:
                sampled_graph[dst] = {src: labels}
            else:
                sampled_graph[src] = {dst: labels}
    return sampled_graph


def n_hop_sample(
        graph: 'DiGraph[NodeType]',
        n_hops: int,
        seeds: Optional[Iterable[ArrowType]] = None,
        n_seeds: int = 1,
        rng: Optional[random.Random] = None) -> 'DiGraph[NodeType]':
    """
    N-hop sampling from random or specified locations.
    Only samples in straight edge direction.
    No control over length of returned graph

    Params:
    - graph: the input graph
    - n_hops: number of hops
    - seeds: to root the walk on specific vertices
    - n_seeds: to start random walk on specified number of uniformly selected vertices
    - rng: random number generator to use, default to random.Random

    Returns: a valid subgraph, where all edges existing between sampled vertices are kept
    """
    if len(graph) == 0 or n_hops <= 0:
        return DiGraph()
    if seeds is None:
        if rng is None:
            rng = random.Random()
        sampled_vertices = set(rng.choices(list(graph), k=n_seeds))
    else:
        sampled_vertices = set(seeds)
    for _ in range(1, n_hops):  # Seed sampling is considered 1st hop
        visited_vertices = set()
        for v in sampled_vertices:
            visited_vertices.update(list(graph[v]))
        sampled_vertices |= visited_vertices
    return graph.subgraph(sampled_vertices)


def clean_selfloops(
    graph: DirectedGraph[NodeType],
    clean_edges: bool=True,
    clean_isolate_nodes: bool=True):
    """Inspect and clean graph self-loops.
    clean_edges=True will remove edges, but isolates nodes
    would possibly be created.
    It's recommended to keep clean_isolate_nodes=True
    """
    selfloop_edges = list(nx.selfloop_edges(graph))
    print(f'\nFollowing {len(selfloop_edges)} selfloop edge(s) are found:\n{selfloop_edges}')
    if clean_edges:
        graph.remove_edges_from(selfloop_edges)
        print(f'{len(selfloop_edges)} selfloop edges are removed.')
    if clean_isolate_nodes:
        clean_isolates(graph)


def clean_isolates(
    graph: DirectedGraph[NodeType],
    clean: bool=True):
    """Clean nodes with zero connections (either in- or outbound)."""
    isolates = list(nx.isolates(graph))
    print(f'Following {len(isolates)} isolate(s) are found:\n{isolates}')
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if clean:
        graph.remove_nodes_from(isolates)
        print(f'{len(isolates)} isolates are removed.')


def init_relation_vectors(relation2id: dict) -> dict:
    """one-hot representation of entities
    equivalent to label_universe required by TrainableDecisionCatModel
    TODO: Parametrize encoding type. OHE could be one of possible encoding schemes.
    In current implementation, other encodings like embedding vectors are not considered.
    """
    nb_relations = len(relation2id)
    return {i: one_hot(i, nb_relations) for i in relation2id.values()}


def augment_graph(graph: DirectedGraph, revers_rels: dict):
    """revers_rels: dict of corresponding opposite relations.
    rels_revers format example: {1: 1, 2: 4, 3: 13, 4: None}
    where 1, 2 are from the existing label universe
    13 is created. None means there is not opposite.
    If opposite relations are created, label_universe must be updated.
    It's recommended to create revers_rels dictionary
    with create_revers_rels method.
    --------------
    Only for Directional, not multirelational graphs (single relation per edge).
    Edge data are created in the format {id: None}.
    If there are custom edge data different from None,
    another function must be created.
    """
    for src, dst, rel in graph.edges(data=True):
        rel_id = list(rel.keys())[0]
        revers_id = revers_rels.get(rel_id)
        if (revers_id and not graph.has_edge(dst, src)):
            graph.add_edge(dst, src)
            graph[dst][src].update({revers_id: None})


def create_revers_rels(revers_rels_str: dict, relation2id: dict) -> 'list(dict, dict, dict)':
    """Create new relation2id and relation_id2vec dictionaries, as well as
    revers_rels dictionary, that should be passed to augment_graph function.
    """
    relation2id_augmented = {}
    relation_id2vec_augmented = {}
    revers_rels = {}
    offset = len(relation2id)
    i = 0
    for rel, revers in revers_rels_str.items():
        rel_id = relation2id[rel]
        relation2id_augmented[rel] = rel_id
        if not revers:
            continue
        if relation2id.get(revers):
            revers_id = relation2id[revers]
        else:
            revers_id = i + offset
            i += 1
        revers_rels[rel_id] = revers_id
        relation2id_augmented[revers] = revers_id
    relation_id2vec_augmented = init_relation_vectors(relation2id_augmented)
    return [relation2id_augmented, relation_id2vec_augmented, revers_rels]
