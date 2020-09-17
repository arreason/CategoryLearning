"""
A library for generating artificial examples for catlearn tests
Variable dependencies are represented by directed acyclic graphs (DAG),
each node is labeled with the generating function to use.
"""
from __future__ import annotations
from typing import (
    Iterable, Dict, FrozenSet, Sequence, Set, TypeVar, Generic,
    Optional, Callable, Tuple, Iterator, Type, DefaultDict, Any)
from itertools import chain, product
from functools import wraps
from collections import abc, defaultdict
import pathlib
import pickle
import importlib.util
import shutil
import random

import torch
import numpy as np

from .causal_graph import CausalGraphDataset


NodeType = TypeVar("NodeType")


def import_module(import_name: str, filename: str):
    """
    import the module corresponding to filename as import_name
    """
    # create module object
    spec = importlib.util.spec_from_file_location(import_name, filename)
    module = importlib.util.module_from_spec(spec)

    # execute module's code and return module object
    if spec.loader is None:
        raise IOError(
            f"Could not load module {import_name} from file: {filename}")

    # initialize module
    spec.loader.exec_module(module)
    return module


class DirectedGraph(Generic[NodeType], abc.MutableMapping):  # pylint: disable=unsubscriptable-object
    """
    A class to encapsulate directed graphs. Initialized from a dictionary:
    Values are iterable which must range over keys of the dictionary.
    For a key k,  v in self[k] means there is a vertex k -> v.
    Different graph composition functions are provided in order to facilitate
    the generation of examples.
    """

    def __init__(self, base_dict: Dict[NodeType, Iterable[NodeType]]) -> None:
        """
        Initialize a new graph object from a dictionary of iterables. All the
        elements of these iterables should themselves be hashables, as they
        will be referenced as new keys of the graphs if they are not already
        so.
        """
        # get all the nodes missing as keys
        missing_nodes = set(chain(*base_dict.values())) - base_dict.keys()

        # initialize the graph
        self._direct: Dict[NodeType, FrozenSet[NodeType]] = {
            **{source: frozenset() for source in missing_nodes},
            **{source: frozenset(targets)
               for source, targets in base_dict.items()}}

        # create the opposite graph as a default dict
        opposite: DefaultDict[
            NodeType, Set[NodeType]] = defaultdict(lambda: set())
        for node, children in self.items():
            for child in children:
                opposite[child].add(node)

        # store opposite graph as standard dict, with frozen set values.
        self._opposite: Dict[NodeType, FrozenSet[NodeType]] = {
            node: frozenset(opposite[node]) for node in self}

    def __getitem__(self, node: NodeType) -> FrozenSet[NodeType]:
        """
        get a frozen set containing child nodes of given node
        """
        return self._direct[node]

    def __delitem__(self, node: NodeType) -> None:
        """
        delete a node from the graph
        """
        # get list of children and parents of node
        children = self._direct[node]
        parents = self._opposite[node]

        # delete node everywhere it appears in direct graph
        for parent in parents:
            children_without_node = frozenset(
                child for child in self._direct[parent]
                if child is not node)
            self._direct[parent] = children_without_node

        # do the same for opposite graph
        for child in children:
            parents_without_node = frozenset(
                parent for parent in self._opposite[child]
                if parent is not node)
            self._opposite[child] = parents_without_node

        # delete node as key of direct and opposite graphs
        del self._direct[node]
        del self._opposite[node]

    def __setitem__(
            self, node: NodeType, children: Iterable[NodeType]) -> None:
        """
        Set or reset children of a node in the graph
        """

        # tag children to be removed and added
        children_set = frozenset(children)
        current_nodes = self[node] if node in self else frozenset()
        to_remove = current_nodes - children_set
        to_add = children_set - current_nodes

        # add missing nodes to graph
        for child in children_set | {node}:
            if child not in self:
                self._direct[child] = frozenset()
                self._opposite[child] = frozenset()

        # add/re-add node to direct graph with new children
        self._direct[node] = children_set

        # remove required children on the opposite graph
        for child in to_remove:
            parents_without_node = self._opposite[child] - {node}
            self._opposite[child] = parents_without_node

        # add required children on the opposite graph
        for child in to_add:
            parents_with_node = self._opposite[child] | {node}
            self._opposite[child] = parents_with_node

    @wraps(dict.__iter__)
    def __iter__(self) -> Iterator[NodeType]:
        return iter(self._direct)

    @wraps(dict.__len__)
    def __len__(self) -> int:
        return len(self._direct)

    @property
    def op(self) -> DirectedGraph[NodeType]:
        """
        returns the opposite graph
        """
        # create an empty opposite graph
        empty_dict: Dict[NodeType, Iterable[NodeType]] = {}
        opposite = type(self)(empty_dict)

        # set manually the direct and opposite graphs of this empty graph
        opposite._direct = self._opposite  # pylint: disable=W0212
        opposite._opposite = self._direct  # pylint: disable=W0212

        return opposite

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
        return type(self)({node: self[node] & nodes_set for node in nodes_set})

    def prune(self, node: NodeType) -> DirectedGraph[NodeType]:
        """
        Prune a node out of the graph, but making sure that for any subgraph
        i -> pruned_node -> j, there is a i -> j vertex added
        """
        # get parents and children of node
        parents = self.op[node]
        children = self[node]

        # delete node
        del self[node]

        # add children to all parents' children sets
        for parent in parents:
            self[parent] |= children

        return self

    def __repr__(self) -> str:
        """
        return string representation of object
        """
        return (
            f"{type(self).__name__}("
            + ", ".join(
                f"{node} > {tuple(children)}"
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
            (False, key): [(False, value) for value in values]
            for key, values in self.items()}
        remap_other = {
            (True, key): [(True, value) for value in values]
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
            (key0, key1): list(product(value0, value1))
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
            (False, key): (
                [(False, value) for value in values]
                + [(True, other_key) for other_key in other_graph])
            for key, values in self.items()}
        remap_other = {
            (True, key): [(True, value) for value in values]
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
            (key0, key1): list(chain(
                ((value0, key1) for value0 in self[key0]),
                ((key0, value1) for value1 in other_graph[key1])))
            for key0, key1 in product(self, other_graph)})

    def __mul__(
            self, other_graph: DirectedGraph[NodeType]
    ) -> DirectedGraph[Tuple[NodeType, NodeType]]:
        """
        Generates a new directed graph by making the directed product of both
        graphs (lexicographic order)
        """
        return type(self)({
            (key0, key1): (
                [(value0, key1) for value0 in values0]
                + [(key, value1) for key, value1 in product(self, values1)])
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
            key_func(key): [key_func(value) for value in values]
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
    def __init__(self, base_dict: Dict[NodeType, Iterable[NodeType]]) -> None:
        """
            Create an acyclic directed graph from a dictinary.
        """

        # init self as directed graph
        super().__init__(base_dict)

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
            "Weights sequence should be of length 4, for or, and, add, mul"
            "operations respectively")
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


class CausalGenerator:
    """
    A representation of a generator for a dataset of causally linked terms
    """
    def __init__(
            self, graph: DirectedAcyclicGraph[str],
            gen_dict: Dict[str, Callable[..., torch.Tensor]]) -> None:
        """
        Initialize new causal generator from directed acyclic graph,
        and a dictionary containing functions
        to be used for generation_ attached to each node.
        Functions should:
            - take n+1 arguments where n is the number of parents of the node,
                these arguments being the generated batches for each parent
            - the first argument is the batch shape
        All keys of the dictionary should be strings
        """
        # check that both dictionaries have the same keys
        assert set(graph.keys()) == set(gen_dict.keys()), (
            "the generators' dictionary should have the same keys as"
            "the graph's dictionary")
        # check that all keys are strings
        assert all(isinstance(key, str) for key in graph.keys()), (
            "Keys of the dictionary should be strings")

        self._graph = graph
        self._generators = gen_dict

    @property
    def graph(self) -> DirectedAcyclicGraph[str]:
        """
        Graph of the causal generator
        """
        return self._graph

    @property
    def generator(self) -> Dict[str, Callable[..., torch.Tensor]]:
        """
        returns the genertor associated with given node of the graph
        """
        return self._generators

    @property
    def ins(self) -> FrozenSet[str]:
        """
        roots of the graph of the generator
        """
        return self.graph.ins

    @property
    def outs(self) -> FrozenSet[str]:
        """
        roots of the opposite of the graph of the generator
        """
        return self.graph.outs

    def dump(self, filename: str) -> None:
        """
        create a picklable representation of the causal generator
        """
        filepath = pathlib.Path(filename + "_graph")
        with open(filepath.with_suffix(".pickle"), "wb") as file:
            generator_names = {
                node: self.generator[node].__name__
                for node in self}
            pickle.dump((dict(self._graph), generator_names), file)

    @staticmethod
    def load(filename: str) -> CausalGenerator:
        """
        Load a generator from a filename. Two files are in fact necessary:
            - A file containing the pickled version of the causal generator
            - A file containing the function definitions in a dictionary named
            generators.
        """
        # load the causal generators
        gen = import_module("gen", f"{filename}_generator.py")

        # unpickle the file
        with open(f"{filename}_graph.pickle", "rb") as file:
            graph, gen_names = pickle.load(file)

        # get generator functions
        generators = {
            node: getattr(gen, name) for node, name in gen_names.items()}

        # recreate the Causal generator
        causal_generator = CausalGenerator(
            DirectedAcyclicGraph(graph), generators)
        return causal_generator

    def __iter__(self) -> Iterator[str]:
        """
        returns iterable over nodes of the generator graph
        """
        return iter(self.graph)


class CausalGraphBatch(CausalGraphDataset):
    """
    A causal graph batch class, to represent batches drawn from existing stored
    datasets
    """
    def __init__(
            self, graph: DirectedAcyclicGraph[str],
            data_dict: Dict[str, torch.Tensor]) -> None:
        """
        Generate a causal batch from given graph and data dictionary
        """
        assert data_dict.keys() <= graph.keys(), (
            "keys of data dictionary should be nodes of the graph")

        # enumerate nodes of the graph
        sorted_names = enumerate(sorted(data_dict, key=pickle.dumps))
        nodes = {node: node_index for node_index, node in sorted_names}

        # create the adjacency matrix as the identity matrix
        # numpy array because of issues with rowwise/columnwise norming of
        # torch tensors
        nb_nodes = len(nodes)
        adj_mat = np.identity(nb_nodes, dtype=np.bool)

        # set all required coefficients to 1 in adjacency matrix
        for node, node_index in nodes.items():
            children_indices = list(nodes[child] for child in graph[node])
            adj_mat[node_index, children_indices] = 1

        # create data tensor
        data = torch.stack(
            tuple(data_dict[node] for node in nodes), -1)

        # register adjacency matrix and data attributes
        self._adj_mat = adj_mat
        self._data = data

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._adj_mat


class CausalDatasetFromGraph:
    """
    A class for causal batches, generated from a CausalGenerator
    """
    def __init__(self, batch_dir: str) -> None:
        """
        Initialize a causal batch object form a directory
        """
        self._dir = pathlib.Path(batch_dir)

        # load graph and names of generator functions
        graph_path = self.directory.joinpath("definition_graph.pickle")
        with open(graph_path, "rb") as graph_file:
            graph, self._gen_names = pickle.load(graph_file)
            self._graph = DirectedAcyclicGraph(graph)

        # verify data exists for all nodes
        assert all(
            self.directory.joinpath(node).is_file()
            for node in self.graph), (
                "Batch is invalid: missing data files for some nodes")

    def get(self, nodes: Optional[Iterable[str]] = None) -> CausalGraphBatch:
        """
        Generate a dataset with data from given nodes of the graph. By default,
        get all nodes of the dataset
        """
        # take care of default case: all nodes of the graph are considered
        nodes_to_get: Optional[Iterable[str]]
        if nodes is None:
            nodes_to_get = self.graph.keys()
        else:
            nodes_to_get = nodes

        # Load all relevant data. Careful, torch.load does not make it
        # a torch.Tensor automatically, can be a numpy.ndarray
        datas = {
            node: torch.tensor(torch.load(self.directory.joinpath(node)))  # pylint: disable=not-callable
            for node in nodes_to_get}

        subgraph: DirectedAcyclicGraph[str] = self.graph.subgraph(nodes_to_get)  # type: ignore

        return CausalGraphBatch(subgraph, datas)

    @property
    def directory(self) -> pathlib.Path:
        """
        directory where one can find the actual data attached to the dataset
        """
        return self._dir

    @property
    def graph(self) -> DirectedAcyclicGraph:
        """
        Get the graph representing causal relations of the dataset
        """
        return self._graph

    @property
    def names(self) -> Dict[str, str]:
        """
        Get names of generating functions corresponding to each node of the
        dataset
        """
        return self._gen_names


def create_dataset(
        causal_generator: CausalGenerator, batch_shape: Iterable[int],
        batch_dir: str) -> CausalDatasetFromGraph:
    """
    Generate a new causal batch from given generator, linked to a directory
    """
    # generate batch starting from the entry node
    current_nodes = causal_generator.ins

    # create directory
    directory = pathlib.Path(batch_dir)
    directory.mkdir(parents=True, exist_ok=True)

    # iteratively generate elements of the dataset until reaching end nodes
    while current_nodes:
        # compute batches for current nodes
        for node in current_nodes:
            save_path = directory.joinpath(node)

            parent_nodes = causal_generator.graph.op[node]

            # load previous batches:
            parent_batches = lambda: (
                torch.load(directory.joinpath(parent_node))
                for parent_node in parent_nodes)

            # draw batch using registered function for current node
            batch = causal_generator.generator[node](
                batch_shape, *parent_batches())

            torch.save(batch, save_path)

        # update current nodes and batches
        current_nodes = frozenset(chain(
            *(causal_generator.graph[node] for node in current_nodes)))

    # dump a pickled rep of the generator
    causal_generator.dump(directory.joinpath("definition").as_posix())

    return CausalDatasetFromGraph(batch_dir)


def generate_dataset(
        generator_filename: str, batch_dir: str, batch_shape: Iterable[int],
        copy_utils: bool = False) -> CausalDatasetFromGraph:
    """
    Load a generator from a filename, and use it to generate batch with
    given shape (batch_shape) in the batch_dir folder.
    copy_utils should be True if one wants to copy also potential utilities
    necessary to define generating functions (such as weights for perceptrons,
                                              etc...)
    """
    # load causal generator
    causal_generator = CausalGenerator.load(generator_filename)

    # generate dataset
    batch = create_dataset(causal_generator, batch_shape, batch_dir)

    # copy function definitions into batch directory
    batch_path = pathlib.Path(batch_dir)
    shutil.copyfile(
        f"{generator_filename}_generator.py",
        batch_path.joinpath("definition_generator.py"))

    # if function definition depends on other informations, should be in
    # a filename_utils folder. Try to copy this folder
    if copy_utils:
        try:
            shutil.copytree(
                f"{causal_generator}_utils", batch_path.joinpath("utils"))
        except IOError:
            raise Warning("Could not copy utils folder.")

    return batch


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a dataset for a given graph generator")
    parser.add_argument(
        "generator_file", type=str,
        help=(
            "filename for generator definition "
            "(pickled graph and functions in a module)"))
    parser.add_argument(
        "save_directory", type=str,
        help="save directory for numpy arrays")
    parser.add_argument(
        "n_samples", type=int, nargs="*",
        help=(
            "number of iid samples to draw along, possibly"
            "along several dimensions"))
    args = parser.parse_args()

    generate_dataset(
        args.generator_file, args.save_directory, args.n_samples)
