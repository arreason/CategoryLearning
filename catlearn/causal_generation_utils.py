"""
A library for generating artificial examples for catlearn tests
Variable dependencies are represented by directed acyclic graphs (DAG),
each node is labeled with the generating function to use.
"""
from __future__ import annotations
from functools import reduce
from operator import mul
from typing import (
    Iterable, Dict, FrozenSet, Optional, Callable, Tuple, Iterator)
from itertools import chain
import pathlib
import pickle
import importlib.util
import shutil

import torch
import numpy as np

from catlearn.graph_utils import DirectedAcyclicGraph


def _draw_batch_tuple_indices(
        num_examples: int,
        transition_mat: np.ndarray,
        tuple_size: int) -> torch.Tensor:
    """
    draw a batch of positive examples given an adjacency matrix
    inputs:
        num_examples: int, size of the batch
        adj_mat: numpy.array, square matrix,
                transition matrix of the causal graph
        tuple_size: int, the size of the tuples to draw
    outputs:
        a tensor of integers of shape (num_examples, tuple_size)
    """
    assert tuple_size >= 1, "Tuple size should be at least 1"

    # to get subsequent indices, given previous indices
    def choice_gen(indices: np.ndarray) -> Iterator[int]:
        """
        Given an array of integer indices, returns for each one a randomly
        drawn index from the possibilities left by the transition matrix
        inputs:
            indices: numpy.ndarray, size (nb_examples,)
        outputs:
            numpy.ndarray; size (nb_examples,)
        """
        for i in indices:
            yield np.random.choice(
                transition_mat.shape[0], 1,
                p=transition_mat[i].numpy())[0]

    def rec_choice(
            indices: Tuple[torch.Tensor, ...],
            order: int) -> Tuple[torch.Tensor, ...]:
        """
        Given the first choices, get the next choices compatible with the
        transition matrix until reaching order
        inputs:
            indices: Sequence[numpy.ndarray], the choices which were already
                    made
            order: int, the number of choices which remain to be made - 1
        outputs:
            Sequence[numpy.ndarray], a sequence of arrays containing all the
            random choices at each order
        """
        if order == 0:
            return indices
        return (rec_choice(indices, order-1)
                + (torch.IntTensor(
                    list(choice_gen(indices[-1]))),))

    # first indices
    firsts = torch.randint(low=0, high=transition_mat.shape[0],
                           size=(num_examples,),
                           dtype=torch.int)

    # get next indices
    all_indices = rec_choice((firsts,), tuple_size-1)

    return torch.stack(all_indices, -1)


class CausalGraphDataset:
    """
    Template class for causal graph datasets. Should contain:
        - a property for returning an adjacency matrix
        - function to generate a dataset
    """

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """ Graph adjacency matrix """
        raise NotImplementedError(
            "You need to define the dataset adjacency matrix property"
            "when sublcassing CausalGraphDataset")

    @property
    def transition_matrix(self) -> torch.FloatTensor:
        """
        Return a row-normalized version of the adjacency matrix
            (cast to float)
        Using a cast to numpy arrays because of torch apparent bug on some
        tensor shape broadcastings.
        """

        adj_mat = self.adjacency_matrix.to(torch.float).numpy()
        return torch.from_numpy(
            adj_mat / adj_mat.sum(axis=1)[:, np.newaxis])

    @property
    def data(self) -> torch.Tensor:
        """ Get underlying data """
        raise NotImplementedError(
            "You need to define the data property when subclassing"
            "CausalGraphDataset")

    @property
    def shape(self) -> Tuple[int]:
        """
        The shape of the dataset
        """
        return self.data.shape

    @property
    def dim(self) -> int:
        """
        The dimension of datapoints of the dataset
        """
        return self.adjacency_matrix.shape[0]

    @property
    def batch_shape(self) -> Tuple[int]:
        """
        The shape of the batch
        """
        return self.data.shape[:-1]

    @property
    def ndim(self) -> int:
        """
        the number of dimensions of the dataset
        """
        return len(self.shape)

    @property
    def batch_ndim(self) -> int:
        """
        the number of dimensions of the batch shape
        """
        return len(self.batch_shape)

    @property
    def numel(self) -> int:
        """
        the number of elements in the batch
        """
        return reduce(mul, self.shape, 1)

    @property
    def numexamples(self) -> int:
        """
        The number of examples in the batch
        """
        return reduce(mul, self.batch_shape, 1)

    def __getitem__(self, *args, **kwargs) -> torch.Tensor:
        """
        Item access: accesses the underlying items of the data attribute
        """
        return self.data.__getitem__(self, *args, **kwargs)

    def draw_positive_batch(self, tuple_size: int) -> torch.Tensor:
        """
        Draw a batch of matching relation tuples of given size.
        inputs:
            tuple_size: int, the size of the positive tuples
        oututs:
            torch.Tensor, of shape self.batch_shape + (tuple_size,)
        """
        indices = _draw_batch_tuple_indices(
            self.numexamples, self.transition_matrix, tuple_size)

        data = self.data.view(self.numexamples, self.dim)

        batch = torch.FloatTensor(
            [[data[i, p] for p in indices[i]]
             for i in range(self.numexamples)])

        return batch.view(self.batch_shape + (tuple_size,))

    def draw_random_batch(self, tuple_size: int) -> torch.Tensor:
        """
        Draw a batch of tuples at random, without regards for matching
        relations.
        inputs:
            tuple_size: int, the size of the positive tuples
        ouputs
            torch.Tensor, of shape self.batch_shape + (tuple_size,)
        """

        indices = torch.randint(low=0, high=self.dim,
                                size=(self.numexamples, tuple_size),
                                dtype=torch.int)

        data = self.data.view(self.numexamples, self.dim)

        batch = torch.FloatTensor(
            [[data[i, p] for p in indices[i]]
             for i in range(self.numexamples)])
        return batch.view(self.batch_shape + (tuple_size,))

    def draw_balanced_batch(self, tuple_size: int) -> torch.Tensor:
        """
        Draw a batch of tuples, half positive, half at random
        inputs:
            tuple_size: int, the size of the positive tuples
        outputs:
            torch.Tensor, of shape (self.batch_shape) +  (2, tuple_size)
        """
        # draw positive and random examples
        positive_batch = self.draw_positive_batch(tuple_size)
        negative_batch = self.draw_random_batch(tuple_size)

        # concatenate
        batch = torch.stack((positive_batch, negative_batch), dim=-1)

        nb_examples = positive_batch.shape[:-1]
        labels = torch.stack((torch.ones(nb_examples),
                              - torch.ones(nb_examples)), dim=-1)

        return batch, labels


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
            self._graph: DirectedAcyclicGraph[str] = DirectedAcyclicGraph(
                graph)

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
