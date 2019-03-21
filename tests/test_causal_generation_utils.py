"""
tests for random causal dataset generation
"""

from typing import (
    Any, Callable, Dict, Tuple, List, Sequence)
from itertools import product
import random
from shutil import copyfile

import torch
import numpy as np
import pytest

from catlearn.causal_generation_utils import (
    DirectedAcyclicGraph, CausalGenerator, generate_dataset)

from tests.test_tools import pytest_generate_tests


DATA_DIR = "./tests/test_causal_generation_utils/"


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


class TestCausalGenerator:
    """
    Unit tests for Causal generator class
    """
    params: Dict[str, List[Any]] = {
        "test_init": [
            dict(
                graph=DirectedAcyclicGraph({0: [], 1: []}),
                generators_dict={
                    0: lambda: torch.ones(0), 1: lambda: torch.ones(0)},
                is_valid_input=False),
            dict(
                graph=DirectedAcyclicGraph({"0x1": ["0x0"]}),
                generators_dict={
                    "0x0": lambda: torch.ones(0), "0x1": lambda x: x},
                is_valid_input=True)
            ],
        "test_dump": [
            dict(filename="_test")
            ]
    }

    @staticmethod
    def test_init(
            graph: DirectedAcyclicGraph[Any],
            generators_dict: Dict[Any, Callable[..., torch.Tensor]],
            is_valid_input: bool) -> None:
        """
        Tests that the assertions of keys being strings and key equality
        of graph and generator dicts work as intended
        """
        if is_valid_input:
            generator = CausalGenerator(graph, generators_dict)
            assert generator.graph == graph
            assert generator.generator == generators_dict
        else:
            with pytest.raises(AssertionError):
                generator = CausalGenerator(graph, generators_dict)

    @staticmethod
    def test_dump(filename: str, tmpdir) -> None:
        """
        Test that dumping works as intended
        """
        generator = CausalGenerator.load(DATA_DIR + filename)

        savepath = tmpdir.join("dump_" + filename)

        # dump and reload generator
        generator.dump(savepath)
        copyfile(DATA_DIR + filename + "_generator.py",
                 savepath + "_generator.py")
        reloaded_generator = CausalGenerator.load(savepath)

        assert generator.graph == reloaded_generator.graph
        assert generator.generator == generator.generator


class TestCausalDatasetFromGraph:
    """
    Unit tests for CausalDatasetFromGraph class
    """
    params: Dict[str, List[Any]] = {
        "test_generate": [
            dict(generator_name="_test",
                 copy_utils=False)
        ],
        "test_batch_shape": [
            dict(generator_name="_test")
        ]
    }

    @staticmethod
    def test_generate(
            generator_name: str, batch_shape: Sequence[int],
            copy_utils: bool, tmpdir: Any) -> None:
        """
        test generation of a dataset object given its save directory.
        Test that it works without problems when directory is valid,
        and returns an appropriate error if it is not
        """
        # load generator

        data_path = tmpdir.join("test_generate")
        generator_path = DATA_DIR + generator_name

        generator = CausalGenerator.load(generator_path)
        causal_dataset = generate_dataset(
            generator_path, data_path, batch_shape, copy_utils=copy_utils)

        # check infos of the dataset
        assert set(causal_dataset.names) == set(generator)
        assert causal_dataset.graph == generator.graph

    @staticmethod
    def test_batch_shape(
            generator_name: str, batch_shape: Sequence[int],
            tmpdir: Any) -> None:
        """
        tests the batch shape of stored data
        """
        data_path = tmpdir.join("test_batch")
        generator_path = DATA_DIR + generator_name

        # generate dataset
        causal_dataset = generate_dataset(
            generator_path, data_path, batch_shape)

        # draw a node
        node = random.choice(list(causal_dataset.graph))

        # load data from the node
        node_path = data_path.join(node)
        data = torch.load(str(node_path))

        assert tuple(data.shape) == tuple(batch_shape)


class TestCausalGraphBatch:
    """
    Unit tests for CausalGraphBatch class
    """
    params: Dict[str, List[Any]] = {
        "test_adjacency_matrix": [
            dict(generator_name="_test")
            ]
        }

    @staticmethod
    def test_adjacency_matrix(
            generator_name: str, batch_shape: Tuple[int, ...], nb_nodes: int,
            tmpdir: Any) -> None:
        """
        test that the batch has the right adjacency matrix
        """

        data_path = tmpdir.join("test_adj")
        generator_path = DATA_DIR + generator_name

        # generate dataset
        causal_dataset = generate_dataset(
            generator_path, data_path, batch_shape)

        # get a random sampling of nodes from the graph
        nodes = random.sample(list(causal_dataset.graph), k=nb_nodes)

        # extract corresponding batch
        batch = causal_dataset.get(nodes)

        # get subgraph corresponding to nodes, convert its keys to integers
        subgraph = causal_dataset.graph.subgraph(nodes).integerify()

        # test adjacency matrix
        # should be 1 for node, children indices
        assert all(
            np.all(batch.adjacency_matrix[node, list(children)])
            for node, children in subgraph.items())
        # should be 1 for node, node indices
        assert all(batch.adjacency_matrix[node, node] for node in subgraph)

        # should be 0 for all other
        assert all(
            not np.any(batch.adjacency_matrix[node, other_node])
            for (node, other_node) in product(subgraph, subgraph)
            if node != other_node and other_node not in subgraph[node])
