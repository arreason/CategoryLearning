from typing import (
    Any, Callable, Hashable, Dict, Iterable, Tuple,
    Set, FrozenSet, Optional, List)
import os
import numpy
import torch
import pickle as pkl
from testfixtures import TempDirectory

from tests.test_tools import pytest_generate_tests

from catlearn.graph_utils import DirectedGraph
from catlearn.data.dataset import Dataset
import catlearn.data.utils as data_utils
from catlearn.data.config import raw_dataset_pat, preproc_pat
from catlearn.tensor_utils import Tsor

class TestDataset:
    """
    Unit tests for Dataset class
    entity2id and relation2id are provided and not generated through Dataset
    due to difficulty of managing multithreading randomness and mapping consistency.
    """
    params: Dict[str, List[Any]] = {
        'test_dataset': [
            dict(initializer_dict={'wn18_graph': (b'a\t_member_of_domain_usage\tb\n'
                                    b'b\t_verb_group\tc\n'
                                    b'a\t_member_of_domain_region\td\n'
                                    b'd\t_member_meronym\tc\n'),
                                    'w2v_dic': {'a': [0.55, 0.45, 0.35],
                                    'b': [0.55, 0.45, 0.35],
                                    'c': [0.55, 0.45, 0.35],
                                    'd': [0.55, 0.45, 0.35]},
                                    'entity2id': b'a\t0\nb\t1\nc\t2\nd\t3\n',
                                    'relation2id': (b'_member_of_domain_usage\t0\n'
                                                    b'_verb_group\t1\n'
                                                    b'_member_of_domain_region\t2\n'
                                                    b'_member_meronym\t3\n')
                                    },
                expected_graph=DirectedGraph(((0, 1, {0: Tsor([1., 0, 0, 0])}),
                                                (1, 2, {1: Tsor([0, 1., 0, 0])}),
                                                (0, 3, {2: Tsor([0, 0, 1., 0])}),
                                                (3, 2, {3 : Tsor([0, 0, 0, 1.])})
                                            ))
                ),
        ]
    }

    @staticmethod
    def test_dataset(
            initializer_dict: Dict,
            expected_graph: DirectedGraph):
        """Check that wn18 dataset format is read and formated properly.
        Current test only covers reading embedding vectors from a file
        (e.g. word2vec pre-computed embeddings). Default randomly generated embeddings
        are not tested due to its trivial implementation and intrinsic randomness,
        that is more difficult to test.

        Args:
            initializer_dict (Dict): String representing wn18 dataset format
            expected_graph (DirectedGraph): Expected graph output
            TODO: add edgecases like plural empty lines, escape characters etc.
        """
        with TempDirectory() as d:
            d.write(raw_dataset_pat['train'], initializer_dict['wn18_graph'])
            d.write(raw_dataset_pat['valid'], initializer_dict['wn18_graph'])
            d.write(raw_dataset_pat['test'], initializer_dict['wn18_graph'])
            d.write(preproc_pat['entity2id'], initializer_dict['entity2id'])
            d.write(preproc_pat['relation2id'], initializer_dict['relation2id'])
            w2v_dict_path = os.path.join(d.path, preproc_pat['word2vec_short'])
            with open(w2v_dict_path, 'wb') as f:
                pkl.dump(initializer_dict['w2v_dic'], f)
            ds: Dataset = Dataset(d.path, 'wn18', node2vec_path=w2v_dict_path)
            graph: DirectedGraph = DirectedGraph(ds.train)
            assert str(graph) == str(expected_graph)
            graph = DirectedGraph(ds.valid)
            assert str(graph) == str(expected_graph)
            graph = DirectedGraph(ds.test)
            assert str(graph) == str(expected_graph)
