# Check if usefull to inherit torch data structure
# import torch.utils.data
import os
# import numpy as np
from typing import Dict, Tuple, Generator, Mapping, List
import torch
# from catlearn.graph_utils import DirectedGraph
from catlearn.data.config import (raw_dataset_pat, preproc_pat)
from catlearn.tensor_utils import Tsor
import catlearn.data.utils as data_utils

class Dataset():
    """Dataset object reads and and formats the dataset datasets
    Dataset must be preprocessed with the related method in data.utils

    """
    def __init__(self,
            path: str=None,
            ds_name: str=None,
            node_vec_dim: int=10,
            ds_pat: Dict=raw_dataset_pat,
            preproc_pat: Dict = preproc_pat,
            node2vec_path: str=None
            ):
        """Initialize dataset object
        Args:
            node_vec_dim (int): [description]
            path (str, optional): [description]. Defaults to None.
            ds_name (str, optional): [description]. Defaults to None.
            ds_pat (Dict, optional): [description]. Defaults to raw_dataset_pat.
            preproc_pat (Dict, optional): [description]. Defaults to preproc_pat.
        """
        self.ds_pat: Dict = ds_pat
        # self.use_cuda = use_cuda
        self.path: str = path
        self.ds_name: str = ds_name
        self.entity_vec_dim: int = node_vec_dim
        self.entity2id: Dict = None
        self.id2entity: Dict = None
        self.relation2id: Dict = None
        self.id2relation: Dict = None
        # not trainamble vector representation
        self.entity_id2vec: Dict = None
        # No trainable one-hot vector. Used for cost calculation
        self.relation_id2vec: Dict = None
        self.node2vec_path: str = node2vec_path
        self.train: Generator[Tuple[str, str, Dict[str, str]]] = None
        self.valid: Generator[Tuple[str, str, Dict[str, str]]] = None
        self.test: Generator[Tuple[str, str, Dict[str, str]]] = None
        if path and ds_name:
            self.from_file()

    def from_file(self, path: str=None):
        """reads dataset from path
        if path argument is provided, it overwrites the default
        path stored in self.path

        Args:
            path (str, optional): [description]. Defaults to None.
        """
        if path:
            self.path = path
        self.read_id_maps()
        self.init_entity_vectors()
        self.init_relation_vectors()
        self.train = self.format_dataset(
            self.read_file(os.path.join(self.path, self.ds_pat['train']))
        )
        self.valid = self.format_dataset(
            self.read_file(os.path.join(self.path, self.ds_pat['valid']))
        )
        self.test = self.format_dataset(
            self.read_file(os.path.join(self.path, self.ds_pat['test']))
        )

    def check_path(self, path: str) -> str:
        """Read dataset file. Line format <entity   relation    entity>"""
        if not os.path.isfile(path):
            raise ValueError(f'Path error: ${path}')
        return os.path.normpath(path)

    def read_file(self, path: str) -> Generator:
        return (line.split(self.ds_pat['sep']) for line in open(self.check_path(path)))

    def format_dataset(self, raw_dataset: Generator) -> Generator:
        if self.ds_name == 'wn18':
            return ((
                    self.entity2id[u.strip()],
                    self.entity2id[v.strip()],
                    {self.relation2id[e.strip()]: self.relation_id2vec[self.relation2id[e.strip()]]}
                ) for u, e, v in raw_dataset)
        else:
            raise ValueError(f'Unknown dataset name ${self.ds_name}')

    def _format_id_map(self, id_map: Generator, ds_name: str) -> Dict[str, int]:
        if ds_name == 'wn18':
            return {e: int(id) for e, id in id_map}
        else:
            raise ValueError(f'Unknown dataset name ${ds_name}')

    def preproc_dataset(self):
        if self.ds_name == 'wn18':
            data_utils.preprocess_wn18(self.path)

    def read_id_maps(self):
        entity2id_path = os.path.join(self.path, preproc_pat['entity2id'])
        relation2id_path = os.path.join(self.path, preproc_pat['relation2id'])
        
        if (not os.path.isfile(entity2id_path)
                or not os.path.isfile(relation2id_path)):
            self.preproc_dataset()
        self.entity2id = self._format_id_map(
            self.read_file(os.path.join(self.path, preproc_pat['entity2id'])),
            self.ds_name
        )
        self.relation2id = self._format_id_map(
            self.read_file(os.path.join(self.path, preproc_pat['relation2id'])),
            self.ds_name
        )
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

    def init_entity_vectors(self):
        """initialize entity2vec and rel2vec
        entity2vec corresponds to data_points parameter for train
        in TrainableDecisionCatModel.
        self.entity_vec_dim is ignored when initializer is provided
        """
        if self.node2vec_path:
            import pickle
            # node2vec_list: List = list(self.read_file(self.node2vec_path))
            with open(self.node2vec_path, 'rb') as f:
                node2vec_dict = pickle.load(f)
            # node2vec_dict:  = list(self.read_file(self.node2vec_path))
            # print(f'node2vec_dict: {node2vec_list}')
            # node2vec_dict: Mapping[int, Tsor] = {k.strip(): v.strip() for k, v in node2vec_list}
            print(f'node2vec_dict: {node2vec_dict}')
            self.entity_id2vec = {k: node2vec_dict[self.id2entity[k]] for k in self.id2entity.keys()}
        else:
            self.entity_id2vec = {k: torch.rand(self.entity_vec_dim) for k in self.id2entity.keys()}

    def init_relation_vectors(self):
        """one-hot representation of entities
        equivalent to label_universe required by TrainableDecisionCatModel
        TODO: label one-hot vectors are redundant in graph. For large libraries of vectors faster to keep only ID
        TODO: Parametrize encoding type. OHE should be only of encoding schemes.
        """
        nb_relations = len(self.relation2id)
        def one_hot(sample_id: int, nb_samples: int):
            enc_sample = torch.zeros(nb_samples)
            enc_sample[sample_id] = 1.0
            return enc_sample
        self.relation_id2vec = {id: one_hot(id, nb_relations) for id in self.id2relation.keys()}