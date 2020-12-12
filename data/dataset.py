"""Dataset reads and preprocesses datasets.
It is a composed object that includes both, dataset and dataloader.
"""
import os
from typing import Dict, Tuple, Generator
import pickle
import torch
from data.config import (raw_dataset_pat, preproc_pat)
from catlearn.utils import one_hot
from catlearn.tensor_utils import Tsor
import data.utils as data_utils

class Dataset():
    """Dataset object reads and formats the dataset datasets
    Dataset must be preprocessed with the related method in data.utils

    """
    # pylint: disable=dangerous-default-value
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
            path: str,
            ds_name: str,
            node_vec_dim: int=10,
            ds_pat: Dict=raw_dataset_pat,
            prep_pat: Dict=preproc_pat,
            node2vec_path: str=None
            ):
        """Initialize dataset object.
        Create train, valid, and test dataset generators
        in the format (srcId: int, dstId: int, Dict[relId=>Tensor])
        Args:
            node_vec_dim (int): [description]
            path (str, optional): [description]. Defaults to None.
            ds_name (str, optional): [description]. Defaults to None.
            ds_pat (Dict, optional): [description]. Default naming patterns from raw_dataset_pat.
            prep_pat (Dict, optional): [description]. Default naming patterns from preproc_pat.
        """
        self.ds_pat: Dict = ds_pat
        self.prep_pat: Dict = prep_pat
        self.path: str = path
        self.ds_name: str = ds_name
        self.entity_vec_dim: int = node_vec_dim
        self.entity2id: Dict = None
        self.id2entity: Dict = None
        self.relation2id: Dict = None
        self.id2relation: Dict = None
        self.entity_id2vec: Dict = None
        self.relation_id2vec: Dict = None
        self.node2vec_path: str = node2vec_path
        self.train: Generator[Tuple[int, int, Dict[int, Tsor]]] = None
        self.valid: Generator[Tuple[int, int, Dict[int, Tsor]]] = None
        self.test: Generator[Tuple[int, int, Dict[int, Tsor]]] = None
        self.from_file()

    def from_file(self):
        """reads dataset from path
        if path argument is provided, it overwrites the default
        path stored in self.path

        Args:
            path (str, optional): [description]. Defaults to None.
        """
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

    @staticmethod
    def check_path(path: str) -> str:
        """Read dataset file. Line format <entity   relation    entity>"""
        if not os.path.isfile(path):
            raise ValueError(f'Path error: ${path}')
        return os.path.normpath(path)

    def read_file(self, path: str) -> Generator:
        """Creates Generator object from dataset file where each line
        coresponds to a triple (h, r, t) separated by a tab.

        Args:
            path (str): complete path to the file

        Returns:
            [Generator]: [description]

        Yields:
            Generator: list with a triple [h, r, t]
        """
        return (line.split(self.ds_pat['sep']) for line in open(Dataset.check_path(path)))

    def format_dataset(self, raw_dataset: Generator) -> Generator:
        """Formats dataset by replacing entities and relations
        with corresponding IDs. Mappings of entities and relations to IDs
        are created automatically during dataset preprocessing.

        Args:
            raw_dataset (Generator): Generator object returned by read_file

        Raises:
            ValueError: if dataset name is not known

        Returns:
            [type]: Generator object for the formated dataset

        Yields:
            Generator: tuple of the format (h_id, r_id, t_id)
        """
        if (self.ds_name in ['wn18', 'fb15']):
            return ((
                    self.entity2id[u.strip()],
                    self.entity2id[v.strip()],
                    {self.relation2id[e.strip()]: None}
                ) for u, e, v in raw_dataset)
        raise ValueError(f'Unknown dataset name ${self.ds_name}')

    def _format_id_map(self, id_map: Generator, ds_name: str) -> Dict[str, int]:
        """Creates a dictionary from read mapping of relation or entity to ID."""
        if (ds_name == 'wn18'
            or self.ds_name == 'fb15'):
            return {e: int(id) for e, id in id_map}
        raise ValueError(f'Unknown dataset name ${ds_name}')

    def preproc_dataset(self):
        """Preprocesses dataset according to specified format (see utils)"""
        if (self.ds_name == 'wn18'
            or self.ds_name == 'fb15'):
            data_utils.preprocess(self.path)

    def read_id_maps(self):
        """Read entity or relation to ID mapping from preprocessed files."""
        entity2id_path = os.path.join(self.path, self.prep_pat['entity2id'])
        relation2id_path = os.path.join(self.path, self.prep_pat['relation2id'])

        if (not os.path.isfile(entity2id_path)
                or not os.path.isfile(relation2id_path)):
            self.preproc_dataset()
        self.entity2id = self._format_id_map(
            self.read_file(os.path.join(self.path, self.prep_pat['entity2id'])),
            self.ds_name
        )
        self.relation2id = self._format_id_map(
            self.read_file(os.path.join(self.path, self.prep_pat['relation2id'])),
            self.ds_name
        )
        self.id2entity = {key: value for value, key in self.entity2id.items()}
        self.id2relation = {key: value for value, key in self.relation2id.items()}

    def init_entity_vectors(self):
        """initialize entity2vec and rel2vec
        entity2vec corresponds to data_points parameter for train
        in TrainableDecisionCatModel.
        self.entity_vec_dim is ignored when initializer is provided
        """
        if self.node2vec_path:
            with open(self.node2vec_path, 'rb') as f:
                node2vec_dict = pickle.load(f)
            self.entity_id2vec = {idx: Tsor(node2vec_dict[value])
                for idx, value in self.id2entity.items()}
        else:
            self.entity_id2vec = {idx: torch.rand(self.entity_vec_dim)
                for idx in self.id2entity.keys()}

    def init_relation_vectors(self):
        """one-hot representation of entities
        equivalent to label_universe required by TrainableDecisionCatModel
        TODO: Parametrize encoding type. OHE could be one of possible encoding schemes.
        In current implementation, other encodings like embedding vectors are not considered.
        """
        nb_relations = len(self.relation2id)
        self.relation_id2vec = {idx: one_hot(idx, nb_relations)
            for idx, value in self.id2relation.items()}

    def load(self):
        """Loads generator objects to memory"""
        if self.train:
            self.train = list(self.train)
        if self.valid:
            self.valid = list(self.valid)
        if self.test:
            self.test = list(self.test)
