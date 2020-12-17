"""Data utils module contains functions to read and process training data"""
from typing import Dict, List, Set, Union, Any
import os
import pickle as pkl
import argparse
import warnings
import logging
import sys
import torch
import numpy as np
from tqdm import tqdm
import word2vec
from data.config import (raw_dataset_pat, preproc_pat)
from catlearn.utils import (init_logger, str_color)


init_logger()


def read_file(path: str) -> List[str]:
    """Safe reading of file, split it by lines and return in a List"""
    if not os.path.isfile(path):
        raise ValueError(f'Dataset path error: ${path}')
    with open(path, 'r') as f:
        lines2list = f.read().splitlines()
    return lines2list


def write_file(path: str, to_file: Union[Dict, Set, List], force: bool=False):
    """Writes dictionary to a file

    Args:
        path (str): complete path to where a file will be stored.
        to_file (Dict or Set): data structure to be stored in a file.
    """
    if os.path.isfile(path) and not force:
        warning_str = (f'File ${path} already exists.'
            f' Set force=True to overwrite. Exiting...')
        warnings.warn(str_color('W', warning_str), UserWarning)
        return
    with open(path, 'w') as f:
        if isinstance(to_file, dict):
            for k, v in to_file.items():
                f.write(str(k) + preproc_pat['sep'] + str(v) + '\n')
        elif isinstance(to_file, set):
            for v in to_file:
                f.write(str(v) + '\n')
        elif (isinstance(to_file, list)
                # pylint: disable=len-as-condition
                # preceded by the condition checking to_file is a list
                and len(to_file)
                and all(type(v) in [int, float, str] for v in to_file)
        ):
            for v in to_file:
                f.write(str(v) + '\n')
        else:
            raise ValueError('Unsupported datatype for writing to a file.')


def get_entities_relations(ds_list: List[str]) -> [List[str], List[str]]:
    """Get all entities and relations as 2 separate lists.
        Used to generate sets of unique values.
    """
    triplets = [line.split(raw_dataset_pat['sep']) for line in ds_list]

    return [item.strip() for [h, _, t] in triplets for item in [h, t]], \
            [r.strip() for _, r, _ in triplets]


def _update_set_with_missing(entities_or_rels: [List[str], List[str]],
                                    set_to_update: Set,
                                    set_type: str='Entity',
                                    set_purpose: str='validation'):
    for er in set(entities_or_rels):
        if not er in set_to_update:
            warning_str = (f'${set_type} ${er} present in ${set_purpose} set'
                f' absent in reference ${set_purpose} set. It was added.')
            warnings.warn(str_color('W', warning_str), UserWarning)
            logging.warning(warning_str)
            set_to_update.add(er)


def preprocess(path: str):
    """Preprocesses dataset stored in a text format <h\tr\tt\n> with N lines
    where N is a number of triplets in a graph.
    Create dictionaries of entities and relations with appropriate mappings.
    Generated sets of unique entities and relations are stored in separate files
    using names from config.
    Saved files:
    entity2id and relation2id are created from a dictionary where key is str(entity_name)
    and value is an integer from 0 to len(entities_set) range
    word_set is specific for wn_18 and saved only a list of unique words

    Args:
        path (str): path where raw dataset files 'train.txt', 'valid.txt',
        and 'test.txt' are stored
    """
    train: List[str] = read_file(os.path.join(path, raw_dataset_pat['train']))
    entities_set, relations_set = get_entities_relations(train)
    entities_set = set(entities_set)
    relations_set = set(relations_set)
    # Check validation set on entities and relations not present in the training set
    valid: List[str] = read_file(os.path.join(path, raw_dataset_pat['valid']))
    entities_rels = get_entities_relations(valid)
    _update_set_with_missing(entities_rels[0],
                                    entities_set,
                                    set_type='Entity',
                                    set_purpose='validation')
    _update_set_with_missing(entities_rels[1],
                                    relations_set,
                                    set_type='Relations',
                                    set_purpose='validation')
    # Check testing set on entities and relations not present in the training set
    test: List[str] = read_file(os.path.join(path, raw_dataset_pat['test']))
    entities_rels = get_entities_relations(test)
    _update_set_with_missing(entities_rels[0],
                                    relations_set,
                                    set_type='Entity',
                                    set_purpose='test')
    _update_set_with_missing(entities_rels[1],
                                    relations_set,
                                    set_type='Relations',
                                    set_purpose='test')
    # Create dictionary with index
    entity2id: Dict = dict(zip(entities_set, range(len(entities_set))))
    write_file(os.path.join(path, preproc_pat['entity2id']), entity2id)
    relation2id: Dict = dict(zip(relations_set, range(len(relations_set))))
    write_file(os.path.join(path, preproc_pat['relation2id']), relation2id)
    word_set: Set = {s.split('.')[0] for s in entities_set}
    write_file(os.path.join(path, preproc_pat['wordset']), word_set)


def preprocess_word2vec(path_word2vec: str, path_dir_wordset: str):
    """Create synthetic word2vec set with vectors for given entities

    Args:
        path_word2vec (str): path to word2vec model file location.
        Either .txt or .pkl
        path_entity_dic (str): path to the directory with entity2id file location
    """
    word_set: List[str] = read_file(os.path.join(path_dir_wordset, preproc_pat['wordset']))
    w2v_model: Any = word2vec.load(path_word2vec)
    w2v_model_vocab: np.ndarray = w2v_model.vocab
    w2v_short: Dict = {}
    vec_dim = len(w2v_model['the'])
    for k in tqdm(word_set):
        # split compound words on members
        members = k.split('_')
        if k in w2v_model_vocab:
            w2v_short[k] = w2v_model[k]
        elif len(members) > 1 and members[-1] in w2v_model_vocab:
            w2v_short[members[-1]] = w2v_model[members[-1]]
        elif len(members) > 2 and members[-2] in w2v_model_vocab:
            w2v_short[members[-2]] = w2v_model[members[-2]]
        elif len(members) > 3 and members[-3] in w2v_model_vocab:
            w2v_short[members[-3]] = w2v_model[members[-3]]
        else:
            w2v_short[k] = torch.rand(vec_dim).numpy()
    with open(os.path.join(path_dir_wordset, preproc_pat['word2vec_short']), 'wb') as f:
        pkl.dump(w2v_short, f)


def parse_args():
    """Parse arguments

    Returns:
        ArgumentParser: object
    """
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--ds_name', help='Specify dataset name one of: wn18, fb15.',
                        type=str)
    parser.add_argument('--path', help=('Path to a folder where raw dataset files are stored.'
        ' For raw files naming format check config.py.'),
                        type=str, default='')
    parser.add_argument('--w2v_path', help=('Preprocess word2vec dataset. Path to word2vec.txt'
        ' model file. Possible only after a dataset was preprocessed.'),
                        type=str, default=None)
    return parser


if __name__ == '__main__':
    arg_parser = parse_args()
    try:
        args = arg_parser.parse_args()
    except ValueError:
        arg_parser.print_help()
        sys.exit()
    if os.path.isdir(args.path):
        if args.ds_name in ['wn18', 'fb15']:
            preprocess(args.path)
        if args.w2v_path:
            preprocess_word2vec(args.w2v_path, args.path)
    else:
        arg_parser.print_help()
