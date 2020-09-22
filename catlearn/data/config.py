from typing import Dict, List, NewType

raw_dataset_pat: Dict[str, str] = {
    'train': 'train.txt',
    'valid': 'valid.txt',
    'test': 'test.txt',
    'sep': '\t'
}

preproc_pat: Dict[str, str] = {
    'relation2id': 'relation_to_id.txt',
    'entity2id': 'entity_to_id.txt',
    'wordset': 'word_set.txt',
    'word2vec_short': 'w2v_short.pkl',
    'train': 'train_pp.txt',
    'valid': 'valid_pp.txt',
    'test': 'test_pp.txt',
    'sep': '\t'
}

