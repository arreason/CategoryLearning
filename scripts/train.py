from typing import Callable, Iterable, Any
import os
import argparse
import warnings
import logging
import sys
import torch
sys.path.append('../catlearn')
from catlearn.data.dataset import Dataset
from catlearn.tensor_utils import (Tsor, DEFAULT_EPSILON)
from catlearn.graph_utils import DirectedGraph
from catlearn.algebra_models import (Algebra, VectAlgebra)
from catlearn.composition_graph import CompositeArrow
from catlearn.categorical_model import (TrainableDecisionCatModel, RelationModel,
                                        ScoringModel)

LOGGING_FILE = 'train_py.log'
logging.basicConfig(format='%(asctime)s %(message)s', filename=LOGGING_FILE, level=logging.DEBUG)
logging.debug('\nStart logging of training process.')

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

def str_color(color_type: str, string: str) -> str:
    """Colorize strings printed in std out.
    Args:
        color_type (str): color according to naming
        string (str): string to be colorized
    Returns:
        str: formated string
    """
    if color_type == 'W':
        return COLOR_SEQ % (30 + COLORS['WARNING']) + string + RESET_SEQ
    elif color_type == 'I':
        return BOLD_SEQ + COLOR_SEQ % (30 + COLORS['INFO']) + string + RESET_SEQ
    elif color_type == 'C':
        return COLOR_SEQ % (30 + COLORS['CRITICAL']) + string + RESET_SEQ
    elif color_type == 'E':
        return COLOR_SEQ % (30 + COLORS['ERROR']) + string + RESET_SEQ
    elif color_type == 'D':
        return COLOR_SEQ % (30 + COLORS['DEBUG']) + string + RESET_SEQ
    else:
        return string

def train_step_log(color_type: str, message: str):
    warning_str = str_color(color_type, message)
    logging.warning(warning_str)
    warnings.warn(warning_str, UserWarning)

class CustomRelation(RelationModel):
    """ Fake relations
    two nodes and rel
    """
    def __init__(self, nb_features: int, nb_labels: int, algebra: Algebra) -> None:
        self.linear = torch.nn.Linear(2 * nb_features + nb_labels, algebra.flatdim)

    @property
    def parameters(self) -> Callable[[], Iterable[Any]]:
        return self.linear.parameters

    def __call__(self, x: Tsor, y: Tsor, l: Tsor) -> Tsor:
        """ Compute x R y """
        return self.linear(torch.cat((x, y, l), -1))

class CustomScore(ScoringModel):
    """ Must be defined. depends on algebra and
    in the scope of definition of the project. """
    def __init__(
            self,
            nb_features: int,
            nb_scores: int,
            algebra: Algebra) -> None:
        self.linear = torch.nn.Linear(
            2 * nb_features + algebra.flatdim, nb_scores + 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    @property
    def parameters(self) -> Callable[[], Iterable[Any]]:
        return self.linear.parameters

    def __call__(self, src: Tsor, dst: Tsor, rel: Tsor) -> Tsor:
        """ Compute S(src, dst, rel) """
        cat_input = torch.cat((src, dst, rel), -1)
        return self.softmax(self.linear(cat_input))[..., :-1]

def train(ds_name: str, ds_path: str):
    """Script to configure and start training.
    """
    if ds_name == 'wn18':
        ds = Dataset(path=ds_path, ds_name='wn18', node_vec_dim=10)
    elif ds_name == 'fb15':
        ds = Dataset(path=ds_path, ds_name='fb15', node_vec_dim=10)
    train_step_log('W', 'Loaded dataste {ds_name}')
    labels = DirectedGraph(ds.train)
    train_step_log('W', 'Created DirectedGraph')
    algebra = VectAlgebra(ds.entity_vec_dim)
    train_step_log('W', 'Created algebra model')

    # ATTENTION: Dummy composite arrow for debug only
    arrow = CompositeArrow(nodes=[1,2,3], arrows=[0,1])
    train_step_log('W', 'Created composite arrow')

    relation_model = CustomRelation(
        nb_features=ds.entity_vec_dim,
        nb_labels=len(ds.relation2id),
        algebra=algebra
    )
    train_step_log('W', 'Created relaton model')

    scoring_model = CustomScore(
        nb_features=ds.entity_vec_dim,
        nb_scores=len(ds.relation2id),
        algebra=algebra
    )
    train_step_log('W', 'Created scoring model')

    model = TrainableDecisionCatModel(
        relation_model=relation_model,
        label_universe=ds.relation_id2vec,
        scoring_model=scoring_model,
        algebra_model=algebra,
        optimizer=torch.optim.Adam,
        epsilon=DEFAULT_EPSILON
    )
    train_step_log('W', 'Created TrainableDecisionCatModel')

    train_step_log('W', 'Started training')
    model.train(
        data_points = ds.entity_id2vec,
        relations = [arrow],
        labels = labels,
        step = True,
        match_negatives=False
    )
    train_step_log('W', 'Finished training')


def parse_args():
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--ds_name', help='Specify dataset name one of: wn18, fb15.',
                        type=str)
    parser.add_argument('--ds_path', help=('Path to a dataset folder where raw dataset files '
        'are stored. For raw files naming format check config.py.'),
                        type=str, default='')
    return parser

if __name__ == '__main__':
    arg_parser = parse_args()
    try:
        args = arg_parser.parse_args()
    except ValueError:
        arg_parser.print_help()
        sys.exit()
    if (os.path.isdir(args.ds_path) and args.ds_name):
        train(ds_name=args.ds_name, ds_path=args.ds_path)
    else:
        arg_parser.print_help()
