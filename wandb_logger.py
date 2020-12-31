"""
A simple module to initialize wandb and define log functionalities for catlearn
 models
"""
import os
from typing import Any, Tuple, Optional, FrozenSet, List
from collections import defaultdict
from itertools import chain, zip_longest
import math

import wandb
import torch

from catlearn.utils.numerics import Tsor
from catlearn.composition_graph import (
    NodeType, ArrowType, DirectedGraph, CompositeArrow,
)
from catlearn.relation_cache import RelationCache
from catlearn.categorical_model import TrainableDecisionCatModel


wandb.login()
wandb.init(project='catlearn', config={})


def save_params(
    model: TrainableDecisionCatModel):
    """
    Save relation and scoring model parameters
    for a trainable decision cat model
    """
    wandb.log({
        "params": {
            name: torch.FloatTensor(param) for (name, param) in model.named_parameters()}
    })


def save_file(file_path: str):
    """Upload file to wandb run session"""
    # Convert if Path object is passed
    file_path = str(file_path)
    if file_path and os.path.isfile(file_path):
        wandb.save(file_path)
    else:
        print(f'Wrong path: {file_path}')
