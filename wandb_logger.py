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

from catlearn.tensor_utils import Tsor
from catlearn.composition_graph import (
    NodeType, ArrowType, DirectedGraph, CompositeArrow,
)
from catlearn.relation_cache import RelationCache
from catlearn.categorical_model import TrainableDecisionCatModel


wandb.login()
wandb.init(project='catlearn', config={})

def safe_list_get(l: list, value: any, default: any):
  try:
    return l.index(value)
  except ValueError:
    return default

def compute_kge_metrics(
    cache: RelationCache[NodeType, ArrowType],
    triplets: List[Tuple[
        Tuple[
            Optional[NodeType], Optional[NodeType],
            Optional[FrozenSet[ArrowType]]],
        Tuple[NodeType, NodeType, FrozenSet[ArrowType]]]]) -> None:
    """
    Compute KGE metrics for a list of tuples containing:
    - a request triplet of the form (src, tar, label) where any of
        the 3 values may be None, meaning we want to find a match for it
    - an evaluation triplet of the form (src, tar, label) with the actual values
    """
    # compute ranks for each triplet
    ranks = {
        triplet: list(cache.sort_relations(*triplet[1]))
        for triplet in triplets
    }

    sum_ranks = 0
    sum_inverse_ranks = 0
    nonhits_at_n = defaultdict(lambda: 0.)
    tripl, rnk_ll = list(ranks.items())[0]
    for triplet, rank_list in ranks.items():
        nb_triplets = len(triplets)
        #         triplet_rank = rank_list.index(triplet[0])
        # NOTE: below is a 'PLUG' solution to keep computation
        # as it takes the worst case if the right tirplet is not returned in the list
        triplet_rank = safe_list_get(rank_list, triplet[0], nb_triplets)
        sum_ranks += triplet_rank + 1.
        sum_inverse_ranks += 1./(triplet_rank + 1.)
        for valid_rank in range(triplet_rank):
            nonhits_at_n[valid_rank] += 1.

    mean_rank = sum_ranks/nb_triplets
    mean_reciprocal_ranks = sum_inverse_ranks/nb_triplets
    hits_at_n = [
        1. - nonhits_at_n[rank]/nb_triplets
        for rank in range(max(nonhits_at_n, default=0) + 1)]

    return {
        'mean_rank': mean_rank,
        'mean_reciprocal_ranks': mean_reciprocal_ranks,
        'hits_at_n': hits_at_n,
    }


def log_kge_metrics(
    cache: RelationCache[NodeType, ArrowType],
    triplets: List[Tuple[
            Optional[NodeType], Optional[NodeType],
            Optional[FrozenSet[ArrowType]]]],
):
    """
    Log KGE metrics for given triplets:
    remove source, then target and make prediction.
    Average results of both
    """
    sourceless_triplets = (
        (None, target, frozenset([label]))
        for (source, target, label) in triplets
    )
    targetless_triplets = (
        (source, None, frozenset([label]))
        for (source, target, label) in triplets
    )
    full_triplets = [
        (source, target, frozenset([label]))
        for (source, target, label) in triplets
    ]
    source_kge_metrics = compute_kge_metrics(
        cache, list(zip(full_triplets, sourceless_triplets)),
    )
    target_kge_metrics = compute_kge_metrics(
        cache, list(zip(full_triplets, targetless_triplets)),
    )
    average_metrics = {
        'mean_rank': (
            source_kge_metrics['mean_rank']
            + target_kge_metrics['mean_rank'])/2.,
        'mean_reciprocal_ranks': (
            source_kge_metrics['mean_reciprocal_ranks']
            + target_kge_metrics['mean_reciprocal_ranks'])/2.,
        'hits_at_n': [sum(tup)/len(tup) for tup in zip_longest(
            source_kge_metrics['hits_at_n'], target_kge_metrics['hits_at_n'],
            fillvalue=0.)]
    }
    wandb.log(average_metrics)


def log_results(
    cache: RelationCache[NodeType, ArrowType],
    matched: DirectedGraph[NodeType],
    **info_to_log: Any) -> None:
    """
        Log results from a training step
    """
    # sanity check: info_to_log shouldn't override default logs
    std_log_keys = ('nb_labels', 'total_match_cost', 'total_causality_cost',
            'cost_per_label', 'arrow_numbers')
    if info_to_log and not all(info_key not in std_log_keys for info_key in info_to_log):
        raise ValueError(
            'cannot provide any of the default keywords to log.'
            'Default keywords are:\n'
            '    nb_labels\n'
            '    total_match_cost\n'
            '    total_causality_cost\n'
            '    cost_per_label\n'
            '    arrow_numbers\n')

    # costs
    total = sum(
        sum(elem for _, elem in costs.values())
        for costs in matched.edges.values())
    nb_labels = sum(len(costs) for costs in matched.edges.values())
    cost_per_label = total/max(nb_labels, 1) + cache.causality_cost

    # arrow by order and causality check
    used_arrows = set(chain(*((
        arr.suspend(src, tar) for (arr, _) in content.values())
        for src, tar, content in matched.edges(data=True))))
    causal_arrows = set(cache.arrows(include_non_causal=False))
    noncausal_arrows = set(cache.arrows()) - causal_arrows

    arrows = defaultdict(lambda: {'used': 0, 'causal': 0, 'noncausal': 0})
    for arr in used_arrows:
        arrows[len(arr)]['used'] += 1
    for arr in causal_arrows:
        arrows[len(arr)]['causal'] += 1
    for arr in noncausal_arrows:
        arrows[len(arr)]['noncausal'] += 1

    wandb.log({
        'nb_labels': nb_labels,
        'total_match_cost': total,
        'total_causality_cost': cache.causality_cost,
        'cost_per_label': cost_per_label,
        'arrow_numbers': arrows,
        **info_to_log
    })


def save_params(
    model: TrainableDecisionCatModel):
    """
    Save relation and scoring model parameters
    for a trainable decision cat model
    """
    for name, param in model.named_parameters():
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
