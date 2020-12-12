"""
A simple module to initialize wandb and define log functionalities for catlearn
 models
"""
from typing import Any
from collections import defaultdict
from itertools import chain
import wandb

from catlearn.tensor_utils import Tsor
from catlearn.composition_graph import NodeType, ArrowType, DirectedGraph
from catlearn.relation_cache import RelationCache
from catlearn.categorical_model import TrainableDecisionCatModel


wandb.login()
wandb.init(project='catlearn', config={})


def log_results(
    cache: RelationCache[NodeType, ArrowType],
    matched: DirectedGraph[NodeType],
    **info_to_log: Any) -> None:
    """
        Log results from a training step
    """


    # sanity check: info_to_log shouldn't override default logs
    if {
            'nb_labels', 'total_match_cost', 'total_causality_cost',
            'cost_per_label', 'arrow_numbers'} and info_to_log:
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
    wandb.log({
        "params": {
            name: param.cpu() for (name, param) in model.named_parameters()}
    })
