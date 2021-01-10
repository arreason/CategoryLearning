import os
from typing import Any, Tuple, Optional, FrozenSet, List, Mapping
from collections import defaultdict
from itertools import chain

from torch.nn.functional import pad

from .numerics import Tsor, get_index_in_list
from ..composition_graph import (
    NodeType, ArrowType, DirectedGraph,
)
from ..relation_cache import RelationCache


ArrowHypothesisTriplet = Tuple[NodeType, NodeType, FrozenSet[ArrowType]]
PartialArrowHypothesisTriplet = Tuple[
        Optional[NodeType], Optional[NodeType],
        Optional[FrozenSet[ArrowType]],
]
TripletMatchPair = Tuple[
    ArrowHypothesisTriplet,
    PartialArrowHypothesisTriplet,
]


def compute_training_metrics(
    cache: RelationCache[NodeType, ArrowType],
    matched: DirectedGraph[NodeType],
) -> Mapping[str, Any]:
    """
        Log some metrics from a training step
    """
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

    return {
        'nb_labels': nb_labels,
        'total_match_cost': total,
        'total_causality_cost': cache.causality_cost,
        'cost_per_label': cost_per_label,
        'arrow_numbers': arrows,
}

from tqdm import tqdm

def compute_eval_triplet_ranks(
    cache: RelationCache[NodeType, ArrowType],
    triplets: List[TripletMatchPair],
    default_rank: int,
    max_rank: Optional[int] = None,
) -> Mapping[ArrowHypothesisTriplet, int]:
    """
    Compute the rank of the true triplet in the list of ranked hypothesis
    generated from the partial triplet
    - default_rank: The rank used if the triplet is not found
    - max_rank: only ranks up to max_rank are computed.
    Hence will default to default_rank above it
    (if None all ranks are taken into account).
    """
    # compute ranks for each triplet
    rank_lists = {
        triplet[0]: cache.sort_relations(*triplet[1], n_items=max_rank)
        for triplet in triplets
    }

    ranks = {}
    for triplet, rank_list in tqdm(rank_lists.items()):
        triplet_rank = get_index_in_list(
            rank_list, triplet, default_index=default_rank,
        )
        ranks[triplet] = triplet_rank
    return ranks


def compute_kge_metrics(
    cache: RelationCache[NodeType, ArrowType],
    triplets: List[TripletMatchPair],
    default_rank: int,
    max_rank: Optional[int] = None,
) -> Mapping[str, Any]:
    """
    Compute KGE metrics for 
    - a list of tuples containing:
        - an evaluation triplet of the form (src, tar, label) with the actual values
        - a request triplet of the form (src, tar, label) where any of
        the 3 values may be None, meaning we want to find a match for it
    - default_rank: the default value of rank when the triplet is not found
    - max_rank: only ranks up to max_rank are computed. Hence will default
    to default_rank above it (if None all ranks are taken into account)
    """
    # compute ranks for each triplet
    ranks = compute_eval_triplet_ranks(
        cache, triplets, default_rank, max_rank=max_rank)
    nb_triplets = len(triplets)

    mean_rank = sum(ranks.values())/nb_triplets
    mean_reciprocal_ranks = sum(
        1./(1. + value) for value in ranks.values())/nb_triplets
    max_found = max((value for value in ranks.values() if value < default_rank), default=default_rank)
    hits_at_n = sum((
        Tsor([float(i >= value) for i in range(max_found + 1)])
        for value in ranks.values()
    ))

    return {
        'mean_rank': mean_rank,
        'mean_reciprocal_ranks': mean_reciprocal_ranks,
        'hits_at_n': hits_at_n,
    }


def compute_average_missing_source_target_kge_metrics(
    cache: RelationCache[NodeType, ArrowType],
    triplets: List[Tuple[
            Optional[NodeType], Optional[NodeType],
            Optional[FrozenSet[ArrowType]]]],
    default_rank: int,
    max_rank: Optional[int] = None,
) -> Mapping[str, Any]:
    """
    Log KGE metrics for given triplets:
    remove source, then target and make prediction.
    Average results of both
    - default_rank: the default value of rank when the triplet is not found
    - max_rank: only ranks up to max_rank are computed. Hence will default
    to default_rank above it (if None all ranks are taken into account)
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
        cache, list(zip(full_triplets, sourceless_triplets)), default_rank,
        max_rank=max_rank
    )
    target_kge_metrics = compute_kge_metrics(
        cache, list(zip(full_triplets, targetless_triplets)), default_rank,
        max_rank=max_rank
    )

    source_hits_at_n = source_kge_metrics['hits_at_n']
    target_hits_at_n = target_kge_metrics['hits_at_n']

    # force hits_at_n 1-d tensors of both dictionaries to have the same size
    max_found = max(len(source_hits_at_n), len(target_hits_at_n))
    source_kge_metrics['hits_at_n'] = pad(
        source_hits_at_n, (0, max_found - len(source_hits_at_n)),
        value=source_hits_at_n[-1]
    )
    target_kge_metrics['hits_at_n'] = pad(
        target_hits_at_n, (0, max_found - len(target_hits_at_n)),
        value=target_hits_at_n[-1]
    )

    average_metrics = {
        key: (source_kge_metrics[key] + target_kge_metrics[key])/2.
        for key in source_kge_metrics
    }
    return average_metrics
