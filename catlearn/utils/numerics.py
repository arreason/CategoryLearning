#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:53:39 2018

@author: christophe_c

various utilities for tensor manipulation
"""
from typing import (
    Optional, Sequence, Iterable, Tuple,
    Any, Callable, List, TypeVar,
)
from heapq import nsmallest, nlargest

import torch
from torch import Tensor as Tsor
from torch.nn.functional import kl_div


_T = TypeVar('_T')


# default precision of computations
DEFAULT_EPSILON = 1e-6


def sorted_nfirst(
    iterable: Iterable[_T],
    key: Optional[Callable[[_T], Any]] = None, reverse: bool = False,
    n_items: Optional[int] = None,
) -> List[_T]:
    """
    Sort the n_items first elements of the given iterable.
    If n_items is None,
    returns all elements of the iterable in order.
    A custom key function can be supplied to customize the sort order,
    and the reverse flag can be set to request the result in descending order.
    """
    if n_items is None:
        return sorted(iterable, key=key, reverse=reverse)

    sorter = nlargest if reverse else nsmallest
    return sorter(n_items, iterable, key=key)


def get_index_in_list(
    this_list: List[_T], value: _T, start: int = 0, stop: Optional[int] = None,
    default_index: Optional[int] = None,
) -> Optional[int]:
    """
    Get index of value in this_list. If value is not in list, returns
    default_index if it is given, or the length of the list.
    """
    list_length = len(this_list)
    try:
        index_stop = stop if stop else list_length
        return this_list.index(value, start, index_stop)
    except ValueError:
        return default_index if default_index else list_length


def repeat_tensor(
        to_repeat: Tsor,
        nb_repeats: int,
        axis: int,
        copy: bool = False) -> Tsor:
    """
    stacks a tensor over itself nb_repeats times, over axis dimension

    /!\\ Default behavior (copy=False) does not allocate new memory.
    Any mutation of the resulting tensor's new axis will be shared
    across the new dimension!

    inputs:
        to_repeat: the tensor to stack
        nb_repeats: the number of times one wants to repeat the tensor
        axis: on which axis to repeat
        copy: should repeat allocate new memory? Otherwise a view is returned
    """
    # Torch repeat expects 0 <= axis <= ndimension
    ndim = to_repeat.ndimension()
    if axis < 0:
        axis += ndim + 1

    existing_dim = 1 if copy else -1
    pattern = [existing_dim for _ in range(ndim + 1)]
    pattern[axis] = nb_repeats
    if copy:
        return to_repeat.unsqueeze(axis).repeat(*pattern)
    return to_repeat.unsqueeze(axis).expand(*pattern)


def is_shape_broadcastable(
        first_shape: Sequence,
        last_shape: Sequence) -> bool:
    """
    Test if two shapes can be broadcasted together, following numpy/pytorch
    conventions
    """
    # iterate on couple of dims from last; will stop when one shape is depleted
    # check individual dimensions from end are broadcastable
    return all((m == n) or (m == 1) or (n == 1)
               for m, n in zip(first_shape[::-1], last_shape[::-1]))


def sorted_weighted_sum(
        to_sum: Tsor,
        weights: Tsor,
        axis: int,
        keepdim: bool,
        largest: bool) -> Tsor:
    """
    Orders elements to sum then sums them with given ponderations
    inputs:
        to_sum: a tensor, will be sumed on the given axes
        weights: weights to apply to the summands. If smaller than number of
                elements to sum, 0 padding is assumed
        axis: the axis on which to perform reduction
        keepdim: wether the reduced axis should be kept in shape
                the tensor after ordered summation on the required axis
        largest: wether the k largest are returned;
                otherwise the k smallest are returned
    outputs:
        Tsor, of shape :
            input_shape[:(axis-1)] + input_shape[(axis+1):] if
                keepdim is false
            input_shape[:(axis-1)] + (1,) + input_shape[(axis+1):] if
                keepdim is true
    """
    nb_weights = weights.numel()
    assert weights.ndimension() == 1
    assert nb_weights <= to_sum.shape[axis]

    # get the top k elements on the required axis (k = number of weights)
    sorted_topk = torch.topk(to_sum, nb_weights, sorted=True,
                             dim=axis, largest=largest)[0]

    # view of weights with the right shape for broadcasting
    nb_dims = to_sum.ndimension()
    reshaped_weights = weights.view([nb_weights
                                     if (ax - axis) % nb_dims == 0 else 1
                                     for ax in range(nb_dims)])

    # weighted sum returned as result
    return torch.sum(reshaped_weights * sorted_topk,
                     dim=axis, keepdim=keepdim)


def tensor_equals(
        left: Tsor,
        right: Tsor,
        precision: float = 1e-6) -> bool:
    """ Are two tensors equal? """
    return torch.lt(
        torch.abs(left.double() - right.double()), precision).all()


def full_like(
        value,  # numbers.Real but mypy does not like it...
        tensor: Tsor,
        shape: Optional[Sequence[int]] = None) -> Tsor:
    """
    Create a new `value`-filled tensor based on an input tensor.
    NB: type, device and layout are conserved
    """
    if shape is None:
        shape = tensor.size()
    return torch.full(
        shape, value, dtype=tensor.dtype,
        device=tensor.device, layout=tensor.layout)


def zeros_like(
        tensor: Tsor,
        shape: Optional[Sequence[int]] = None) -> Tsor:
    """
    Create a new 0-filled tensor based on an input tensor.
    NB: type, device and layout are conserved
    """
    return full_like(0.0, tensor, shape)


def ones_like(
        tensor: Tsor,
        shape: Optional[Sequence[int]] = None) -> Tsor:
    """
    Create a new 1-filled tensor based on an input tensor.
    NB: type, device and layout are conserved
    """
    return full_like(1.0, tensor, shape)


def one_hot(label: int, nb_labels):
    enc = torch.zeros(nb_labels)
    enc[label] = 1.0
    return enc


def clip_proba(
        proba_vector: Tsor,
        dim: int = -1,
        epsilon: float = DEFAULT_EPSILON) -> Tsor:
    """
    clip a probability vector to remove pure 0. and 1. values, to avoid
    numerical errors. Also works on subprobability vectors.

    Arguments:
        proba_vector: the tensor of probability vectors
        dim: the dimension on which values can be interpreted as probabilities
        epsilon: the clipping numerical tolerance
    """
    if proba_vector.ndimension() == 0:
        vector_size = 1.
    else:
        vector_size = float(proba_vector.shape[dim])
    return (1. - vector_size * epsilon) * proba_vector + epsilon


def subproba_kl_div(
        predicted: Tsor, labels: Tsor,
        epsilon: float = DEFAULT_EPSILON,
        dim: int = -1, keepdim: bool = False) -> Tsor:
    """
    compute KL-divergence of subprobability vectors, extending them to sum to 1
    The dim on which they are assumed to be subprobability vectors is -1
    by default
    """
    # compute complement vectors
    complement_predicted = 1. - predicted.sum(dim=dim, keepdim=keepdim)
    complement_labels = 1. - labels.sum(dim=dim, keepdim=keepdim)

    # compute raw KL-div (unsumed) of vectors and their complement
    kl_direct = kl_div(
        torch.log(clip_proba(predicted, dim=dim, epsilon=epsilon)),
        labels, reduction="none")
    kl_complement = kl_div(
        torch.log(clip_proba(complement_predicted, dim=dim, epsilon=epsilon)),
        complement_labels, reduction="none")

    # sum all components of kl and return
    return kl_direct.sum(dim=dim, keepdim=keepdim) + kl_complement


def remap_subproba(
        to_remap: Tsor, reference: Tsor,
        dim: int = -1) -> Tsor:
    """
    remap a subprobability vector to an other subprobability vector, making
    sure that remapping the reference yields a vector of total proba 1.
    This is done by considering coordinates as squares of coordinates
    of vectors in euclidian space.
    Considering the following vectors of norm 1:
                p = sqrt(to_remap)                      , sqrt(1-sum(to_remap))
                r0 = sqrt(reference)/sqrt(sum(reference)), 0.
                r = sqrt(reference)                     , sqrt(1-sum(reference)
                v = 0 ... 0                             , 1.
        we want to apply to p a rotation
        in the plane [r0, v] which brings r to r0.
        The angle theta of rotation verifies:
                cos(theta) = sum(reference)
        Hence we get:
            p - <r0, p>r0 - <v,p>v
            + (<r0, p>cos(theta) - <v, p>sin(theta))r0
            + (<r0, p>sin(theta) + <v, p>cos(theta))v
        Taking the square of all coordinates except the last then yields
        the result of remap_subproba
    """
    # compute total proba of both vectors
    to_remap_proba = to_remap.sum(dim=dim, keepdim=True)
    reference_proba = reference.sum(dim=dim, keepdim=True)

    # compute square roots, to use as vectors in euclidian vector space
    to_remap_sqrt = to_remap.sqrt()
    reference_sqrt = reference.sqrt()

    # inner product of sqrt vectors and total correction factor
    inner_product = torch.sum(
        to_remap_sqrt * reference_sqrt, dim=dim, keepdim=True)
    compl_proba_sqrt = torch.sqrt(
        (1. - to_remap_proba) * (1. - reference_proba))
    corr_factor = (
        inner_product * (1. - 1./reference_proba.sqrt())
        + compl_proba_sqrt)/reference_proba.sqrt()

    return (to_remap_sqrt + corr_factor * reference_sqrt) ** 2


# Abstract type definitions
# NB: docstring temporarily disabled
# pylint: disable=missing-docstring
class AbstractModel:
    def parameters(self, recurse: bool = True) -> Iterable[Tsor]:
        return (param for (_, param) in self.named_parameters(recurse=recurse))

    def named_parameters(
        self, recurse: bool = True
    ) -> Iterable[Tuple[str, Tsor]]:
        raise NotImplementedError()

    def freeze(self) -> None:
        raise NotImplementedError()

    def unfreeze(self) -> None:
        raise NotImplementedError()
