#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utilities for device placement
"""

from torch import nn


def to_device(device: str, model: nn.Module) -> None:
    """ Send model to specified device

    Accepts 'cpu' or 'cuda' for generic placement,
    or specific device placement eg 'cuda:1' to move
    to 2nd available GPU.
    """
    for p in model.parameters():
        if device == 'cpu':
            p.cpu()
        elif device == 'cuda':
            p.cuda()
        else:
            p.device(device)
