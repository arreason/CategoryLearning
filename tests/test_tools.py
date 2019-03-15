#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:10:12 2018
@author: christophe_c
a module for useful functions for testing
"""

from typing import Callable, Iterator
import random


def random_int_list(min_length: int,
                    max_length: int,
                    min_member: int,
                    max_member: int) -> Callable[[], Iterator[int]]:
    """
    Draws a list of integers of random size

    params:
    - min_length: the minimum length of the list
    - max_length: the maximum length of the list
    - member_min: the minimum value of members of the list
    - member_max: the maximum value of members of the list
    """
    assert min_length >= 0, "a list cannot have negative length"
    assert max_length >= min_length
    assert max_member >= min_member

    # draw list length
    length = random.randint(min_length, max_length)

    # return the generator
    return lambda: (random.randint(min_member, max_member)
                    for _ in range(length))
