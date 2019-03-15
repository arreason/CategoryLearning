#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:10:12 2018
@author: christophe_c
a module for useful functions for testing
"""

from typing import Callable, Iterator
import random


def pytest_generate_tests(metafunc):
    """
    called once for each test function. Decorates the test runs with the right
    parametrization. The parametrization of each function is to be found in a
    params dictionary in the same scope as the function, in a list associated
    to the key being the name of the function.
    """

    # collect parameters list for test
    func_name = metafunc.function.__name__
    if hasattr(metafunc.cls, "params") and func_name in metafunc.cls.params:
        funcarglist = metafunc.cls.params[func_name]
    else:
        # if no parameters are declared, do as if the params list is empty
        funcarglist = []

    # if specific arguments are declared, execute tests using those
    if funcarglist:
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames, [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist]
            )
    else:
        # no specific parameters. Any existing fixture will still be used.
        print(
            f"No specific parametrization found"
            f" for {metafunc.cls.__name__}.{func_name}")
        metafunc.parametrize([], [])


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
