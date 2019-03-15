#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:36:07 2018

@author: christophe_c
"""

from typing import Iterable

import numpy as np
import numpy.random as nprand


def gauss_gen(batch_shape: Iterable[int], *arrs: np.ndarray):
    return nprand.randn(*batch_shape) + sum(arrs)
