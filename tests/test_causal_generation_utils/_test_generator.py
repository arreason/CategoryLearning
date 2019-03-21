"""
The main module for generation of new functions
"""

from typing import Iterable, Any, Union
from pathlib import Path

import numpy as np
import numpy.random as nprand
import re

UTILS_FOLDER = Path(re.sub("_generator.py", "_utils", __file__))
DIM = np.load(UTILS_FOLDER.joinpath("dim.npy"))


class ParamsDict(dict):
    """
    The class for extra parameters of the generating functions
    """
    def __missing__(self, key: Union[int, str]) -> np.ndarray:
        """
        Specify what happens when accessing a missing key
        """
        # generate and save dictionary with the new value
        if key == "bias":
            array = nprand.randn(DIM)
        else:
            array = nprand.randn(DIM, DIM)
        np.save(UTILS_FOLDER, dict(self))

        # return the new value
        return array


def __getattr__(name: str) -> Any:
    """
    Attribute getter to get functions called with their right parameters
    """
    # load existing parameters or generate a new parameters dictionary
    path = UTILS_FOLDER.joinpath(name).with_suffix(".npy")
    if path.is_file():
        params = ParamsDict(np.load(path))
    else:
        params = ParamsDict()

    def gauss_gen(
            batch_shape: Iterable[int],
            *arrs: np.ndarray) -> np.ndarray:
        return (
            params["bias"][..., np.newaxis]
            + params["rand_filter"] @ nprand.randn(*(batch_shape + (DIM,)))[
                ..., np.newaxis]
            + sum(
                params[idx] @ arrs[idx][..., np.newaxis]
                for idx in range(len(arrs))))[..., 0]

    return gauss_gen
