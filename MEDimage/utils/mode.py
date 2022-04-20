#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import numpy as np


def mode(x, return_counts=False) -> Union[np.ndarray, np.ndarray]:
    """Implementation of mode that also returns counts,
    unlike the standard statistics.mode.

    Args:
        x (ndarray): n-dimensional array of which to find mode.
        return_counts (bool): If True, also return the number 
            of times each unique item appears in x.
    Returns:
        ndarray: Array of the modal (most common) value in the
            given array.
        ndarray: Array of the counts.

    """

    unique_values, counts = np.unique(x, return_counts=True)

    if return_counts:
        return unique_values[np.argmax(counts)], np.max(counts)

    return unique_values[np.argmax(counts)]
