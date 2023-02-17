from collections.abc import Sequence

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ceil_divide",
    "debug_print_array",
    "iseven",
    "interleave",
]


def ceil_divide(n: ArrayLike, d: ArrayLike) -> NDArray:
    """
    Return the smallest integer greater than or equal to the quotient of the inputs.

    Computes integer division of dividend `n` by divisor `d`, rounding up instead of
    truncating.

    Parameters
    ----------
    n : array_like
        Numerator.
    d : array_like
        Denominator.

    Returns
    -------
    q : numpy.ndarray
        Quotient.
    """
    n = np.asanyarray(n)
    d = np.asanyarray(d)
    return (n + d - np.sign(d)) // d


def debug_print_array(name: str, arr: da.Array) -> None:
    """ """
    print(f"{name}: shape={arr.shape} | chunksize={arr.chunksize} | numblocks={arr.numblocks} | dtype={arr.dtype}")


def iseven(n: int) -> bool:
    """Check if the input is even-valued."""
    return n % 2 == 0


def interleave(s1: Sequence, s2: Sequence) -> list:
    """
    Interleave two sequences of the same length.

    Parameters
    ----------
    s1, s2 : sequence
        Two sequences with the same length.

    Returns
    -------
    interleaved : list
        A list containing alternating elements from the input sequences.
    """
    if len(s1) != len(s2):
        raise ValueError(
            f"length mismatch: sequence lengths must be equal, got {len(s1)} and"
            f" {len(s2)}"
        )
    return [val for pair in zip(s1, s2) for val in pair]
