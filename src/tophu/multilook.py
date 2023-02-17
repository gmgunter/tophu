import warnings
from collections.abc import Iterable
from typing import SupportsInt, Tuple, Union, cast

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike

from . import util

__all__ = [
    "multilook",
]


IntOrInts = Union[SupportsInt, Iterable[SupportsInt]]


def normalize_nlooks_tuple(nlooks: IntOrInts, ndim: int) -> Tuple[int, ...]:
    """
    Normalize `nlooks` to a tuple of ints.

    Normalize `nlooks` into a tuple with length equal to `ndim`. If `nlooks` was a
    scalar, take the same number of looks along each axis in the array.

    Parameters
    ----------
    nlooks : int or iterable of int
        Number of looks along each axis of the array.
    ndim : int
        The rank of the array to be multilooked.

    Returns
    -------
    nlooks : tuple of int
        A tuple with length `ndim` containing the number of looks to take along each
        axis.
    """
    try:
        n = int(nlooks)  # type: ignore
        nlooks = (n,) * ndim
    except TypeError:
        nlooks = tuple([int(n) for n in nlooks])  # type: ignore
        if len(nlooks) != ndim:
            raise ValueError(
                f"length mismatch: length of nlooks ({len(nlooks)}) must match input"
                f" array rank ({ndim})"
            )

    # Convince static type checkers that `nlooks` is a tuple of ints now.
    nlooks = cast(Tuple[int, ...], nlooks)

    return nlooks


def validate_nlooks(nlooks: Tuple[int, ...], arr: ArrayLike) -> None:
    """
    Check that `nlooks` was valid.

    Raise a exception if the specified number of looks was invalid for input array
    `arr`, or else emit a warning if the multilooking operation might have unexpected
    results.

    Parameters
    ----------
    nlooks : int or iterable of int
        Number of looks along each axis of the array.
    arr : array_like
        The array to be multilooked.

    Raises
    ------
    ValueError
        If the number of looks was negative or zero for any axis.
    ValueError
        If the number of looks exceeded the input array shape.
    """
    # The number of looks must be at least 1 and at most the size of the input array
    # along the corresponding axis.
    for m, n in zip(arr.shape, nlooks):
        if n < 1:
            raise ValueError("number of looks must be >= 1")
        elif n > m:
            raise ValueError("number of looks should not exceed array shape")

    # Warn if the number of looks along any axis is even-valued.
    if any(map(util.iseven, nlooks)):
        warnings.warn(
            "one or more components of nlooks is even-valued -- this will result in a"
            " phase delay in the multilooked data equivalent to a half-bin shift",
            RuntimeWarning,
        )

    # Warn if any array dimensions are not integer multiples of `nlooks`.
    if any(m % n != 0 for (m, n) in zip(arr.shape, nlooks)):
        warnings.warn(
            "input array shape is not an integer multiple of nlooks -- remainder"
            " samples will be excluded from output",
            RuntimeWarning,
        )


def multilook(arr: da.Array, nlooks: IntOrInts) -> da.Array:
    """
    Multilook an array by simple averaging.

    Performs spatial averaging and decimation. Each element in the output array is the
    arithmetic mean of neighboring cells in the input array.

    Parameters
    ----------
    arr : dask.array.Array
        Input array.
    nlooks : int or iterable of int
        Number of looks along each axis of the input array.

    Returns
    -------
    out : dask.array.Array
        Multilooked array.

    Notes
    -----
    If the length of the input array along a given axis is not evenly divisible by the
    specified number of looks, any remainder samples from the end of the array will be
    discarded in the output.
    """
    # Sanitize input `nlooks`.
    nlooks = normalize_nlooks_tuple(nlooks, arr.ndim)
    validate_nlooks(nlooks, arr)

    # If the input array shape is not a multiple of the multilook window shape, first
    # truncate the remainder samples from the end each axis.
    out_shape = np.floor_divide(arr.shape, nlooks)
    valid_shape = out_shape * nlooks
    cropped_arr = arr[tuple(slice(n) for n in valid_shape)]

    # Multilook by reshaping the array to insert new axes and reducing along them.
    # To demonstrate the approach, first consider a 1-D input array with length N*K,
    # where K is the number of looks. In order to multilook this array, we could reshape
    # it to a row-major 2-D array of N rows by K columns, and then simply average along
    # each row. This procedure is straightforward to extend to N-dimensional inputs. We
    # reshape the array to split each input axis into a pair of axes, then reduce along
    # each second axis.
    tmp_shape = tuple(util.interleave(out_shape, nlooks))
    reshaped_arr = cropped_arr.reshape(tmp_shape)
    axes = tuple(np.ogrid[1:reshaped_arr.ndim:2])
    return da.mean(reshaped_arr, axis=axes)
