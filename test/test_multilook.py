import warnings

import dask.array as da
import numpy as np
import pytest

import tophu


class TestMultilook:
    def test_multilook_1d(self):
        # Expected output array.
        expected = da.arange(4, dtype=np.float64)

        # Build the input array by repeating each element `nlooks` times.
        nlooks = 11
        input = da.repeat(expected, repeats=nlooks)

        # Multilook.
        output = tophu.multilook(input, nlooks=nlooks)

        # Check results.
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        assert da.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_multilook_2d(self):
        # Expected output array.
        expected = da.arange(12, dtype=np.float64).reshape(3, 4)

        # Build the input array by repeating each element `nlooks` times.
        nlooks = (7, 5)
        input = expected
        for axis, n in enumerate(nlooks):
            input = da.repeat(input, repeats=n, axis=axis)

        # Multilook.
        output = tophu.multilook(input, nlooks=nlooks)

        # Check results.
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        assert da.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_multilook_3d(self):
        # Expected output array.
        expected = da.arange(60, dtype=np.float64).reshape(3, 4, 5)

        # Build the input array by repeating each element `nlooks` times.
        nlooks = (5, 1, 3)
        input = expected
        for axis, n in enumerate(nlooks):
            input = da.repeat(input, repeats=n, axis=axis)

        # Multilook.
        output = tophu.multilook(input, nlooks=nlooks)

        # Check results.
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        assert da.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_multilook_complex(self):
        # Expected output array.
        expected = (
            da.arange(12, dtype=np.float64).reshape(3, 4)
            + 1.0j * da.arange(12, 24, dtype=np.float64).reshape(3, 4)
        )

        # Build the input array by repeating each element `nlooks` times.
        nlooks = 3
        input = expected
        for axis in range(expected.ndim):
            input = da.repeat(input, repeats=nlooks, axis=axis)

        # Multilook.
        output = tophu.multilook(input, nlooks=nlooks)

        # Check results.
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        assert da.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_nlooks_length_mismatch(self):
        # Check that `multilook()` fails if length of `nlooks` doesn't match `arr.ndim`.
        arr = da.zeros((15, 15), dtype=np.float64)
        errmsg = r"length of nlooks \(3\) must match input array rank \(2\)"
        with pytest.raises(ValueError, match=errmsg):
            tophu.multilook(arr, nlooks=(1, 2, 3))

    def test_zero_or_negative_nlooks(self):
        # Check that `multilook()` fails if `nlooks` has zero or negative values.
        arr = da.zeros((15, 15), dtype=np.float64)
        with pytest.raises(ValueError, match="number of looks must be >= 1"):
            tophu.multilook(arr, nlooks=(-1, 3))
        with pytest.raises(ValueError, match="number of looks must be >= 1"):
            tophu.multilook(arr, nlooks=(5, 0))

    def test_nlooks_too_large(self):
        # Check that `multilook()` fails if `nlooks` is larger than the input array
        # shape (along any axis).
        arr = da.zeros((15, 15), dtype=np.float64)
        errmsg = "number of looks should not exceed array shape"
        with pytest.raises(ValueError, match=errmsg):
            tophu.multilook(arr, nlooks=(1, 16))

    def test_even_nlooks_warning(self):
        # Check that a warning is emitted if any component of `nlooks` is even-valued.
        arr = da.zeros((15, 16), dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            # Run `multilook()` with all warnings enabled.
            warnings.simplefilter("always")
            tophu.multilook(arr, nlooks=(3, 4))

            # Check that a single warning was emitted.
            assert len(w) == 1

            # Check the warning category and message.
            assert issubclass(w[0].category, RuntimeWarning)
            substr = "one or more components of nlooks is even-valued"
            assert substr in str(w[0].message)

    def test_throwaway_samples_warning(self):
        # Check that a warning is emitted if there are throwaway samples due to
        # the input array shape not being an integer multiple of `nlooks`.
        arr = da.zeros((23, 21), dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            # Run `multilook()` with all warnings enabled.
            warnings.simplefilter("always")
            output = tophu.multilook(arr, nlooks=(3, 5))

            # Check that a single warning was emitted.
            assert len(w) == 1

            # Check the warning category and message.
            assert issubclass(w[0].category, RuntimeWarning)
            substr = "input array shape is not an integer multiple of nlooks"
            assert substr in str(w[0].message)

        # Check the output shape.
        assert output.shape == (7, 4)
