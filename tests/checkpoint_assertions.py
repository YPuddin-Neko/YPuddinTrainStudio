"""Strict recursive state comparisons, including optimizer and all RNG tensors."""

import numpy as np
import torch


def assert_checkpoint_value_exact(actual, expected, path="state"):
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_checkpoint_value_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_checkpoint_value_exact(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, path
