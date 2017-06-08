import pytest
import numpy as np
from elfie.acquisition import GridAcquisition


class DummyModel(object):
    def __init__(self, bounds):
        self.input_dim = len(bounds)
        self.bounds = bounds
    def evaluate(self, v):
        raise NotImplementedError


def test_acquisition_works_with_simple_grid():
    model = DummyModel([[0, 1], [2,4]])
    tics = [[0,0.5,1], [2,4]]
    acq = GridAcquisition(model=model, tics=tics)
    assert np.array_equal(acq.acquire(n_values=1), [[0,2],])
    assert np.array_equal(acq.acquire(n_values=1), [[0.5,2],])
    assert np.array_equal(acq.acquire(n_values=1), [[1,2],])
    assert np.array_equal(acq.acquire(n_values=1), [[0,4],])
    assert np.array_equal(acq.acquire(n_values=1), [[0.5,4],])
    assert np.array_equal(acq.acquire(n_values=1), [[1,4],])

