import numpy as np
import scipy as sp

from elfi.methods.bo.acquisition import AcquisitionBase
from elfi.methods.bo.utils import minimize, stochastic_optimization

import logging
logger = logging.getLogger(__name__)

""" Additional BO acquisition functions """



class GridAcquisition(AcquisitionBase):
    """Acquisition from a grid

    Parameters
    ----------
    tics : list of lists containing axis locations defining the grid
           example: [[1,2,3], [2,4]] -> grid (1,2), (1,4), (2,2), (2,4), (3,2), (3,4)
    """

    def __init__(self, tics, *args, **kwargs):
        self.tics = tics
        self.next_idx = 0
        super(GridAcquisition, self).__init__(*args, **kwargs)

    def acquire(self, n_values, pending_locations=None, t=0):
        ret = np.zeros((n_values, len(self.model.bounds)))
        for i in range(n_values):
            idx = self.next_idx
            self.next_idx += 1
            for j, tics in enumerate(self.tics):
                l = len(tics)
                mod = idx % l
                idx = int(idx / l)
                ret[i, j] = tics[mod]
        return ret


def soft_plus(x):
    return np.log(1.0 + np.exp(float(x)))

def phi(x, x2, L, M):
    mean, var = self.model.predict(x2, noiseless=True)
    z = 1.0 / np.sqrt(2.0 * var) * (L * abs(x - x2) - M + mean)
    return 0.5 * sp.special.erfc(-z)

class GPLCA(AcquisitionBase):
    """ Gaussian Process Lipschitz Constant Approximation (Gonzalez et al. 2016)
        http://proceedings.mlr.press/v51/gonzalez16a.pdf

    Parameters
    ----------
    a : AcquisitionBase
        core acquisition function
    g : callable
        x -> [0, inf), positive squashing function
    p : callable
        (x, x', L, M) -> [0, 1], penalization kernel
    """
    def __init__(self, a, g=soft_plus, p=phi):
        self.a = a
        self.g = g
        self.p = p
        super(GPLCA, self).__init__(model=self.a.model)

    def acquire(self, n_values, pending_locations=None, t=0):
        ret = np.zeros((n_values, len(self.model.bounds)))
        if pending_locations is None:
            pl = None
        else:
            pl = pending_locations[:]
        self.L = self.estimate_L()
        self.M = self.estimate_M()
        for i in range(n_values):
            r = self._acq(pl)
            print(r)
            r = np.atleast_2d(r)
            print(pl, r)
            if len(pl) == 0:
                pl = r
            else:
                pl = np.vstack((pl, r))
            ret[i] = r
        return ret

    def estimate_M(self):
        obj = lambda x: self.a.evaluate(x, t=0)
        loc, val = stochastic_optimization(obj, self.model.bounds, maxiter=1000, polish=True, seed=0)
        return val

    def estimate_L(self):
        grad_obj = lambda x: -np.abs(np.linalg.norm(self.a.evaluate_gradient(x, t=0)))
        print(self.model.bounds)
        loc, val = stochastic_optimization(grad_obj, self.model.bounds, maxiter=1000, polish=True, seed=0)
        return abs(val)

    def _acq(self, pending_locations, t=0):
        obj = [lambda x: self.g(self.a.evaluate(x, t))]
        if pending_locations is None:
            for p in pending_locations:
                obj.append(lambda x: self.phi(x, p, self.L, self.M) * obj[-1](x))
        loc, val = stochastic_optimization(obj[-1], self.model.bounds, maxiter=1000, polish=True, seed=0)
        return loc
