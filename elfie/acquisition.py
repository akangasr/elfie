from functools import partial
import numpy as np
import scipy as sp

from elfi.methods.bo.acquisition import AcquisitionBase
from elfi.methods.bo.utils import minimize

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
    """ As in Gonzalez et al. 2016 """
    return float(np.log(1.0 + np.exp(float(x))))

def phi(x, x2, L, M, mean, var):
    """ As in Gonzalez et al. 2016 except for dimension-wise L """
    Ldiag = np.diag(L)
    diff = x - x2
    mult = np.dot(Ldiag, diff)
    norm = np.linalg.norm(mult)
    z = 1.0 / np.sqrt(2.0 * var) * (norm - M + mean)
    ret = 0.5 * sp.special.erfc(-z)
    return float(ret)

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

    def acquire(self, n_values, pending_locations=None, t=None):
        ret = np.zeros((n_values, len(self.model.bounds)))
        if pending_locations is None:
            pl = None
        else:
            pl = pending_locations[:]
        self.L = self.estimate_L(t)
        self.M = self.estimate_M(t)
        for i in range(n_values):
            r = self._acq(pl, t)
            ret[i] = r
            r2 = np.atleast_2d(r)
            if len(pl) == 0:
                pl = r2
            else:
                pl = np.vstack((pl, r2))
        return ret

    def estimate_M(self, t):
        """ Estimate function maximum value """
        obj = lambda x: self.a.evaluate(x, t=t)  # minimization
        loc, val = minimize(obj, self.model.bounds, random_state=self.a.random_state)
        return -1.0 * float(val)  # maximization

    def estimate_L(self, t):
        """ Return a list of acq surface gradient absolute value max for each dimension """
        L = list()
        for i in range(len(self.model.bounds)):
            grad_obj = lambda x: -np.abs(float(self.a.evaluate_gradient(x, t=t)[0][i]))  # abs max
            loc, val = minimize(grad_obj, self.model.bounds, random_state=self.a.random_state)
            L.append(abs(val))
        return L

    def _acq(self, pending_locations, t):
        phis = []
        if pending_locations is not None:
            for pl in pending_locations:
                print("Pending: {}".format(pl))
                mean, var = self.model.predict(pl, noiseless=True)
                mean = -1.0 * float(mean)  # maximization
                var = float(var)
                phis.append((pl, mean, var))

        def trans(x, t):
            # negation as the GPLCA formulation is for maximization
            return self.g(-1.0 * float(self.a.evaluate(x, t)))

        def pend(x):
            val = 1.0
            for pl, mean, var in phis:
                val *= self.p(x, pl, self.L, self.M, mean, var)
            return val

        def obj(x, t):
            # negation as we use a minimizer to solve a maximization problem
            return -1.0 * trans(x, t) * pend(x)

        loc, val = minimize(partial(obj, t=t), self.model.bounds, random_state=self.a.random_state)

        if True:
            self._debug_print("GP mean", lambda x: self.a.model.predict(x, noiseless=True)[0])
            self._debug_print("GP std", lambda x: self.a.model.predict(x, noiseless=True)[1])
            self._debug_print("Original surface", partial(self.a.evaluate, t=t))
            self._debug_print("Transformed surface", partial(trans, t=t))
            self._debug_print("Pending points modifier", pend)
            self._debug_print("Final surface (M={:.2f}, L={})".format(self.M,
                                                                      "".join(["{:.2f} ".format(l) for l in self.L])),
                              partial(obj, t=t), loc=loc)

        return loc


    def _debug_print(self, name, obj, loc=None):
        """debug printout"""
        tics = 20
        maxv = float("-inf")
        minv = float("inf")
        mind = float("inf")
        minl = None
        vals = list()
        xtics = np.linspace(*self.model.bounds[0], tics)
        ytics = np.linspace(*self.model.bounds[1], tics)
        for y in ytics:
            vals.append(list())
            for x in xtics:
                l = np.array([x, y])
                v = float(obj(l))
                vals[-1].append(v)
                maxv = max(maxv, v)
                minv = min(minv, v)
                if loc is not None:
                    d = np.linalg.norm(l-loc)
                    if d < mind:
                        minl = l
                        mind = d

        print("{} :".format(name))
        for y in reversed(range(tics)):
            line = ["|"]
            for x in range(tics):
                l = np.array([xtics[x], ytics[y]])
                if maxv == minv:
                    v = 1.0
                else:
                    v = (float(vals[y][x])-minv)/(maxv-minv)
                if loc is not None and np.allclose(minl, l):
                    line.append("@@")
                elif v > 0.95:
                    line.append("XX")
                elif v > 0.8:
                    line.append("xx")
                elif v > 0.6:
                    line.append("::")
                elif v > 0.4:
                    line.append(":.")
                elif v > 0.2:
                    line.append("..")
                elif v > 0.05:
                    line.append(". ")
                else:
                    line.append("  ")
            line.append("|")
            print("".join(line))
        if loc is not None:
            print("Acquired {}".format(loc))

