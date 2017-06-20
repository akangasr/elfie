import numpy as np
import scipy as sp

from elfi.methods.bo.acquisition import AcquisitionBase, LCBSC
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


class BetterLCBSC(LCBSC):

    def acquire(self, n_values, pending_locations=None, t=None):
        """ Replaced multivariate_normal with set of independent truncnorms.
            Noise cov should be a list now, not 2D array.
        """

        logger.debug('Acquiring {} values'.format(n_values))

        obj = lambda x: self.evaluate(x, t)
        grad_obj = lambda x: self.evaluate_grad(x, t)
        minloc, minval = minimize(obj, grad_obj, self.model.bounds, self.prior, self.n_inits, self.max_opt_iters)
        x = np.tile(minloc, (n_values, 1))

        # add some noise for more efficient exploration
        if max(self.noise_cov) <= 0:
            return x

        for i in range(n_values):
            for j in range(x.shape[1]):
                bounds = self.model.bounds[j]
                mean = x[i][j]
                try:
                    # maybe list
                    std = self.noise_cov[j]
                except:
                    # assume value
                    std = self.noise_cov
                a, b = (bounds[0] - mean) / std, (bounds[1] - mean) / std
                x[i][j] = sp.stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=1, random_state=self.random_state)

        return x


