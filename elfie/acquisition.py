import numpy as np

from elfi.methods.bo.acquisition import AcquisitionBase

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

