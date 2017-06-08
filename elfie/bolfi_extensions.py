import numpy as np
import GPy

import elfi
from elfi.bo.gpy_regression import GPyRegression
from elfi.methods import BOLFI
from elfi.results import BolfiPosterior

import logging
logger = logging.getLogger(__name__)


class BolfiParams():
    """ Encapsulates BOLFI parameters
    """
    def __init__(self,
            bounds=((0,1),),
            n_samples=0,
            n_initial_evidence=0,
            sampling_type="BO",
            grid_tics=None,
            parallel_batches=1,
            batch_size=1,
            kernel_class=GPy.kern.RBF,
            noise_var=0.05,
            kernel_var=0.05,
            kernel_scale=0.1,
            gp_params_optimizer="scg",
            gp_params_max_opt_iters=50,
            gp_params_update_interval=0,
            acq_delta=0.1,
            acq_noise_cov=0.1,
            acq_opt_iterations=100):
        for k, v in locals().items():
            seattr(self, k, v)

    def to_dict(self):
        ret = self.__dict__.copy()
        ret["kernel_class"] = self.kernel_class.__name__
        return ret


class BolfiFactory():
    """ Constructs BOLFI inference objects from ElfiModel BolfiParams

    Parameters
    ----------
    model : ElfiModel
    params : BolfiParams
    """
    def __init__(self, model, params):
        self.model = model
        self.params = params
        if len(self.model.parameters) < 1:
            raise ValueError("Task must have at least one parameter.")
        if len(self.params.bounds) != len(self.model.parameters):
            raise ValueError("Number of ask parameters (was {}) must agree with bounds (was {})."\
                    .format(len(self.model.parameters), len(self.params.bounds)))

    def _gp(self):
        input_dim = len(self.params.bounds)
        kernel = self.params.kernel_class(input_dim=input_dim,
                                          variance=self.params.kernel_var,
                                          lengthscale=self.params.kernel_scale)
        gp = GPy.model.GPRegression(kernel=kernel,
                                    noise_var=self.params.noise_var)
        return GPyRegression(input_dim=input_dim,
                        bounds=self.params.bounds,
                        optimizer=self.params.gp_params_optimizer,
                        max_opt_iters=self.params.gp_params_max_opt_iters,
                        gp=gp)

    def _acquisition(self, gp):
        if self.params.sampling_type == "uniform":
            return UniformAcquisition(model=gp)
        if self.params.sampling_type == "grid":
            return GridAcquisition(tics=self.params.grid_tics,
                                   model=gp)
        if self.params.sampling_type == "BO":
            return LCBSC(delta=self.params.acq_delta,
                         max_opt_iter=self.params.acq_opt_iterations,
                         noise_cov=self.params.acq_noise_cov,
                         model=gp)
        logger.critical("Unknown sampling type '{}', aborting!".format(self.params.sampling_type))
        assert False

    def get(self):
        """ Returns new BOLFI inference object
        """
        gp = self._gp()
        acquisition = self._acquisition(gp)
        return BOLFI(model=self.model,
                     target="discrepancy",
                     target_model=gp,
                     acquisition_method=acquisition,
                     bounds=self.params.bounds,
                     initial_evidence=self.params.n_initial_evidence,
                     update_interval=self.params.gp_params_update_interval,
                     batch_size=self.params.batch_size,
                     max_parallel_batches=self.params.parallel_batches,
                     seed=self.params.seed)

