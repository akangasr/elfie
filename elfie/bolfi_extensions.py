import copy
import numpy as np
import GPy

from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.methods import BOLFI
from elfi.methods.results import BolfiPosterior
from elfi.methods.bo.acquisition import UniformAcquisition, LCBSC
from elfi.store import OutputPool

from elfie.inference import InferenceTask
from elfie.acquisition import GridAcquisition
from elfie.utils import eval_2d_mesh

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
            acq_opt_iterations=100,
            seed=1,
            discrepancy_node_name="discrepancy"):
        for k, v in locals().items():
            setattr(self, k, v)

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
        return GPyRegression(input_dim=input_dim,
                        bounds=self.params.bounds,
                        optimizer=self.params.gp_params_optimizer,
                        max_opt_iters=self.params.gp_params_max_opt_iters,
                        kernel=kernel,
                        noise_var=self.params.noise_var)

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
        """ Returns new BolfiExperiment object
        """
        gp = self._gp()
        acquisition = self._acquisition(gp)
        pool = OutputPool((self.params.discrepancy_node_name, ) + tuple(self.model.parameters))
        bolfi = BOLFI(model=self.model,
                      target=self.params.discrepancy_node_name,
                      target_model=gp,
                      acquisition_method=acquisition,
                      bounds=self.params.bounds,
                      initial_evidence=self.params.n_initial_evidence,
                      update_interval=self.params.gp_params_update_interval,
                      batch_size=self.params.batch_size,
                      max_parallel_batches=self.params.parallel_batches,
                      seed=self.params.seed,
                      pool=pool)
        return BolfiInferenceTask(bolfi, copy.copy(self.params), pool, self.model.parameters, self.params.discrepancy_node_name)

    def to_dict(self):
        return {
                "model": "model",  # TODO
                "params": self.params.to_dict(),
                }


class BolfiInferenceTask(InferenceTask):
    def __init__(self, bolfi, params, pool, paramnames, discname):
        self.bolfi = bolfi
        self.params = params
        self.post = None
        self.samples = dict()
        self.ML = dict()
        self.ML_val = None
        self.MAP = dict()
        self.MAP_val = None
        self.order = None
        self.pool = pool
        self.paramnames = paramnames
        self.discname = discname

    def do_sampling(self):
        """ Computes BO samples """
        self.bolfi.infer(self.params.n_samples)

    def compute_samples_and_ML(self):
        """ Extracts samples from pool and computes ML sample """
        self.samples = dict()
        self.ML = dict()
        self.ML_val = None
        idx = 0
        while True:
            X = self.pool.get_batch(idx, self.paramnames)
            Y = self.pool.get_batch(idx, [self.discname])
            if len(X) == 0 or len(Y) == 0:
                break
            y = float(Y[self.discname])
            x = dict()
            for k, v in X.items():
                x[k] = float(v)
            if self.ML_val is None or y < self.ML_val:
                self.ML_val = y
                self.ML = x
            self.samples[idx] = {"X": x, "Y": y}
            idx += 1

    def compute_posterior(self):
        """ Constructs posterior """
        self.post = self.bolfi.infer_posterior()

    def compute_MAP(self):
        """ Computes MAP sample """
        self.MAP = dict()
        self.MAP_val = None
        self.order = None
        for idx, sample in self.samples.items():
            print(sample)
            x = np.array([sample["X"][k] for k in self.paramnames])
            lp = self.post.logpdf(x)
            if self.MAP_val is None or lp > self.ML_val:
                self.MAP_val = lp
                self.MAP = sample["X"]

    def plot(self, plotter):
        if self.result.model._gp is None:
            plotter.plot_text_page("No model to plot")
            return
        fig = pl.figure(figsize=plotter.figsize)
        try:
            self.post.model._gp.plot()
        except:
            fig.text(0.02, 0.02, "Was not able to plot GP model")
        plotter.pdf.savefig()
        pl.close()

        if len(self.order) == 2:
            fig, ax = pl.subplots(1,1,figsize=plotter.figsize)
            try:
                ax.set_xlabel(self.order[0], fontsize=20)
                ax.set_ylabel(self.order[1], fontsize=20)
                vals = eval_2d_mesh(bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1], 100, 100, self.post.pdf)
                CS = ax.contourf(vals[0], vals[1], vals[2] / np.max(vals[2]), cmap='hot')
                cbar_ax = fig.add_axes([0.95, 0.2, 0.03, 0.65]) # left, bottom, width, height
                fig.colorbar(CS, cax=cbar_ax)
                pl.show()
            except:
                fig.text(0.02, 0.02, "Was not able to plot posterior")
            plotter.pdf.savefig()
            pl.close()





