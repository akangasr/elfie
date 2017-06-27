import copy
import traceback
import numpy as np
import GPy

import matplotlib
from matplotlib import pyplot as pl

from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.parameter_inference import BOLFI, Rejection
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.bo.acquisition import UniformAcquisition, LCBSC
from elfi.store import OutputPool

from elfie.acquisition import GridAcquisition, GPLCA
from elfie.outputpool_extensions import SerializableOutputPool
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
            ARD=False,
            gp_params_optimizer="scg",
            gp_params_max_opt_iters=50,
            gp_params_update_interval=0,
            acq_delta=0.1,
            acq_noise_cov=0.1,
            acq_opt_iterations=100,
            seed=1,
            simulator_node_name="simulator",
            observed_node_name="summary",
            discrepancy_node_name="discrepancy",
            pool=None):
        d = {k: v for k, v in locals().items()}
        for k, v in d.items():
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
        if len(self.model.parameter_names) < 1:
            raise ValueError("Task must have at least one parameter.")
        if len(self.params.bounds) != len(self.model.parameter_names):
            raise ValueError("Number of ask parameters (was {}) must agree with bounds (was {})."\
                    .format(len(self.model.parameter_names), len(self.params.bounds)))

    def _gp(self):
        input_dim = len(self.params.bounds)
        kernel = self.params.kernel_class(input_dim=input_dim,
                                          variance=self.params.kernel_var,
                                          lengthscale=self.params.kernel_scale,
                                          ARD=self.params.ARD)
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
        if self.params.sampling_type == "bo":
            return GPLCA(LCBSC(delta=self.params.acq_delta,
                               max_opt_iters=self.params.acq_opt_iterations,
                               noise_cov=self.params.acq_noise_cov,
                               model=gp))
        logger.critical("Unknown sampling type '{}', aborting!".format(self.params.sampling_type))
        assert False

    def get(self):
        """ Returns new BolfiExperiment object
        """
        gp = self._gp()
        acquisition = self._acquisition(gp)
        if self.params.pool is None:
            pool = SerializableOutputPool((self.params.discrepancy_node_name, ) + tuple(self.model.parameter_names))
        else:
            pool = self.params.pool
        bolfi = BOLFI(model=self.model,
                      target_name=self.params.discrepancy_node_name,
                      target_model=gp,
                      acquisition_method=acquisition,
                      bounds=self.params.bounds,
                      initial_evidence=self.params.n_initial_evidence,
                      update_interval=self.params.gp_params_update_interval,
                      batch_size=self.params.batch_size,
                      max_parallel_batches=self.params.parallel_batches,
                      seed=self.params.seed,
                      pool=pool)
        return BolfiInferenceTask(bolfi,
                                  self.model,
                                  copy.copy(self.params),
                                  pool,
                                  self.model.parameter_names,
                                  self.params.simulator_node_name,
                                  self.params.observed_node_name,
                                  self.params.discrepancy_node_name)

    def to_dict(self):
        return {
                "model": "model",  # TODO
                "params": self.params.to_dict(),
                }


class BolfiInferenceTask():
    def __init__(self, bolfi, model, params, pool, paramnames, simuname, obsnodename, discname):
        self.bolfi = bolfi
        self.model = model
        self.params = params
        self.post = None
        self.samples = dict()
        self.ML = dict()
        self.ML_val = None
        self.MAP = dict()
        self.MAP_val = None
        self.pool = pool
        self.paramnames = paramnames
        self.simuname = simuname
        self.obsnodename = obsnodename
        self.discname = discname

    def do_sampling(self):
        """ Computes BO samples """
        self.bolfi.infer(self.params.n_samples)

    def compute_samples_and_MD(self):
        """ Extracts samples from pool and computes min discrepancy sample """
        self.samples = dict()
        self.MD = dict()
        self.MD_val = None
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
            if self.MD_val is None or y < self.MD_val:
                self.MD_val = y
                self.MD = x
            self.samples[idx] = {"X": x, "Y": y}
            idx += 1

    def compute_posterior(self):
        """ Constructs posterior """
        self.post = self.bolfi.extract_posterior()

    def compute_ML(self):
        """ Computes ML sample """
        self.ML = dict()
        self.ML_val = None
        for idx, sample in self.samples.items():
            x = np.array([sample["X"][k] for k in self.paramnames])
            logl = self.post._unnormalized_loglikelihood(x)
            if self.ML_val is None or logl > self.ML_val:
                self.ML_val = logl
                self.ML = sample["X"]

    def compute_MAP(self):
        """ Computes MAP sample """
        self.MAP = dict()
        self.MAP_val = None
        for idx, sample in self.samples.items():
            x = np.array([sample["X"][k] for k in self.paramnames])
            logp = self.post.logpdf(x)
            if self.MAP_val is None or logp > self.MAP_val:
                self.MAP_val = logp
                self.MAP = sample["X"]

    def simulate_data(self, with_values):
        return self.model.generate(with_values=with_values)[self.simuname]

    def compute_from_model(self, node_names, with_values_list, new_data=None):
        """ Compute values from model nodes, in parallel, using parameter values from with_values_list
            and optionally coditioned on on the new observation data.
        """
        if type(with_values_list) != list:
            logger.critical("With_values should be a list of elfi-compatible 'with_values'-dictionaries")
            assert False
        pool = SerializableOutputPool(node_names)
        for i, with_values in enumerate(with_values_list):
            pool.add_batch(with_values, batch_index=i)
        rej = Rejection(self.model, discrepancy_name=self.model[self.discname], pool=pool, batch_size=1)
        if new_data is not None:
            old_data = self.model.computation_context.observed[self.obsnodename]
            self.model.computation_context.observed[self.obsnodename] = new_data
        rej.sample(len(with_values_list), quantile=1)
        if new_data is not None:
            self.model.computation_context.observed[self.obsnodename] = old_data
        ret = list()
        for i, values in enumerate(with_values_list):
            r = dict()
            batch = pool.get_batch(i)
            for n in node_names:
                r[n] = batch[n][0]  # assume length 1
            logger.info("Computed values of nodes {} with values {}".format(node_names, values))
            ret.append(r)
        return ret


    def plot_post(self, pdf, figsize):
        if self.post is None:
            return

        logger.debug("Plotting GP model")
        fig = pl.figure(figsize=figsize)
        try:
            self.post.model._gp.plot()
        except Exception as e:
            fig.text(0.02, 0.02, "Was not able to plot GP model: {}".format(e))
            tb = traceback.format_exc()
            logger.critical(tb)
        pdf.savefig()
        pl.close()

        bounds = list()
        names = list()
        for k in sorted(self.params.bounds.keys()):
            bounds.append(self.params.bounds[k])
            names.append(k)

        for fname, fun in [("GP mean", lambda x: self.post.model.predict(x)[0][:,0]),
                           ("GP std", lambda x: self.post.model.predict(x)[1][:,0]),
                           ("Prior density", self.post.prior.pdf),
                           ("Unnormalized likelihood", self.post._unnormalized_likelihood),
                           ("Unnormalized posterior", self.post.pdf)]:
            if len(self.paramnames) == 1:
                logger.debug("Plotting {}".format(fname))
                fig = pl.figure(figsize=figsize)
                try:
                    pl.xlabel(names[0], fontsize=20)
                    pl.ylabel(fname, fontsize=20)
                    locs = np.linspace(bounds[0][0], bounds[0][1], 100)
                    vals = [fun(np.array(l)) for l in locs]
                    pl.plot(locs, vals)
                    pl.show()
                except Exception as e:
                    fig.text(0.02, 0.02, "Was not able to plot {}: {}".format(fname, e))
                    tb = traceback.format_exc()
                    logger.critical(tb)
                pdf.savefig()
                pl.close()

            if len(self.paramnames) == 2:
                logger.debug("Plotting {}".format(fname))
                fig, ax = pl.subplots(1,1,figsize=figsize)
                try:
                    ax.set_title(fname)
                    ax.set_xlabel(names[0], fontsize=20)
                    ax.set_ylabel(names[1], fontsize=20)
                    vals = eval_2d_mesh(bounds[0][0],
                                        bounds[1][0],
                                        bounds[0][1],
                                        bounds[1][1],
                                        100, 100, fun)
                    CS = ax.contourf(vals[0], vals[1], vals[2] / max(np.max(vals[2]), 1e-5), cmap='hot')
                    cbar_ax = fig.add_axes([0.91, 0.2, 0.03, 0.65]) # left, bottom, width, height
                    fig.colorbar(CS, cax=cbar_ax)
                    pl.show()
                except Exception as e:
                    fig.text(0.02, 0.02, "Was not able to plot {}: {}".format(fname, e))
                    tb = traceback.format_exc()
                    logger.critical(tb)
                pdf.savefig()
                pl.close()

