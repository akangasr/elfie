import copy
import traceback
import numpy as np
import scipy as sp
import GPy

import matplotlib
from matplotlib import pyplot as pl

from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.parameter_inference import BOLFI, Rejection
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.bo.acquisition import UniformAcquisition, LCBSC
from elfi.store import OutputPool
from elfi.methods.bo.utils import minimize

from elfie.acquisition import GridAcquisition, GPLCA
from elfie.outputpool_extensions import SerializableOutputPool
from elfie.utils import eval_2d_mesh

import logging
logger = logging.getLogger(__name__)


class BolfiParams():
    """ Encapsulates BOLFI parameters
    """
    def __init__(self,
            bounds=dict(),
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
            kernel_prior=None,
            L=None,
            ARD=False,
            gp_params_optimizer="scg",
            gp_params_max_opt_iters=50,
            gp_params_update_interval=0,
            noisy_posterior=False,
            model_scale=1.0,
            abc_threshold_delta=0,
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
        if self.params.kernel_prior is not None:
            kernel.lengthscale.set_prior(
                GPy.priors.Gamma.from_EV(self.params.kernel_prior["scale_E"], self.params.kernel_prior["var_E"]), warning=False)
            kernel.variance.set_prior(
                GPy.priors.Gamma.from_EV(self.params.kernel_prior["scale_V"], self.params.kernel_prior["var_V"]), warning=False)
        return GPyRegression(parameter_names=self.model.parameter_names,
                        bounds=self.params.bounds,
                        optimizer=self.params.gp_params_optimizer,
                        max_opt_iters=self.params.gp_params_max_opt_iters,
                        kernel=kernel,
                        noise_var=self.params.noise_var)

    def _acquisition(self, gp):
        if self.params.sampling_type == "uniform":
            return UniformAcquisition(model=gp)
#        if self.params.sampling_type == "grid":
#            return GridAcquisition(tics=self.params.grid_tics,
#                                   model=gp)
        if self.params.sampling_type == "bo":
            return GPLCA(LCBSC(delta=self.params.acq_delta,
                               max_opt_iters=self.params.acq_opt_iterations,
                               noise_var=self.params.acq_noise_cov,
                               model=gp),
                         L=self.params.L)
        logger.critical("Unknown sampling type '{}', aborting!".format(self.params.sampling_type))
        assert False

    def get(self):
        """ Returns new BolfiExperiment object
        """
        if self.params.sampling_type in ["grid"]:
            return BolfiInferenceTask(None,
                                      self.model,
                                      copy.copy(self.params),
                                      None,
                                      self.model.parameter_names,
                                      self.params.simulator_node_name,
                                      self.params.observed_node_name,
                                      self.params.discrepancy_node_name,
                                      None,
                                      self.params.grid_tics)
        if self.params.sampling_type in ["lbfgsb", "neldermead"]:
            return BolfiInferenceTask(None,
                                      self.model,
                                      copy.copy(self.params),
                                      None,
                                      self.model.parameter_names,
                                      self.params.simulator_node_name,
                                      self.params.observed_node_name,
                                      self.params.discrepancy_node_name,
                                      self.params.sampling_type)
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


def _compute(model, node_names, with_values_list, discname, obsnodename, new_data=None):
    """ Compute values from model nodes, in parallel, using parameter values from with_values_list
        and optionally coditioned on on the new observation data.
    """
    if type(with_values_list) != list:
        logger.critical("With_values should be a list of elfi-compatible 'with_values'-dictionaries")
        assert False
    if len(with_values_list) == 0:
        return None
    pool_nodes = node_names
    for k in with_values_list[0].keys():
        if k not in pool_nodes:
            pool_nodes.append(k)
    pool = SerializableOutputPool(pool_nodes)
    for i, with_values in enumerate(with_values_list):
        wv = {k: np.atleast_2d(v) for k, v in with_values.items()}
        pool.add_batch(wv, batch_index=i)
    rej = Rejection(model, discrepancy_name=discname, pool=pool, batch_size=1)
    if new_data is not None:
        old_data = model.computation_context.observed[self.obsnodename]
        model.computation_context.observed[obsnodename] = new_data
    rej.sample(len(with_values_list), quantile=1)
    if new_data is not None:
        model.computation_context.observed[obsnodename] = old_data
    ret = list()
    for i, values in enumerate(with_values_list):
        r = dict()
        batch = pool.get_batch(i)
        for n in node_names:
            while len(batch[n].shape) > 1 and batch[n].shape[0] == 1:
                batch[n] = batch[n][0]  # unwrap numpy arrays
            r[n] = batch[n].tolist()
        #logger.info("Computed values of nodes {} with values {}".format(node_names, values))
        #logger.info("Result was {}".format(r))
        ret.append(r)
    return ret


class BolfiInferenceTask():
    def __init__(self, bolfi, model, params, pool, paramnames, simuname, obsnodename, discname, opt=None, grid=None):
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
        self.opt = opt
        self.grid = grid

    def do_sampling(self):
        """ Computes BO samples """
        if self.opt is not None:
            self._optimize()
            return
        if self.grid is not None:
            self._grid_search()
            return
        self.bolfi.infer(self.params.n_samples)
        try:
            self.kernel_params = {p.name: p.values.tolist() for p in self.bolfi.target_model._gp.kern.parameters}
            self.kernel_params["noise_variance"] = float(self.bolfi.target_model._gp.Gaussian_noise.variance)
        except Exception as e:
            self.kernel_params = None
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

    def _optimize(self):
        class _Target():
            def __init__(self, calls, model, bounds, parameters, discname, obsnodename):
                self.calls = calls
                self.samples = list()
                self.md = model
                self.bd = bounds
                self.pn = parameters
                self.dn = discname
                self.on = obsnodename
            def __call__(self, x):
                if self.calls < 1:
                    raise ValueError("No calls left")
                wv = {p: v for p, v in zip(self.pn, x)}
                logger.info("Evaluating at {}, {} calls left".format(wv, self.calls))
                for xi, bi in zip(x, self.bd):
                    if xi < bi[0] or xi > bi[1]:
                        logger.info("Outside bounds, returning large discrepancy")
                        return 1e10
                self.calls -= 1
                ret = _compute(self.md, [self.dn], [wv], self.dn, self.on)
                logger.info("Result: {}".format(ret))
                ret = float(ret[0][self.dn][0])
                self.samples.append((wv, ret))
                return ret

        self.MD = dict()
        self.MD_val = None
        bounds = [self.params.bounds[p] for p in self.paramnames]
        target = _Target(int(self.params.n_samples),
                         self.model,
                         bounds,
                         self.paramnames,
                         self.discname,
                         self.obsnodename)
        logger.info("Starting optimization with bounds {}, total {} function calls".format(bounds, target.calls))
        locs = list()
        vals = list()
        samples = list()
        while target.calls > 0:
            logger.info("Optimization starting, {} function calls left".format(target.calls))
            x0 = [np.random.uniform(*b) for b in bounds]  # TODO: randomstate?
            if self.opt == "lbfgsb":
                method = "L-BFGS-B"
                options = {"eps": 1e-3}
                logger.info("x0 {}".format(x0))
            if self.opt == "neldermead":
                method = "Nelder-Mead"
                initial_simplex = np.array([[np.random.uniform(*b) for b in bounds] for i in range(len(bounds)+1)])  # TODO: randomstate?
                logger.info("initial simplex: {}".format(initial_simplex))
                options = {"xatol": 0.005,
                           "fatol": 0.005,
                           "initial_simplex": initial_simplex,
                           "disp": True}
                logger.info("method: {}, bounds: {}".format(method, bounds))
            try:
                r = sp.optimize.minimize(fun=target,
                                         x0=x0,
                                         method=method,
                                         bounds=bounds,
                                         options=options)
                loc = {p: float(v) for p, v in zip(self.paramnames, r.x)}
                val = float(r.fun)
                logger.info("Optimum at {} (value {}), {} function calls remaining".format(loc, val, target.calls))
                locs.append(loc)
                vals.append(val)
            except ValueError as e:
                logger.info(e)
                if len(locs) == 0:
                    # if optimization ends before the first convergence, then
                    # extract sampled location with smallest observed value
                    minloc = None
                    minval = None
                    for loc, val in target.samples:
                        if minval is None or minval > val:
                            minloc = loc
                            minval = val
                    logger.info("Smallest value observed at {} (value {}), {} function calls remaining".format(minloc, minval, target.calls))
                    locs.append(minloc)
                    vals.append(minval)
            samples.extend(target.samples)
            target.samples = list()
        self.samples = dict()
        idx = 0
        for loc, val in samples:
            self.samples[idx] = {"X": loc, "Y": val}
            idx += 1
        for loc, val in zip(locs, vals):
            if self.MD_val is None or self.MD_val > val:
                self.MD = loc
                self.MD_val = val
        logger.info("MD sample at {}, value {}".format(self.MD, self.MD_val))

    def _grid_search(self):
        logger.info("Starting grid search")
        logger.info("Grid: {}".format(self.grid))
        self.MD = dict()
        self.MD_val = None
        self.samples = dict()
        wvs = list()
        for idx in range(int(self.params.n_samples)):
            loc = list()
            for j, tics in enumerate(self.grid):
                l = len(tics)
                mod = idx % l
                idx = int(idx / l)
                loc.append(tics[mod])
            wvs.append({p: v for p, v in zip(self.paramnames, loc)})
        ret = _compute(self.model, [self.discname], wvs, self.discname, self.obsnodename)
        for result, loc in zip(ret, wvs):
            val = float(result[self.discname][0])
            self.samples[idx] = {"X": loc, "Y": val}
            logger.info("Observed {} at {}".format(val, loc))
            if self.MD_val is None or val < self.MD_val:
                self.MD = loc
                self.MD_val = val
        logger.info("MD sample at {}, value {}".format(self.MD, self.MD_val))


    def compute_from_model(self, node_names, with_values_list, new_data=None):
        return _compute(self.model, node_names, with_values_list, self.discname, self.obsnodename, new_data=None)

    def compute_posterior(self):
        """ Constructs posterior """
        #bounds = list()
        #for k in sorted(self.params.bounds.keys()):
        #    bounds.append(self.params.bounds[k])
        #loc, val = minimize(lambda x: self.bolfi.target_model.predict(x, noiseless=True)[0], bounds)
        #mean, std = self.bolfi.target_model.predict(loc, noiseless=True)
        minval = None
        for idx, sample in self.samples.items():
            x = np.array([sample["X"][k] for k in self.paramnames])
            mean, std = self.bolfi.target_model.predict(x, noiseless=True)
            mean = float(mean)
            if minval is None or mean < minval:
                minval = mean
        self.threshold = minval + self.params.abc_threshold_delta
        self.post = self.bolfi.extract_posterior(threshold=self.threshold)

    def sample_from_likelihood(self):
        logger.info("MCMC sampling likelihood..")
        if self.params.noisy_posterior:
            fun = lambda x: self.params.model_scale * self.post.model._gp.posterior_samples(np.array(x), size=1, full_cov=True)
        else:
            fun = self.post._unnormalized_loglikelihood
        self.lik_samples, self.lik_acc_prop = MCMC_sample(fun, self.post.model.bounds, [0.5*(b[1]+b[0]) for b in self.post.model.bounds], noisy=self.params.noisy_posterior)
        logger.info("Acceptance probability {:.4f}".format(self.lik_acc_prop))
        self.LM = {k: v for k, v in zip(self.paramnames, np.mean(self.lik_samples, axis=0))}

    def sample_from_posterior(self):
        logger.info("MCMC sampling posterior..")
        if self.params.noisy_posterior:
            print("Posterior sampling for noisy posterior not implemented")
            raise NotImplementedError()
        else:
            fun = self.post.logpdf
        self.post_samples, self.post_acc_prop = MCMC_sample(fun, self.post.model.bounds, [0.5*(b[1]+b[0]) for b in self.post.model.bounds], noisy=self.params.noisy_posterior)
        logger.info("Acceptance probability {:.4f}".format(self.post_acc_prop))
        self.PM = {k: v for k, v in zip(self.paramnames, np.mean(self.post_samples, axis=0))}

    def compute_MED(self):
        """ Computes minimum expected discrepancy sample """
        self.MED = dict()
        self.MED_val = None
        for idx, sample in self.samples.items():
            x = np.array([sample["X"][k] for k in self.paramnames])
            mean, std = self.post.model.predict(x, noiseless=True)
            if self.MED_val is None or mean < self.MED_val:
                self.MED_val = mean
                self.MED = sample["X"]

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


    def plot_post(self, pdf, figsize):
        bounds = list()
        names = list()
        for k in sorted(self.params.bounds.keys()):
            bounds.append(self.params.bounds[k])
            names.append(k)

        if self.post is None:
            if self.params.grid_tics is not None:
                logger.debug("Plotting Grid")
                multiplot(plot_grid, loc=self.MD, title="Grid", pdf=pdf, figsize=figsize, bounds=bounds, samples=self.samples, grid_tics=self.params.grid_tics, names=names)
            return

        if hasattr(self, "LM"):
            loc = self.LM
        elif hasattr(self, "MAP"):
            loc = self.MAP
        elif hasattr(self, "ML"):
            loc = self.ML
        elif hasattr(self, "MED"):
            loc = self.MED

        logger.debug("Plotting GP model")
        multiplot(self.post.model._gp.plot, loc, pdf=pdf)

        ret = dict()
        logger.debug("Plotting GP model residuals")
        ret["residuals"] = plot_residuals(self.samples, self.post.model, figsize, pdf)

        for fname, fun in [("GP mean", lambda x: self.post.model.predict(x)[0][:,0]),
                           ("GP std", lambda x: self.post.model.predict(x)[1][:,0]),
                           ("Prior density", self.post.prior.pdf),
                           ("Unnormalized likelihood", self.post._unnormalized_likelihood),
                           ("Unnormalized posterior", self.post.pdf)]:
            n_tics = 150
            logger.debug("Plotting {}".format(fname))
            multiplot(plot_1d2d, loc, title=fname, pdf=pdf, figsize=figsize, fun=fun, bounds=bounds, n_tics=n_tics)


        if len(bounds) == 2 and hasattr(self, "LM"):
            # TODO: plot slices?
            try:
                if self.params.noisy_posterior:
                    fun1 = lambda x: self.params.model_scale * float(self.post.model.predict(x)[0])
                    fun2 = sp.stats.gaussian_kde(np.array(self.lik_samples).T, bw_method=0.1)
                else:
                    fun1 = self.post._unnormalized_loglikelihood
                    fun2 = self.post._unnormalized_likelihood
                logger.debug("Plotting Unnormalized log-likelihood")
                plot_2d_gridworld_likelihood(fun1, self.post.model.bounds, figsize, n_tics=n_tics, title="Unnormalized log-likelihood and samples", samples=self.lik_samples)
                if pdf is not None:
                    pdf.savefig()
                    pl.close()
                logger.debug("Plotting Unnormalized likelihood")
                plot_2d_gridworld_likelihood(fun2, self.post.model.bounds, figsize, n_tics=n_tics, title="Unnormalized likelihood", samples=self.lik_samples, plotsamples=False)
                if pdf is not None:
                    pdf.savefig()
                    pl.close()
            except Exception as e:
                tb = traceback.format_exc()
                logger.critical(tb)

        return ret

def find_sample(loc, samples, names):
    for s in samples.values():
        found = True
        for j in range(len(loc)):
            if abs(s["X"][names[j]] - loc[j]) > 1e-7:
                found = False
                break
        if found is True:
            return s
    return None

def plot_grid(samples, grid_tics, bounds, names, figsize, fixed_inputs=list(), pdf=None):
    try:
        idx = set(range(len(bounds))) - set([f[0] for f in fixed_inputs])
        if len(idx) == 1:
            idx = min(idx)
            loc = [None] * len(bounds)
            for i, v in fixed_inputs:
                loc[i] = v
            x = grid_tics[idx]
            y = list()
            for xi in x:
                loc[idx] = xi
                s = find_sample(loc, samples, names)
                y.append(float(s["Y"]))
            fig = pl.figure(figsize=figsize)
            pl.plot(x, y)
            pl.show()
        elif len(idx) == 2:
            idx1 = min(idx)
            idx2 = max(idx)
            loc = [None] * len(bounds)
            for i, v in fixed_inputs:
                loc[i] = v
            x = grid_tics[idx1]
            y = grid_tics[idx2]
            img = np.empty((len(y), len(x)))
            for i, xi in enumerate(x):
                for j, yi in enumerate(y):
                    loc[idx1] = xi
                    loc[idx2] = yi
                    s = find_sample(loc, samples, names)
                    img[len(y)-j-1][i] = float(s["Y"])
            mx = float(np.max(np.max(img)))
            mn = float(np.min(np.min(img)))
            img_s = (img - mn) / max(mx - mn, 1e-10)
            fig = pl.figure(figsize=figsize)
            im = pl.imshow(img_s, cmap='hot',
                    extent=[bounds[idx1][0], bounds[idx1][1], bounds[idx2][0], bounds[idx2][1]],
                    aspect=(bounds[idx1][1]-bounds[idx1][0])/(bounds[idx2][1]-bounds[idx2][0]))
            pl.show()
        else:
            print("dimension mismatch, bounds={}, fixed_inputs={}".format(len(bounds), fixed_inputs))
            return
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical(tb)
    if pdf is not None:
        pdf.savefig()
        pl.close()

def multiplot(printer, loc, title=None, pdf=None, **kwargs):
    try:
        n_params = len(loc)
        param_names = sorted(loc.keys())
        vals = [float(loc[k]) for k in param_names]
        for i in range(n_params):
            for j in range(i, n_params):
                fixed_inputs = list()
                for k in range(n_params):
                    if k != i and k != j:
                        fixed_inputs.append((k, vals[k]))
                printer(fixed_inputs=fixed_inputs, **kwargs)
                pl.xlabel(param_names[i])
                if i != j:
                    pl.ylabel(param_names[j])
                if title is not None:
                    pl.title(title)
                if pdf is not None:
                    pdf.savefig()
                    pl.close()
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical(tb)
    if pdf is not None:
        pl.close()

def plot_1d2d(fun, bounds, n_tics, figsize, fixed_inputs=list(), pdf=None, cmap="hot", return_fig=False):
    try:
        idx = set(range(len(bounds))) - set([f[0] for f in fixed_inputs])
        if len(idx) == 1:
            idx = min(idx)
            loc = [None] * len(bounds)
            for i, v in fixed_inputs:
                loc[i] = v
            x = np.linspace(bounds[idx][0], bounds[idx][1], n_tics)
            y = list()
            for xi in x:
                loc[idx] = xi
                y.append(float(fun(loc)))
            fig = pl.figure(figsize=figsize)
            pl.plot(x, y)
            pl.show()
        elif len(idx) == 2:
            idx1 = min(idx)
            idx2 = max(idx)
            loc = [None] * len(bounds)
            for i, v in fixed_inputs:
                loc[i] = v
            x = np.linspace(bounds[idx1][0], bounds[idx1][1], n_tics)
            y = np.linspace(bounds[idx2][0], bounds[idx2][1], n_tics)
            img = np.empty((n_tics, n_tics))
            for i, xi in enumerate(x):
                for j, yi in enumerate(y):
                    loc[idx1] = xi
                    loc[idx2] = yi
                    img[n_tics-j-1][i] = float(fun(loc))
            mx = float(np.max(np.max(img)))
            mn = float(np.min(np.min(img)))
            img_s = (img - mn) / max(mx - mn, 1e-10)
            fig = pl.figure(figsize=figsize)
            im = pl.imshow(img_s, cmap=cmap,
                    extent=[bounds[idx1][0], bounds[idx1][1], bounds[idx2][0], bounds[idx2][1]],
                    aspect=(bounds[idx1][1]-bounds[idx1][0])/(bounds[idx2][1]-bounds[idx2][0]))
            #cbar_ax = fig.add_axes([0.91, 0.2, 0.03, 0.65]) # left, bottom, width, height
            #fig.colorbar(im, cax=cbar_ax)
            if return_fig is True:
                return fig
            pl.show()
        else:
            print("dimension mismatch, bounds={}, fixed_inputs={}".format(len(bounds), fixed_inputs))
            return
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical(tb)
    if pdf is not None:
        pdf.savefig()
        pl.close()

def get_residuals(samples, gp):
    errs = list()
    aerrs = list()
    stds = list()
    for sample in samples.values():
        x = list()
        for name in sorted(sample["X"].keys()):
            x.append(float(sample["X"][name]))
        val = float(sample["Y"])
        mean, std = gp.predict(x, noiseless=True)
        err = float(mean - val)
        aerr = abs(err)
        errs.append(err)
        aerrs.append(aerr)
        stds.append(float(std))
    return errs, aerrs, stds

def plot_residuals(samples, gp, figsize, pdf=None):
    fig, (ax1, ax2) = pl.subplots(2,1,figsize=figsize)
    ret = dict()
    try:
        errs, aerrs, stds = get_residuals(samples, gp)
        minv = -max(aerrs)
        maxv = max(aerrs)
        x1 = np.linspace(minv, maxv, 100)
        ax1.hist(np.array(errs), 30, (minv-1e-5, maxv+1e-5), normed=True)
        ax1.set_title("Residual errors")
        x2 = np.linspace(0, maxv, 100)
        ax2.hist(np.array(aerrs), 30, (-1e-5, maxv+1e-5), normed=True)
        ax2.set_title("Residual absolute errors")
        if ret is not None:
            ret = {"errs": errs, "stds": stds}
    except Exception as e:
        fig.text(0.02, 0.02, "Was not able to plot GP model: {}".format(e))
        tb = traceback.format_exc()
        logger.critical(tb)
    if pdf is not None:
        pdf.savefig()
        pl.close()
    return ret


def MCMC_sample(logl, bounds, init, noisy=False, n_samples=20000, burnout=1000, thinning=10, propstd=0.1):
    loc1 = init
    if noisy is False:
        logl1 = logl(loc1)
    samples = list()
    prop = lambda : np.random.normal(loc=[0.0]*len(bounds), scale=[propstd]*len(bounds))
    n_acc = 0
    n_rej = 0
    for i in range(burnout + n_samples*thinning):
        while True:
            loc2 = loc1 + prop()
            if not out_of_bounds(loc2, bounds):
                break
        if noisy is True:
            logl1, logl2 = logl([loc1, loc2])  # draw from posterior
            ratio = min(1, np.exp(logl2 - logl1))
        else:
            logl2 = logl(loc2)
            ratio = min(1, np.exp(logl2 - logl1))
        if ratio > np.random.uniform():
            n_acc += 1
            loc1 = loc2
            if noisy is False:
                logl1 = logl2
        else:
            n_rej += 1
        if i >= burnout and i % thinning == 0:
            samples.append(loc1)
    return samples, float(n_acc)/(n_acc+n_rej)

def out_of_bounds(loc, bounds):
    for v, b in zip(loc, bounds):
        if v < b[0] or v > b[1]:
            return True
    return False

def plot_2d_gridworld_likelihood(fun, bounds, figsize, n_tics, ML=None, title="", samples=list(), plotsamples=True):
    plot_1d2d(fun, bounds, n_tics, figsize, cmap="hot", return_fig=True)
    pl.xlabel("Feature 1")
    pl.ylabel("Feature 2")
    pl.title(title)
    if plotsamples is True and len(samples) > 0:
        pl.scatter([s[0] for s in samples], [s[1] for s in samples], alpha=0.25,
                   color="black", marker=",", s=1, label="Samples")
    if len(samples) > 1:
        pl.scatter(*np.mean(samples, axis=0), color="green", marker="s", s=80,
                   label="Sample mean", edgecolor="black", linewidth="1")
    if ML is not None:
        pl.scatter(ML["p00_feature1_value"], ML["p01_feature2_value"], color="green", marker="D", s=80,
                   label="ML", edgecolor="black", linewidth="1")
    pl.scatter(-0.33, -0.67, color="black", edgecolor="white", linewidth="1",
               marker="*", s=200, label="Ground truth")
    pl.xlim(-1,0)
    pl.ylim(-1,0)
    pl.gca().legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), ncol=1, scatterpoints=1)
    pl.show()

