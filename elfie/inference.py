import time
import traceback
import numpy as np

from elfie.outputpool_extensions import SerializableOutputPool
from elfie.utils import read_json_file, pretty

import logging
logger = logging.getLogger(__name__)


def inference_experiment(inference_factory,
                         ground_truth=None,
                         obs_data=None,
                         test_data=None,
                         pdf=None,
                         figsize=(8.27, 11.69),
                         skip_post=False,
                         plot_data=None,
                         n_cores=1):

        inference_task = inference_factory.get()
        ret = dict()

        phases = [
            SamplingPhase(n_cores=n_cores),
            PosteriorAnalysisPhase(skip=skip_post),
            PointEstimateSimulationPhase(),
            PlottingPhase(pdf=pdf, figsize=figsize, obs_data=obs_data, test_data=test_data, plot_data=plot_data),
            GroundTruthErrorPhase(ground_truth=ground_truth),
            PredictionErrorPhase(test_data=test_data),
            ]

        for phase in phases:
            ret = phase.run(inference_task, ret)
        return ret


class InferencePhase():
    def __init__(self, name, requirements):
        self.name = name
        self.req = requirements

    def run(self, inference_task, ret):
        for k in self.req:
            if ret is None or k not in ret.keys():
                logger.warning("Can not run {} phase as {} does not exist!".format(self.name, k))
                return
        try:
            logger.info("Running {} phase".format(self.name))
            return self._run(inference_task, ret)
        except:
            logger.critical("Error during {} phase!".format(self.name))
            tb = traceback.format_exc()
            logger.critical(tb)
            return ret

    def _run(self, inference_task, ret):
        return ret


class SamplingPhase(InferencePhase):

    def __init__(self, name="Sampling", requirements=[], n_cores=1):
        InferencePhase.__init__(self, name, requirements)
        self.n_cores = n_cores

    def _run(self, inference_task, ret):
        sampling_start = time.time()
        inference_task.do_sampling()
        sampling_end = time.time()
        ret["n_cores"] = self.n_cores
        ret["sampling_duration"] = sampling_end - sampling_start
        if inference_task.pool is not None:
            ret["sample_pool"] = inference_task.pool.to_dict()
        ret["MD"] = inference_task.MD
        ret["MD_val"] = inference_task.MD_val
        return ret


class PosteriorAnalysisPhase(InferencePhase):

    def __init__(self, name="Posterior analysis", requirements=["MD"], skip=False):
        self.skip = skip
        InferencePhase.__init__(self, name, requirements)

    def _run(self, inference_task, ret):
        if self.skip is True:
            logger.info("Skipping")
            return ret
        post_start = time.time()
        inference_task.compute_posterior()
        inference_task.compute_ML()
        inference_task.compute_MAP()
        post_end = time.time()
        ret["post_duration"] = post_end - post_start
        ret["post"] = "TODO" #inference_task.post.to_dict()
        ret["ML"] = inference_task.ML
        ret["ML_val"] = float(inference_task.ML_val)
        ret["MAP"] = inference_task.MAP
        ret["MAP_val"] = float(inference_task.MAP_val)  # TODO: find points with ML and MAP values, also extract discerpancies
        return ret


class PointEstimateSimulationPhase(InferencePhase):

    def __init__(self, name="Point estimate simulation", requirements=["MD"], replicates=10, region_size=0.02):
        InferencePhase.__init__(self, name, requirements)
        self.replicates = replicates
        self.region_size = region_size

    def _simulate_around(self, inference_task, loc):
        nodes = [inference_task.obsnodename, inference_task.discname]
        bounds = inference_task.params.bounds
        wv_dicts = list()
        logger.info("Simulating near: {}".format(loc))
        for i in range(self.replicates):
            wv = dict()
            for k, v in loc.items():
                bound_width = bounds[k][1] - bounds[k][0]
                region_width = bound_width * self.region_size
                low = max(bounds[k][0], v - (region_width/2.0))
                high = min(bounds[k][1], v + (region_width/2.0))
                val = float(np.random.uniform(low, high))
                wv[k] = val
            wv_dicts.append(wv)
        logger.info("Simulating at: {}".format(wv_dicts))
        sims = inference_task.compute_from_model(nodes, wv_dicts)
        for i in range(len(sims)):
            sims[i]["_at"] = wv_dicts[i]
        return sims

    def _run(self, inference_task, ret):
        logger.info("Simulating near MD")
        ret["MD_sim"] = self._simulate_around(inference_task, ret["MD"])
        if "MAP" in ret.keys():
            logger.info("Simulating near ML")
            ret["ML_sim"] = self._simulate_around(inference_task, ret["ML"])
            logger.info("Simulating near MAP")
            ret["MAP_sim"] = self._simulate_around(inference_task, ret["MAP"])
        return ret


class PlottingPhase(InferencePhase):

    def __init__(self, name="Plotting", requirements=["MD"], pdf=None, figsize=(8.27, 11.69),
                 obs_data=None, test_data=None, plot_data=None):
        InferencePhase.__init__(self, name, requirements)
        self.pdf = pdf
        self.figsize = figsize
        self.obs_data = obs_data
        self.test_data = test_data
        self.plot_data = plot_data

    def _plot_datas(self, inference_task, sims, name):
        logger.info("Plotting {} samples".format(name))
        for sim in sims:
            try:
                string = "{} sample at {} (discrepancy {:.2f})".format(name, pretty(sim["_at"]), sim[inference_task.discname][0])
            except:
                string = "asd"
                print(name, sim)
            self.plot_data(self.pdf, self.figsize, sim[inference_task.obsnodename], string)

    def _run(self, inference_task, ret):
        if self.pdf is not None:
            logger.info("Plotting posterior")
            inference_task.plot_post(self.pdf, self.figsize)
            if self.plot_data is not None:
                if "MD_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["MD_sim"], "Minimum discrepancy")
                if "ML_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["ML_sim"], "ML")
                if "MAP_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["MAP_sim"], "MAP")
                if self.obs_data is not None:
                    logger.info("Plotting observation data")
                    self.plot_data(self.pdf, self.figsize, self.obs_data, "Observation data")
                if self.test_data is not None:
                    logger.info("Plotting test data")
                    self.plot_data(self.pdf, self.figsize, self.test_data, "Test data")
        else:
            logger.info("Pass, no pdf file.")
        return ret


class GroundTruthErrorPhase(InferencePhase):

    def __init__(self, name="Ground truth error computation", requirements=["MD"], ground_truth=None):
        InferencePhase.__init__(self, name, requirements)
        self.ground_truth = ground_truth

    def _run(self, inference_task, ret):
        if self.ground_truth is not None:
            ret["MD_err"] = rmse(ret["MD"], self.ground_truth)
            if "MAP" in ret.keys():
                ret["ML_err"] = rmse(ret["ML"], self.ground_truth)
                ret["MAP_err"] = rmse(ret["MAP"], self.ground_truth)
        else:
            logger.info("Pass, no ground truth parameters.")
        return ret


class PredictionErrorPhase(InferencePhase):

    def __init__(self, name="Prediction error computation", requirements=["MD"], test_data=None):
        InferencePhase.__init__(self, name, requirements)
        self.test_data = test_data

    def _compute_errors(self, inference_task, sims):
        wv_dicts = [{inference_task.obsnodename: sim[inference_task.obsnodename]} for sim in sims]
        disc = inference_task.compute_from_model([inference_task.discname], wv_dicts, new_data=self.test_data)
        return [float(d[inference_task.discname][0]) for d in disc]

    def _run(self, inference_task, ret):
        if self.test_data is not None:
            logger.info("Estimating MD prediction error")
            ret["MD_errs"] = self._compute_errors(inference_task, ret["MD_sim"])
            if "MAP" in ret.keys():
                logger.info("Estimating ML prediction error")
                ret["ML_errs"] = self._compute_errors(inference_task, ret["ML_sim"])
                logger.info("Estimating MAP prediction error")
                ret["MAP_errs"] = self._compute_errors(inference_task, ret["MAP_sim"])
        else:
            logger.info("Pass, no test data.")
        return ret


def get_sample_pool(filename):
    data = read_json_file(filename)
    return SerializableOutputPool.from_dict(data["sample_pool"])


def print_graph(data, name):
    grid = 20
    logger.info(name)
    lmax = max(errors)
    lmin = min(errors)
    delta = (lmax - lmin) / float(grid)
    for n in reversed(range(grid+2)):
        lim = lmin + (n-1)*delta
        st = ["{: >+7.3f}".format(lim)]
        for e in errors:
            if e >= lim:
                st.append("*")
            else:
                st.append(" ")
        logger.info("".join(st))


def plot_graph(data, name, pdf, figsize):
    fig = pl.figure(figsize=figsize)
    t = range(len(errors))
    pl.plot(t, errors)
    pl.xlabel("Samples")
    pl.ylabel(name)
    pl.ylim(min(errors)-0.1, max(errors)+0.1)
    pl.title("{} over time".format(name))
    pdf.savefig()
    pl.close()


def rmse(a, b):
    """ RMSE for dictionaries """
    sqerr = []
    for k in a.keys():
        sqerr.append((float(a[k]) - float(b[k])) ** 2)
    return float(np.sqrt(np.mean(sqerr)))

