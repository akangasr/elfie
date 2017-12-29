import time
import traceback
import numpy as np
import random

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
                         plot_data=None,
                         types=["MD"],
                         n_cores=1,
                         replicates=10,
                         region_size=0.05):

        inference_task = inference_factory.get()
        ret = dict()
        ret["n_cores"] = n_cores

        phases = [
            SamplingPhase(),
            PosteriorAnalysisPhase(types=types),
            PointEstimateSimulationPhase(replicates=replicates, region_size=region_size),
            #LikelihoodSamplesSimulationPhase(replicates=replicates),
            #PosteriorSamplesSimulationPhase(replicates=replicates),
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
                return ret
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

    def __init__(self, name="Sampling", requirements=[]):
        InferencePhase.__init__(self, name, requirements)

    def _run(self, inference_task, ret):
        sampling_start = time.time()
        inference_task.do_sampling()
        sampling_end = time.time()
        ret["sampling_duration"] = sampling_end - sampling_start
        if inference_task.pool is not None:
            ret["sample_pool"] = inference_task.pool.to_dict()
        ret["samples"] = inference_task.samples
        ret["n_samples"] = len(inference_task.samples)
        if hasattr(inference_task, "kernel_params"):
            ret["kernel_params"] = inference_task.kernel_params
            logger.info("Kernel: {}".format(ret["kernel_params"]))
        return ret


class PosteriorAnalysisPhase(InferencePhase):

    def __init__(self, name="Posterior analysis", requirements=["samples"], types=["MD"]):
        self.types = types
        InferencePhase.__init__(self, name, requirements)

    def _run(self, inference_task, ret):
        if "MED" in self.types or "ML" in self.types or "MAP" in self.types or "LIK" in self.types:
            post_start = time.time()
            inference_task.compute_posterior()
            inference_task.compute_MED()
            inference_task.compute_ML()
            inference_task.compute_MAP()
            post_end = time.time()
            ret["post_duration"] = post_end - post_start
            ret["threshold"] = inference_task.threshold
            if "LIK" in self.types:
                try:
                    lik_s_start = time.time()
                    inference_task.sample_from_likelihood()
                    ret["lik_samples"] = inference_task.lik_samples
                    ret["lik_acc_prop"] = inference_task.lik_acc_prop
                    ret["LM"] = inference_task.LM
                    logger.info("LM at {}".format(ret["LM"]))
                    lik_s_end = time.time()
                    ret["lik_sampling_duration"] = lik_s_end - lik_s_start
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.critical(tb)
            if "POST" in self.types:
                try:
                    post_s_start = time.time()
                    inference_task.sample_from_posterior()
                    ret["post_samples"] = inference_task.post_samples
                    ret["post_acc_prop"] = inference_task.post_acc_prop
                    ret["PM"] = inference_task.PM
                    logger.info("PM at {}".format(ret["PM"]))
                    post_s_end = time.time()
                    ret["post_sampling_duration"] = post_s_end - post_s_start
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.critical(tb)
        if "MD" in self.types:
            ret["MD"] = inference_task.MD
            ret["MD_val"] = float(inference_task.MD_val)
            logger.info("MD at {} (val {})".format(ret["MD"], ret["MD_val"]))
        if "MED" in self.types:
            ret["MED"] = inference_task.MED
            ret["MED_val"] = float(inference_task.MED_val)
            logger.info("MED at {} (val {})".format(ret["MED"], ret["MED_val"]))
        if "ML" in self.types:
            ret["ML"] = inference_task.ML
            ret["ML_val"] = float(inference_task.ML_val)
            logger.info("ML at {} (val {})".format(ret["ML"], ret["ML_val"]))
        if "MAP" in self.types:
            ret["MAP"] = inference_task.MAP
            ret["MAP_val"] = float(inference_task.MAP_val)
            logger.info("MAP at {} (val {})".format(ret["MAP"], ret["MAP_val"]))
        return ret


class PointEstimateSimulationPhase(InferencePhase):

    def __init__(self, name="Point estimate simulation", requirements=["samples"], replicates=10, region_size=0.05):
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
        if self.replicates < 1:
            logger.info("No point estimates")
            return ret
        if "MD" in ret.keys():
            logger.info("Simulating near MD")
            ret["MD_sim"] = self._simulate_around(inference_task, ret["MD"])
        if "MED" in ret.keys():
            logger.info("Simulating near MED")
            ret["MED_sim"] = self._simulate_around(inference_task, ret["MED"])
        if "ML" in ret.keys():
            logger.info("Simulating near ML")
            ret["ML_sim"] = self._simulate_around(inference_task, ret["ML"])
        if "MAP" in ret.keys():
            logger.info("Simulating near MAP")
            ret["MAP_sim"] = self._simulate_around(inference_task, ret["MAP"])
        if "LM" in ret.keys():
            logger.info("Simulating near LM")
            ret["LM_sim"] = self._simulate_around(inference_task, ret["LM"])
        if "PM" in ret.keys():
            logger.info("Simulating near PM")
            ret["PM_sim"] = self._simulate_around(inference_task, ret["PM"])
        return ret

class LikelihoodSamplesSimulationPhase(InferencePhase):

    def __init__(self, name="Likelihood samples simulation", requirements=["lik_samples"], replicates=10):
        InferencePhase.__init__(self, name, requirements)
        self.replicates = replicates

    def _run(self, inference_task, ret):
        if self.replicates < 1:
            logger.info("No simulations")
            return ret
        nodes = [inference_task.obsnodename, inference_task.discname]
        bounds = inference_task.params.bounds
        wv_dicts = list()
        for sample in random.sample(ret["lik_samples"], self.replicates):
            wv = dict()
            for val, name in zip(sample, inference_task.paramnames):
                wv[name] = val
            wv_dicts.append(wv)
        logger.info("Simulating at: {}".format(wv_dicts))
        sims = inference_task.compute_from_model(nodes, wv_dicts)
        for i in range(len(sims)):
            sims[i]["_at"] = wv_dicts[i]
        ret["lik_sim"] = sims
        return ret

class PosteriorSamplesSimulationPhase(InferencePhase):

    def __init__(self, name="Posterior samples simulation", requirements=["post_samples"], replicates=10):
        InferencePhase.__init__(self, name, requirements)
        self.replicates = replicates

    def _run(self, inference_task, ret):
        if self.replicates < 1:
            logger.info("No simulations")
            return ret
        nodes = [inference_task.obsnodename, inference_task.discname]
        bounds = inference_task.params.bounds
        wv_dicts = list()
        for sample in random.sample(ret["post_samples"], self.replicates):
            wv = dict()
            for val, name in zip(sample, inference_task.paramnames):
                wv[name] = val
            wv_dicts.append(wv)
        logger.info("Simulating at: {}".format(wv_dicts))
        sims = inference_task.compute_from_model(nodes, wv_dicts)
        for i in range(len(sims)):
            sims[i]["_at"] = wv_dicts[i]
        ret["post_sim"] = sims
        return ret

class PlottingPhase(InferencePhase):

    def __init__(self, name="Plotting", requirements=["samples"], pdf=None, figsize=(8.27, 11.69),
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
            ret["plots"] = inference_task.plot_post(self.pdf, self.figsize)
            if self.plot_data is not None:
                if "MD_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["MD_sim"], "Minimum discrepancy")
                if "MED_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["MED_sim"], "MED")
                if "ML_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["ML_sim"], "ML")
                if "MAP_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["MAP_sim"], "MAP")
                if "LM_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["LM_sim"], "LM")
                if "lik_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["lik_sim"], "Likelihood samples")
                if "PM_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["PM_sim"], "PM")
                if "post_sim" in ret.keys():
                    self._plot_datas(inference_task, ret["post_sim"], "Posterior samples")
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

    def __init__(self, name="Ground truth error computation", requirements=["samples"], ground_truth=None):
        InferencePhase.__init__(self, name, requirements)
        self.ground_truth = ground_truth

    def _run(self, inference_task, ret):
        if self.ground_truth is not None:
            if "MD" in ret.keys():
                ret["MD_gt_err"] = rmse(ret["MD"], self.ground_truth)
            if "MED" in ret.keys():
                ret["MED_gt_err"] = rmse(ret["MED"], self.ground_truth)
            if "ML" in ret.keys():
                ret["ML_gt_err"] = rmse(ret["ML"], self.ground_truth)
            if "MAP" in ret.keys():
                ret["MAP_gt_err"] = rmse(ret["MAP"], self.ground_truth)
            if "LM" in ret.keys():
                ret["LM_gt_err"] = rmse(ret["LM"], self.ground_truth)
            if "PM" in ret.keys():
                ret["PM_gt_err"] = rmse(ret["PM"], self.ground_truth)
        else:
            logger.info("Pass, no ground truth parameters.")
        return ret


class PredictionErrorPhase(InferencePhase):

    def __init__(self, name="Prediction error computation", requirements=["samples"], test_data=None):
        InferencePhase.__init__(self, name, requirements)
        self.test_data = test_data

    def _compute_errors(self, inference_task, sims):
        wv_dicts = [{inference_task.obsnodename: sim[inference_task.obsnodename]} for sim in sims]
        disc = inference_task.compute_from_model([inference_task.discname], wv_dicts, new_data=self.test_data)
        return [float(d[inference_task.discname][0]) for d in disc]

    def _run(self, inference_task, ret):
        if self.test_data is not None:
            if "MD_sim" in ret.keys():
                logger.info("Estimating MD prediction error")
                ret["MD_errs"] = self._compute_errors(inference_task, ret["MD_sim"])
                logger.info("Average MD error: {}".format(np.mean(ret["MD_errs"])))
            if "MED_sim" in ret.keys():
                logger.info("Estimating MED prediction error")
                ret["MED_errs"] = self._compute_errors(inference_task, ret["MED_sim"])
                logger.info("Average MED error: {}".format(np.mean(ret["MED_errs"])))
            if "ML_sim" in ret.keys():
                logger.info("Estimating ML prediction error")
                ret["ML_errs"] = self._compute_errors(inference_task, ret["ML_sim"])
                logger.info("Average ML error: {}".format(np.mean(ret["ML_errs"])))
            if "MAP_sim" in ret.keys():
                logger.info("Estimating MAP prediction error")
                ret["MAP_errs"] = self._compute_errors(inference_task, ret["MAP_sim"])
                logger.info("Average MAP error: {}".format(np.mean(ret["MAP_errs"])))
            if "LM_sim" in ret.keys():
                logger.info("Estimating LM prediction error")
                ret["LM_errs"] = self._compute_errors(inference_task, ret["LM_sim"])
                logger.info("Average LM error: {}".format(np.mean(ret["LM_errs"])))
            if "lik_sim" in ret.keys():
                logger.info("Estimating Likelihood samples prediction error")
                ret["lik_errs"] = self._compute_errors(inference_task, ret["lik_sim"])
                logger.info("Average LIK error: {}".format(np.mean(ret["lik_errs"])))
            if "PM_sim" in ret.keys():
                logger.info("Estimating PM prediction error")
                ret["PM_errs"] = self._compute_errors(inference_task, ret["PM_sim"])
                logger.info("Average PM error: {}".format(np.mean(ret["PM_errs"])))
            if "post_sim" in ret.keys():
                logger.info("Estimating Posterior samples prediction error")
                ret["post_errs"] = self._compute_errors(inference_task, ret["post_sim"])
                logger.info("Average POST error: {}".format(np.mean(ret["post_errs"])))
        else:
            logger.info("Pass, no test data.")
        return ret


def get_sample_pool(filename):
    data = read_json_file(filename)
    return SerializableOutputPool.from_dict(data["sample_pool"])


def rmse(a, b):
    """ RMSE for dictionaries """
    sqerr = []
    for k in a.keys():
        k2 = k[4:]  # strip prefix
        sqerr.append((float(a[k]) - float(b[k2])) ** 2)
    return float(np.sqrt(np.mean(sqerr)))

