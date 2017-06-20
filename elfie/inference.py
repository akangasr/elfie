import time
import traceback
import numpy as np

from elfie.outputpool_extensions import SerializableOutputPool
from elfie.utils import read_json_file

import logging
logger = logging.getLogger(__name__)


def inference_experiment(inference_factory,
                         ground_truth=None,
                         test_data=None,
                         pdf=None,
                         figsize=(8.27, 11.69)):

        inference_task = inference_factory.get()
        ret = dict()

        phases = [
            SamplingPhase(),
            PosteriorAnalysisPhase(),
            PointEstimateSimulationPhase(),
            PlottingPhase(pdf=pdf, figsize=figsize),
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
            if k not in ret.keys():
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

    def __init__(self, name="Sampling", requirements=[]):
        InferencePhase.__init__(self, name, requirements)

    def _run(self, inference_task, ret):
        sampling_start = time.time()
        inference_task.do_sampling()
        inference_task.compute_samples_and_MD()
        sampling_end = time.time()
        ret["sampling_duration"] = sampling_end - sampling_start
        ret["sample_pool"] = inference_task.pool.to_dict()
        ret["MD"] = inference_task.MD
        ret["MD_val"] = inference_task.MD_val
        return ret


class PosteriorAnalysisPhase(InferencePhase):

    def __init__(self, name="Posterior analysis", requirements=["sample_pool"]):
        InferencePhase.__init__(self, name, requirements)

    def _run(self, inference_task, ret):
        post_start = time.time()
        inference_task.compute_posterior()
        inference_task.compute_MAP()
        post_end = time.time()
        ret["post_duration"] = post_end - post_start
        ret["post"] = "TODO" #inference_task.post.to_dict()
        ret["MAP"] = inference_task.MAP
        ret["MAP_val"] = float(inference_task.MAP_val)
        return ret


class PointEstimateSimulationPhase(InferencePhase):

    def __init__(self, name="Point estimate simulation", requirements=["ML", "MAP"]):
        InferencePhase.__init__(self, name, requirements)

    def _run(self, inference_task, ret):
        ret["ML_sim"], ret["MAP_sim"] = inference_task.compute_from_model([inference_task.obsnodename],
                                                                          [ret["ML"], ret["MAP"]])
        return ret


class PlottingPhase(InferencePhase):

    def __init__(self, name="Plotting", requirements=["sample_pool"], pdf=None, figsize=(8.27, 11.69)):
        InferencePhase.__init__(self, name, requirements)
        self.pdf = pdf
        self.figsize = figsize

    def _run(self, inference_task, ret):
        if self.pdf is not None:
            inference_task.plot(self.pdf, self.figsize)
        else:
            logger.info("Pass, no pdf file.")
        return ret


class GroundTruthErrorPhase(InferencePhase):

    def __init__(self, name="Ground truth error computation", requirements=["MD", "ML", "MAP"], ground_truth=None):
        InferencePhase.__init__(self, name, requirements)
        self.ground_truth = ground_truth

    def _run(self, inference_task, ret):
        if self.ground_truth is not None:
            ret["MD_err"] = rmse(ret["MD"], self.ground_truth)
            ret["ML_err"] = rmse(ret["ML"], self.ground_truth)
            ret["MAP_err"] = rmse(ret["MAP"], self.ground_truth)
        else:
            logger.info("Pass, no ground truth parameters.")
        return ret


class PredictionErrorPhase(InferencePhase):

    def __init__(self, name="Prediction error computation", requirements=["MD", "ML", "MAP"], test_data=None):
        InferencePhase.__init__(self, name, requirements)
        self.test_data = test_data

    def _run(self, inference_task, ret):
        if self.test_data is not None:
            ret["MD_sim"], ret["ML_sim"], ret["MAP_sim"] = inference_task.compute_from_model([inference_task.discname],
                                                                                             [ret["MD"], ret["ML"], ret["MAP"]],
                                                                                             new_data=test_data)
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

