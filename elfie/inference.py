import time
import numpy as np

import logging
logger = logging.getLogger(__name__)

#with PdfPages(self.pdf_file) as pdf:

def inference_experiment(inference_factory,
                         ground_truth=None,
                         test_data=None,
                         pdf=None,
                         figsize=(8.27, 11.69)):

        inference_task = inference_factory.get()

        ret = dict()
        logger.info("Starting inference task")
        sampling_start = time.time()
        inference_task.do_sampling()
        inference_task.compute_samples_and_ML()
        sampling_end = time.time()
        ret["sampling_duration"] = sampling_end - sampling_start
        ret["samples"] = inference_task.samples
        ret["ML"] = inference_task.ML
        ret["ML_val"] = inference_task.ML_val

        logger.info("Analyzing posterior")
        post_start = time.time()
        inference_task.compute_posterior()
        inference_task.compute_MAP()
        post_end = time.time()
        ret["post_duration"] = post_end - post_start
        ret["post"] = "TODO" #inference_task.post.to_dict()
        ret["MAP"] = inference_task.MAP
        ret["MAP_val"] = inference_task.MAP_val

        if pdf is not None:
            logger.info("Plotting")
            inference_task.plot(pdf, figsize)

        if ground_truth is not None:
            logger.info("Computing errors to ground truth")
            ret["ML_err"] = rmse(ret["ML"], ground_truth)
            ret["MAP_err"] = rmse(ret["MAP"], ground_truth)

        if test_data is not None:
            logger.info("Computing prediction errors")
            ret["ML_disc"] = inference_task.compute_discrepancy_with_data(ret["ML"], test_data)
            ret["MAP_disc"] = inference_task.compute_discrepancy_with_data(ret["MAP"], test_data)


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

