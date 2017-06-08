import numpy as np

from elfie.experiment import Experiment, ExperimentPhase


class InferenceTask():
    """ encapsulated inference task """
    def __init__(self):
        self.result = None
    def run(self):
        pass
    def plot(self, plotter):
        pass
    def to_dict(self):
        return {}


class InferenceExperiment(Experiment):
    def __init__(self,
            inference_factory,
            ground_truth=None,
            plotter=None,
            error_measures=[]):
        name = "Inference Experiment"
        super(InferenceExperiment, self).__init__(name, plotter)
        self.inference_factory = inference_factory
        phase1 = InferencePhase(parents=[],
                                inference_task=self.inference_factory.get())
        self.add_phase(phase1)
        self.add_phase(ErrorPhase(parents=[phase1],
                                  error_measures=error_measures))

    def to_dict(self):
        ret = super(InferenceExperiment, self).to_dict()
        ret["inference_factory"] = self.inference_factory.to_dict()
        return ret


class InferencePhase(ExperimentPhase):

    def __init__(self, parents, inference_task):
        name = "Inference Phase"
        super(InferencePhase, self).__init__(name, parents)
        self.inference_task = inference_task

    def _run(self):
        self.inference_task.run()

    def _plot(self, plotter):
        self.inference_task.plot(plotter)

    def to_dict(self):
        ret = super(ExperimentPhase, self).to_dict()
        ret["inference_task"] = self.inference_task.to_dict()
        return ret


class ErrorPhase(ExperimentPhase):
    def __init__(self, parents, error_measures=list()):
        name = "Error Computation Phase"
        super(ExperimentPhase, self).__init__(name, parents)
        self.error_measures = error_measures

    def _run(self):
        result = self.parents[0].inference_task.result
        for error_measure in self.error_measures:
            error_measure.compute(result)

    def _plot(self, plotter):
        for error_measure in self.error_measures:
            error_measure.plot(plotter)

    def to_dict(self):
        ret = super(ExperimentPhase, self).to_dict()
        ret["error_measures"] = dict()
        for error_measure in self.error_measures:
            ret["error_measures"][error_measure.name] = error_measure.to_dict()
        return ret


class ErrorMeasure():
    def __init__(self, name):
        self.name = name
        self.errors = None

    def compute(self, results):
        pass

    def plot(self, plotter):
        pass

    def to_dict(self):
        return self.__dict__

    def printout(self):
        grid = 20
        logger.info(self.name)
        lmax = max(self.errors)
        lmin = min(self.errors)
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

    def _plot(self, plotter):
        fig = pl.figure(figsize=plotter.figsize)
        t = range(len(self.errors))
        pl.plot(t, self.errors)
        pl.xlabel("Samples")
        pl.ylabel(self.name)
        pl.ylim(min(errors)-0.1, max(errors)+0.1)
        pl.title("{} over time".format(self.name))
        plotter.pdf.savefig()
        pl.close()


class DiscrepancyError(ErrorMeasure):
    """ Discrepancy of point estimate """
    def __init__(self, MAP=True, esimated=False, inference_factory=None):
        super(DiscrepancyError, self).__init__("Discrepancy")
        self.MAP = MAP
        self.estimated = estimated
        self.inference_factory = inference_factory

    def compute(self, results):
        assert issubclass(results, BolfiPosterior), type(results)  # TODO
        if self.MAP is True:
            loc, _ = results.MAP
        else:
            loc, _ = results.ML
        # TODO: handle whole storage
        if self.estimated is True:
            self.errors = [results.model._gp.predict_mean(loc)]
        else:
            self.errors = [min(results.model._gp.Y)]
            # TODO: simulate at min loc

    def to_dict(self):
        ret = super(DiscrepancyError, self).to_dict()
        ret["inference_factory"] = "none"  # TODO


class GroundTruthError(ErrorMeasure):
    """ Error of point estimate to ground truth """
    def __init__(self, ground_truth=[], errorf=lambda x, y: 0):
        super(GroundTruthError, self).__init__(self.__class__.__name__)
        self.ground_truth = np.atleast_1d(ground_truth)
        self.errorf = errorf

    def compute(self, results):
        assert issubclass(results, BolfiPosterior), type(results)  # TODO
        if self.MAP is True:
            loc, _ = results.MAP
        else:
            loc, _ = results.ML
        self.errors = [self.errorf(self.ground_truth, loc)]

    def to_dict(self):
        ret = super(GroundTruthError, self).to_dict()
        ret["ground_truth"] = [str(v) for v in self.ground_truth]
        return ret


def L2Error(a, b):
    """ L2 distance """
    return np.linalg.norm(a - b, ord=2)


def OrderError(a, b):
    """ Hamming distance of ranks """
    order_a = np.argsort(a)
    order_b = np.argsort(b)
    return sum([o1 != o2 for o1, o2 in zip(order_a, order_b)])


def ProportionError(a, b):
    """ log10 L2 distance between vectors of parameter proportions """
    prop_a = list()
    prop_b = list()
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            prop_a.append(float(a[i])/float(a[j]))
            prop_b.append(float(b[i])/float(b[j]))
    return np.log10(float(np.linalg.norm(np.array(prop_a) - np.array(prop_b), ord=2)))

