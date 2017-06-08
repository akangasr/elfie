from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import logging
logger = logging.getLogger(__name__)


class ExperimentPlotter():
    """ facilitates producing a pdf plot of the experiment """
    def __init__(self, pdf_file, figsize=(8.27, 11.69)):  # A4 portrait
        self.pdf_file = pdf_file
        self.pdf = None
        self.figsize = figsize

    def do_pdf(self, kallable):
        with PdfPages(self.pdf_file) as pdf:
            self.pdf = pdf
            kallable()
        logger.info("Plotted results to {}".format(self.pdf_file))

    def plot_text_page(self, text, locx=0.02, locy=0.02):
        if self.pdf is not None:
            fig = pl.figure(figsize=self.figsize)
            fig.text(locx, locy, text)
            self.pdf.savefig()
            pl.close()

    def to_dict(self):
        return {
                "pdf_file": self.pdf_file,
                "figsize": self.figsize,
                }


class Experiment():
    """ single complete experiment """
    def __init__(self, name, plotter):
        self.name = name
        self.phases = set()
        self.plotter = plotter

    def run(self):
        logger.info("Running {}".format(self.name))
        if self.plotter is not None:
            self.plotter.do_pdf(self._run)
        else:
            self._run()
        logger.info("{} ready".format(self.name))

    def _run(self):
        parents = set().union(*[set(p.parents) for p in self.phases])
        for phase in self.phases:
            if phase not in parents:
                phase.run(self.plotter)

    def add_phase(self, phase):
        self.phases.add(phase)

    def to_dict(self):
        return {
                "name": self.name,
                "plotter": self.plotter.to_dict()
                "phases": [p.to_dict() for p in self.phases]
                }


class ExperimentPhase():
    """ single phase of experiment """

    def __init__(self, name, parents=list()):
        self.name = name
        self.parents = parents

    def run(self, plotter=None):
        """ Runs the experiment phase
        """
        logger.info("Running {}".format(self.name))
        for p in self.parents:
            p.run(plotter)
        self._run()
        self.plot(plotter)
        self.printout()
        logger.info("{} ready".format(self.name))

    def _run(self):
        pass

    def plot(self, plotter=None):
        if plotter is None:
            return
        self._plot(plotter)

    def printout(self):
        pass

    def _plot(self, plotter):
        pass

    def to_dict(self):
        return self.__dict__

