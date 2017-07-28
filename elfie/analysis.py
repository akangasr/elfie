import numpy as np
import scipy as sp
import json
import matplotlib
import matplotlib.pyplot as pl

class ExperimentLog():
    def __init__(self, filename, method, samples):
        self.method = method
        self.samples = samples
        with open(filename) as f:
            full_log = json.load(f)
        for k, v in full_log.items():
            setattr(self, k, v)

    def debug_print(self):
        print("Experiment:")
        for k, v in self.__dict__.items():
            print("* {} = {}".format(k, v))


class ExperimentGroup():
    def __init__(self, experiments):
        self.exp = experiments
        self._index("samples", lambda e: e.samples)
        self._index("methods", lambda e: e.method)

    def get_filt_agg(self, getter, filt, aggregate):
        return aggregate([getter(e) for e in self.exp if filt(e)])

    def _index(self, name, getter):
        vals = dict()
        for e in self.exp:
            val = getter(e)
            if val not in vals.keys():
                vals[val] = [e]
            else:
                vals[val].append(e)
        setattr(self, name, vals)

    def print_value_mean_std(self, name, getter, formatter=None):
        print("{} mean and standard deviation".format(name))
        for s in sorted(self.samples.keys()):
            print("* {} samples".format(s))
            for m in sorted(self.methods.keys()):
                try:
                    mean = self.get_filt_agg(getter, lambda e: e.method==m and e.samples==s, np.mean)
                    std = self.get_filt_agg(getter, lambda e: e.method==m and e.samples==s, np.std)
                    n = self.get_filt_agg(getter, lambda e: e.method==m and e.samples==s, len)
                    if formatter is None:
                        print("  - {}: mean={:.2f}, std={:.2f} (N={})".format(m, mean, std, n))
                    else:
                        print("  - {}: mean={}, std={} (N={})".format(m, formatter(mean), formatter(std), n))
                except Warning:
                    print("  - {}: skip".format(m))
        print("")

    def plot_value_mean_std(self, name, getters, colors=None, alpha=0.25):
        if colors is None:
            colors = {
                    "BO": "skyblue",
                    "ABC ML": "dodgerblue",
                    "ABC MAP": "blue",
                    "ABC": "blue",
                    "GRID": "green",
                    "LBFGSB": "orangered",
                    "NELDERMEAD": "m"}
        if type(getters) is not dict:
            getters = {"": getters}
        drawn = False
        for m in sorted(self.methods.keys()):
            for n, getter in getters.items():
                means = list()
                stds = list()
                samples = list()
                label = "{} {}".format(m, n).upper().strip()
                if label == "BO ML":
                    label = "ABC ML"
                if label == "BO MAP":
                    label = "ABC MAP"
                if label == "BO ABC":
                    label = "ABC"
                for s in sorted(self.samples.keys()):
                    try:
                        mean = self.get_filt_agg(getter, lambda e: e.method==m and e.samples==s, np.mean)
                        std = self.get_filt_agg(getter, lambda e: e.method==m and e.samples==s, np.std)
                        means.append(mean)
                        stds.append(std)
                        samples.append(s)
                    except Exception as e:
                        print("skip {} ({})".format(label, e))
                        pass
                if len(means) > 0:
                    drawn = True
                    means = np.array(means)
                    stds = np.array(stds)
                    pl.plot(samples, means, marker=".", color=colors[label], label=label)
                    #pl.fill_between(samples, means+stds, means-stds, facecolor=colors[label], alpha=alpha)
        if drawn is True:
            pl.title("{} (mean and std)".format(name))
            pl.legend(loc=1)
            pl.xlabel("Samples")
            pl.show()


