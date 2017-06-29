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
                    mean = self.get_filt_agg(lambda e: getter(e), lambda e: e.method==m and e.samples==s, np.mean)
                    std = self.get_filt_agg(lambda e: getter(e), lambda e: e.method==m and e.samples==s, np.std)
                    n = self.get_filt_agg(lambda e: getter(e), lambda e: e.method==m and e.samples==s, len)
                    if formatter is None:
                        print("  - {}: mean={:.2f}, std={:.2f} (N={})".format(m, mean, std, n))
                    else:
                        print("  - {}: mean={}, std={} (N={})".format(m, formatter(mean), formatter(std), n))
                except Warning:
                    print("  - {}: skip".format(m))
        print("")


