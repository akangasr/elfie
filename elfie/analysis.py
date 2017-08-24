import numpy as np
import scipy as sp
import json

class ExperimentLog():
    def __init__(self, filename, method, samples):
        self.method = method
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
        self._index("methods", lambda e: e.method)
        self._index("n_samples", lambda e: e.n_samples)

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

    def get_value_mean_std(self, getters, x_getters=None, verbose=False):
        if type(getters) is not dict:
            getters = {"": getters}
        if type(x_getters) is not dict:
            x_getters = {"": lambda e: e.n_samples}
        datas = dict()
        for m in sorted(self.methods.keys()):
            for n, getter in getters.items():
                if n in x_getters.keys():
                    x_getter = x_getters[n]
                else:
                    x_getter = x_getters[""]
                means = list()
                stds = list()
                x = list()
                label = "{} {}".format(m, n).upper().strip()
                for s in sorted(self.n_samples.keys()):
                    try:
                        mean = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.mean)
                        std = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.std)
                        xloc = self.get_filt_agg(x_getter, lambda e: e.method==m and e.n_samples==s, np.mean)
                        x.append(xloc)
                        means.append(mean)
                        stds.append(std)
                    except Exception as e:
                        if verbose is True:
                            print("- skip {} s={} ({})".format(label, s, e))
                        pass
                if len(means) > 0:
                    means = np.array(means)
                    stds = np.array(stds)
                    x = np.array(x)
                    datas[label] = (means, stds, x)
                    if verbose is True:
                        print("- {} means={} stds={} x={}".format(m, means, stds, x))
        return datas

