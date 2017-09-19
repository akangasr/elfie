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
                if type(getter) == dict:
                    if m in getter.keys():
                        getter = getter[m]
                    else:
                        continue
                label = "{} {}".format(m, n).upper().strip()
                if label in x_getters.keys():
                    x_getter = x_getters[label]
                else:
                    x_getter = x_getters[""]
                means = list()
                stds = list()
                p05s = list()
                p95s = list()
                x = list()
                ns = list()
                for s in sorted(self.n_samples.keys()):
                    try:
                        mean = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.mean)
                        std = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.std)
                        p05 = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, lambda x: np.percentile(x, 5))
                        p95 = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, lambda x: np.percentile(x, 95))
                        xloc = self.get_filt_agg(x_getter, lambda e: e.method==m and e.n_samples==s, np.mean)
                        n = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, len)
                        means.append(mean)
                        stds.append(std)
                        p05s.append(p05)
                        p95s.append(p95)
                        x.append(xloc)
                        ns.append(n)
                    except Exception as e:
                        if verbose is True:
                            print("- skip {} s={} ({})".format(label, s, e))
                        pass
                if len(means) > 0:
                    means = np.array(means)
                    stds = np.array(stds)
                    p05s = np.array(p05s)
                    p95s = np.array(p95s)
                    x = np.array(x)
                    ns = np.array(ns)
                    datas[label] = (means, stds, x, p05s, p95s, ns)
                    if verbose is True:
                        print("- {} means={} stds={} x={} ns={}".format(m, means, stds, x, ns))
        return datas

