import numpy as np
import scipy as sp
import json
import matplotlib
import matplotlib.pyplot as pl

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

    def print_value_mean_std(self, name, getter, formatter=None):
        print("{} mean and standard deviation".format(name))
        for s in sorted(self.n_samples.keys()):
            print("* {} samples".format(s))
            for m in sorted(self.methods.keys()):
                try:
                    mean = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.mean)
                    std = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.std)
                    n = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, len)
                    if formatter is None:
                        print("  - {}: mean={:.2f}, std={:.2f} (N={})".format(m, mean, std, n))
                    else:
                        print("  - {}: mean={}, std={} (N={})".format(m, formatter(mean), formatter(std), n))
                except Warning:
                    print("  - {}: skip".format(m))
        print("")

    def plot_value_mean_std(self, plotdef, getters, x_getters=None, relabeler=lambda x:x):
        if type(getters) is not dict:
            getters = {"": getters}
        if type(x_getters) is not dict:
            x_getters = {"": lambda e: e.n_samples}
        drawn = False
        datas = list()
        maxmeanlen = 0
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
                label = relabeler(label)
                for s in sorted(self.n_samples.keys()):
                    try:
                        mean = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.mean)
                        std = self.get_filt_agg(getter, lambda e: e.method==m and e.n_samples==s, np.std)
                        xloc = self.get_filt_agg(x_getter, lambda e: e.method==m and e.n_samples==s, np.mean)
                        x.append(xloc)
                        means.append(mean)
                        stds.append(std)
                    except Exception as e:
                        print("- skip {} s={} ({})".format(label, s, e))
                        pass
                if len(means) > 0:
                    drawn = True
                    means = np.array(means)
                    stds = np.array(stds)
                    print("- {} means={} stds={}".format(m, means, stds))
                    datas.append((label, means, stds))
                    maxmeanlen = max(len(means), maxmeanlen)
        if maxmeanlen > 1:
            plot_trendlines(datas, plotdef)
        elif maxmeanlen > 0:
            plot_barchart(datas, plotdef)


class Plotdef():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def plot_trendlines(datas, pd):
    fig = pl.figure(figsize=pd.figsize)
    pl.rc('text', usetex=True)
    pl.rc('font', **{'family':'sans-serif','sans-serif':['Avant Garde']})
    for label, means, stds in datas:
        pl.plot(x, means, marker=pd.markers[label], color=pd.colors[label], label=label)
        if pd.errbars is True:
            pl.fill_between(x, means+stds, means-stds, facecolor=pd.colors[label], alpha=pd.alpha)

    pl.title("pd.title")
    pl.legend(loc=pd.legend_loc)
    pl.show()

def plot_barchart(datas, pd):
    ind = np.arange(len(datas)+1)
    bar_width = 1.0

    fig, ax = pl.subplots(figsize=pd.figsize)
    #pl.rc('text', usetex=True)
    #pl.rc('font', **{'family':'sans-serif','sans-serif':['Avant Garde']})
    ax.grid(True)
    ax.grid(zorder=0)
    for line in ax.get_xgridlines():
        line.set_color("white")
        line.set_linestyle("")
    for line in ax.get_ygridlines():
        line.set_color("lightgrey")
        line.set_linestyle("-")

    bars = list()
    i = 1
    for label, means, stds in datas:
        bar = ax.bar(ind[i], means[0], bar_width, color=pd.colors[label],
                     hatch=pd.hatches[label], edgecolor="black", zorder=3)
        if hasattr(pd, "errbars"):
            ax.errorbar(ind[i], means[0], fmt=" ",
                     yerr=stds[0], ecolor="black", capsize=5, zorder=4)
        bars.append(bar)
        i += 1

    ax.set_ylabel(pd.ylabel, fontsize=16)
    ax.set_title(pd.title, fontsize=20)
    ax.set_xticks(ind[1:])
    ax.set_xticklabels([d[0] for d in datas], fontsize=16)
    pl.tick_params(axis='x',which='both',bottom='off',top='off')
    pl.setp(ax.get_yticklabels(), fontsize=16)
    pl.xlim(ind[0]-0.1, ind[-1]+0.1+bar_width)

    if hasattr(pd, "ylim"):
        pl.ylim(pd.ylim)
    if hasattr(pd, "yscale"):
        pl.yscale(pd.yscale)

    #if pd.legend_loc == "in":
    #    ax.legend([b[0] for b in bars], [d["name"] for d in datas], loc=2,
    #          ncol=pd.legend_cols, fontsize=16)
    #if ed.legend_loc == "out":
    #    ax.legend([b[0] for b in bars], [d["name"] for d in datas], loc='upper center',
    #          bbox_to_anchor=(0.5, -0.1), ncol=pd.legend_cols, fontsize=16)
    pl.show()
