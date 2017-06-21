import numpy as np

import elfi

def uniform_prior(minv, maxv, name, **kwargs):
    return elfi.Prior("uniform", minv, maxv - minv, name=name)

def truncnorm_prior(minv, maxv, mean, std, name, **kwargs):
    return elfi.Prior("truncnorm", (minv - mean)/std, (maxv - mean)/std, mean, std, name=name)

def constant_prior(val, name, **kwargs):
    return elfi.Constant(val, name=name)


class ModelParams():

    priors = {
        "uniform": uniform_prior,
        "truncnorm": truncnorm_prior,
        "constant": constant_prior,
        }

    def __init__(self, parameters):
        self.parameters = parameters
        for i, p in enumerate(self.parameters):  # BOLFI works in mysterious ways
            p["name"] = "p{:02d}_{}".format(i, p["name"])

    def get_elfi_params(self):
        ret = list()
        for p in self.parameters:
            if p["distr"] not in self.priors.keys():
                raise ValueError("Unsupported distribution: {}".format(p["distr"]))
            ret.append(self.priors[p["distr"]](**p))
        return ret

    def get_bounds(self):
        return [(p["minv"], p["maxv"]) for p in self.parameters if p["distr"] is not "constant"]

    def get_acq_noises(self):
        return [p["acq_noise"] for p in self.parameters if p["distr"] is not "constant"]

    def get_grid_tics(self):
        return [np.linspace(p["minv"], p["maxv"], p["ntics"]).tolist() for p in self.parameters if p["distr"] is not "constant"]

