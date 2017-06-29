import json
import traceback
import numpy as np

import elfi

import logging
logger = logging.getLogger(__name__)


def write_json_file(filename, data):
    try:
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        logger.info("Wrote {}".format(filename))
    except Exception as e:
        logger.critical("Could not write the following to file as JSON:")
        logger.critical(data)
        tb = traceback.format_exc()
        logger.critical(tb)

def read_json_file(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    logger.info("Read {}".format(filename))
    return data

def eval_2d_mesh(xmin, ymin, xmax, ymax, nx, ny, eval_fun):
    """
        Evaluate 'eval_fun' at a grid defined by max and min
        values with number of points defined by 'nx' and 'ny'.
    """
    if xmin > xmax:
        raise ValueError("xmin (%.2f) was greater than"
                         "xmax (%.2f)" % (xmin, xmax))
    if ymin > ymax:
        raise ValueError("ymin (%.2f) was greater than"
                         "ymax (%.2f)" % (xmin, xmax))
    if nx < 1 or ny < 1:
        raise ValueError("nx (%.2f) or ny (%.2f) was less than 1" % (nx, ny))
    X = np.linspace(xmin, xmax, nx)
    lenx = len(X)
    Y = np.linspace(ymin, ymax, ny)
    leny = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((leny, lenx))
    for i in range(leny):
        for j in range(lenx):
            Z[i][j] = eval_fun(np.array([X[i][j], Y[i][j]]))
    return (X, Y, Z)

def pretty(param_dict):
    ret = ["("]
    for k in sorted(param_dict.keys()):
        ret.append("{}={:.3f}".format(k.split("_",1)[1], param_dict[k]))  # assume there is the ugly prefix, remove it
        ret.append(",")
    ret[-1] = ")"
    return "".join(ret)

def pretty_time(seconds):
    ret = list()
    minutes = 0
    hours = 0
    days = 0
    if seconds > 60:
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
    if seconds > 0:
        ret.append("{:.0f}s".format(seconds))
    if minutes > 60:
        hours = int(minutes / 60)
        minutes = minutes - hours * 60
    if minutes > 0:
        ret.append("{:d}m ".format(minutes))
    if hours > 24:
        days = int(hours / 24)
        hours = hours - days * 24
    if hours > 0:
        ret.append("{:d}h ".format(hours))
    if days > 0:
        ret.append("{:d}d ".format(days))
    return "".join(reversed(ret))

