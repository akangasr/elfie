import json

import logging
logger = logging.getLogger(__name__)


def write_json_file(filename, data):
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
    logger.info("Wrote {}".format(filename))


def read_json_file(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    logger.info("Read {}".format(filename))
    return data

