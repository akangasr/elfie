import pytest
import numpy as np

from elfi.examples.ma2 import get_model
from elfie.bolfi_extensions import BolfiParams, BolfiFactory
from elfie.inference import inference_experiment

def test_simple_inference_experiment_can_be_run():
    n_samples = 4
    true_params = [0.5, 0.5]  # t1, t2
    params = BolfiParams(bounds=((0,1),(0,1)),
                         n_samples=n_samples,
                         n_initial_evidence=2,
                         sampling_type="uniform",
                         seed=1,
                         simulator_node_name="MA2",
                         discrepancy_node_name="d")
    model = get_model(n_obs=1000, true_params=true_params, seed_obs=1)
    bf = BolfiFactory(model, params)
    test_data = model.generate(1)["MA2"][0]
    ground_truth = {"t1":0.5, "t2":0.5}
    inference_experiment(bf, ground_truth=ground_truth, test_data=test_data)

