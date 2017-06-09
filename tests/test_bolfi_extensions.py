import pytest
import json
import numpy as np

from elfi.examples.ma2 import get_model
from elfie.bolfi_extensions import BolfiParams, BolfiFactory

def test_bolfi_params_can_be_serialized():
    bp = BolfiParams()
    json.dumps(bp.__str__())

def test_simple_inference_can_be_run():
    n_samples = 4
    true_params = [0.5, 0.5]  # t1, t2
    for sampling_type in ["grid", "uniform", "BO"]:
        params = BolfiParams(bounds=((0,1),(0,1)),
                             n_samples=n_samples,
                             n_initial_evidence=2,
                             sampling_type=sampling_type,
                             grid_tics=[[0.25, 0.75], [0.33, 0.66]],
                             seed=1,
                             simulator_node_name="MA2",
                             discrepancy_node_name="d")
        model = get_model(n_obs=1000, true_params=true_params, seed_obs=1)
        results=list()
        bf = BolfiFactory(model, params)
        exp = bf.get()
        exp.do_sampling()
        exp.compute_samples_and_ML()
        assert len(exp.samples) == n_samples
        assert exp.ML_val is not None
        exp.compute_posterior()
        assert exp.post is not None
        exp.compute_MAP()
        assert exp.MAP_val is not None
        ML_sim = exp.simulate_data(exp.ML)
        MAP_sim = exp.simulate_data(exp.MAP)
        ML_disc = exp.compute_discrepancy_with_data(exp.ML, ML_sim)
        MAP_disc = exp.compute_discrepancy_with_data(exp.MAP, MAP_sim)


@pytest.mark.skip(reason="elfi does not work yet deterministically")
def test_simple_inference_can_be_run_consistently():
    for sampling_type in ["grid", "uniform", "BO"]:
        params = BolfiParams(bounds=((0,1),(0,1)),
                             n_samples=4,
                             n_initial_evidence=2,
                             sampling_type=sampling_type,
                             grid_tics=[[0.25, 0.75], [0.33, 0.66]],
                             seed=1,
                             discrepancy_node_name="d")
        model = get_model()
        results=list()
        bf = BolfiFactory(model, params)
        for i in range(2):
            post = bf.get().run()
            results.append(post.ML[0])
        np.testing.assert_array_almost_equal(results[0], results[1])

