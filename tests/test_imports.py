import numpy as np
import numpy.testing as npt

import optimalcontrol


def test_version() -> None:
    assert optimalcontrol.__version__ == "0.2.0"


def test_set_random_seed_controls_numpy_global_state() -> None:
    optimalcontrol.set_random_seed(1234)
    first = np.random.random(5)

    optimalcontrol.set_random_seed(1234)
    second = np.random.random(5)

    npt.assert_allclose(first, second, rtol=0.0, atol=0.0)
