import numpy as np

from inclusivekinematicfit.utility import calculate_chi2_prob


def test_calculate_chi2_prob():
    ndf = 3
    test_vals = (np.array([1, 2, 3]), np.array([0.801252, 0.572407, 0.391625]))

    chi2_probs = calculate_chi2_prob(test_vals[0], ndf=ndf)

    np.testing.assert_almost_equal(chi2_probs, test_vals[1], decimal=4)
