import pytest
import numpy as np

from inclusivekinematicfit.covariance import (
    number_of_off_diagonal_values,
    create_symmetric_matrix,
)


@pytest.mark.parametrize("test_input,expected", [(2, 1), (3, 3), (4, 6)])
def test_number_of_off_diagonal_values(test_input, expected):
    assert number_of_off_diagonal_values(test_input) == expected


def test_create_symmetric_matrix():
    m = create_symmetric_matrix(3, np.array([1, 2, 3]), np.array([4, 5, 6]))
    true_m = np.array([[1, 4, 5], [4, 2, 6], [5, 6, 3]])
    np.testing.assert_array_equal(m, true_m)

    m = create_symmetric_matrix(2, np.array([1, 3]), np.array([5]))
    true_m = np.array([[1, 5], [5, 3]])
    np.testing.assert_array_equal(m, true_m)

    with pytest.raises(ValueError):
        m = create_symmetric_matrix(4, np.array([1, 2, 3]), np.array([4, 5, 6]))
        m = create_symmetric_matrix(4, np.array([1, 2, 3, 4]), np.array([4, 5, 6]))
        m = create_symmetric_matrix(3, np.array([[1, 2, 3]]), np.array([4, 5, 6]))
        m = create_symmetric_matrix(3, np.array([1, 2, 3]), np.array([[4, 5, 6]]))
