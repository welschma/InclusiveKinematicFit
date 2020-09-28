"""This module contains utility code to create and handle covariance matrices used for the fit.
"""
import numpy as np


def number_of_off_diagonal_values(dimension: int) -> int:
    """Calculates the number of off diagonal elements
     in the upper triangle of the matrix.

    :param dimension: Dimension of the matrix.
    :type dimension: int
    :return: Number of off diagonal elements in upper matrix triangle.
    :rtype: int
    """
    return ((dimension * dimension) - dimension) // 2


def create_symmetric_matrix(
    dimension: int, diag_values: np.ndarray, off_diag_values: np.ndarray
) -> np.ndarray:
    """Creates a symmetric matrix from given diagonal and
    off diagonal values.

    :param dimension: Dimension of the matrix.
    :type dimension: int
    :param diag_values: On diagonal values of the matrix
    :type diag_values: np.ndarray
    :param off_diag_values: [description]
    :type off_diag_values: np.ndarray
    """
    m = np.zeros((dimension, dimension), dtype=np.double)

    if len(diag_values.shape) != 1 or len(off_diag_values.shape) != 1:
        raise ValueError("Shape of given matrix elements has to be 1-dimensional")

    if diag_values.size != dimension:
        raise ValueError(
            "Number of given diagonal values not compatible with matrix dimension"
        )

    if off_diag_values.size != number_of_off_diagonal_values(dimension):
        raise ValueError(
            "Number of given off diagonal values not compatible with matrix dimension"
        )

    row_diag, column_diag = np.diag_indices(dimension)
    row_off_diag, column_off_diag = np.triu_indices(dimension, k=1)

    m[row_diag, column_diag] = diag_values
    m[row_off_diag, column_off_diag] = off_diag_values
    m[column_off_diag, row_off_diag] = off_diag_values

    return m
