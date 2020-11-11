"""This module contains utility code to create and handle covariance matrices used for the fit.
"""
from typing import List, Callable, Dict, Tuple

import numpy as np
import pandas as pd

from inclusivekinematicfit.utility import rms


__all__ = [
    "number_of_off_diagonal_values",
    "create_symmetric_matrix",
    "get_covariance_by_phase_space",
    "get_rms_diag_error_matrix",
    "get_rms_error_matrix",
    "get_gaussian_cov_matrix",
    "get_gaussian_diag_cov_matrix",
]


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


def get_covariance_by_phase_space(
    df: pd.DataFrame,
    phase_space_bin_columns: List[str],
    cov_mat_estimator: Callable[[pd.DataFrame], np.ndarray],
) -> Dict[Tuple[pd.Interval, ...], np.ndarray]:
    cov_mat_dict = dict()

    df_grouped = df.groupby(phase_space_bin_columns)
    for group_name, group_df in df_grouped:
        cov_mat_dict[group_name] = cov_mat_estimator(group_df)

    return cov_mat_dict


def get_rms_diag_error_matrix(
    df: pd.DataFrame, four_momentum_columns: List[str]
) -> np.ndarray:
    return (
        np.diag([rms(df.loc[:, x_var]) for x_var in four_momentum_columns]).astype(
            np.float64
        )
        ** 2
    )


def get_rms_error_matrix(
    df: pd.DataFrame, four_momentum_columns: List[str]
) -> np.ndarray:
    diag_rms = np.sqrt((get_rms_diag_error_matrix(df, four_momentum_columns)))
    correlation = np.corrcoef(df.loc[:, four_momentum_columns], rowvar=False)
    return diag_rms @ correlation @ diag_rms


def get_gaussian_cov_matrix(
    df: pd.DataFrame, four_momentum_columns: List[str]
) -> np.ndarray:
    return np.cov(df.loc[:, four_momentum_columns], rowvar=False)


def get_gaussian_diag_cov_matrix(
    df: pd.DataFrame, four_momentum_columns: List[str]
) -> np.ndarray:
    return np.diag(np.diag(get_gaussian_cov_matrix(df, four_momentum_columns)))
