import math

from typing import List


def add(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    """
    Add two matrices (lists of lists) together.

    :param x: First matrix
    :param y: Second matrix
    :return: Sum of matrices
    """
    return [[x[i][j] + y[i][j] for j in range(len(x[0]))] for i in range(len(x))]


def subtract(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    """
    Subtract one matrix (list of lists) from another.

    :param x: First matrix
    :param y: Second matrix
    :return: Difference of matrices
    """
    return [[x[i][j] - y[i][j] for j in range(len(x[0]))] for i in range(len(x))]


def elementwise_multiply(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    """
    Multiply two matrices (lists of lists) element-wise.

    :param x: First matrix
    :param y: Second matrix
    :return: Element-wise product of matrices
    """
    return [[x[i][j] * y[i][j] for j in range(len(x[0]))] for i in range(len(x))]


def multiply(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    """
    Multiply two matrices (lists of lists) together.

    :param x: First matrix
    :param y: Second matrix
    :return: Product of matrices
    """
    return [[sum(x[i][k] * y[k][j] for k in range(len(y))) for j in range(len(y[0]))] for i in range(len(x))]


def divide(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    """
    Divide one matrix (list of lists) by another.

    :param x: First matrix
    :param y: Second matrix
    :return: Quotient of matrices
    """
    return [[x[i][j] / y[i][j] for j in range(len(x[0]))] for i in range(len(x))]


def scalar_multiply(x: List[List[float]], y: float) -> List[List[float]]:
    """
    Multiply a matrix (list of lists) by a scalar.

    :param x: Matrix
    :param y: Scalar
    :return: Product of matrix and scalar
    """
    return [[x[i][j] * y for j in range(len(x[0]))] for i in range(len(x))]


def scalar_power(x: List[List[float]], y: float) -> List[List[float]]:
    """
    Raise a matrix (list of lists) to a scalar power.

    :param x: Matrix
    :param y: Scalar
    :return: Matrix raised to scalar power
    """
    return [[x[i][j] ** y for j in range(len(x[0]))] for i in range(len(x))]


def scalar_add(x: List[List[float]], y: float) -> List[List[float]]:
    """
    Add a scalar to a matrix (list of lists).

    :param x: Matrix
    :param y: Scalar
    :return: Sum of matrix and scalar
    """
    return [[x[i][j] + y for j in range(len(x[0]))] for i in range(len(x))]


def exp(x: List[List[float]]) -> List[List[float]]:
    """
    Compute the exponential of a matrix (list of lists).

    :param x: Matrix
    :return: Exponential of matrix
    """
    return [[math.exp(x[i][j]) for j in range(len(x[0]))] for i in range(len(x))]


def transpose(x: List[List[float]]) -> List[List[float]]:
    """
    Transpose a matrix (list of lists).

    :param x: Matrix to transpose
    :return: Transpose of matrix
    """
    return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]


def matrix_of_zeros(rows: int, cols: int) -> List[List[float]]:
    """
    Create a matrix (list of lists) of zeros.

    :param rows: Number of rows
    :param cols: Number of columns
    :return: Matrix of zeros
    """
    return [[0 for _ in range(cols)] for _ in range(rows)]


def matrix_of_ones(rows: int, cols: int) -> List[List[float]]:
    """
    Create a matrix (list of lists) of ones.

    :param rows: Number of rows
    :param cols: Number of columns
    :return: Matrix of ones
    """
    return [[1 for _ in range(cols)] for _ in range(rows)]


def identity_matrix(size: int) -> List[List[float]]:
    """
    Create an identity matrix (list of lists).

    :param size: Number of rows and columns
    :return: Identity matrix
    """
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]
