from .functions import (
    multiply,
    transpose,
    add,
    elementwise_multiply,
    matrix_of_zeros,
    matrix_of_ones,
    identity_matrix,
    scalar_multiply,
    scalar_power,
    scalar_add,
    exp,
)

from typing import (
    List,
    Union,
)


def cast_list_items_to_float(x: List[List[Union[int, float]]]) -> List[List[float]]:
    return [[float(x[i][j]) for j in range(len(x[0]))] for i in range(len(x))]


class Matrix:

    def __init__(self, data: List[List[Union[int, float]]]):
        if len(data) == 0 or len(data[0]) == 0:
            raise ValueError('Matrix must have at least one row and one column.')

        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError('All rows must have the same length.')

        self._data: List[List[float]] = cast_list_items_to_float(data)
        self._shape = (len(data), len(data[0]))

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def data(self) -> List[List[float]]:
        return self._data

    def __repr__(self) -> str:
        return f'Matrix({self._data})'

    def __str__(self):
        return '\n'.join(str(row) for row in self._data)

    def __getitem__(self, key: Union[int, tuple]) -> Union[float, List]:
        if isinstance(key, int):
            return self._data[key]
        elif isinstance(key, tuple):
            return self._data[key[0]][key[1]]
        else:
            raise TypeError('Invalid key type.')

    def __setitem__(self, key: Union[int, tuple], value: Union[float, List]):
        if isinstance(key, int):
            self._data[key] = value
        elif isinstance(key, tuple):
            self._data[key[0]][key[1]] = value
        else:
            raise TypeError('Invalid key type.')

    def __add__(self, other: Union[int, float, 'Matrix']) -> 'Matrix':
        if isinstance(other, float) or isinstance(other, int):
            return Matrix(scalar_add(self._data, other))
        return Matrix(add(self._data, other._data))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        if isinstance(other, float) or isinstance(other, int):
            return Matrix(scalar_add(self._data, other * -1))
        return Matrix(add(self._data, scalar_multiply(other._data, -1)))

    def __rsub__(self, other: 'Matrix') -> 'Matrix':
        return other - self

    def __mul__(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        if isinstance(other, float) or isinstance(other, int):
            return Matrix(scalar_multiply(self._data, other))
        return Matrix(elementwise_multiply(self._data, other._data))

    def __rmul__(self, other: float) -> 'Matrix':
        return self * other

    def __pow__(self, power: float) -> 'Matrix':
        return Matrix(scalar_power(self._data, power))

    def __neg__(self):
        return self * -1

    def __truediv__(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        if isinstance(other, float) or isinstance(other, int):
            return self * (other ** -1)
        return self * (other ** -1)

    def mm(self, other: 'Matrix') -> 'Matrix':
        if self.shape[1] != other.shape[0]:
            raise ValueError('Matrices cannot be multiplied.')
        return Matrix(multiply(self._data, other._data))

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        return self.mm(other)

    def transpose(self) -> 'Matrix':
        return Matrix(transpose(self._data))

    @property
    def T(self) -> 'Matrix':
        return self.transpose()

    def exp(self) -> 'Matrix':
        return Matrix(exp(self._data))

    def sum(self) -> float:
        return sum(sum(row) for row in self._data)

    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Matrix':
        return Matrix(matrix_of_zeros(rows, cols))

    @classmethod
    def ones(cls, rows: int, cols: int) -> 'Matrix':
        return Matrix(matrix_of_ones(rows, cols))

    @classmethod
    def identity(cls, size: int) -> 'Matrix':
        return Matrix(identity_matrix(size))
