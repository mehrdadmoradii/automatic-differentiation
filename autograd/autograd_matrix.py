from .matrix import Matrix

from .matrix import autograd_functions as F

from typing import (
    List,
    Union,
)


class AutogradMatrix(Matrix):

    def __init__(self, data: List[List[Union[int, float]]]):

        super().__init__(data)
        self._grad = Matrix.zeros(*self.shape)
        self._calculate_grad = lambda: Matrix.ones(*self.shape)
        self._previous_nodes = set()

    def backward(self):
        self._grad += self._calculate_grad()

    def start_backpropagation(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._previous_nodes:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node.backward()

    def reset_grad(self):
        self._grad = Matrix.zeros(*self.shape)

    @property
    def grad(self):
        return self._grad

    def add_prev(self, *prev):
        self._previous_nodes.update(prev)

    def reset_grad(self):
        self._grad = Matrix.ones(*self.shape)

    def __mul__(self, other):
        return F.elementwise_multiply(self, other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return F.add(self, other)

    def __radd__(self, other):
        return self + other

    def __matmul__(self, other):
        return F.matmul(self, other)

    def __pow__(self, power):
        return F.power(self, power)

    def __truediv__(self, other):
        return F.divide(self, other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self) -> 'Matrix':
        return F.exp(self)

    def sum(self):
        return super().sum()
