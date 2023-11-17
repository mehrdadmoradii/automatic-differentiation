from auto_grad.functions import (
    Addition,
    Multiplication,
    Power,
)


class Value:

    __slots__ = ['data', 'gradient', '_calculate_gradient', '_prev']

    def __init__(self, data):
        self.data = data
        self.gradient = 0
        self._prev = set()
        self._calculate_gradient = lambda: 1

    def backward(self):
        self.gradient += self._calculate_gradient()

    def reset_gradient(self):
        self.gradient = 0

    def run_backpropagation(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node.backward()

    def add_prev(self, *prev):
        self._prev.update(prev)

    def __repr__(self):
        return f'Value({self.data})'

    def __add__(self, other):
        out = Addition.apply(self, other)
        return out

    def __mul__(self, other):
        out = Multiplication.apply(self, other)
        return out

    def __pow__(self, power):
        out = Power.apply(self, power)
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1
