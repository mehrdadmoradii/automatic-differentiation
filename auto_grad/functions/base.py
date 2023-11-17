import abc

from typing import Optional


def rhs_required(func):
    def wrapper(lhs, rhs):
        if rhs is None:
            raise ValueError('Right hand side is required for this operation.')
        return func(lhs, rhs)
    return wrapper


class BaseOperation(abc.ABC):

    @classmethod
    def apply(cls, lhs, rhs=None):
        from scalar import Value

        if not isinstance(lhs, Value):
            raise ValueError('Left hand side must be a Value instance.')

        if rhs is not isinstance(rhs, Value):
            res = cls.forward(lhs.data, rhs)
            output = Value(res)
            lhs._calculate_gradient = lambda: cls.backward(lhs.data, rhs) * output.gradient
            output.add_prev(lhs)
            return output

        res = cls.forward(lhs.data, rhs.data)
        output = Value(res)
        lhs._calculate_gradient = lambda: cls.backward(lhs.data, rhs.data) * output.gradient
        rhs._calculate_gradient = lambda: cls.backward(rhs.data, lhs.data) * output.gradient
        output.add_prev(lhs, rhs)
        return output

    @staticmethod
    @abc.abstractmethod
    def forward(lhs: float, rhs: Optional[float]):
        pass

    @staticmethod
    @abc.abstractmethod
    def backward(lhs: float, rhs: Optional[float]):
        pass
