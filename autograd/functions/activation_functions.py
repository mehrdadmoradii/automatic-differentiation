import math

from .base import (
    BaseOperation,
)


class Relu(BaseOperation):

    @staticmethod
    def forward(lhs: float, rhs: float):
        return max(0, lhs)

    @staticmethod
    def backward(lhs: float, rhs: float):
        return 1 if lhs > 0 else 0


class Sigmoid(BaseOperation):

    @staticmethod
    def forward(lhs: float, rhs: float):
        return 1 / (1 + math.exp(-lhs))

    @staticmethod
    def backward(lhs: float, rhs: float):
        return (1 / (1 + math.exp(-lhs))) * (1 - (1 / (1 + math.exp(-lhs))))


class Tanh(BaseOperation):

    @staticmethod
    def forward(lhs: float, rhs: float):
        return math.tanh(lhs)

    @staticmethod
    def backward(lhs: float, rhs: float):
        epsilon = 1e-10
        derivative = 1 - math.tanh(lhs) ** 2
        return max(derivative, epsilon)  # Ensures derivative is never zero
