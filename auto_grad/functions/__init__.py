from .operations import (
    Addition,
    Multiplication,
    Power,
)

from .activation_functions import (
    Sigmoid,
    Tanh,
    Relu,
)


def add(lhs, rhs):
    return Addition.apply(lhs, rhs)


def mul(lhs, rhs):
    return Multiplication.apply(lhs, rhs)


def pow(lhs, rhs):
    return Power.apply(lhs, rhs)


def sigmoid(x):
    return Sigmoid.apply(x)


def tanh(x):
    return Tanh.apply(x)


def relu(x):
    return Relu.apply(x)


__all__ = [
    'Addition',
    'Multiplication',
    'Power',
    'Sigmoid',
    'Tanh',
    'Relu',
    'add',
    'mul',
    'pow',
    'sigmoid',
    'tanh',
    'relu',
]
