from .base import (
    BaseOperation,
    rhs_required,
)

from typing import (
    Optional,
)


class Multiplication(BaseOperation):

    @staticmethod
    @rhs_required
    def forward(lhs: float, rhs: Optional[float]):
        return lhs * rhs

    @staticmethod
    @rhs_required
    def backward(lhs: float, rhs: Optional[float]):
        return rhs


class Addition(BaseOperation):

    @staticmethod
    @rhs_required
    def forward(lhs: float, rhs: Optional[float]):
        return lhs + rhs

    @staticmethod
    @rhs_required
    def backward(lhs: float, rhs: Optional[float]):
        return 1


class Power(BaseOperation):

    @staticmethod
    @rhs_required
    def forward(lhs: float, rhs: Optional[float]):
        return lhs ** rhs

    @staticmethod
    @rhs_required
    def backward(lhs: float, rhs: Optional[float]):
        return rhs * lhs ** (rhs - 1)
