import abc

from .matrix import Matrix


def add(x, y):
    return Addition.apply(x, y)


def elementwise_multiply(x, y):
    return Multiplication.apply(x, y)


def matmul(x, y):
    return MatrixMultiply.apply(x, y)


def power(x, y):
    return Power.apply(x, y)


def divide(x, y):
    return Division.apply(x, y)


def exp(x):
    return Exp.apply(x)


class BaseFunction(abc.ABC):

    commutative: bool

    @classmethod
    def apply(cls, x, y=None):
        from autograd import AutogradMatrix

        if not isinstance(x, AutogradMatrix):
            raise TypeError('Left hand side must be an AutogradMatrix instance.')

        if y is None or isinstance(y, int) or isinstance(y, float):
            res = cls.forward(x.data, y)
            output = AutogradMatrix(res.data)
            x._calculate_grad = lambda: cls.backward(x.data, y, output.grad)
            output.add_prev(x)
            return output

        res = cls.forward(x.data, y.data)
        output = AutogradMatrix(res.data)
        x._calculate_grad = lambda: cls.backward(x.data, y.data, output.grad)
        y._calculate_grad = lambda: cls.backward(y.data, x.data, output.grad)
        output.add_prev(x, y)
        return output

    @staticmethod
    def forward(x, y=None):
        pass

    @staticmethod
    def backward(x, y, output_grad):
        pass


class Addition(BaseFunction):

    commutative = True

    @staticmethod
    def forward(x, y=None):
        if isinstance(y, int) or isinstance(y, float):
            return Matrix(x) + (Matrix.ones(*Matrix(x).shape) * y)
        return Matrix(x) + Matrix(y)

    @staticmethod
    def backward(x, y, output_grad):
        return output_grad * Matrix.ones(len(x), len(x[0]))


class Multiplication(BaseFunction):

    commutative = True

    @staticmethod
    def forward(x, y):
        if isinstance(y, int) or isinstance(y, float):
            return Matrix(x) * (Matrix.ones(*Matrix(x).shape) * y)
        return Matrix(x) * Matrix(y)

    @staticmethod
    def backward(x, y, output_grad):
        return Matrix(y) * output_grad


class MatrixMultiply(BaseFunction):

    commutative = False

    @classmethod
    def apply(cls, x, y):
        from autograd import AutogradMatrix

        if not isinstance(x, AutogradMatrix) and not isinstance(y, AutogradMatrix):
            raise TypeError('Both left and right hand sides must be an AutogradMatrix instance.')

        res = Matrix(x.data) @ Matrix(y.data)
        output = AutogradMatrix(res.data)
        x._calculate_grad = lambda: output.grad @ y.T
        y._calculate_grad = lambda: x.T @ output.grad
        output.add_prev(x, y)
        return output


class Power(BaseFunction):

    commutative = False

    @classmethod
    def apply(cls, x, y):
        from autograd import AutogradMatrix

        if not isinstance(x, AutogradMatrix):
            raise TypeError('Left hand side must be an AutogradMatrix instance.')

        if not isinstance(y, int) and not isinstance(y, float):
            raise TypeError('Right hand side must be an int or float.')

        res = Matrix(x.data) ** y
        output = AutogradMatrix(res.data)
        x._calculate_grad = lambda: (y * (x ** (y - 1))) * output.grad
        output.add_prev(x)
        return output


class Division(BaseFunction):

    commutative = False

    @classmethod
    def apply(cls, x, y):
        from autograd import AutogradMatrix

        if not isinstance(x, AutogradMatrix):
            raise TypeError('Left hand side must be an AutogradMatrix instance.')

        if not isinstance(y, int) and not isinstance(y, float):
            raise TypeError('Right hand side must be an int or float.')

        res = Matrix(x.data) / y
        output = AutogradMatrix(res.data)
        x._calculate_grad = lambda: (1/y) * output.grad
        output.add_prev(x)
        return output


class Exp(BaseFunction):

    commutative = False

    @staticmethod
    def forward(x, y=None):
        return Matrix(x).exp()

    @staticmethod
    def backward(x, y, output_grad):
        return output_grad * Matrix(x).exp()
