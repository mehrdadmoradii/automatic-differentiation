import unittest

from autograd import AutogradMatrix

from autograd.matrix import autograd_functions as F


def eval_numerical_gradient_array_for_x(x, y, f, h=1e-5):
    grad = AutogradMatrix.zeros(*x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            old_value = x[i][j]
            x[i][j] = old_value + h
            pos = f(x, y)
            x[i][j] = old_value - h
            neg = f(x, y)
            x[i][j] = old_value

            current_grad = (pos - neg)
            current_grad = current_grad / (2 * h)
            grad[i][j] = current_grad.sum()

    return grad


def eval_numerical_gradient_array_for_y(x, y, f, h=1e-5):
    grad = AutogradMatrix.zeros(*y.shape)

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            old_value = y[i][j]
            y[i][j] = old_value + h
            pos = f(x, y)
            y[i][j] = old_value - h
            neg = f(x, y)
            y[i][j] = old_value

            current_grad = (pos - neg)
            current_grad = current_grad / (2 * h)
            grad[i][j] = current_grad.sum()

    return grad


class TestOperations(unittest.TestCase):

    def setUp(self):
        self.x = AutogradMatrix([
            [1, 2],
            [3, 4],
        ])

        self.y = AutogradMatrix([
            [5, 6],
            [7, 8],
        ])

        self.h = 1e-5

    def matrices_almost_equal(self, expected, actual):
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i][j], actual[i][j])

    def test_addition_against_numerical_gradient(self):
        z = self.x + self.y
        z._grad = AutogradMatrix.ones(*z.shape)
        self.x.backward()
        x2, y2 = AutogradMatrix(self.x.data), AutogradMatrix(self.y.data)
        numerical_grad = eval_numerical_gradient_array_for_x(x2, y2, F.Addition.apply)
        self.matrices_almost_equal(numerical_grad, self.x.grad)

    def test_mul_against_numerical_gradient(self):
        z = self.x * self.y
        z._grad = AutogradMatrix.ones(*z.shape)
        self.x.backward()
        x2, y2 = AutogradMatrix(self.x.data), AutogradMatrix(self.y.data)
        numerical_grad = eval_numerical_gradient_array_for_x(x2, y2, F.Multiplication.apply)
        self.matrices_almost_equal(numerical_grad, self.x.grad)

    def test_matmul_against_numerical_gradient(self):
        z = self.x @ self.y
        z._grad = AutogradMatrix.ones(*z.shape)
        self.x.backward()
        self.y.backward()

        x2, y2 = AutogradMatrix(self.x.data), AutogradMatrix(self.y.data)
        numerical_grad = eval_numerical_gradient_array_for_x(x2, y2, F.MatrixMultiply.apply)
        self.matrices_almost_equal(numerical_grad, self.x.grad)

        x2, y2 = AutogradMatrix(self.x.data), AutogradMatrix(self.y.data)
        numerical_grad = eval_numerical_gradient_array_for_y(x2, y2, F.MatrixMultiply.apply)
        self.matrices_almost_equal(numerical_grad, self.y.grad)

    def test_power_against_numerical_gradient(self):
        z = self.x ** 2
        z._grad = AutogradMatrix.ones(*z.shape)
        self.x.backward()
        x2 = AutogradMatrix(self.x.data)
        numerical_grad = eval_numerical_gradient_array_for_x(x2, 2, F.Power.apply)
        self.matrices_almost_equal(numerical_grad, self.x.grad)

    def test_div_against_numerical_gradient(self):
        z = self.x / 3
        z._grad = AutogradMatrix.ones(*z.shape)
        self.x.backward()
        x2 = AutogradMatrix(self.x.data)
        numerical_grad = eval_numerical_gradient_array_for_x(x2, 3, F.Division.apply)
        self.matrices_almost_equal(numerical_grad, self.x.grad)

    def test_exp_against_numerical_gradient(self):
        z = self.x.exp()
        z._grad = AutogradMatrix.ones(*z.shape)
        self.x.backward()
        x2 = AutogradMatrix(self.x.data)
        numerical_grad = eval_numerical_gradient_array_for_x(x2, None, F.Exp.apply)
        self.matrices_almost_equal(numerical_grad, self.x.grad)


class TestBackprop(unittest.TestCase):

    def setUp(self):
        self.x = AutogradMatrix([
            [1, 2, 3, 4],
        ])

        self.w1 = AutogradMatrix([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ])

        self.b1 = AutogradMatrix([
            [0.1, 0.2, 0.3],
        ])

        self.w2 = AutogradMatrix([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ])

        self.b2 = AutogradMatrix([
            [0.1, 0.2],
        ])

        self.h = 1e-5

    def matrices_almost_equal(self, expected, actual):
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i][j], actual[i][j])

    def test_simple(self):
        z = self.x @ self.w1
        z.start_backpropagation()

        x_copy, w1_copy = AutogradMatrix(self.x.data), AutogradMatrix(self.w1.data)

        def f(x, y):
            return x @ y

        numerical_grad = eval_numerical_gradient_array_for_x(x_copy, w1_copy, f)
        self.matrices_almost_equal(numerical_grad, self.x.grad)
        numerical_grad = eval_numerical_gradient_array_for_y(x_copy, w1_copy, f)
        self.matrices_almost_equal(numerical_grad, self.w1.grad)

    def test_multilayer_expression(self):
        z = self.x @ self.w1 + self.b1
        z = z @ self.w2 + self.b2
        z.start_backpropagation()

        x_copy, w1_copy, b1_copy, w2_copy, b2_copy = AutogradMatrix(self.x.data), AutogradMatrix(self.w1.data), \
                                                     AutogradMatrix(self.b1.data), AutogradMatrix(self.w2.data), \
                                                     AutogradMatrix(self.b2.data)

        def f(x, y):
            z = x_copy @ w1_copy + b1_copy
            z = z @ x + y
            return z

        numerical_grad = eval_numerical_gradient_array_for_x(w2_copy, b2_copy, f)
        self.matrices_almost_equal(numerical_grad, self.w2.grad)
        numerical_grad = eval_numerical_gradient_array_for_y(w2_copy, b2_copy, f)
        self.matrices_almost_equal(numerical_grad, self.b2.grad)

        def f(x, y):
            z = x_copy @ x + y
            z = z @ w2_copy + b2_copy
            return z

        numerical_grad = eval_numerical_gradient_array_for_x(w1_copy, b1_copy, f)
        self.matrices_almost_equal(numerical_grad, self.w1.grad)
        numerical_grad = eval_numerical_gradient_array_for_y(w1_copy, b1_copy, f)
        self.matrices_almost_equal(numerical_grad, self.b1.grad)

        def f(x, y):
            z = x @ w1_copy + b1_copy
            z = z @ w2_copy + y
            return z

        numerical_grad = eval_numerical_gradient_array_for_x(x_copy, b2_copy, f)
        self.matrices_almost_equal(numerical_grad, self.x.grad)


if __name__ == '__main__':
    unittest.main()
