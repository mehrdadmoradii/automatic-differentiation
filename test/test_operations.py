import unittest

from autograd.functions import (
    Addition,
    Multiplication,
    Power,
    Sigmoid,
    Tanh,
    Relu,
)

from autograd.value import (
    Value,
)


class TestBaseOperation(unittest.TestCase):

    def setUp(self):
        self.x = Value(2)
        self.y = Value(3)

    def test_left_hand_side_type_value_required(self):
        """
        Test that the left hand side of the operation is a Value object.
        """
        with self.assertRaises(TypeError):
            Addition.apply(1, 2)

        with self.assertRaises(TypeError):
            result = Addition.apply(2, self.x)
            self.assertEqual(result.data, 4)

        result = Addition.apply(self.x, 2)
        self.assertEqual(result.data, 4)

    def test_return_type_is_value(self):
        """
        Test that the return type of the operation is a Value object.
        """
        result = Addition.apply(self.x, self.y)
        self.assertIsInstance(result, Value)

    def test_apply_method_attaches_previous_nodes_to_resulting_object(self):
        """
        Test that the apply method attaches the previous nodes to the
        resulting Value object.
        """
        result = Addition.apply(self.x, self.y)
        self.assertTrue(self.x in result._prev and self.y in result._prev)

        result = Addition.apply(self.x, 2)
        self.assertTrue(self.x in result._prev)

    def test_apply_method_attaches_calculate_gradient_function_to_previous_nodes(self):
        """
        Test that the apply method attaches the calculate gradient function to
        the previous nodes.
        """
        self.assertEqual(self.x._calculate_gradient(), 1)
        self.assertEqual(self.y._calculate_gradient(), 1)

        result = Multiplication.apply(self.x, self.y)
        expected_gradient = Multiplication.backward(self.x.data, self.y.data) * result.gradient
        self.assertEqual(self.x._calculate_gradient(), expected_gradient)
        expected_gradient = Multiplication.backward(self.y.data, self.x.data) * result.gradient
        self.assertEqual(self.y._calculate_gradient(), expected_gradient)

        result = Multiplication.apply(self.x, 2)
        expected_gradient = Multiplication.backward(self.x.data, 2) * result.gradient
        self.assertEqual(self.x._calculate_gradient(), expected_gradient)


class TestAddition(unittest.TestCase):

    def setUp(self):
        self.x = Value(2)

    def test_forward_method(self):
        """
        Test that the forward method returns the correct value.
        """
        result = Addition.forward(self.x.data, 2)
        self.assertEqual(result, 4)

    def test_backward_method(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Addition.backward(self.x.data, 2)
        h = 1e-5
        expected = (Addition.forward(self.x.data, 2 + h) - Addition.forward(self.x.data, 2)) / h
        self.assertAlmostEqual(result, expected)


class TestMultiplication(unittest.TestCase):

    def setUp(self):
        self.x = Value(3)

    def test_forward_method(self):
        """
        Test that the forward method returns the correct value.
        """
        result = Multiplication.forward(self.x.data, 2)
        self.assertEqual(result, 6)

    def test_backward_method(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Multiplication.backward(self.x.data, 2)
        h = 1e-5
        expected = (Multiplication.forward(self.x.data + h, 2) - Multiplication.forward(self.x.data, 2)) / h
        self.assertAlmostEqual(result, expected)

    def test_backward_method_with_negative_value(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Multiplication.backward(self.x.data, -2)
        h = 1e-5
        expected = (Multiplication.forward(self.x.data + h, -2) - Multiplication.forward(self.x.data, -2)) / h
        self.assertAlmostEqual(result, expected)


class TestPower(unittest.TestCase):

    def setUp(self):
        self.x = Value(2)
        self.y = Value(3)

    def test_forward_method(self):
        """
        Test that the forward method returns the correct value.
        """
        result = Power.forward(self.y.data, 2)
        self.assertAlmostEqual(result, 9)

    def test_forward_method_with_negative_value(self):
        """
        Test that the forward method returns the correct value.
        """
        result = Power.forward(self.y.data, -2)
        self.assertAlmostEqual(result, 1 / 9)

    def test_backward_method(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Power.backward(self.x.data, 2)
        h = 1e-5
        expected = (Power.forward(self.x.data + h, 2) - Power.forward(self.x.data, 2)) / h
        self.assertAlmostEqual(result, expected, places=4)

    def test_backward_method_with_negative_value(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Power.backward(self.x.data, -2)
        h = 1e-5
        expected = (Power.forward(self.x.data + h, -2) - Power.forward(self.x.data, -2)) / h
        self.assertAlmostEqual(result, expected, places=4)


class TestSigmoid(unittest.TestCase):

    def setUp(self):
        self.x = Value(2)

    def test_forward_method(self):
        """
        Test that the forward method returns the correct value.
        """
        result = Sigmoid.forward(self.x.data, None)
        self.assertAlmostEqual(result, 0.8807970779778823)

    def test_backward_method(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Sigmoid.backward(self.x.data, None)
        h = 1e-5
        expected = (Sigmoid.forward(self.x.data + h, None) - Sigmoid.forward(self.x.data, None)) / h
        self.assertAlmostEqual(result, expected, places=4)


class TestTanh(unittest.TestCase):

    def setUp(self):
        self.x = Value(2)

    def test_forward_method(self):
        """
        Test that the forward method returns the correct value.
        """
        result = Tanh.forward(self.x.data, None)
        self.assertAlmostEqual(result, 0.9640275800758169)

    def test_backward_method(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Tanh.backward(self.x.data, None)
        h = 1e-5
        expected = (Tanh.forward(self.x.data + h, None) - Tanh.forward(self.x.data, None)) / h
        self.assertAlmostEqual(result, expected, places=4)


class TestRelu(unittest.TestCase):

    def setUp(self):
        self.x = Value(2)

    def test_forward_method(self):
        """
        Test that the forward method returns the correct value.
        """
        result = Relu.forward(self.x.data, None)
        self.assertAlmostEqual(result, 2)

    def test_backward_method(self):
        """
        Test that the backward method returns the correct value.
        """
        result = Relu.backward(self.x.data, None)
        h = 1e-5
        expected = (Relu.forward(self.x.data + h, None) - Relu.forward(self.x.data, None)) / h
        self.assertAlmostEqual(result, expected, places=4)


if __name__ == '__main__':
    unittest.main()
