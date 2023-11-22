import unittest

from autograd.value import Value


class TestValueClass(unittest.TestCase):

    def setUp(self):
        self.x = Value(2)
        self.y = Value(3)
        self.h = 1e-4

    def test_data_attribute(self):
        """
        Test that the data attribute returns the correct value.
        """
        self.assertEqual(self.x.data, 2)
        self.assertEqual(self.y.data, 3)

    def test_initial_gradient_attribute(self):
        """
        Test that the gradient attribute returns the correct value.
        """
        self.assertEqual(self.x.gradient, 0)
        self.assertEqual(self.y.gradient, 0)

    def test_initial_prev_attribute(self):
        """
        Test that the prev attribute returns the correct value.
        """
        self.assertEqual(self.x._prev, set())
        self.assertEqual(self.y._prev, set())

    def test_initial_calculate_gradient_attribute(self):
        """
        Test that the _calculate_gradient attribute returns the correct value.
        """
        self.assertEqual(self.x._calculate_gradient(), 1)
        self.assertEqual(self.y._calculate_gradient(), 1)

    def test_add_prev_method(self):
        """
        Test that the add_prev method returns the correct value.
        """
        a, b, c = Value(2), Value(3), Value(4)
        self.x.add_prev(a, b, c)
        self.assertEqual(self.x._prev, {a, b, c})

    def test_backward_method(self):
        """
        Test that the backward method returns the correct value.
        """
        a = Value(2)
        a._calculate_gradient = lambda: 23
        a.backward()
        self.assertEqual(a.gradient, 23)

        a.backward()
        self.assertEqual(a.gradient, 46)    # gradient should be accumulated

    def test_reset_gradient_method(self):
        """
        Test that the reset_gradient method returns the correct value.
        """
        a = Value(2)
        a._calculate_gradient = lambda: 23
        a.backward()
        self.assertEqual(a.gradient, 23)
        a.reset_gradient()
        self.assertEqual(a.gradient, 0)

    def test_simple_backpropagation(self):
        """
        Test backpropagation in a simple computation graph.
        """
        a = Value(0.2)
        b = a * 3
        c = b + 1
        d = c ** 2
        e = d / 2

        e.run_backpropagation()

        # Check the gradient at the end of the graph
        self.assertEqual(e.gradient, 1)

        self.assertAlmostEqual(d.gradient, 0.5)
        self.assertAlmostEqual(c.gradient, d.gradient * 2 * c.data)
        self.assertAlmostEqual(b.gradient, c.gradient)
        self.assertAlmostEqual(a.gradient, b.gradient * 3)

    def test_numerical_differentiation(self):
        """
        Test backpropagation against numerical differentiation.
        """
        a = Value(0.2)
        b = a * 3
        c = b + 1
        d = c ** 2
        res = d / 2

        res.run_backpropagation()
        self.assertAlmostEqual(res.gradient, 1)

        a1 = Value(0.2)
        b1 = a1 * 3
        c1 = b1 + 1
        d1 = (c1 ** 2) + self.h
        res1 = d1 / 2

        res1.run_backpropagation()
        expected_gradient = (res1.data - res.data) / self.h
        self.assertAlmostEqual(d1.gradient, expected_gradient, places=3)

        a2 = Value(0.2)
        b2 = a2 * 3
        c2 = b2 + 1 + self.h
        d2 = (c2 ** 2)
        res2 = d2 / 2

        res2.run_backpropagation()
        expected_gradient = (res2.data - res.data) / self.h
        self.assertAlmostEqual(c2.gradient, expected_gradient, places=3)

        a3 = Value(0.2)
        b3 = a3 * 3 + self.h
        c3 = b3 + 1
        d3 = (c3 ** 2)
        res3 = d3 / 2

        res3.run_backpropagation()
        expected_gradient = (res3.data - res.data) / self.h
        self.assertAlmostEqual(b3.gradient, expected_gradient, places=3)

        a4 = Value(0.2) + self.h
        b4 = a4 * 3
        c4 = b4 + 1
        d4 = (c4 ** 2)
        res4 = d4 / 2

        res4.run_backpropagation()
        expected_gradient = (res4.data - res.data) / self.h
        self.assertAlmostEqual(a4.gradient, expected_gradient, places=3)


if __name__ == '__main__':
    unittest.main()
