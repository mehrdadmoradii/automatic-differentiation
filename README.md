# AutoGrad Library

## Introduction
AutoGrad is an automatic differentiation library tailored for educational purposes. Inspired by Andrej Karpathy's Micrograd, it offers a practical and easy-to-understand approach to automatic differentiation, a fundamental concept in machine learning. This library aims to demystify the inner workings of gradient computation and backpropagation in neural networks.

## Quick Start

### Matrix Values

AutoGrad now supports operations with matrices, enabling a wide range of mathematical computations. Below is an example illustrating basic matrix operations using AutoGrad:

```python
from autograd import AutogradMatrix


# Defining matrices
A = AutogradMatrix([
    [1, 2],
    [3, 4]
])
B = AutogradMatrix([
    [5, 6],
    [7, 8]
])
C = AutogradMatrix([
    [9, 8],
    [7, 6]
])

D = A @ B  # Matrix multiplication
E = D * C  # Element-wise multiplication
F = E + A.T  # Adding transpose of matrix A

# Backpropagation
F.start_backpropagation()

# Accessing gradients
print(A.gradient)  # Gradient of A after operations
print(B.gradient)  # Gradient of B after operations
print(C.gradient)  # Gradient of C after operations
```

### Scalar Values

```python
from autograd.value import Value

a = Value(0.2)
b = a * 3
c = b + 1
d = c ** 2
e = d / 2

e.run_backpropagation()

print(a.gradient)  # Gradient of a
print(b.gradient)  # Gradient of b
```
