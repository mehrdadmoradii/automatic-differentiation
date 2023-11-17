# AutoGrad Library

## Introduction
AutoGrad is an automatic differentiation library tailored for educational purposes. Inspired by Andrej Karpathy's Micrograd, 
it offers a practical and easy-to-understand approach to automatic differentiation, a fundamental concept in machine learning. 
This library aims to demystify the inner workings of gradient computation and backpropagation in neural networks,

## Quick Start

```python
from auto_grad.value import Value

a = Value(0.2)
b = a * 3
c = b + 1
d = c ** 2
e = d / 2

e.run_backpropagation()

print(a.gradient)  # Gradient of a
print(b.gradient)  # Gradient of b
```
