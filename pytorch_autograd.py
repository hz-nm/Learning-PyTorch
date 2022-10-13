# Autograd is what makes PyTorch flexible and fast for building Machine Learning projects.
# It allows for the rapid and easy computation of multiple partial derivatives also called gradients over a complex computation.
# This operation is central to backpropagation based neural network learning.

# Autograd traces your computation dynamically at runtime, hence the computation will still be traced correctly.
# Autograd tracks the history of every computation. Every computed tensor in PyTorch model carries a history of its input tensors and the function used to create it.

# ! Let's look at AUTOGRAD in practice.
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

# ? Create an input tensor
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)

# * Setting the flag means that in every computation that follows, autograd will be accumulating the history of the computation in the output tensors of that computation.
print(a)

b = torch.sin(a)

plt.plot(a.detach(), b.detach())

# * When we print the tensor 'b' we will see an indicator that the computation is being tracked.
print(b)
# This grad_fn gives us a hint that when we execute the backpropagation step and compute gradients, 
# we’ll need to compute the derivative of sin(x) for all this tensor’s inputs.
# let's perform some more computations
c = 2 * b
print(c)

d = c + 1
print(d)

out = d.sum()
print(out)

# Each grad_fn stored with our tensors allows us to walk the computation all the way back to its inputs with its
# next_functions property

print('d:')
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn)

# ! with all this machinery in place we call backward() method on the output, and check the input's grad property to inspect the gradients.
out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach())

