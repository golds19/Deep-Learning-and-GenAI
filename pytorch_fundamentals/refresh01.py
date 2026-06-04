'''
understand pytorch autograd and how pytorch computes gradients
'''
import torch
import numpy as np

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

z = x * w # creates MulBackward node, stores x and w
y = z + b # creates AddBackward node, srores z and b
L = y ** 2 # creates PowBackward node, store y

L.backward()

print(x.grad)
print(w.grad)
print(b.grad)