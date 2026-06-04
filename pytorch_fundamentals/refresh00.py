"""
Math + Pytorch implementation
"""
import numpy as np
import torch

'''
tensor creation
'''
# x = torch.tensor([[1,2,3,4,5], [0,85,3,6,5]])
# y = torch.tensor([[1,2,3,4,5], [0,85,3,6,5]])
# w = torch.tensor([[10,11,12,13,14], [10,11,12,13,14]])
# z = torch.cat([x, y, w], dim=0) # dim=0 across the the 0th dimension
# print(x.shape)
# print(y.shape)
# print(z.shape)
# print(z)

'''
Reshaping pytorch tensors
'''

x = torch.tensor([0,1,2,3,4,5,6,7,8,9])
print(x)

'''
reshape
'''
y = x.reshape((2,5)) # reshape a tensor into another tensor shape
print(y)

'''
flatten
'''
a = torch.tensor([[1,2,3,4,5,6,7,8],
                  [1,2,3,4,5,6,7,8]])

b = torch.randn(1,4,8)

a_flatten = torch.flatten(a, 1)
b_flatten = torch.flatten(b, 1)
print(a_flatten)
print(b_flatten.shape)


'''
using view
view is used to change a tensor in 2 dim format.
Example, rows and columns
'''

c = torch.FloatTensor([24, 56, 10, 20, 30, 40, 50, 1, 2, 3, 4, 5])

print(c.view(4,3))
print(c.view(3,4))
print(c.view(2,6))

'''
using squeeze and unsqueeze
'''

d = torch.tensor([[1,2,3,4], [1,2,3,4]])
print(d.shape)

d_unsqueeze = torch.unsqueeze(d, dim=0)
print(d_unsqueeze.shape)


# squeeze on dim 2
d_squeeze = torch.squeeze(d_unsqueeze, dim=1)
print(d_squeeze.shape)

'''
tensor contiguity
'''
j = torch.tensor([[1,2,3,4],
                  [1,2,3,4]])

y = j.T

y = y.contiguous()
print(y.view(-1)) # throws error since, y is not contiguous (logical view doesnt matchb physical memory)

'''
expand and repeat
'''
x = torch.tensor([[1,2,3], [2,3,4]])

x_repeated = x.repeat(2, 1) # tensor is repeated twice along the first dimension

y = torch.tensor([1,2,3])

# adds a dimension using unsqueeze and repeat
y_unsqueeze = y.unsqueeze(0) # dim=0

y_unsqueeze_repeat = y_unsqueeze.repeat(2,1)
print(y_unsqueeze_repeat)