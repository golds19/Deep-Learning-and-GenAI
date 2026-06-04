"""
Math + Pytorch implementation
"""
import numpy as np
import torch

'''
tensor creation
'''
x = torch.tensor([[1,2,3,4,5], [0,85,3,6,5]])
y = torch.tensor([[1,2,3,4,5], [0,85,3,6,5]])
z = torch.cat([x, y], dim=1)
print(x.shape)
print(y.shape)
print(z.shape)
print(z)
