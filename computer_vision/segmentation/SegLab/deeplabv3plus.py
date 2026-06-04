"""
Implement the DeepLabV3plus model
"""
import torch
import torch.nn as nn


"""
Depthwise separable convolution
Encoder-decoder with Atrous Convolution
"""