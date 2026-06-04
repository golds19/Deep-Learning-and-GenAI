'''
implementing 4 variants of attention mechanisms
simplified sekf-attention
Self-attention
Causal attention
Multi-head attention
'''

import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step
)
"""
Implementing self-attention simplified
"""

'''
Step 1 computing the score between all inputs and input (2)
'''
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product between tensors

print(attn_scores_2)

'''
Step 2: compute the normalized scores - using softmax
'''

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print(f"Attention weights: {attn_weights_2_naive}")
print(f"Sum: {attn_weights_2_naive.sum()}")

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(f"attn_weight_2: {attn_weights_2}")

'''
compute the context vector z(2) by multiplying the embedded input tokens x(i) 
with attention weights and sum the resulting vector
'''

context_vec_2 = torch.zeros(query.shape[0])
for i, x_i in enumerate(inputs):
    context_vec_2 += x_i * attn_weights_2[i]

print(context_vec_2)

'''
compute the attention weights for all tokens
'''

attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

'''
simple way
'''
attn_scores = inputs @ inputs.T
print(f"attn_scores: {attn_scores}")

# attention weights
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)

# computing all context_vecs
all_context_vecs = attn_weights @ inputs
print(f"all context vecs: {all_context_vecs}")

"""
self-attention with trainable weights
"""
# jere we compute 3 vectors (query, key, value)
x_2   = inputs[1]
d_in  = inputs.shape[1] # embedding size 3
d_out = 2 # output embedding size 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

# compute the query, key, value vectors with respect to 2nd element
query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value

print(f"query_2: {query_2}")

# compute keys and values for all inputs
keys   = inputs @ W_key
values = inputs @ W_value

print(f"keys.shape: {keys.shape}")
print(f"values.shape: {values.shape}")

# next we compute the unormalized attention scores by computing the dot product between query and each key vector
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# next - we compute the attention weights
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(attn_weights_2)

# next - compute context vector with respect to inputs 2 element
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

'''
implementing a compact self attention class
'''
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key   = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value

        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec  = attn_weights @ values

        return context_vec
    

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec  = attn_weights @ values

        return context_vec
    
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(f"context vectors of inputs: {sa_v2(inputs)}")



