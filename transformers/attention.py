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

'''
Applying causal attention mask
'''
# implementing a compact causal self attention class
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out           = d_out
        self.W_query         = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key           = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value         = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.droput          = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores  = queries @ keys.transpose(1,2)
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.droput(attn_weights)
        context_vec  = attn_weights @ values

        return context_vec
    
torch.manual_seed(123)
context_length = batch.shape[1]
ca           = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)

print(f"context_vecs with causal attention: {context_vecs}")
print(f"context_vecs.shape: {context_vecs.shape}")

# Extending single-head attention layers
'''
Multi-head attention
'''

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self,x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out    = 3,2
mha            = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(f"context_vecs with mha: {context_vecs}")
print(f"context_ves shape: {context_vecs.shape}")


'''
multi-head attention with weight splits
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads # reduce the projection dim to match desired output dim

        self.W_query         = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key           = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value         = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj        = nn.Linear(d_out, d_out)
        self.droput          = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys    = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values  = values.transpose(1,2)

        # compute scaled dot-product attention
        attn_scores = queries @ keys.transpose(2,3) # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # original mask truncated to the number of tokens
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.droput(attn_weights)

        # shape (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2)

        # combine heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
    

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(f"MHA context_vecs: {context_vecs}")
print(f"context_vecs shape - mha: {context_vecs.shape}")


'''
MultiheadAttention using with Pytorch scaled dot product and Flashattention
'''
class MHAPytorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.num_heads      = num_heads
        self.context_length = context_length
        self.head_dim       = d_out // num_heads
        self.d_out          = d_out

        self.qkv            = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj           = nn.Linear(d_in, d_out)
        self.dropout        = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) -> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) -> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) -> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2,0,3,1,4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )

        # combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec