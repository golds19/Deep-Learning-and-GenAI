import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tiktoken
from attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257, # vocab size
    "context_length": 1024, # context length
    "embed_dim": 768,   # Embedding dimension
    "n_heads": 12,    # Number of attention heads
    "n_layers": 12,   # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # QKV values
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["embed_dim"])
        self.out_head   = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds          = self.tok_emb(in_idx)
        pos_embeds          = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x      = tok_embeds * pos_embeds
        s      = self.drop_emb(x)
        x      = self.trf_blocks(x)
        x      = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05):
        super().__init__()
        # The parameters here are just to mimic teh LayerNorm interface

    def forward(self, x): 
        # This layer does nothing and just returns its input
        return x
    

'''
implementation
'''
tokenizer = tiktoken.get_encoding('gpt2')
batch = []
txt1  = "Every effort moves you"
txt2  = "every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print(f"output shape: {logits.shape}")
print(logits)

'''
Normalizing activations with layer normalization
'''
torch.manual_seed(123)
# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2,5)

layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out   = layer(batch_example)
print(f"out: {out}")

# lets compute the mean and variance
mean     = out.mean(dim=-1, keepdim=True)
variance = out.var(dim=-1, keepdim=True)

print(f"Mean: {mean}")
print(f"Variance: {variance}")

out_norm = (out - mean) / torch.sqrt(variance)
print(f"normalized layer outputs: {out_norm}")

mean = out_norm.mean(dim=-1, keepdim=True)
var  = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)
print(f"Mean: {mean}")
print(f"var: {var}")


'''
LayerNormCLass
'''
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
ln = LayerNorm(emb_dim=6)
out_ln = ln(out)
mean = out_ln.mean(dim=-1, keepdim=True)
var  = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print(f"Mean - with LN: {mean}")
print(f"Var - with LN: {var}")


'''
Implementing a feed forward network with GELU activations
'''
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * 5 * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"])
        )

    def forward(self, x):
        return self.layers(x)
    
ffn = FeedForward(GPT_CONFIG_124M)
# input shape: [batch_size, tokens, emb_size]
x   = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

'''
Connecting attention and linear layers in a transformer block
'''

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        
        self.ff = FeedForward(cfg)
        self.norml         = LayerNorm(cfg["embed_dim"])
        self.norm2         = LayerNorm(cfg["embed_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norml(x)
        x = self.att(x) # shape [Batch, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut # the original input

        # shortcut connection for head forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    

torch.manual_seed(123)
x = torch.randn(2,4,768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print(f"block input shape: {x.shape}")
print(f"Output shape: {output.shape}")

'''
Coding the GPT model
'''
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["embed_dim"])
        self.out_head   = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds * pos_embeds # shape (batch, num_tokens, emb_size)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print(f"Input batch: {batch}")
print(f"Output shape: {out.shape}")
print(out)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # Apply softmax to get probs
        probas = torch.softmax(logits, dim=-1) # batch, vocab_size

        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # batch, 1

        # append sampled index to running sequence
        idx = torch.cat((idx, idx_next), dim=-1) # (batch, num_tokens+1)

    return idx

start_context = "Hello, i am"
encoded = tokenizer.encode(start_context)
print(f"encoded: {encoded}")

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(f"encoded tensor shape: {encoded_tensor.shape}")

model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print(f"output with simple word gen: {out}")
print(F"Output length: {len(out[0])}")

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)    
# plotting relu and gelu
# gelu, relu = nn.GELU(), nn.ReLU()

# # some sample data
# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)

# plt.figure(figsize=(8,3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
#     plt.subplot(1,2,i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)

# plt.tight_layout()
# plt.show()