# PyTorch Refresher → DL → LLM → VLM
## A focused plan for engineers coming back to the foundations

---

> **Core philosophy**: Learn each concept as a chain —
> `tensor operation → mathematical meaning → model component → training behavior → debugging skill`
>
> Do not learn `nn.Embedding` as syntax. Learn it as a learnable lookup table,
> trained via backprop, that maps token IDs to dense vectors, and whose weights
> are often tied to the output head in LLMs. That is the level of understanding you want throughout.

---

## How to use this plan

- Run **math and code in parallel** — do not finish all the math before opening PyTorch. You solidify the math by seeing it in code.
- Each phase ends with a **mini project**. Do not skip them. Projects are where understanding becomes skill.
- Sections marked **[defer]** are real topics you will need eventually — but not during this refresher. Come back to them when you hit them in practice.

---

## Phase 1 — PyTorch Core

**Goal**: Rebuild muscle memory with tensors, gradients, and the training loop. Everything in LLMs and VLMs is built on top of this.

---

### 1.1 Tensors

The primary data structure for everything you will do.

| Topic | Why it matters |
|---|---|
| Tensor creation, shape, dtype, device | Core PyTorch object |
| Indexing and slicing | Data selection |
| Reshaping: `reshape`, `view`, `permute`, `transpose` | Model input/output formatting |
| `unsqueeze`, `squeeze`, `expand`, `repeat` | Broadcast-compatible shapes |
| Tensor contiguity | Why `.view()` sometimes fails |

Deep understanding goal:

> Given any tensor shape, predict the output shape after every operation.
> This skill becomes critical for attention, CNNs, and VLMs.

---

### 1.2 Broadcasting

One of the most important tensor concepts. Essential for attention masks, normalization, and loss computation.

```python
x    = torch.randn(32, 128)   # [batch, features]
bias = torch.randn(128)        # [features]
y    = x + bias                # bias broadcasts across batch
```

Be comfortable with shapes like:

```
[batch, seq_len, hidden] + [hidden]
[batch, heads, seq_len, seq_len] + [batch, 1, 1, seq_len]
```

---

### 1.3 Autograd

You must understand what PyTorch is actually doing when it computes gradients.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # 4.0
```

| Topic | Why it matters |
|---|---|
| `requires_grad`, computational graph | How PyTorch records operations |
| `.backward()`, `.grad` | Gradient computation and storage |
| `torch.no_grad()` | Inference without graph overhead |
| `.detach()` | Stop gradient flow deliberately |
| Leaf vs non-leaf tensors | Common source of gradient confusion |

Critical pattern — know exactly why this order matters:

```python
optimizer.zero_grad()   # clear old gradients
loss.backward()         # compute new gradients
optimizer.step()        # update weights
```

---

### 1.4 `nn.Module`

The backbone of every PyTorch model.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)
```

| Topic | Why it matters |
|---|---|
| `__init__` and `forward` | Structure of every model |
| `model.parameters()` | What gets passed to the optimizer |
| `model.train()` vs `model.eval()` | Controls dropout and batch norm behavior |
| `torch.no_grad()` | Disables gradient tracking — not the same as eval |

These three are not interchangeable:

```python
model.train()       # sets training mode — affects dropout, batch norm
model.eval()        # sets eval mode — disables dropout, uses running stats
torch.no_grad()     # disables gradient computation — saves memory
```

---

### 1.5 Essential layers

Do not just use these. Understand what each one does mathematically.

| Layer | What it does |
|---|---|
| `nn.Linear` | Matrix multiplication + bias |
| `nn.Embedding` | Learnable lookup table: token ID → dense vector |
| `nn.LayerNorm` | Normalizes across features — standard in transformers |
| `nn.BatchNorm2d` | Normalizes across batch — standard in CNNs |
| `nn.Dropout` | Zeroes random activations during training |
| `nn.Conv2d` | Spatial feature extraction |
| `nn.MultiheadAttention` | Transformer foundation |
| `nn.Sequential`, `nn.ModuleList` | Model composition |

---

### 1.6 Loss functions

| Loss | Used for |
|---|---|
| `nn.MSELoss` | Regression |
| `nn.CrossEntropyLoss` | Multi-class classification, language modeling |
| `nn.BCEWithLogitsLoss` | Binary classification |

Critical distinction:

```python
# Correct — CrossEntropyLoss expects raw logits
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, labels)

# Wrong — do not apply softmax before CrossEntropyLoss
probs = torch.softmax(model(x), dim=-1)
loss = nn.CrossEntropyLoss()(probs, labels)
```

Know the difference between: logits, probabilities, class indices, one-hot labels.

---

### 1.7 Optimizers and schedulers

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

| Optimizer | When to use |
|---|---|
| SGD | Classical baseline |
| Adam | General purpose |
| AdamW | Default for transformers |

Essential scheduler concepts:

| Concept | Why it matters |
|---|---|
| Warmup | Prevents unstable early updates |
| Cosine decay | Smooth learning rate reduction |
| Weight decay | Regularization baked into AdamW |
| Gradient clipping | Prevents exploding gradients in deep nets |
| Gradient accumulation | Simulates larger batch with limited memory |

---

### 1.8 Dataset and DataLoader

```python
class CustomDataset(Dataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

| Topic | Why it matters |
|---|---|
| `__len__` and `__getitem__` | Interface the DataLoader expects |
| `batch_size`, `shuffle` | Training stability and generalization |
| `num_workers` | Parallelizes data loading |
| `collate_fn` | Handles variable-length sequences — critical for LLMs |
| Padding, truncation, attention masks | Core of LLM input pipelines |

---

### 1.9 Training loop

You should be able to write this from scratch without copying.

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        val_loss = criterion(logits, y)
```

---

### 1.10 Device management

```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
x = x.to(device)
```

The most common error at this stage:

```
RuntimeError: Expected all tensors to be on the same device
```

Debug it immediately by printing `x.device` and `model.parameters().__next__().device`.

---

### 1.11 Saving and loading

```python
# Save
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "loss": loss,
}, "checkpoint.pt")

# Load
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

Always save `state_dict`, not the full model object.

---

### 1.12 Debugging toolkit

Build these habits now. They save hours later.

```python
print(x.shape, x.dtype, x.device)
print(loss.item())
print(torch.isnan(loss))

for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm().item())
```

| Problem | Common cause |
|---|---|
| Loss is NaN | LR too high, bad data, log of zero |
| Loss does not decrease | Wrong labels, frozen params, bad LR |
| GPU OOM | Batch too large, sequence too long, graph not freed |
| Accuracy stuck | Output/loss shape mismatch, eval mode not set |
| Training slow | DataLoader bottleneck, data staying on CPU |

---

### Phase 1 Mini Project — MLP from scratch

**Dataset**: MNIST or any tabular dataset (e.g. Titanic, California Housing).

**What to build**:

```
CustomDataset → DataLoader → MLP with nn.Linear + ReLU + Dropout
→ CrossEntropyLoss → AdamW → training loop → validation loop
→ checkpoint saving → accuracy + loss plots
```

**Requirements**:
- Write the training loop from scratch — no Trainer, no Lightning
- Implement proper `model.train()` / `model.eval()` switching
- Log train loss and val loss per epoch and plot them
- Save and reload a checkpoint, verify the loss matches

**What you will discover**:
- Why `optimizer.zero_grad()` must come before `loss.backward()`
- How batch size affects training stability
- What overfitting looks like in a loss curve
- Why val loss diverging from train loss is the key signal

---

## Phase 2 — Deep Learning Foundations

**Goal**: Understand why modern architectures are built the way they are. CNN, normalization, residuals, and attention are the direct building blocks of VLMs.

---

### 2.1 Math grounded in code

Run this in parallel with phase 1 and 2 — not as a prerequisite.

**Linear algebra — what you actually need**:

| Topic | Where it appears in PyTorch |
|---|---|
| Matrix multiplication | `nn.Linear`, attention (`Q @ K.T`) |
| Dot product | Attention scores, cosine similarity |
| Norms | Gradient clipping, LayerNorm, embeddings |
| Transpose, reshape, broadcast | Everywhere |

**Calculus — what you actually need**:

| Topic | Where it appears |
|---|---|
| Chain rule | Backpropagation |
| Partial derivatives | Gradients through multi-input ops |
| Gradient direction | Why weights move the way they do |
| Vanishing/exploding gradients | Why residuals and normalization exist |

**Probability — what you actually need**:

| Topic | Where it appears |
|---|---|
| Probability distributions | Softmax output, generative models |
| Cross-entropy | Classification loss, language modeling loss |
| Maximum likelihood | What cross-entropy is actually optimizing |
| Sampling | Text generation (temperature, top-k, top-p) |

Defer KL divergence, eigenvalues, SVD, and Bayes rule to the LLM/VLM phase where they are motivated by concrete use cases.

---

### 2.2 CNN fundamentals

Important for VLMs — vision encoders are either CNNs or ViTs, and patch embeddings borrow from both.

| Topic | Why it matters |
|---|---|
| `nn.Conv2d` — filters, stride, padding | Spatial feature extraction |
| Channels and feature maps | How visual information is represented |
| Receptive field | How much context a neuron sees |
| Pooling | Spatial compression |
| BatchNorm | Training stability in CNNs |
| ResNet-style residual blocks | How deep networks stay trainable |

CNN tensor shapes:

```
Input:  [batch, channels, height, width]
Conv2d: [batch, out_channels, h_out, w_out]
```

---

### 2.3 Normalization and initialization

Do not skip these — training instability almost always traces back here.

| Topic | Used in |
|---|---|
| BatchNorm | CNNs |
| LayerNorm | Transformers |
| RMSNorm | Modern LLMs (Llama, Mistral) |
| Kaiming initialization | ReLU networks |
| Xavier initialization | Sigmoid/tanh networks |
| Residual connections | Everything deep |
| Weight tying | LLM embedding and output head share weights |

---

### 2.4 Sequence modeling fundamentals

Before transformers, understand the vocabulary they use.

| Concept | What it means |
|---|---|
| Tokens | Discrete units of a sequence |
| Embeddings | Dense vector representation of a token |
| Padding | Making variable-length sequences the same length |
| Attention mask | Tells the model which positions to ignore |
| Autoregressive generation | Each token is predicted from all previous tokens |
| Teacher forcing | During training, feed ground-truth tokens as input |

---

### 2.5 Attention and transformers

This is mandatory before any LLM or VLM work.

The core equation — understand every part intuitively:

```
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
```

| Term | What it represents |
|---|---|
| Q (Query) | What this token is looking for |
| K (Key) | What each token offers |
| V (Value) | What gets passed forward if attention is high |
| `QKᵀ` | Similarity scores between every pair of tokens |
| `/ sqrt(d_k)` | Prevents softmax from saturating on large values |
| `softmax(...)` | Converts scores to a probability distribution |
| `... V` | Weighted sum of value vectors |

| Transformer concept | Why it matters |
|---|---|
| Multi-head attention | Attends to multiple relationship types simultaneously |
| Causal masking | Prevents a token from seeing future tokens |
| Positional encoding | Adds order information — attention has no built-in notion of position |
| Feed-forward block | Per-token transformation after attention |
| Residual connections | Stabilizes training in deep stacks |
| LayerNorm placement | Pre-norm vs post-norm affects training stability |
| KV cache | Avoids recomputing past tokens during generation |

Encoder vs decoder:

| Architecture | Examples | Used for |
|---|---|---|
| Encoder-only | BERT | Classification, embeddings |
| Decoder-only | GPT, Llama | Text generation |
| Encoder-decoder | T5, Whisper | Translation, summarization |

Transformer tensor shapes — memorize these:

```
input_ids:      [batch, seq_len]
embeddings:     [batch, seq_len, hidden_dim]
attention mask: [batch, seq_len]
logits:         [batch, seq_len, vocab_size]
labels:         [batch, seq_len]
```

---

### Phase 2 Mini Project A — CNN image classifier

**Dataset**: CIFAR-10.

**What to build**:

```
Custom augmentation pipeline → Conv2d + BatchNorm + ReLU + MaxPool
→ ResNet-style skip connection → CrossEntropyLoss → AdamW + cosine LR
→ training loop → confusion matrix → checkpoint
```

**Requirements**:
- Implement at least one residual block manually
- Use data augmentation: random crop, horizontal flip, normalization
- Plot train vs val accuracy and loss curves
- Generate a confusion matrix on the test set

**What you will discover**:
- How residual connections prevent degradation as depth increases
- Why BatchNorm placement matters
- How augmentation changes the val accuracy curve
- Which classes the model confuses most and why

---

### Phase 2 Mini Project B — Mini character-level language model

This is one of the best projects for building transformer intuition.

**Dataset**: Tiny Shakespeare or any plain text file (~1MB).

**What to build**:

```
Character tokenizer → token embeddings + positional embeddings
→ causal self-attention block → feed-forward block
→ stacked transformer layers → linear head over vocab
→ CrossEntropyLoss → AdamW → training loop
→ autoregressive text generation with temperature sampling
```

**Requirements**:
- Implement multi-head causal self-attention from scratch with `torch.tril` masking
- Implement the full transformer decoder block: attention → residual → layernorm → FFN → residual → layernorm
- Train until you see coherent word patterns emerging (not just random characters)
- Implement generation with greedy decoding and temperature sampling, compare results

**What you will discover**:
- Exactly how causal masking prevents future leakage
- Why residuals and layernorm are not optional
- How temperature changes the distribution of generated text
- What perplexity actually measures

> This project is the bridge between PyTorch mechanics and LLM intuition.
> Do not skip it.

---

## Phase 3 — LLM Foundations

**Goal**: Understand how LLMs are trained, adapted, and used in practice.

---

### 3.1 Tokenization

| Topic | Why it matters |
|---|---|
| BPE (GPT-2, GPT-4) | Subword tokenization — splits on frequency |
| WordPiece (BERT) | Similar to BPE, used by Google models |
| SentencePiece (Llama) | Language-agnostic tokenization |
| Vocabulary size | Output dimension of the language model head |
| Special tokens: `[BOS]`, `[EOS]`, `[PAD]` | Control structure of model input/output |

Key intuition: tokenization determines what the model sees. "ChatGPT" might be 1 token or 3 depending on the tokenizer. This affects everything from prompt length to model behavior on rare words.

---

### 3.2 Language modeling objective

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
```

The model is trained to predict the next token at every position simultaneously. The loss is cross-entropy averaged over all positions and all tokens in the batch.

This is called the **teacher forcing** setup — during training, the model sees the ground-truth previous tokens, not its own predictions.

---

### 3.3 Decoding strategies

| Strategy | Behavior |
|---|---|
| Greedy | Always picks the highest probability token |
| Temperature sampling | Divides logits by T before softmax — higher T = more random |
| Top-k | Samples only from the top k most likely tokens |
| Top-p (nucleus) | Samples from the smallest set of tokens covering p probability mass |
| Beam search | Keeps the top B sequences at each step — more coherent but slower |

---

### 3.4 Fine-tuning and efficient adaptation

Important distinction — do not confuse these:

```
Pretraining    → learns general language patterns from massive data
Fine-tuning    → adapts weights for a specific task or behavior
RAG            → retrieves external knowledge at inference time
Prompting      → steers behavior without changing weights
```

| Technique | What it does |
|---|---|
| Full fine-tuning | Updates all parameters — expensive |
| LoRA | Adds small low-rank weight updates — only trains those |
| QLoRA | LoRA on a quantized base model — runs on consumer GPUs |
| PEFT (Hugging Face) | Library wrapping LoRA, prefix tuning, and others |

LoRA intuition:

```
Instead of updating W (large matrix), learn two small matrices A and B
where the update is ΔW = A @ B. Rank r is typically 4–64.
```

---

### 3.5 Hugging Face essentials

| Tool | What it does |
|---|---|
| `AutoModel`, `AutoTokenizer` | Load any model family by name |
| `AutoProcessor` | For VLMs — handles image + text preprocessing |
| `datasets` | Load and preprocess standard datasets |
| `Trainer` | High-level training loop |
| `generate()` | Text generation with decoding strategies |
| `PEFT` | LoRA, QLoRA, prefix tuning |
| `Accelerate` | Distributed training and mixed precision |

Know both the high-level Hugging Face workflow and the raw PyTorch training loop. The high-level tools are fast for prototyping. The raw loop gives you understanding and control.

---

### 3.6 Mixed precision

| Format | Description |
|---|---|
| FP32 | Standard — full precision |
| FP16 | Faster and lighter — less stable on some ops |
| BF16 | Same memory as FP16, better numerical range — preferred for LLM training |
| INT8 / INT4 | Quantized — used for inference, enables larger models on smaller hardware |

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### Phase 3 Mini Project — Fine-tune a small LLM

**Base model**: GPT-2 (124M) or a small Llama variant via Hugging Face.

**Task**: Pick one — instruction following, sentiment classification, text summarization, or domain Q&A.

**What to build**:

```
Load pretrained model + tokenizer → prepare dataset with correct prompt format
→ implement collate_fn with padding + attention masks + causal labels
→ fine-tune with LoRA via PEFT → evaluate perplexity + qualitative output
→ compare outputs before and after fine-tuning
```

**Requirements**:
- Use LoRA — do not full fine-tune at this stage
- Implement the data pipeline manually, not just `Trainer` defaults
- Evaluate both quantitatively (loss, perplexity) and qualitatively (read the outputs)
- Log and plot training loss vs steps

**What you will discover**:
- Why prompt formatting is not trivial — the model is sensitive to it
- How LoRA rank affects the quality-parameter tradeoff
- Why perplexity going down does not always mean the outputs are better
- What catastrophic forgetting looks like when fine-tuning too aggressively

---

## Phase 4 — VLM Foundations

**Goal**: Understand how models combine visual and textual information.

---

### 4.1 Vision encoders

| Architecture | Description |
|---|---|
| CNN (ResNet, EfficientNet) | Convolutional feature extractor — spatially local |
| ViT (Vision Transformer) | Splits image into patches, treats them as tokens |
| CLIP ViT | ViT trained with contrastive image-text objective |

ViT intuition: a 224×224 image with 16×16 patches becomes a sequence of 196 patch tokens. Each patch is projected to a vector and processed by a standard transformer encoder. There is nothing new — it is the same architecture as a text transformer, applied to image patches.

Vision transformer shapes:

```
image:        [batch, channels, height, width]  → e.g. [B, 3, 224, 224]
patches:      [batch, num_patches, patch_dim]   → e.g. [B, 196, 768]
embeddings:   [batch, num_patches, hidden_dim]  → same as text transformer
```

---

### 4.2 Image-text alignment (CLIP)

The core idea:

> Images and text must be mapped into the same representation space
> so that "a photo of a cat" and an actual photo of a cat are close together.

CLIP trains two encoders — one for images, one for text — with a contrastive objective:

```
For a batch of N (image, caption) pairs:
- Push matched pairs closer together
- Push mismatched pairs further apart
- Loss is cross-entropy over an N×N similarity matrix
```

```python
image_features = image_encoder(images)          # [B, embed_dim]
text_features  = text_encoder(texts)            # [B, embed_dim]
similarity     = image_features @ text_features.T  # [B, B]
labels         = torch.arange(B)                # diagonal = correct pairs
loss = (cross_entropy(similarity, labels) + cross_entropy(similarity.T, labels)) / 2
```

---

### 4.3 Multimodal fusion

After CLIP-style alignment, VLMs need to combine image and text in a single model.

| Approach | How it works | Examples |
|---|---|---|
| Cross-attention | Text tokens attend to image tokens | Flamingo |
| Token concatenation | Image tokens prepended to text tokens | LLaVA, InternVL |
| Projection layer | Linear or MLP maps image embeddings to text token space | LLaVA |

VLM shapes:

```
image embeddings: [batch, image_tokens, hidden_dim]
text embeddings:  [batch, text_tokens, hidden_dim]
combined:         [batch, image_tokens + text_tokens, hidden_dim]
```

The combined sequence is then processed by a standard transformer decoder — which is why understanding transformers deeply in Phase 2 is prerequisite to this.

---

### 4.4 Image preprocessing pipeline

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

Know why normalization uses these specific values (ImageNet statistics) and what happens if you skip it.

---

### Phase 4 Mini Project A — CLIP-style image-text retrieval

**Dataset**: MS-COCO captions or Flickr30k (small subset is fine).

**What to build**:

```
Image encoder (pretrained ResNet or ViT) → projection head
Text encoder (pretrained transformer) → projection head
Contrastive loss over similarity matrix → training loop
→ Recall@1, Recall@5 on val set
→ given a text query, retrieve the top-5 matching images
→ given an image, retrieve the top-5 matching captions
```

**Requirements**:
- Implement the contrastive loss manually — do not use a library wrapper
- Build the N×N similarity matrix and verify its shape
- Evaluate both image→text and text→image retrieval
- Visualize: pick 5 text queries and show the top-3 retrieved images

**What you will discover**:
- Why the similarity matrix must be square and symmetric
- How projection layers affect alignment quality
- What Recall@K actually measures and why K=1 vs K=5 tells different stories
- How representation space geometry changes during training

---

### Phase 4 Mini Project B — Visual question answering with a pretrained VLM

**Model**: LLaVA-1.5 (7B) or InternVL2 (2B) via Hugging Face.

**What to build**:

```
Load VLM + processor → preprocess image + question → run inference
→ build a small VQA evaluation loop over 100–200 examples
→ analyze failure modes: what kinds of questions does it get wrong?
→ try prompt variations and observe how answers change
```

**Requirements**:
- Use `AutoProcessor` to handle image + text jointly — understand what it produces
- Inspect the combined input tensor: where do image tokens sit relative to text tokens?
- Evaluate on at least three question types: factual, spatial, counting
- Document five specific failure cases with your hypothesis about why they fail

**What you will discover**:
- How image tokens are represented in the token sequence
- Why spatial and counting questions are harder than factual ones
- How much prompt wording changes the answer
- What the model's confidence (token probabilities) looks like on correct vs wrong answers

---

## What you are deferring (and when to return)

These are real topics — just not for this refresher.

| Topic | When to return |
|---|---|
| Distributed training: DDP, FSDP, DeepSpeed | When you are training models that do not fit on one GPU |
| Quantization (INT8, INT4, GPTQ, AWQ) | When you need to run large models on limited hardware |
| Deployment: ONNX, TorchScript, vLLM, TGI | When you are building production inference |
| Experiment tracking: W&B, MLflow | When you are running systematic experiments |
| RLHF, DPO, preference optimization | After you understand fine-tuning deeply |
| KL divergence, SVD, eigenvalues | They will surface naturally in LLM/VLM reading — learn them then |

---

## Reference — tensor shapes by model type

```
MLP:
  input:    [batch, features]
  output:   [batch, classes]

CNN:
  input:    [batch, channels, height, width]
  output:   [batch, channels_out, h_out, w_out]

Transformer / LLM:
  input_ids:       [batch, seq_len]
  embeddings:      [batch, seq_len, hidden_dim]
  attention_mask:  [batch, seq_len]
  logits:          [batch, seq_len, vocab_size]
  labels:          [batch, seq_len]

Vision Transformer:
  image:           [batch, 3, 224, 224]
  patches:         [batch, num_patches, patch_dim]
  embeddings:      [batch, num_patches, hidden_dim]

VLM (combined):
  image_tokens:    [batch, n_img_tokens, hidden_dim]
  text_tokens:     [batch, n_txt_tokens, hidden_dim]
  combined:        [batch, n_img_tokens + n_txt_tokens, hidden_dim]
```

---

## Minimum checklist before moving beyond this plan

```
[ ] Can write a training loop from scratch without referencing anything
[ ] Can debug a shape mismatch immediately by reading the error
[ ] Understands why optimizer.zero_grad() must come before loss.backward()
[ ] Understands the difference between model.eval() and torch.no_grad()
[ ] Can explain what CrossEntropyLoss is actually computing
[ ] Has built and trained an MLP, a CNN, and a transformer from scratch
[ ] Understands causal masking and why it exists
[ ] Can explain the CLIP contrastive objective and implement the loss
[ ] Understands what image tokens are and how they enter a VLM
[ ] Has read generated output from a model they fine-tuned
```

If all boxes are checked, you have the foundation. Everything in DL, GenAI, LLMs, VLMs, MLOps, and agentic systems is built on top of what is in this plan.