# Core imports and reproducibility
from pathlib import Path
from urllib.request import urlretrieve
import json
import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

print('\n')
SEED = 1337
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")

# Choose the dataset.
USE_CUSTOM_DATA = False
CUSTOM_DATA_PATH = "my_training_text.txt"

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DEFAULT_DATA_PATH = Path("tiny_shakespeare.txt")

if USE_CUSTOM_DATA:
    data_path = Path(CUSTOM_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Upload your UTF-8 text file and set CUSTOM_DATA_PATH. "
            f"Could not find: {data_path}"
        )
else:
    data_path = DEFAULT_DATA_PATH
    if not data_path.exists():
        print("Downloading the small training corpus...")
        urlretrieve(DATA_URL, data_path)

text = data_path.read_text(encoding="utf-8")
if len(text) < 10_000:
    raise ValueError(
        "The dataset is too small"
    )

split_index = int(0.90 * len(text))
train_text = text[:split_index]
val_text = text[split_index:]

print(f"Total characters:      {len(text):,}")
print(f"Training characters:   {len(train_text):,}")
print(f"Validation characters: {len(val_text):,}")
print("\nDataset sample:\n")
print(text[:500])

# Build the vocabulary and encoder/decoder.
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {character: index for index, character in enumerate(chars)}
itos = {index: character for character, index in stoi.items()}

def encode(string):
    """Convert a string into token IDs, with a clear error for unknown tokens."""
    unknown = sorted(set(string) - set(stoi))
    if unknown:
        raise ValueError(f"Prompt contains characters outside the vocabulary: {unknown}")
    return [stoi[character] for character in string]

def decode(token_ids):
    """Convert token IDs back into a string."""
    return "".join(itos[int(token_id)] for token_id in token_ids)

sample_text = "hello" if all(character in stoi for character in "hello") else text[:5]
sample_ids = encode(sample_text)

print(f"Vocabulary size: {vocab_size}")
print(f"Example text:    {sample_text!r}")
print(f"Encoded:         {sample_ids}")
print(f"Decoded:         {decode(sample_ids)!r}")
assert decode(sample_ids) == sample_text

# Convert both splits into PyTorch tensors.
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

# A context window is the maximum number of previous tokens provided at once.
# Smaller CPU settings keep the notebook usable without a GPU.
BATCH_SIZE = 64 if device == "cuda" else 32
BLOCK_SIZE = 128 if device == "cuda" else 64

def get_batch(split):
    """Create a batch of input sequences and one-token-shifted targets."""
    source = train_data if split == "train" else val_data
    starting_positions = torch.randint(
        0,
        len(source) - BLOCK_SIZE - 1,
        (BATCH_SIZE,),
    )
    x = torch.stack(
        [source[position : position + BLOCK_SIZE] for position in starting_positions]
    )
    y = torch.stack(
        [source[position + 1 : position + BLOCK_SIZE + 1] for position in starting_positions]
    )
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(f"Input batch shape:  {tuple(xb.shape)}")
print(f"Target batch shape: {tuple(yb.shape)}")
print("\nOne short input example:")
print(repr(decode(xb[0, :40].tolist())))
print("The matching targets are shifted by one token:")
print(repr(decode(yb[0, :40].tolist())))

@torch.no_grad()
def estimate_loss(model, eval_iters=40):
    """Estimate mean train and validation loss without updating weights."""
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for iteration in range(eval_iters):
            inputs, targets = get_batch(split)
            _, loss = model(inputs, targets)
            losses[iteration] = loss.detach().cpu()
        results[split] = losses.mean().item()
    model.train()
    return results

def train_model(model, steps, learning_rate, label, eval_interval=None):
    """Train a model and return loss history for plotting."""
    if eval_interval is None:
        eval_interval = max(50, steps // 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    start_time = time.time()
    model.train()

    for step in range(steps):
        if step % eval_interval == 0 or step == steps - 1:
            metrics = estimate_loss(model, eval_iters=20 if device == "cpu" else 40)
            history.append((step, metrics["train"], metrics["val"]))
            print(
                f"{label} | step {step:5d}/{steps - 1:5d} | "
                f"train {metrics['train']:.4f} | val {metrics['val']:.4f}"
            )

        inputs, targets = get_batch("train")
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    elapsed = time.time() - start_time
    print(f"Finished {label} training in {elapsed:.1f} seconds.")
    return history

def make_prompt(prompt=None):
    """Create a one-row prompt tensor using only known characters."""
    if prompt is None:
        prompt = "ROMEO:\n" if all(c in stoi for c in "ROMEO:\n") else train_text[:20]
    return torch.tensor([encode(prompt)], dtype=torch.long, device=device)

def plot_history(history, title):
    steps, train_losses, val_losses = zip(*history)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, train_losses, marker="o", label="training loss")
    plt.plot(steps, val_losses, marker="o", label="validation loss")
    plt.xlabel("training step")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly indexes one row of next-token logits.
        self.next_token_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices, targets=None):
        logits = self.next_token_table(indices)  # (batch, time, vocabulary)
        loss = None
        if targets is not None:
            batch, time_steps, channels = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch * time_steps, channels),
                targets.reshape(batch * time_steps),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, indices, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(indices)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_index), dim=1)
        return indices

bigram_model = BigramLanguageModel(vocab_size).to(device)
print(f"Bigram parameters: {sum(p.numel() for p in bigram_model.parameters()):,}")

print("\nUntrained bigram output:\n")
untrained_bigram = bigram_model.generate(make_prompt(), max_new_tokens=250)
print(decode(untrained_bigram[0].tolist()))

BIGRAM_STEPS = 500 if device == "cuda" else 200
bigram_history = train_model(
    bigram_model,
    steps=BIGRAM_STEPS,
    learning_rate=1e-2,
    label="bigram",
)
plot_history(bigram_history, "Bigram model learning curve")

print("\nTrained bigram output:\n")
trained_bigram = bigram_model.generate(make_prompt(), max_new_tokens=500)
print(decode(trained_bigram[0].tolist()))

class CausalAverageLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)))
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, vocab_size),
        )

    def forward(self, indices, targets=None):
        batch, time_steps = indices.shape
        token_vectors = self.token_embedding(indices)
        position_vectors = self.position_embedding(
            torch.arange(time_steps, device=indices.device)
        )
        x = token_vectors + position_vectors

        # Every row is normalized so the available past positions sum to one.
        weights = self.causal_mask[:time_steps, :time_steps]
        weights = weights / weights.sum(dim=1, keepdim=True)
        x = weights @ x

        logits = self.network(x)
        loss = None
        if targets is not None:
            batch, time_steps, channels = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch * time_steps, channels),
                targets.reshape(batch * time_steps),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, indices, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            conditioned = indices[:, -self.block_size :]
            logits, _ = self(conditioned)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_index), dim=1)
        return indices

DEMO_EMBD = 64
context_model = CausalAverageLanguageModel(
    vocab_size=vocab_size,
    n_embd=DEMO_EMBD,
    block_size=BLOCK_SIZE,
).to(device)

CONTEXT_STEPS = 700 if device == "cuda" else 250
context_history = train_model(
    context_model,
    steps=CONTEXT_STEPS,
    learning_rate=3e-3,
    label="causal average",
)
plot_history(context_history, "Context model learning curve")

print("\nContext model output:\n")
context_output = context_model.generate(make_prompt(), max_new_tokens=500)
print(decode(context_output[0].tolist()))

class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        _, time_steps, _ = x.shape
        keys = self.key(x)
        queries = self.query(x)

        # Scaled dot-product attention scores.
        weights = queries @ keys.transpose(-2, -1)
        weights = weights * (keys.shape[-1] ** -0.5)
        weights = weights.masked_fill(
            self.causal_mask[:time_steps, :time_steps] == 0,
            float("-inf"),
        )
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        values = self.value(x)
        output = weights @ values
        if return_weights:
            return output, weights
        return output

# Visualize the mechanics of an untrained attention head.
# The upper triangle must be empty because future positions are masked.
demo_inputs, _ = get_batch("val")
demo_inputs = demo_inputs[:1, :16]
time_steps = demo_inputs.shape[1]

with torch.no_grad():
    demo_x = context_model.token_embedding(demo_inputs)
    demo_x = demo_x + context_model.position_embedding(
        torch.arange(time_steps, device=device)
    )
    demo_head = Head(
        n_embd=DEMO_EMBD,
        head_size=16,
        block_size=BLOCK_SIZE,
        dropout=0.0,
    ).to(device)
    _, demo_weights = demo_head(demo_x, return_weights=True)

labels = [repr(character)[1:-1] for character in decode(demo_inputs[0].tolist())]
plt.figure(figsize=(7, 6))
plt.imshow(demo_weights[0].cpu(), cmap="magma", vmin=0)
plt.colorbar(label="attention weight")
plt.xticks(range(time_steps), labels)
plt.yticks(range(time_steps), labels)
plt.xlabel("position being attended to")
plt.ylabel("current position")
plt.title("Untrained causal attention head")
plt.tight_layout()
plt.show()

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [
                Head(n_embd, head_size, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(combined))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPTLanguageModel(nn.Module):
    """A small decoder-only transformer language model."""

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(n_embd, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.language_model_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed(self, indices):
        _, time_steps = indices.shape
        token_vectors = self.token_embedding(indices)
        position_vectors = self.position_embedding(
            torch.arange(time_steps, device=indices.device)
        )
        return token_vectors + position_vectors

    def forward(self, indices, targets=None):
        x = self.embed(indices)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        loss = None
        if targets is not None:
            batch, time_steps, channels = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch * time_steps, channels),
                targets.reshape(batch * time_steps),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        indices,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            conditioned = indices[:, -self.block_size :]
            logits, _ = self(conditioned)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                top_k = min(top_k, logits.shape[-1])
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_index), dim=1)
        return indices

    @torch.no_grad()
    def first_head_attention(self, indices):
        """Return attention weights from the first head of the first block."""
        self.eval()
        indices = indices[:, -self.block_size :]
        x = self.embed(indices)
        first_block = self.blocks[0]
        normalized = first_block.layer_norm_1(x)
        _, weights = first_block.attention.heads[0](
            normalized,
            return_weights=True,
        )
        return weights

# QUICK_MODE is designed for a first Colab run. Turn it off for a larger model
# and a longer training run after the full notebook works.
QUICK_MODE = False

if QUICK_MODE:
    N_EMBD = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    DROPOUT = 0.15
    FULL_STEPS = 1_500 if device == "cuda" else 400
else:
    N_EMBD = 256
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.20
    FULL_STEPS = 5_000 if device == "cuda" else 1_000

model_config = {
    "vocab_size": vocab_size,
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
}

gpt = GPTLanguageModel(**model_config).to(device)
parameter_count = sum(parameter.numel() for parameter in gpt.parameters())

test_inputs, test_targets = get_batch("train")
test_logits, test_loss = gpt(test_inputs, test_targets)

print(json.dumps(model_config, indent=2))
print(f"Total parameters: {parameter_count:,}")
print(f"Logit tensor shape: {tuple(test_logits.shape)}")
print(f"Initial loss: {test_loss.item():.4f}")

PROMPT = "ROMEO:\n" if all(c in stoi for c in "ROMEO:\n") else train_text[:20]

print("Untrained transformer output:\n")
untrained_output = gpt.generate(
    make_prompt(PROMPT),
    max_new_tokens=350,
    temperature=1.0,
    top_k=30,
)
print(decode(untrained_output[0].tolist()))

gpt_history = train_model(
    gpt,
    steps=FULL_STEPS,
    learning_rate=3e-4,
    label="transformer",
)
plot_history(gpt_history, "Transformer training and validation loss")

print("Trained transformer output:\n")
trained_output = gpt.generate(
    make_prompt(PROMPT),
    max_new_tokens=800,
    temperature=0.9,
    top_k=40,
)
print(decode(trained_output[0].tolist()))

for temperature in (0.5, 0.9, 1.3):
    generated = gpt.generate(
        make_prompt(PROMPT),
        max_new_tokens=350,
        temperature=temperature,
        top_k=40,
    )
    print("\n" + "=" * 70)
    print(f"TEMPERATURE = {temperature}")
    print("=" * 70)
    print(decode(generated[0].tolist()))

attention_prompt = PROMPT[-min(len(PROMPT), 24) :]
attention_indices = make_prompt(attention_prompt)
trained_weights = gpt.first_head_attention(attention_indices)[0].cpu()
attention_labels = [
    repr(character)[1:-1]
    for character in decode(attention_indices[0].tolist())
]

plt.figure(figsize=(7, 6))
plt.imshow(trained_weights, cmap="magma", vmin=0)
plt.colorbar(label="attention weight")
plt.xticks(range(len(attention_labels)), attention_labels)
plt.yticks(range(len(attention_labels)), attention_labels)
plt.xlabel("position being attended to")
plt.ylabel("current position")
plt.title("First trained attention head")
plt.tight_layout()
plt.show()

checkpoint_path = Path("lera_lm_checkpoint.pt")
checkpoint = {
    "model_state_dict": gpt.state_dict(),
    "model_config": model_config,
    "stoi": stoi,
    "itos": itos,
    "prompt": PROMPT,
    "history": gpt_history,
    "seed": SEED,
}
torch.save(checkpoint, checkpoint_path)
print(f"Saved checkpoint to: {checkpoint_path.resolve()}")

