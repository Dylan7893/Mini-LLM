import torch

# Select the best available hardware backend for PyTorch execution.
# CUDA is used for NVIDIA GPUs, MPS is used for Apple Silicon GPUs,
# and CPU is used as a fallback option.
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)


from pathlib import Path
from urllib.request import urlretrieve

# Tiny Shakespeare provides a small text dataset for language model experiments.
# The dataset contains Shakespeare's writing and works well for initial testing.
data_url = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# Local file path for storing the downloaded dataset.
data_path = Path("tiny_shakespeare.txt")

# Download the dataset only when the local copy does not already exist.
# Prevents unnecessary downloads during future program runs.
if not data_path.exists():
    urlretrieve(data_url, data_path)


# Load the complete text dataset into memory.
# The dataset size is small enough for direct memory loading.
text = data_path.read_text(encoding="utf-8")

# Display a preview of the dataset to verify successful loading.
print(text[:500])

# Display the total number of characters available for training.
print(f"Characters: {len(text):,}")


# Split the dataset into training and validation sections.
# Training data is used for learning patterns.
# Validation data is used for checking model performance.
split_index = int(0.90 * len(text))

train_text = text[:split_index]
val_text = text[split_index:]


print(f"Training characters:   {len(train_text):,}")
print(f"Validation characters: {len(val_text):,}")


# Verify dataset integrity before continuing.
# These checks confirm sufficient data size, correct split proportions,
# and preservation of the original text.
assert len(text) > 10_000
assert len(train_text) > len(val_text)
assert train_text + val_text == text

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {character: index for index, character in enumerate(chars)}
itos = {index: character for character, index in stoi.items()}

print(chars)
print(f"Vocabulary size: {vocab_size}")

def encode(string):
    unknown = sorted(set(string) - set(stoi))
    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")
    return [stoi[character] for character in string]

def decode(token_ids):
    return "".join(itos[int(token_id)] for token_id in token_ids)

phrase = "hello"
token_ids = encode(phrase)

print(token_ids)
print(decode(token_ids))
assert decode(token_ids) == phrase

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

BATCH_SIZE = 64 if device == "cuda" else 32
BLOCK_SIZE = 128 if device == "cuda" else 64

def get_batch(split):
    source = train_data if split == "train" else val_data
    starts = torch.randint(
        0,
        len(source) - BLOCK_SIZE - 1,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        source[start : start + BLOCK_SIZE]
        for start in starts
    ])
    y = torch.stack([
        source[start + 1 : start + BLOCK_SIZE + 1]
        for start in starts
    ])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")

assert xb.shape == yb.shape
assert xb.shape == (BATCH_SIZE, BLOCK_SIZE)
assert torch.equal(xb[:, 1:], yb[:, :-1])

print(repr(decode(xb[0, :40].tolist())))
print(repr(decode(yb[0, :40].tolist())))
