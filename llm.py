import torch

# Select the best available hardware backend for PyTorch execution.
# CUDA is used for NVIDIA GPUs, MPS is used for Apple Silicon GPUs,
# and CPU is used as a fallback option.
if torch.cuda.is_available():
    device = "\ncuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "\nmps"
else:
    device = "\ncpu"

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
