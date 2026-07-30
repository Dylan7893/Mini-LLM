import torch

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)


from pathlib import Path
from urllib.request import urlretrieve

data_url = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
data_path = Path("tiny_shakespeare.txt")

if not data_path.exists():
    urlretrieve(data_url, data_path)

text = data_path.read_text(encoding="utf-8")
print(text[:500])
print(f"Characters: {len(text):,}")


split_index = int(0.90 * len(text))
train_text = text[:split_index]
val_text = text[split_index:]

print(f"Training characters:   {len(train_text):,}")
print(f"Validation characters: {len(val_text):,}")


assert len(text) > 10_000
assert len(train_text) > len(val_text)
assert train_text + val_text == text
print("Mission 1 passed.")