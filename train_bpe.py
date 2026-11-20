from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
import json

# Read training data
with open("data/train_mega.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Data size: {len(text)} chars")

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = NFD()
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "User:", "Bot:", "AI:", "\n"]
)
tokenizer.train_from_iterator([text], trainer=trainer)

# Save
tokenizer.save("data/bpe_tokenizer.json")
print("Saved bpe_tokenizer.json")

# Also export simple vocab.txt and merges.txt for C++ loading
with open("data/bpe_tokenizer.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vocab = data["model"]["vocab"]
merges = data["model"]["merges"]

with open("data/bpe_vocab.txt", "w", encoding="utf-8") as f:
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        f.write(f"{token} {idx}\n")

with open("data/bpe_merges.txt", "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
    for merge in merges:
        f.write(f"{merge}\n")

print(f"Vocab size: {len(vocab)}")
print(f"Merges: {len(merges)}")
print("Saved bpe_vocab.txt and bpe_merges.txt")