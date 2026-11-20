from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.normalizers import NFC
import json

with open("train_full3.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Data size: {len(text)} chars ({len(text)/1024/1024:.1f} MB)")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = WhitespaceSplit()

trainer = BpeTrainer(
    vocab_size=12000,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", " ", "\n", "User:", "Bot:", "AI:"]
)
tokenizer.train_from_iterator([text], trainer=trainer)

tokenizer.save("bpe_tokenizer.json")

with open("bpe_tokenizer.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vocab = data["model"]["vocab"]
merges = data["model"]["merges"]

with open("bpe_vocab.txt", "w", encoding="utf-8") as f:
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if token == " ":
            f.write(f"SPACE {idx}\n")
        elif token == "\n":
            f.write(f"NEWLINE {idx}\n")
        else:
            f.write(f"{token} {idx}\n")

with open("bpe_merges.txt", "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
    for merge in merges:
        if isinstance(merge, list):
            f.write(f"{merge[0]} {merge[1]}\n")
        else:
            f.write(f"{merge}\n")

print(f"Vocab size: {len(vocab)}")
print(f"Merges: {len(merges)}")

import sys
sys.stdout.reconfigure(encoding="utf-8")
enc = tokenizer.encode("USER: привіт як справи\nAI: нормально")
print(f"tokens: {enc.tokens}")
print(f"ids: {enc.ids}")
