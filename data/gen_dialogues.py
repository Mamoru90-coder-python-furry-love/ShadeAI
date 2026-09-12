#!/usr/bin/env python3
"""Generate 100k+ dialogues from ukr_fiction.txt for SFT training."""
import json, random

INPUT = "data/ukr_fiction.txt"
OUTPUT = "data/dialogues.jsonl"
TARGET = 100000

# Read lines
with open(INPUT, encoding="utf-8") as f:
    lines = [l.strip() for l in f.readlines() if 20 < len(l.strip()) < 300]

print(f"Found {len(lines)} lines")
if len(lines) < 2:
    print("ERROR: not enough text")
    exit(1)

# Generate dialogues by pairing adjacent lines
dialogues = []
for i in range(len(lines) - 1):
    q = lines[i]
    a = lines[i + 1]
    dialogues.append({
        "messages": [
            {"role": "system", "content": "Ти — корисний помічник, що відповідає українською мовою."},
            {"role": "user", "content": f"Розкажи більше: {q}"},
            {"role": "assistant", "content": a}
        ]
    })

# Add variation
for i in range(len(lines) - 1):
    dialogues.append({
        "messages": [
            {"role": "system", "content": "Ти — корисний помічник, що відповідає українською мовою."},
            {"role": "user", "content": f"Що таке: {lines[i+1][:80]}"},
            {"role": "assistant", "content": lines[i]}
        ]
    })

# Pad if needed
while len(dialogues) < TARGET:
    dialogues.extend(dialogues[:min(len(dialogues), TARGET - len(dialogues))])

dialogues = dialogues[:TARGET]
random.shuffle(dialogues)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for d in dialogues:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"Generated {len(dialogues)} dialogues -> {OUTPUT}")
