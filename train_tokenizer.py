# train_tokenizer.py
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os

# ==================== CONFIG ====================
INPUT_FILE = "darija_preprocessed.txt"
TOKENIZER_DIR = "darija_tokenizer"
VOCAB_SIZE = 8000
# ================================================

print("=" * 50)
print("TRAINING DARIJA BPE TOKENIZER")
print("=" * 50)

# 1. Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="||<<||<|unk|>"))

# 2. Pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),
    pre_tokenizers.Punctuation(),
    pre_tokenizers.Digits(individual_digits=False),
])

# 3. Trainer
special_tokens = ["||<<||<|pad|>", "||<<||<|unk|>", "||<<||<|bos|>", "||<<||<|eos|>"]

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=special_tokens,
    min_frequency=2,
    show_progress=True,
)

# 4. Train
print(f"Training on {INPUT_FILE}...")
files = [INPUT_FILE]
tokenizer.train(files, trainer)

# 5. Get special token IDs
pad_id = tokenizer.token_to_id("||<<||<|pad|>")
unk_id = tokenizer.token_to_id("||<<||<|unk|>")
bos_id = tokenizer.token_to_id("||<<||<|bos|>")
eos_id = tokenizer.token_to_id("||<<||<|eos|>")

print(f"\nSpecial token IDs:")
print(f"  PAD: {pad_id}")
print(f"  UNK: {unk_id}")
print(f"  BOS: {bos_id}")
print(f"  EOS: {eos_id}")

# 6. Save
os.makedirs(TOKENIZER_DIR, exist_ok=True)
tokenizer.save(os.path.join(TOKENIZER_DIR, "tokenizer.json"))

# Save vocab reference + special IDs
with open(os.path.join(TOKENIZER_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
    vocab = tokenizer.get_vocab()
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        f.write(f"{idx}: {token}\n")

with open(os.path.join(TOKENIZER_DIR, "special_tokens.txt"), "w") as f:
    f.write(f"pad_id={pad_id}\n")
    f.write(f"unk_id={unk_id}\n")
    f.write(f"bos_id={bos_id}\n")
    f.write(f"eos_id={eos_id}\n")

print(f"\nTokenizer saved to: {TOKENIZER_DIR}/")
print(f"Actual vocab size: {len(tokenizer.get_vocab())}")

# 7. Test with manual BOS/EOS
print("\n" + "=" * 50)
print("TESTING TOKENIZER")
print("=" * 50)

test_sentences = [
    "شحال هادي ما شفتكش",
    "3andek ch7al men flouss",
    "يا خويا واش راك بخير",
    "Cava merci bzf",
    "هذا البلاصة راهي زينة بصح",
]

for text in test_sentences:
    encoded = tokenizer.encode(text)
    ids_with_special = [bos_id] + encoded.ids + [eos_id]
    tokens_with_special = ["||<<||<|bos|>"] + encoded.tokens + ["||<<||<|eos|>"]
    decoded = tokenizer.decode(encoded.ids)

    print(f"\nOriginal:     {text}")
    print(f"Tokens:       {tokens_with_special}")
    print(f"IDs:          {ids_with_special}")
    print(f"Decoded:      {decoded}")
    print(f"N tokens:     {len(ids_with_special)}")