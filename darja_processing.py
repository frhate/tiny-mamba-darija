# preprocess_darija.py
import re
import os
from datasets import load_dataset

# ==================== CONFIG ====================
INPUT_FILE = "darija_clean.txt"
OUTPUT_FILE = "darija_preprocessed.txt"
MIN_LINE_LENGTH = 15      # Skip very short lines
MAX_LINE_LENGTH = 2000    # Skip absurdly long lines (likely garbage)
# ================================================


def remove_html_tags(text):
    """Remove HTML tags like <br>, <div>, etc."""
    clean = re.sub(r'<[^>]+>', ' ', text)
    return clean


def remove_urls(text):
    """Remove http/https/www links"""
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    return text


def remove_emojis(text):
    """
    Remove emojis and other Unicode symbols.
    Covers most emoji ranges including flags, skin tones, etc.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols etc
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(' ', text)


def normalize_whitespace(text):
    """Collapse multiple spaces/newlines into single space"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_repeated_chars(text):
    """
    Darija users love stretching words: 'ههههههه' or 'bzzzzzf'
    Limit repetition to max 3 of the same char
    """
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    return text


def remove_mentions_hashtags(text):
    """Remove @mentions and #hashtags (or keep if you want)"""
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    return text


def remove_email_phone(text):
    """Remove email addresses and phone numbers"""
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)  # emails
    text = re.sub(r'\b\d{8,}\b', ' ', text)    # long numbers (phones)
    return text


def normalize_punctuation(text):
    """
    Keep useful punctuation but normalize:
    - Multiple !!! or ??? into single
    - Arabic punctuation into standard forms
    - Remove weird special chars
    """
    # Normalize repeated punctuation
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{3,}', '...', text)  # Keep ... but not .......
    text = re.sub(r'\.{4,}', '...', text)

    # Remove non-linguistic special characters but keep essentials
    # Keep: Arabic letters, Latin letters, numbers (for Franco-Arab), basic punctuation, spaces
    # Remove: other symbols, math symbols, arrows, box drawing, etc.
    allowed = r'[^\w\s.,!?؛،؟\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    text = re.sub(allowed, ' ', text)

    return text


def normalize_numbers_for_franco(text):
    """
    In Franco-Arab Darija, numbers represent Arabic letters:
    3 = ع, 7 = ح/خ, 9 = ق, 5 = خ, 2 = ء, etc.
    We keep them as-is (the tokenizer will learn them),
    but we normalize some common issues:
    - '3a' vs '3 a' → keep as '3a'
    """
    # Ensure numbers stick to following letters (Franco convention)
    # e.g., "3 andek" → not needed, but "3andek" is fine
    # Just clean up spaces around single digits used as letters
    text = re.sub(r'(\d)\s+([a-zA-Z])', r'\1\2', text)
    return text


def filter_line_quality(text):
    """
    Heuristic filters to remove garbage lines:
    - Too short
    - Too long (likely copy-paste spam)
    - Mostly numbers
    - Mostly non-alphabetic
    """
    if len(text) < MIN_LINE_LENGTH:
        return None
    if len(text) > MAX_LINE_LENGTH:
        return None

    # Must contain at least some Arabic or Latin letters
    letter_count = sum(1 for c in text if c.isalpha())
    if letter_count < len(text) * 0.3:  # Less than 30% letters
        return None

    # Must contain at least some Darija-like content
    # (either Arabic script or Franco-Arab indicators)
    has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
    has_franco = any(c in '379524' for c in text)  # Common Franco-Arab digits
    has_latin = any(c.isascii() and c.isalpha() for c in text)

    # Keep if it has Arabic script OR looks like Franco-Arab (Latin + numbers)
    if not (has_arabic or (has_latin and has_franco)):
        # Exception: pure Arabic script is fine
        if not has_arabic:
            return None

    return text


def preprocess_line(text):
    """Full preprocessing pipeline for one line"""
    if not text or not isinstance(text, str):
        return None

    # Step-by-step cleaning
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_mentions_hashtags(text)
    text = remove_email_phone(text)
    text = normalize_repeated_chars(text)
    text = normalize_numbers_for_franco(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    # Quality filter
    text = filter_line_quality(text)

    return text


# ==================== MAIN ====================

print("=" * 50)
print("DARIJA TEXT PREPROCESSING")
print("=" * 50)

# Load raw data
if not os.path.exists(INPUT_FILE):
    print(f"{INPUT_FILE} not found. Downloading from HuggingFace...")
    dataset = load_dataset("ayoubkirouane/Algerian-Darija", split="train")
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(ex["Text"].strip() + "\n\n")
    print(f"Saved raw data to {INPUT_FILE}")

print(f"Reading {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

print(f"Raw lines: {len(raw_lines)}")

# Process
cleaned_lines = []
removed_count = 0

for i, line in enumerate(raw_lines):
    line = line.strip()
    cleaned = preprocess_line(line)

    if cleaned:
        cleaned_lines.append(cleaned)
    else:
        removed_count += 1

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1}/{len(raw_lines)} lines...")

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        f.write(line + "\n")

# Stats
raw_size = os.path.getsize(INPUT_FILE) / 1024 / 1024
clean_size = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
raw_chars = sum(len(l) for l in raw_lines)
clean_chars = sum(len(l) for l in cleaned_lines)

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)
print(f"Raw lines:      {len(raw_lines)}")
print(f"Cleaned lines:  {len(cleaned_lines)}")
print(f"Removed:        {removed_count} ({100*removed_count/len(raw_lines):.1f}%)")
print(f"Raw size:       {raw_size:.2f} MB")
print(f"Clean size:     {clean_size:.2f} MB")
print(f"Raw chars:      {raw_chars:,}")
print(f"Clean chars:    {clean_chars:,}")
print(f"Output file:    {OUTPUT_FILE}")

# Show 5 random samples
import random
print("\n--- 5 RANDOM SAMPLES ---")
for line in random.sample(cleaned_lines, min(5, len(cleaned_lines))):
    print(f"> {line[:200]}")