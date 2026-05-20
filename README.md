# Tiny Mamba for Algerian Darija 🇩🇿

A from-scratch implementation of a **Mamba State Space Model (SSM)** trained to speak **Algerian Darija** — the colloquial Arabic dialect of Algeria, written in both Arabic script and Franco-Arab (Latin + numbers).

Built with pure PyTorch. No Transformers. No attention. Just selective state spaces.

&gt; **"تعجبني هذيك اللقطة كي كنت ف المريكان . و الله غير يعطيك الصحة بالتوفيق"**

---

## 🚀 Features

- **Pure PyTorch Mamba** — no `mamba-ssm` CUDA kernels required
- **Custom BPE Tokenizer** trained on real Algerian social media text
- **Mixed-script support** — handles Arabic script (`شحال`) and Franco-Arab (`3andek`)
- **RTX 3060 Ti optimized** — trains on 8GB VRAM
- **Repetition penalty + top-k sampling** for clean generation
- **Resume training** from checkpoints automatically

---

## 📁 Project Structure
.
├── darija_preprocessed.txt      # Cleaned training corpus (not in repo, too big)
├── darija_tokenizer/            # Trained BPE tokenizer
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── special_tokens.txt
├── checkpoints/                 # Saved model checkpoints (not in repo, too big)
│   └── final_model.pt
├── darja_processing.py          # Step 1: Clean raw text
├── train_tokenizer.py           # Step 2: Train BPE tokenizer
├── mamba_darija.py              # Step 3: Train Mamba model (pure PyTorch)
├── infernce.py                  # Step 4: Generate text
├── requirements.txt
└── README.md



---

## 🛠️ Installation

```bash
git clone https://github.com/frhate/tiny-mamba-darija.git
cd tiny-mamba-darija

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt



## 📚 Dataset
Uses the Algerian-Darija dataset by Ayoub Kirouane.
23 MB of raw Algerian social media text
Preprocessed to remove HTML, emojis, URLs, mentions, and duplicate characters



🏋️# Training from Scratch
Step 1: Download & Preprocess
python darja_processing.py

Downloads from HuggingFace and cleans the text.
Output: darija_preprocessed.txt



Step 2: Train Tokenizer

python train_tokenizer.py
Output: darija_tokenizer/tokenizer.json


Step 3: Train Mamba Model
python mamba_darija.py


| Hyperparameter  | Value                            |
| --------------- | -------------------------------- |
| Architecture    | Mamba SSM (no attention)         |
| Layers          | 6                                |
| Dimension       | 384                              |
| SSM State       | 16                               |
| Vocab Size      | 8,000                            |
| Sequence Length | 512                              |
| Parameters      | ~8.7M                            |
| Batch Size      | 8 (effective 16 with grad accum) |


🗣️# Inference
Command Line (Recommended for Arabic)

python infernce.py "شحال هادي"
python infernce.py "واش راك"
python infernce.py "يا خويا"

🎯 Example Outputs (10k steps)
| Prompt    | Generated Darija                                                       |
| --------- | ---------------------------------------------------------------------- |
| `شحال`    | تعجبني هذيك اللقطة كي كنت ف المريكان . و الله غير يعطيك الصحة بالتوفيق |
| `راني`    | ن توحش كل يوم الحلقة الاخيرة تع ليوم و وليت نعاود فيه في رمضان         |
| `الجزائر` | الجميلة و المظلوم في الارض                                             |


## Acknowledgments:
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
https://arxiv.org/abs/2312.00752

Algerian-Darija Dataset
https://huggingface.co/datasets/ayoubkirouane/Algerian-Darija
