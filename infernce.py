# infernce.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import os
import sys

# ==================== CONFIG ====================
TOKENIZER_PATH = "darija_tokenizer/tokenizer.json"
CHECKPOINT_PATH = "checkpoints/final_model.pt"

# Model config (must match training)
VOCAB_SIZE = 8000
DIM = 384
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2
MAX_SEQ_LEN = 512

# Generation settings
TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW_TOKENS = 50
REPETITION_PENALTY = 1.2

# Special tokens
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
# ================================================


class PureMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, l, d = x.shape
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_conv, z = xz.chunk(2, dim=-1)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :l]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_ssm = x_conv
        B, C = self.x_proj(x_ssm).chunk(2, dim=-1)
        delta = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log.float())

        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y = torch.empty(b, l, self.d_inner, device=x.device, dtype=x.dtype)

        for t in range(l):
            dt = delta[:, t, :].unsqueeze(1)
            A_bar = torch.exp(dt * A.unsqueeze(0))
            B_bar = dt * B[:, t, :].unsqueeze(1)
            h = A_bar * h + B_bar * x_conv[:, t, :].unsqueeze(-1)
            y[:, t, :] = (C[:, t, :].unsqueeze(1) * h).sum(dim=-1)

        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class TinyMamba(nn.Module):
    def __init__(self, vocab_size, dim=384, n_layers=6, d_state=16,
                 d_conv=4, expand=2, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            PureMambaBlock(dim, d_state, d_conv, expand, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, idx):
        x = self.embedding(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None,
                 repetition_penalty=1.0):
        self.eval()
        generated_ids = idx[0].tolist()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0 and len(generated_ids) > 1:
                for token_id in set(generated_ids[1:]):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            next_id = idx_next.item()

            if next_id == EOS_ID:
                break

            generated_ids.append(next_id)
            idx = torch.cat((idx, idx_next), dim=1)

        return generated_ids


def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    model = TinyMamba(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n_params/1e6:.2f}M parameters")
    print(f"Trained for {checkpoint.get('step', '?')} steps")
    return model


def generate_text(model, tokenizer, prompt_text, device,
                  max_tokens=50, temperature=0.8, top_k=40,
                  repetition_penalty=1.2):
    encoded = tokenizer.encode(prompt_text)
    prompt_ids = [BOS_ID] + encoded.ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )

    output_ids = generated_ids[len(prompt_ids):]
    if EOS_ID in output_ids:
        output_ids = output_ids[:output_ids.index(EOS_ID)]

    return tokenizer.decode(output_ids)


def safe_input(prompt_text):
    """
    Bulletproof input that handles any terminal encoding.
    Reads raw bytes from stdin and decodes safely.
    """
    print(prompt_text, end="", flush=True)
    try:
        # Try normal input first
        return input().strip()
    except (UnicodeDecodeError, EOFError):
        # Fallback: read raw bytes and decode with replacement
        import select
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            raw = sys.stdin.buffer.read1().decode('utf-8', errors='replace').strip()
            return raw
        return ""


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        checkpoint_files = sorted([f for f in os.listdir("checkpoints") if f.endswith('.pt')])
        if not checkpoint_files:
            print("No checkpoint found! Train first.")
            return
        ckpt_path = os.path.join("checkpoints", checkpoint_files[-1])
        print(f"Final model not found, using latest: {ckpt_path}")
    else:
        ckpt_path = CHECKPOINT_PATH

    model = load_model(ckpt_path, device)

    print(f"\n{'='*50}")
    print("DARIJA MAMBA — GENERATION")
    print(f"{'='*50}")
    print(f"Settings: temp={TEMPERATURE}, top_k={TOP_K}, rep_penalty={REPETITION_PENALTY}")
    print(f"{'='*50}")

    # ===== COMMAND-LINE MODE (RECOMMENDED FOR ARABIC) =====
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = generate_text(
            model, tokenizer, prompt, device,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY
        )
        print(f"\nPrompt:  {prompt}")
        print(f"Output:  {result}")
        return

    # ===== INTERACTIVE MODE =====
    print("\n--- AUTO TEST PROMPTS ---")
    test_prompts = [
        "شحال",
        "يا خويا واش",
        "راني",
        "كيفاش",
        "الجزائر",
    ]
    for prompt in test_prompts:
        result = generate_text(
            model, tokenizer, prompt, device,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY
        )
        print(f"  '{prompt}' → {result}")
    print("---")

    print("\nInteractive mode started. Type 'quit' to exit.")
    print("NOTE: If Arabic typing doesn't work here, use:")
    print("  python infernce.py \"your prompt here\"")
    print("-" * 40)

    while True:
        prompt = safe_input("Prompt > ")
        if not prompt:
            continue
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        result = generate_text(
            model, tokenizer, prompt, device,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY
        )
        print(f"Generated: {result}\n")


if __name__ == "__main__":
    main()