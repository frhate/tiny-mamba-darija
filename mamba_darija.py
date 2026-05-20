# train_mamba_pure.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tokenizers import Tokenizer
import os
import time
import math

# ==================== CONFIG ====================
TEXT_FILE = "darija_preprocessed.txt"
TOKENIZER_PATH = "darija_tokenizer/tokenizer.json"
CHECKPOINT_DIR = "checkpoints"

# Model (slightly smaller for pure PyTorch speed)
VOCAB_SIZE = 8000
DIM = 384
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2
DROPOUT = 0.1
MAX_SEQ_LEN = 512

# Training (RTX 3060 Ti 8GB)
BATCH_SIZE = 8
GRAD_ACCUM = 2          # Effective batch = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_STEPS = 10000
WARMUP_STEPS = 1000
EVAL_EVERY = 500
SAVE_EVERY = 2000
GEN_EVERY = 1000
MAX_GRAD_NORM = 1.0

# Special tokens
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
# ================================================


class PureMambaBlock(nn.Module):
    """
    Pure PyTorch Mamba SSM block.
    No causal-conv1d, no mamba_ssm — just PyTorch ops.
    Sequential scan is O(L) but fine for tiny models & learning.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state

        # Input projection -> split into x (conv/SSM) and z (gate)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)

        # A: structured state matrix (repeat pattern)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Norm
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, l, d = x.shape
        residual = x
        x = self.norm(x)

        # Project and split
        xz = self.in_proj(x)                          # (b, l, 2*d_inner)
        x_conv, z = xz.chunk(2, dim=-1)               # each (b, l, d_inner)

        # Causal convolution
        x_conv = x_conv.transpose(1, 2)               # (b, d_inner, l)
        x_conv = self.conv1d(x_conv)[:, :, :l]        # trim padding -> causal
        x_conv = x_conv.transpose(1, 2)               # (b, l, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters (input-dependent)
        x_ssm = x_conv
        B, C = self.x_proj(x_ssm).chunk(2, dim=-1)    # (b, l, d_state)
        delta = F.softplus(self.dt_proj(x_ssm))       # (b, l, d_state)

        A = -torch.exp(self.A_log.float())            # (d_inner, d_state)

        # Selective scan (sequential but batched across d_inner)
        # h: (b, d_inner, d_state)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y = torch.empty(b, l, self.d_inner, device=x.device, dtype=x.dtype)

        for t in range(l):
            dt = delta[:, t, :].unsqueeze(1)          # (b, 1, d_state)
            A_bar = torch.exp(dt * A.unsqueeze(0))    # (b, d_inner, d_state)
            B_bar = dt * B[:, t, :].unsqueeze(1)      # (b, 1, d_state)

            # State update
            h = A_bar * h + B_bar * x_conv[:, t, :].unsqueeze(-1)

            # Output projection
            y[:, t, :] = (C[:, t, :].unsqueeze(1) * h).sum(dim=-1)

        # Skip connection + gating
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)

        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class TinyMamba(nn.Module):
    def __init__(self, vocab_size, dim=384, n_layers=6, d_state=16,
                 d_conv=4, expand=2, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            PureMambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie weights

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyMamba (Pure PyTorch): {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=PAD_ID
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class DarijaDataset(Dataset):
    def __init__(self, text_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print("Loading and tokenizing dataset...")
        self.tokens = []
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) < 5:
                    continue
                encoded = tokenizer.encode(line)
                ids = [BOS_ID] + encoded.ids + [EOS_ID]
                self.tokens.extend(ids)

        # Pack into overlapping sequences
        self.samples = []
        stride = max_length // 2
        for i in range(0, len(self.tokens) - max_length, stride):
            chunk = self.tokens[i:i + max_length + 1]
            if len(chunk) == max_length + 1:
                self.samples.append(chunk)

        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Total sequences: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_lr(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Dataset
    dataset = DarijaDataset(TEXT_FILE, tokenizer, max_length=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Model
    print("\nBuilding model...")
    model = TinyMamba(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')

    # Checkpoint dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Resume if exists
    start_step = 0
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
    if checkpoint_files:
        latest = checkpoint_files[-1]
        print(f"\nResuming from {latest}...")
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        print(f"Resumed at step {start_step}")

    # Training loop
    print(f"\n{'='*50}")
    print("TRAINING")
    print(f"{'='*50}")
    print(f"Steps: {start_step} -> {MAX_STEPS}")
    print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"Max seq len: {MAX_SEQ_LEN}")
    print("NOTE: Pure PyTorch scan is slower than CUDA kernels. ~1-2s/step expected.")
    print("Let it run overnight. 50k steps ≈ 15-20 hours on RTX 3060 Ti.\n")

    model.train()
    step = start_step
    data_iter = iter(dataloader)
    total_loss = 0.0
    start_time = time.time()

    while step < MAX_STEPS:
        optimizer.zero_grad()
        accum_loss = 0.0

        for accum_step in range(GRAD_ACCUM):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                logits, loss = model(x, y)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # LR schedule
        lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Step
        scaler.step(optimizer)
        scaler.update()

        total_loss += accum_loss
        step += 1

        # Logging
        if step % EVAL_EVERY == 0:
            avg_loss = total_loss / EVAL_EVERY
            perplexity = math.exp(min(avg_loss, 20))  # cap for stability
            elapsed = time.time() - start_time
            steps_per_sec = EVAL_EVERY / elapsed

            print(f"Step {step:>6} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                  f"LR: {lr:.2e} | Steps/s: {steps_per_sec:.2f}")

            total_loss = 0.0
            start_time = time.time()

        # Generation sample
        if step % GEN_EVERY == 0:
            print("\n--- GENERATION SAMPLE ---")
            model.eval()

            prompts = ["شحال", "يا خويا", "3andek", "هذا"]
            for prompt_text in prompts:
                encoded = tokenizer.encode(prompt_text)
                prompt_ids = [BOS_ID] + encoded.ids
                idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

                generated = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=40)
                output_ids = generated[0].tolist()
                output_ids = output_ids[1:]  # Remove BOS
                if EOS_ID in output_ids:
                    output_ids = output_ids[:output_ids.index(EOS_ID)]
                decoded = tokenizer.decode(output_ids)
                print(f"  '{prompt_text}' -> {decoded}")

            print("---")
            model.train()

        # Save checkpoint
        if step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{step:06d}.pt")
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': accum_loss * GRAD_ACCUM,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_path)
    print(f"\nFinal model saved: {final_path}")


if __name__ == "__main__":
    main()