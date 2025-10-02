# mini_gpt_char.py
# Small, pedagogical GPT-style language model “from scratch” (char-level).
# Trains on own dataset (corpus.txt) and generates text.
# Can generate text with stop tokens.

import math, os, io, random, argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
@dataclass
class Config:
    data_path: str = "corpus.txt"     # your own dataset
    block_size: int = 64             # context length (number of tokens)
    batch_size: int = 32
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 128                 # embedding/hidden size
    dropout: float = 0.1
    lr: float = 3e-4
    max_iters: int = 3000             # iterations (enough for small dataset)
    eval_interval: int = 300
    eval_iters: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    ckpt_path: str = "mini_gpt_char.pt"

cfg = Config()

# ----------------------------
# Helper functions
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File '{path}' not found. Create corpus.txt and add your own text."
        )
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()

def train_val_split(data: torch.Tensor, val_fraction=0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]

def load_checkpoint(checkpoint_path: str, device: str):
    """Load model and tokenizer from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration and vocabulary
    cfg_dict = checkpoint['cfg']
    vocab = checkpoint['vocab']
    
    # Create tokenizer from loaded vocabulary
    tokenizer = CharTokenizer("")
    itos = vocab['itos']
    stoi = vocab['stoi']
    tokenizer.itos = itos
    tokenizer.stoi = stoi
    tokenizer.chars = [itos[i] for i in range(len(itos))]  # stable order by index
    tokenizer.vocab_size = len(tokenizer.chars)
    
    # Create model with loaded configuration
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg_dict['n_embd'],
        n_layers=cfg_dict['n_layers'],
        n_heads=cfg_dict['n_heads'],
        block_size=cfg_dict['block_size'],
        dropout=cfg_dict['dropout'],
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Loaded model with vocab_size={tokenizer.vocab_size}, block_size={cfg_dict['block_size']}")
    return model, tokenizer, cfg_dict

# ----------------------------
# Char-level tokenization
# ----------------------------
class CharTokenizer:
    def __init__(self, text: str):
        # Set vocabulary from all characters in the dataset
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

# ----------------------------
# Dataloader (micro-batch)
# ----------------------------
class CharDataset:
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def get_batch(self, batch_size: int, device: str):
        # Select random starting indices
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)

# ----------------------------
# Small GPT-style model
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask (alakulmainen)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # Shape to (B, heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layers, n_heads, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "T > block_size"
        token_embeddings = self.tok_emb(idx)              # (B,T,C)
        position_embeddings = self.pos_emb[:, :T, :]      # (1,T,C)
        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                             # (B,T,vocab)

        loss = None
        if targets is not None:
            # Combine batch and time, calculate cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_tokens=None, temperature=1.0):
        # Autoregressive generation
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # restrict to context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # temperature: <1 makes choices peakier (more deterministic)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            
            # Check stop tokens
            if stop_tokens is not None:
                last_token = next_id.item()
                if last_token in stop_tokens:
                    break
                    
        return idx

# ----------------------------
# Training and evaluation
# ----------------------------
def estimate_loss(model, train_loader, val_loader, iters, device):
    model.eval()
    losses = {}
    for split, loader in (("train", train_loader), ("val", val_loader)):
        loss_sum = 0.0
        with torch.no_grad():
            for _ in range(iters):
                xb, yb = loader.get_batch(cfg.batch_size, device)
                _, loss = model(xb, yb)
                loss_sum += loss.item()
        losses[split] = loss_sum / iters
    model.train()
    return losses

def generate_text(model, tokenizer, prompt: str, max_tokens: int, device: str, 
                 stop_tokens=None, temperature=1.0):
    """Generate text using a trained model."""
    print(f"\nGenerating text with prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    
    # Encode the prompt
    try:
        start_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    except KeyError as e:
        print(f"Error: Character '{e}' in prompt is not in the model's vocabulary.")
        print(f"Available characters: {sorted(tokenizer.chars)}")
        return None
    
    # Convert stop tokens from characters to IDs if provided
    stop_token_ids = None
    if stop_tokens:
        try:
            stop_token_ids = [tokenizer.stoi[char] for char in stop_tokens]
            print(f"Stop tokens: {stop_tokens} (IDs: {stop_token_ids})")
        except KeyError as e:
            print(f"Warning: Stop token '{e}' not in vocabulary, ignoring stop tokens")
            stop_token_ids = None
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(start_ids, max_new_tokens=max_tokens, 
                                  stop_tokens=stop_token_ids, temperature=temperature).squeeze(0).tolist()
    
    # Decode and return
    generated_text = tokenizer.decode(output_ids)
    return generated_text

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mini GPT Character-level Model')
    parser.add_argument('--load', type=str, help='Path to checkpoint file to load for testing')
    parser.add_argument('--prompt', type=str, default='machine learning', help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation (higher = more random)')
    parser.add_argument('--stop-tokens', type=str, help='Stop generation when these characters appear (e.g., ".,!")')
    args = parser.parse_args()

    set_seed(cfg.seed)

    # Check if we should load from checkpoint
    if args.load:
        # Test mode: load existing model
        try:
            model, tokenizer, loaded_cfg = load_checkpoint(args.load, cfg.device)
            
            # Default: generate text with loaded model
            stop_tokens = list(args.stop_tokens) if args.stop_tokens else None
            generated_text = generate_text(model, tokenizer, args.prompt, args.max_tokens, cfg.device,
                                            stop_tokens=stop_tokens, temperature=args.temperature)
            if generated_text:
                print("\n=== Generated Text ===\n")
                print(generated_text)
                print("\n=====================\n")
            
            return
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

    # Training mode: train new model
    raw = load_text(cfg.data_path)
    print(f"Loaded corpus: {len(raw):,} chars")

    tokenizer = CharTokenizer(raw)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Convert entire text to sequence of IDs (Tensor)
    data = torch.tensor(tokenizer.encode(raw), dtype=torch.long)

    # Split train/val
    train_data, val_data = train_val_split(data, val_fraction=0.1)
    
    # Check that block_size is appropriate
    min_data_size = min(len(train_data), len(val_data))
    if cfg.block_size >= min_data_size:
        print(f"Warning: block_size ({cfg.block_size}) is too large for dataset (min size: {min_data_size})")
        print(f"Reducing block_size to {min_data_size - 1}")
        cfg.block_size = min_data_size - 1
    
    train_ds = CharDataset(train_data, cfg.block_size)
    val_ds = CharDataset(val_data, cfg.block_size)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg.n_embd,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Training
    pbar = tqdm(range(1, cfg.max_iters + 1), desc="Training")
    for it in pbar:
        xb, yb = train_ds.get_batch(cfg.batch_size, cfg.device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % cfg.eval_interval == 0 or it == 1:
            losses = estimate_loss(model, train_ds, val_ds, cfg.eval_iters, cfg.device)
            pbar.set_postfix(loss=loss.item(), train_loss=losses["train"], val_loss=losses["val"])

    # Save checkpoint
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "vocab": {
            "itos": tokenizer.itos,
            "stoi": tokenizer.stoi,
        }
    }, cfg.ckpt_path)
    print(f"Saved checkpoint to {cfg.ckpt_path}")

    # Generation after training
    stop_tokens = list(args.stop_tokens) if args.stop_tokens else None
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_tokens, cfg.device,
                                 stop_tokens=stop_tokens, temperature=args.temperature)
    if generated_text:
        print("\n=== Generated Text ===\n")
        print(generated_text)
        print("\n=====================\n")

if __name__ == "__main__":
    main()
