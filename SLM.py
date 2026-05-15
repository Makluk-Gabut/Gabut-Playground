import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import math
import sys
import json
import random
import numpy as np
import time

# --- 0. CONFIGURATION & REPRODUCIBILITY ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@dataclass
class NanoConfig:
    block_size: int = 256
    n_layer: int = 6 # Ditambah untuk kapasitas lebih besar
    n_head: int = 8
    n_embd: int = 384 # Ditambah dari 256
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 5e-4
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 40
    warmup_iters: int = 200
    checkpoint_name: str = "nanocore_beta_v2.pt"
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

config = NanoConfig()

# --- 1. BYTE-LEVEL TOKENIZER (Fix: No more <UNK>) ---
class ByteTokenizer:
    def __init__(self):
        # Menggunakan 256 byte dasar + special tokens
        self.vocab_size = 256 + 1 # +1 untuk End of Text jika perlu
        
    def encode(self, s):
        # Mengonversi string ke daftar byte (0-255)
        return list(s.encode('utf-8'))

    def decode(self, l):
        # Mengonversi daftar byte kembali ke string, ignore error untuk partial bytes
        return bytes(l).decode('utf-8', errors='replace')

# --- 2. ARCHITECTURE (RoPE & Causal Attention) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0), persistent=True)
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0), persistent=True)

    def forward(self, T):
        return self.cos[:, :, :T, :], self.sin[:, :, :T, :]

def apply_rope(q, k, cos, sin):
    q_rot = torch.cat((-q[..., q.size(-1)//2:], q[..., :q.size(-1)//2]), dim=-1)
    k_rot = torch.cat((-k[..., k.size(-1)//2:], k[..., :k.size(-1)//2]), dim=-1)
    return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.block_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(config.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=config.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class NanoCore_Beta(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.token_emb.weight = self.head.weight 

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# --- 3. CHECKPOINT SYSTEM (Fix: Full Resume) ---
def save_full_checkpoint(model, optimizer, scheduler, scaler, step, loss, filename):
    state = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(state, filename)

def load_full_checkpoint(model, optimizer, scheduler, scaler, filename):
    if os.path.exists(filename):
        ckpt = torch.load(filename, map_location=config.device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        return ckpt['step'], ckpt['loss']
    return 0, float('inf')

# --- 4. DATA LOADER ---
def get_batch(data, config):
    ix = torch.randint(0, len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)

# --- 5. SAMPLING (Fix: Top-K & Top-P) ---
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=config.device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Top-K
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Top-P (Nucleus)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        
        token_str = tokenizer.decode([next_token.item()])
        print(token_str, end='', flush=True)
        
    print()

# --- 6. TRAINING ENGINE ---
if __name__ == "__main__":
    if not os.path.exists('corpus.txt'):
        with open('corpus.txt', 'w') as f: f.write("Contoh data training untuk NanoCore Beta v2.")
    
    with open('corpus.txt', 'r', encoding='utf-8') as f: text = f.read()
    tokenizer = ByteTokenizer()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    model = NanoCore_Beta(tokenizer.vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)
    scaler = torch.amp.GradScaler(enabled=(config.device == 'cuda'))
    
    start_step, best_val_loss = load_full_checkpoint(model, optimizer, scheduler, scaler, config.checkpoint_name)
    
    print(f"NanoCore_Beta v2: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    try:
        for i in range(start_step, config.max_iters):
            model.train()
            xb, yb = get_batch(train_data, config)
            
            with torch.amp.autocast(device_type=('cuda' if 'cuda' in config.device else 'cpu'), enabled=(config.device == 'cuda')):
                _, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if i % config.eval_interval == 0:
                model.eval()
                # Estimate loss & Perplexity
                val_xb, val_yb = get_batch(val_data, config)
                with torch.no_grad():
                    _, v_loss = model(val_xb, val_yb)
                
                perplexity = math.exp(v_loss.item())
                print(f"Step {i} | Loss: {v_loss.item():.4f} | PPL: {perplexity:.2f}")
                
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    save_full_checkpoint(model, optimizer, scheduler, scaler, i, v_loss.item(), config.checkpoint_name)
                    
    except KeyboardInterrupt:
        print("\nSaving checkpoint before exit...")
        save_full_checkpoint(model, optimizer, scheduler, scaler, i, loss.item(), "interrupt_backup.pt")

    print("\n--- Testing NanoCore_Beta v2 Generation ---")
    generate(model, tokenizer, "U: Halo, siapa ini?\nA: ")
