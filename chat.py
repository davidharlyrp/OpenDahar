import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json
try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ==========================================
# 1. Konfigurasi Bobot (Hyperparameters)
# (HARUS SAMA dengan konfigurasi saat training)
# ==========================================
block_size = 256        # Context length
n_embd = 256           # Dimensi vektor embedding 
n_head = 8             # Jumlah attention heads
n_layer = 6            # Jumlah blok Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. Muat Metadata Vocabulary
# ==========================================
data_dir = "data"
vocab_path = os.path.join(data_dir, "vocab.json")

print(f"Memuat vocabulary dari '{vocab_path}'...")
with open(vocab_path, "r", encoding="utf-8") as f:
    chars = json.load(f)

vocab_size = len(chars)
print(f"Ukuran vocabulary: {vocab_size} karakter")

# Konversi String ke Indeks (Integer) dan sebaliknya
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    # Lakukan filtering karakter jika karakter tidak ada dalam vocab training
    return [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ==========================================
# 3. Arsitektur Model: Decoder-only Transformer
# ==========================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PyTorchGenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, stream=False):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if stream:
                char_str = decode(idx_next[0].tolist())
                print(char_str, end='', flush=True)
                
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 4. Memuat Model dan Menjalankan Inference
# ==========================================
print(f"Membangun model dan memuat bobot ke '{device}'...")
model = PyTorchGenModel()

model_path_pth = "model_geoteknik.pth"
model_path_safe = "model_geoteknik.safetensors"

if HAS_SAFETENSORS and os.path.exists(model_path_safe):
    print(f"Memuat bobot model dari '{model_path_safe}' (Safetensors)...")
    model.load_state_dict(load_file(model_path_safe, device=device))
elif os.path.exists(model_path_pth):
    print(f"Memuat bobot model dari '{model_path_pth}' (PyTorch legacy)...")
    model.load_state_dict(torch.load(model_path_pth, map_location=device, weights_only=True))
else:
    print(f"[Error] File model tidak ditemukan di '{model_path_safe}' maupun '{model_path_pth}'.")
    exit(1)
model = model.to(device)
model.eval() # Set mode evaluasi agar dropout dimatikan saat inference

print("\nModel berhasil dimuat! Masuk ke mode Chat Interaktif...")
print("Ketik 'keluar' atau 'exit' untuk menghentikan program.")

while True:
    try:
        print("\n" + "="*50)
        user_prompt = input("Masukkan prompt awal Anda: ")
        
        if user_prompt.lower() in ['keluar', 'exit']:
            print("Menghentikan program...")
            break
            
        if len(user_prompt) == 0:
            continue
            
        print(f"\n[AI Menjawab]: {user_prompt}", end='', flush=True)
        # Menyiapkan tensor integer untuk prompt, lalu menambah dimensi Batch menggunakan .unsqueeze(0)
        encoded_prompt = encode(user_prompt)
        # Jika prompt mengandung karakter yang tidak dikenal vocab, maka kita kasih pesan
        if len(encoded_prompt) == 0:
            print("\n[Error System]: Karakter pada prompt tidak dikenali model. Harap gunakan bahasa dari data.")
            continue
            
        context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
        
        # Panggil generate dengan stream=True untuk melihat proses berfikirnya karakter demi karakter
        _ = model.generate(context, max_new_tokens=150, stream=True)
        print("\n")
        
    except KeyboardInterrupt:
        print("\nMenghentikan program...")
        break
