import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json
try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from tqdm import tqdm
except ImportError:
    # Jika tqdm tidak ada, kita buat wrapper kosong (dummy) agar kode tidak error
    print("[Warning] tqdm tidak ditemukan. Disarankan install: pip install tqdm")
    def tqdm(iterable, **kwargs):
        return iterable

# ==========================================
# 1. Konfigurasi Bobot (Hyperparameters)
# ==========================================
batch_size = 16        # Jumlah rentetan sekuens yang diproses paralel per iterasinya
block_size = 256        # Context length (jumlah karakter maksimum yang diingat untuk prediksi berikutnya)
n_embd = 256           # Dimensi vektor embedding 
n_head = 8             # Jumlah attention heads (sehingga head_size = n_embd // n_head = 32)
n_layer = 6            # Jumlah blok Transformer
max_iters = 10000      # Total iterasi pelatihan
learning_rate = 1e-3   # Laju pembelajaran
eval_interval = 100    # Interval pencetakan loss pelatihan
eval_iters = 50        # Frekuensi iterasi sampel untuk mengestimasi rerata validation & train loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# ==========================================
# 2. Dataset Sampel Internal (Multi-File)
# ==========================================
data_dir = "data"
text = ""

# Membaca semua file .txt di dalam folder 'data'
print(f"Membaca file data dari folder '{data_dir}':")
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            # Membaca isi file, membuang baris baru menjadi spasi
            content = f.read().replace('\n', ' ')
            text += content + " " # Tambahkan spasi antar-dokumen
        # print(f" - Membaca {filename} ({len(content)} karakter)")

print(f"Total Karakter Keseluruhan: {len(text)/1e6:.2f} Juta ({len(text)})\n")

# ==========================================
# 3. Manual Tokenizer (Karakter-Level)
# ==========================================
vocab_path = os.path.join(data_dir, "vocab.json")
model_path_pth = "model_geoteknik.pth"
model_path_safe = "model_geoteknik.safetensors"

# Memuat vocabulary lama jika model sudah ada untuk menjaga konsistensi shape matriks
if os.path.exists(vocab_path) and (os.path.exists(model_path_pth) or (HAS_SAFETENSORS and os.path.exists(model_path_safe))):
    print(f"[Info] Memuat vocabulary lama dari '{vocab_path}' (Menjaga konsistensi shape matriks)...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        chars = json.load(f)
else:
    print(f"[Info] Membuat vocabulary baru dari dataset...")
    chars = sorted(list(set(text)))

vocab_size = len(chars)

# Konversi String ke Indeks (Integer) dan sebaliknya
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    # Hanya konversi karakter yang eksis di dalam vocab (mencegah KeyError jika ada huruf tak dikenal)
    return [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# Persiapan data split train dan validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% latih, 10% validasi
train_data = data[:n]
val_data = data[n:]

# Fungsi batch sampling acak berukuran (batch_size, block_size)
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==========================================
# 4. Arsitektur Model: Decoder-only Transformer
# ==========================================

class Head(nn.Module):
    """Sebuah head untuk layer self-attention tunggal"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril adalah pseudo-mask matriks segitiga bawah (Lower Triangular Matrix).
        # Tujuannya untuk masking agar node tidak bisa melihat input dari node token di sisi masa depan saat Autoregression
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Hitung attention scores (Affinities). 
        # Matriks Q dikali transpos dari K. Transpos hanya pada 2 matrix dimension terakhir (-2 dan -1).
        # Dibagi sqrt(head_size) alias (C**-0.5) untuk kestabilan normalisasi gradient (Scaled Dot-Product)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # Output size = (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
        
        # Masking attention matrix jika nilainya = 0 pada tril buffer, ubah menjadi angka negatif ekstrem (-inf)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        # Softmax mengubah output nilai tadi menjadi distribusi probabilitas dimana sisi mask (-inf) bakal dihitung jadi probabilitas nyaris 0
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        # Hasil weight affinity yang valid lalu dikalikan agregat V (Value) untuk menghasilkan output dari Head ini
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Paralelisasi beberapa Self-Attention Head dalam satu komputasi besar"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Proyeksi linear akhir sebagai penggabungan seluruh state attention dari heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Concatenate output semua hasil heads pada dimensi terakhir
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Layer Non-linear sederhana layaknya MLP setelah operasi Attention selesai"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Standar transformasi di transformer, inner dimension = 4xn_embd
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Kembalikan lagi ke dimensi model embedding
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Satu buah kompartemen utuh blok Transformer: Komunikasi (Attention) + Komputasi (FNN)"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # Pre-Layer Normalization untuk stabilisasi sebaran data gradient secara internal
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Terdapat skip connection (+ x) atau Residual Connection untuk menghindari Vanishing Gradient
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PyTorchGenModel(nn.Module):
    """Root module untuk Generative Text Language Model"""
    def __init__(self):
        super().__init__()
        # Table matrix pemetaan Token Karakter ke Dimensi Model
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Table matrix pemetaan Urutan Posisi (Positional Encoding).
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Instance blocks transformer n_layer kali berurutan
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm sebelum dilinearkan jadi vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx dan targets merupakan tensor of integers dengan ukuran matriks (B, T) alias (batch_size, block_size/Time)
        tok_emb = self.token_embedding_table(idx) # Akan menghasilkan ukuran (B,T,n_embd)
        # Positional Encoding array integer (0 sampai T-1) diubah ke dimensi (T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        
        # Gabungkan nilai Token Embedding dengan Positional Embedding
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.blocks(x)    # Terapkan blok komputasi transformer: (B,T,n_embd)
        x = self.ln_f(x)      # Normalisasi akhir (B, T, n_embd)
        
        logits = self.lm_head(x) # Diproyeksikan ke output vocab: menghasilkan ukuran (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # Karena F.cross_entropy PyTorch menerima multdimensi seperti BxC atau multidimensi di Flatten dulu, 
            # maka matriks logits dan targets diubah ukuran Matrix-nya
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, stream=False):
        # Autoregressive generation process (Generate karakter satu persatu)
        for _ in range(max_new_tokens):
            # Batasi context idx supaya tidak lebih dari 'block_size'
            idx_cond = idx[:, -block_size:]
            # Lakukan perhitungan probability ke depan tanpa butuh target/loss
            logits, loss = self(idx_cond)
            # Fokus ambil Prediksi Probability Logits PADA KARAKTER TERAKHIR SAJA (Time Step terakhir)
            logits = logits[:, -1, :] # Menjadi ukuran (B, C)
            # Terapkan Softmax untuk menghasilkan skor probabilitas absolut
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Ambil sampel (secara acak bergantung dari sebaran distribusi probabilitas di C vocab_size) untuk 1 prediksi token baru
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            if stream:
                # Menampilkan karakter yang baru diprediksi secara real-time
                char_str = decode(idx_next[0].tolist())
                print(char_str, end='', flush=True)
                
            # Menggabungkan token baru ini ke dalam list idx untuk iterasi berulang prediksi huruf selanjutnya
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = PyTorchGenModel()

# --- MODIFIKASI: Memuat Model Jika Sudah Ada ---
# (Path model sudah didefinisikan di bagian Tokenizer di atas)

if HAS_SAFETENSORS and os.path.exists(model_path_safe):
    print(f"\n[Info] Ditemukan bobot model safetensors di '{model_path_safe}'.")
    model.load_state_dict(load_file(model_path_safe, device=device))
elif os.path.exists(model_path_pth):
    print(f"\n[Info] Ditemukan bobot model PyTorch lama di '{model_path_pth}'.")
    print("Memuat bobot untuk melanjutkan training (Fine-Tuning/Resume)...")
    model.load_state_dict(torch.load(model_path_pth, map_location=device, weights_only=True))
else:
    print(f"\n[Info] Tidak ada file model lama. Memulai training dari awal (Scratch)...")

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 5. Training Loop & Inference
# ==========================================
if __name__ == "__main__":
    total_params = sum(p.numel() for p in model.parameters())
    print(f"=====================================")
    print(f"Total Parameter Model: {total_params / 1e6:.2f} Juta ({total_params})")
    print(f"Menjalankan training dengan Device: {device}...")
    print(f"=====================================")
    
    # Progress bar untuk memantau proses training
    pbar = tqdm(range(max_iters), desc="Training Model", unit="iter")
    
    for iter in pbar:
        # Cetak metrik Evaluasi reguler ke progress bar (tidak print line baru)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            pbar.set_postfix({
                'train_loss': f"{losses['train']:.4f}", 
                'val_loss': f"{losses['val']:.4f}"
            })

        # Ambil sampel batch baru
        xb, yb = get_batch('train')
        
        # Evaluasi nilai loss
        logits, loss = model(xb, yb)
        
        # Backpropagation (pengkinian Gradien Bobot)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    print("\nTraining Selesai. Menyimpan bobot dan vocabulary...")

    # Simpan vocabulary (metadata)
    vocab_path = os.path.join(data_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(chars, f)
    print(f"Vocabulary disimpan di '{vocab_path}'")

    # Simpan bobot model
    model_path_pth = "model_geoteknik.pth"
    model_path_safe = "model_geoteknik.safetensors"
    
    if HAS_SAFETENSORS:
        save_file(model.state_dict(), model_path_safe)
        print(f"Bobot model (safetensors) disimpan di '{model_path_safe}'")
        # Hapus file .pth lama jika ada agar tidak membingungkan
        if os.path.exists(model_path_pth):
            os.remove(model_path_pth)
    else:
        torch.save(model.state_dict(), model_path_pth)
        print(f"Bobot model (.pth) disimpan di '{model_path_pth}'")
    
    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"\n[Info] Total Parameter Model setelah training: {total_params_after / 1e6:.2f} Juta ({total_params_after})")
    print("\nProses Training & Saving Selesai! Anda siap menggunakan chat.py.")
