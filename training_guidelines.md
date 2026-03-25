# Panduan Manajemen Model (Train vs. Reset)

Berikut adalah panduan ringkas untuk menentukan kapan Anda cukup melanjutkan training dan kapan harus melakukan reset total.

### 1. Lanjutkan Training (Just Train)
*Cukup jalankan `main.py` tanpa menghapus apa pun.*

*   **Data Baru yang Sesuai:** Anda menambahkan file `.txt` baru dengan topik serupa (misal: dokumen geoteknik tambahan).
*   **Optimasi Akurasi:** Nilai *loss* masih terlihat menurun dan Anda ingin model lebih memahami pola bahasa yang sudah dipelajari.
*   **Hemat Waktu:** Memanfaatkan pemahaman yang sudah ada di model lama agar lebih ahli dalam waktu singkat (*Fine-tuning*).

### 2. Reset Total (Hapus `.safetensors` & `.json`)
*Hapus `model_geoteknik.safetensors` dan `data/vocab.json` sebelum menjalankan `main.py`.*

*   **Ubah Konfigurasi (Hyperparameters):** Jika Anda mengubah nilai `n_embd`, `n_head`, `n_layer`, atau `block_size`. Struktur model baru tidak akan kompatibel dengan file bobot lama.
*   **Simbol/Karakter Baru:** Jika data baru mengandung banyak simbol unik, angka, atau karakter yang sebelumnya tidak ada sama sekali di data awal. Reset diperlukan agar model bisa membangun "kamus" baru di `vocab.json`.
*   **Model "Macet" atau Berantakan:** Jika model mulai memberikan jawaban repetitif atau *loss* tidak kunjung turun (indikasi *overfitting* atau *gradient explosion*), memulai dari nol dengan data yang lebih bersih seringkali membantu.

> [!TIP]
> **Selalu backup** file `model_geoteknik.safetensors` sebelum melakukan reset jika Anda ingin membandingkan hasil model lama dengan yang baru.
