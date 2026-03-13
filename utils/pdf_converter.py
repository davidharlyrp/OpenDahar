import os
import fitz  # PyMuPDF

def extract_text_from_pdfs(input_dir, output_dir):
    """
    Membaca semua file PDF dari input_dir, mengekstrak teksnya, 
    dan menyimpannya sebagai file .txt di output_dir.
    """
    if not os.path.exists(input_dir):
        print(f"Folder '{input_dir}' tidak ditemukan. Membuat folder secara otomatis...")
        os.makedirs(input_dir, exist_ok=True)
        print("Silakan masukkan file PDF Anda ke dalam folder tersebut dan jalankan ulang script ini.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if len(pdf_files) == 0:
        print(f"Tidak ada file PDF yang ditemukan di folder '{input_dir}'.")
        return

    print(f"Ditemukan {len(pdf_files)} file PDF. Memulai ekstraksi...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        
        # Buat nama file .txt yang sepadan dengan nama .pdf
        base_name = os.path.splitext(pdf_file)[0]
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        print(f" -> Memproses '{pdf_file}'...")
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text("text") + "\n"
                
            # Bersihkan sedikit baris baru berlebih jika perlu, dan simpan
            full_text = full_text.replace('\n\n', '\n')
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
                
            print(f"    Selesai. Disimpan sebagai '{txt_filename}'. ({len(full_text)} karakter)")
            
        except Exception as e:
            print(f"    [Error] Gagal memproses '{pdf_file}': {e}")

if __name__ == "__main__":
    # Path folder relatif dari root project (asumsi script dijalankan dari root)
    INPUT_FOLDER = os.path.join("data", "raw_pdf")
    OUTPUT_FOLDER = "data"
    
    print("====================================")
    print("=== PDF to TXT Converter Utility ===")
    print("====================================")
    extract_text_from_pdfs(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\nProses ekstraksi selesai!")
    print("Anda bisa langsung menjalankan 'python main.py' untuk mentraining model dengan teks baru ini.")
