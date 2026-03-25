import os
import re

def clean_text(text):
    # 1. Remove common noise/headers/footers
    noise_patterns = [
        r'© ASCE',
        r'J\. Geotech\. Geoenviron\. Eng\.',
        r'P R O O F O N L Y',
        r'Article\s+in\s+Journal of Geotechnical and Geoenvironmental Engineering',
        r'DOI:\s+10\.1061/\(ASCE\)GT\.[0-9.-]+',
        r'https://orcid\.org/[0-9-]+',
        r'ISSN\s+1090-0241',
        r'F[0-9]+:[0-9]+', # Figure IDs
        r'T[0-9]+:[0-9]+', # Table IDs
    ]
    
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue
            
        # Skip lines that are just single numbers (likely page numbers)
        if re.match(r'^\d+$', line):
            continue
            
        # Skip noise patterns
        is_noise = False
        for pattern in noise_patterns:
            if re.search(pattern, line):
                is_noise = True
                break
        if is_noise:
            continue
            
        cleaned_lines.append(line)

    # 2. Join lines into paragraphs
    final_text = ""
    for i in range(len(cleaned_lines)):
        current_line = cleaned_lines[i]
        if not current_line:
            final_text += "\n\n"
            continue
            
        if i + 1 < len(cleaned_lines):
            next_line = cleaned_lines[i+1]
            # If current line doesn't end with punctuation and next line starts with lowercase
            # OR if current line is short but doesn't end in punctuation
            if (not re.search(r'[.?!:]$', current_line) and 
                next_line and 
                (next_line[0].islower() or re.match(r'^[a-z]', next_line))):
                final_text += current_line + " "
            else:
                final_text += current_line + "\n"
        else:
            final_text += current_line

    # 3. Final cleanup
    final_text = re.sub(r' +', ' ', final_text) # Multiple spaces to single
    final_text = re.sub(r'\n{3,}', '\n\n', final_text) # Max 2 newlines
    
    return final_text.strip()

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            print(f"Processing {filename}...")
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            cleaned_content = clean_text(content)
            
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

if __name__ == "__main__":
    input_folder = r"c:\AI_Local\AI Dahar\data"
    output_folder = r"c:\AI_Local\AI Dahar\data_cleaned"
    process_directory(input_folder, output_folder)
