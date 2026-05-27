#!/usr/bin/env python3
import json
import csv
import os
from itertools import zip_longest

def main():
    input_file = 'Input.txt'
    json_file = 'aligned_english_afrikaans.json'
    csv_file = 'aligned_english_afrikaans.csv'

    if not os.path.exists(input_file):
        print(f"❌ Error: '{input_file}' not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. SPLIT BY SEPARATOR
    if '===' not in content:
        print("❌ Error: Could not find the '===' separator in Input.txt")
        return

    parts = content.split('===')
    if len(parts) != 2:
        print("❌ Error: Input.txt should have exactly one '===' separator.")
        return

    # 2. PROCESS ENGLISH (Top Half)
    source_lines = parts[0].strip().split('\n')
    source_lang = source_lines[0].strip()  # Should be "English"
    # Keep only lines that have text, ignoring the language header
    source_text = [line.strip() for line in source_lines[1:] if line.strip()]

    # 3. PROCESS AFRIKAANS (Bottom Half)
    target_lines = parts[1].strip().split('\n')
    target_lang = target_lines[0].strip()  # Should be "Afrikaans"
    # Keep only lines that have text, ignoring the language header
    target_text = [line.strip() for line in target_lines[1:] if line.strip()]

    print(f"📖 Found {len(source_text)} lines for {source_lang}")
    print(f"📖 Found {len(target_text)} lines for {target_lang}")

    # 4. VALIDATE PARITY
    if len(source_text) != len(target_text):
        print("\n⚠️ WARNING: Line count mismatch!")
        print(f"  {source_lang} has {len(source_text)} lines.")
        print(f"  {target_lang} has {len(target_text)} lines.")
        print("  The script will align them, but look for empty values at the end of the JSON/CSV.\n")

    aligned_data = []
    
    # 5. ALIGN AND PAIR (zip_longest ensures we don't lose data if lengths mismatch)
    for src, tgt in zip_longest(source_text, target_text, fillvalue=""):
        aligned_data.append({
            "english": src,
            "afrikaans": tgt
        })

    # 6. SAVE TO JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(aligned_data, f, indent=4, ensure_ascii=False)
    
    # 7. SAVE TO CSV
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["English", "Afrikaans"])
        for pair in aligned_data:
            writer.writerow([pair["english"], pair["afrikaans"]])

    print(f"✅ Successfully aligned {len(aligned_data)} pairs!")
    print(f"📁 Generated: {json_file}")
    print(f"📁 Generated: {csv_file}")

if __name__ == '__main__':
    main()