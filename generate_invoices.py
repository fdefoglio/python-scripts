#!/usr/bin/env python3
import re
import os

def parse_multi_student_tsv(filepath):
    """Reads concatenated TSV blocks, auto-stripping markdown code fences."""
    students = []
    current_record = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('```') or not line:
                continue
            if '\t' in line:
                key, val = line.split('\t', 1)
                key, val = key.strip(), val.strip()
                if key == 'Name Parent' and current_record:
                    students.append(current_record)
                    current_record = {}
                current_record[key] = val
        if current_record:
            students.append(current_record)
    return students

def get_surname(full_name):
    clean = re.sub(r'^(Mnr|Mev|Mr|Mrs|Dr|Prof)\.?\s*', '', full_name.strip()).strip()
    parts = clean.split()
    if not parts: return 'STUDENT'
    if len(parts) >= 3 and parts[-3].lower() in ('van', 'de', 'der', 'du'):
        return ' '.join(parts[-3:]).upper()
    if len(parts) >= 2 and parts[-2].lower() in ('van', 'de', 'der', 'du'):
        return ' '.join(parts[-2:]).upper()
    return parts[-1].upper()

def get_salutation(full_name):
    t = full_name.strip().lower()
    if t.startswith(('mev', 'mrs', 'ms')):
        return 'mev.'
    return 'mnr.'

def main():
    input_file = 'student.txt'
    if not os.path.exists(input_file):
        print(f"❌ Error: '{input_file}' not found.")
        return

    students = parse_multi_student_tsv(input_file)
    if not students:
        print("❌ No student records found.")
        return

    print(f"📖 Found {len(students)} student(s) in {input_file}")
    
    # Month is usually the same for a batch, so ask once
    new_month = input("📅 Billing Month (e.g., April 2026): ").strip()

    csv_header = "Name,Month,Customer ID,E-mail,Invoice #"

    for data in students:
        parent_name = data.get('Name Parent', '')
        student_name = data.get('Student', '')

        # Per-student prompts
        print(f"\n👤 Processing: {student_name} ({parent_name})")
        new_invoice = input(f"🧾 Invoice # for {student_name}: ").strip()
        new_lessons = input(f"🎸 Lessons for {student_name}: ").strip()

        csv_name    = re.sub(r'^(Mnr|Mev|Mr|Ms|Dr|Prof)\.?\s*', '', parent_name.strip()).strip()
        surname     = get_surname(parent_name)
        salutation  = get_salutation(parent_name)
        email_addr  = data.get('E-mail', '')
        cust_id     = data.get('Customer ID', '')
        instrument  = data.get('Instrument', '')

        # TSV Block
        tsv = "\n".join([
            f"Name Parent\t{csv_name}",
            f"Address 1\t{data.get('Address 1', '')}",
            f"Address 2\t{data.get('Address 2', '')}",
            f"Customer ID\t{cust_id}",
            f"E-mail\t{email_addr}",
            f"Invoice #\t{new_invoice}",
            f"Month\t{new_month}",
            f"Student\t{student_name}",
            f"Number of lessons\t{new_lessons}",
            f"Instrument\t{instrument}"
        ])

        csv_row = f"{csv_name},{new_month},{cust_id},{email_addr},{new_invoice}"
        subject = f"DE FOGLIO STUDIOS {new_invoice}: {surname}"

        email_body = f'''"{parent_name}" <{email_addr}>

Geagte mnr./mev. {surname},

Vind asseblief aangeheg u nuutste faktuur en staat. Die faktuur is vir dienste gelewer, en die staat verskaf 'n opsomming van u rekeningaktiwiteit.
Laat weet my asseblief indien u enige vrae het.
===
<i>Dear Mr/Mrs {surname},

Please find attached your latest invoice and statement. The invoice is for services rendered, and the statement provides a summary of your account activity.
Please let me know if you have any questions.</i>

Vriendelike groete / <i>Kind regards</i>'''

        # Combine & Save
        output_content = (
            f"```tsv\n{tsv}\n```\n\n"
            f"```csv\n{csv_header}\n{csv_row}\n```\n\n"
            f"```text\nSubject line: {subject}\n```\n\n"
            f"```text\n{email_body}\n```"
        )

        safe_student = student_name.replace(' ', '_').strip('.').strip()
        filename = f"{new_invoice}_{safe_student}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output_content)

        print(f"✅ Saved: {filename}")

    print(f"\n🎉 Successfully processed {len(students)} student(s)!")

if __name__ == '__main__':
    main()