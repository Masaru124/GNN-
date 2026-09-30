import pypdf, os, hashlib

pdf_dir = 'materials-screening-ai/research/literature/pdfs'
for fname in sorted(os.listdir(pdf_dir)):
    if fname.endswith('.pdf'):
        p = os.path.join(pdf_dir, fname)
        with open(p, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
        reader = pypdf.PdfReader(p)
        print(f"=== {fname} ({len(reader.pages)} pages, SHA256: {h}) ===")
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            lines = txt.splitlines()
            for line in lines:
                for term in ['1.67', '1.30', '1.75', '2.45', '3.20', '3.25', '6.48', 'Table ']:
                    if term in line:
                        print(f"  [Page {i+1}] {line.strip()[:100]}")
