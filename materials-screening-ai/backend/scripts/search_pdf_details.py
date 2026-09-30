import pypdf, os

pdf_dir = 'materials-screening-ai/research/literature/pdfs'
for fname in sorted(os.listdir(pdf_dir)):
    if fname.endswith('.pdf'):
        p = os.path.join(pdf_dir, fname)
        reader = pypdf.PdfReader(p)
        print(f"=== {fname} ===")
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            for line in txt.splitlines():
                if any(w in line.lower() for w in ['gap', 'band', 'qsgw', 'b3pw', 'b1-wc', 'ev', 'hse']):
                    for kw in ['1.67', '1.30', '1.75', '2.45', '3.20', '3.25', '6.48', 'cs', 'mapb', 'cssn', 'batio', 'srtio']:
                        if kw in line.lower():
                            print(f"  [p.{i+1}] {line.strip()[:120]}")
