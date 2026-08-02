#!/usr/bin/env python3
import os
import docx
from docx.shared import Pt, Emu

docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '知识库')

def check_font_sizes(path):
    doc = docx.Document(path)
    sizes = {}

    for para in doc.paragraphs:
        for run in para.runs:
            try:
                if run.font and run.font.size:
                    size_pt = run.font.size.pt
                    size_twips = run.font.size.pt * 20  # 1pt = 20 twips

                    # 公文字号对照（以Points为单位）
                    # 1号=28pt, 2号=22pt, 3号=16pt, 4号=14pt
                    if size_pt >= 20 and size_pt <= 30:
                        doc_size = "2号"
                    elif size_pt >= 14 and size_pt < 20:
                        doc_size = "3号"
                    elif size_pt >= 12 and size_pt < 14:
                        doc_size = "4号"
                    elif size_pt >= 26 and size_pt <= 30:
                        doc_size = "1号"
                    else:
                        doc_size = f"{size_pt}pt"

                    key = f"{run.font.name}_{doc_size}_{size_pt}pt"
                    if key not in sizes:
                        sizes[key] = 0
                    sizes[key] += 1
            except:
                pass

    return sizes

files = [f for f in os.listdir(docs_dir) if f.endswith('.docx')][:5]

for fname in files:
    path = os.path.join(docs_dir, fname)
    print(f"\n=== {fname} ===")
    sizes = check_font_sizes(path)
    for k, v in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}处")

print("\n\n=== 字号对照说明 ===")
print("公文规范中：1号=28pt, 2号=22pt, 3号=16pt, 4号=14pt")