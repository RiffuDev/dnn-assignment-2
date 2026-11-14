"""Build <StudentID>_DNN_Assignment.zip from the submission tree only. Usage: python make_zip.py <StudentID>"""

import sys, zipfile
from pathlib import Path

sid = sys.argv[1]
files = ["README.md", "requirements.txt", "results.csv", "report.pdf"]
files += [f"problem{n}/problem{n}.ipynb" for n in (1, 2)]
files += [str(p) for n in (1, 2) for p in sorted(Path(f"problem{n}/figures").glob("*.png"))]
files += ["problem2/generated_samples.txt"]
missing = [f for f in files if not Path(f).exists()]
assert not missing, f"missing: {missing}"

# forbidden layers in Problem 2, Drive paths anywhere; the reference attention only in the P2.1 test cell
nb2 = Path("problem2/problem2.ipynb").read_text()
for bad in ("nn.MultiheadAttention", "nn.Transformer"):
    assert bad not in nb2, f"{bad} found in problem2"
for n in (1, 2):
    assert "/content/drive" not in Path(f"problem{n}/problem{n}.ipynb").read_text(), f"Drive path in problem{n}"
assert nb2.count("scaled_dot_product_attention") == 2, "reference attention outside the P2.1 test"

out = Path(f"{sid}_DNN_Assignment.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f"{sid}_DNN_Assignment/{f}")
mb = out.stat().st_size / 1e6
assert mb < 25, f"{mb:.1f} MB"
print(f"{out} {mb:.2f} MB, {len(files)} files, compliance checks passed")
