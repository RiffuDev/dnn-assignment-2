"""report.md -> report.html (pandoc, math as TeX spans) -> MathJax SVG (node) -> report.pdf (headless Chrome)."""

import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "report_build"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

md = (ROOT / "report.md").read_text()

# title block: first lines up to the first "## "
head, body = md.split("\n## ", 1)
lines = [l for l in head.splitlines() if l.strip()]
title = lines[0].lstrip("# ").strip()
meta = lines[1:]
body = "## " + body


# figure lines: image followed by its "Figure N." caption -> raw HTML figure
def fig(m):
    src, cap = m.group(1), m.group(2)
    num, rest = cap.split(".", 1)
    return f'\n<figure><img src="{src}"><figcaption><b>{num}.</b>{rest}</figcaption></figure>\n'


body = re.sub(r"!\[[^\]]*\]\(([^)]+)\)\n\n(Figure \d+\..*)", fig, body)
# images inside a pair div: bare figures; caption paragraph right after the div
body = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", r'<figure><img src="\1"></figure>', body)
body = re.sub(r"^(Figure \d+(?: \(left\))?)\.", r'<span class="fcap"><b>\1.</b></span>', body, flags=re.M)
body = re.sub(r"(Figure \d+ \(right\))\.", r"<b>\1.</b>", body)
# table captions
body = re.sub(
    r"^(Table \d+)\.(.*)$", lambda m: f'<p class="tcap"><b>{m.group(1)}.</b>{m.group(2)}</p>', body, flags=re.M
)
# paragraph lead-ins like "Memory." or "(a) Gradient norm."
body = re.sub(
    r"^(\([a-d]\) [A-Z][^.]{2,40}\.)",
    r'<span class="lead">\1</span>',
    body,
    flags=re.M,
)
# start Problem 2 and the AI statement on a new page is not forced; keep flow

(B / "body.md").write_text(body)
html = subprocess.run(
    [
        "pandoc",
        str(B / "body.md"),
        "-f",
        "markdown+tex_math_dollars+pipe_tables",
        "-t",
        "html5",
        "--mathjax",
        "--wrap=none",
    ],
    capture_output=True,
    text=True,
    check=True,
).stdout

meta_html = "".join(
    f'<p class="meta"><b>{l.split(":", 1)[0]}:</b>{l.split(":", 1)[1]}</p>'
    if ":" in l.split(".")[0]
    else f'<p class="meta">{l}</p>'
    for l in meta
)
meta_html = subprocess.run(
    ["pandoc", "-f", "markdown+tex_math_dollars", "-t", "html5", "--mathjax", "--wrap=none"],
    input="\n\n".join(meta),
    capture_output=True,
    text=True,
    check=True,
).stdout
meta_html = re.sub(r"<p>(Name|Student ID|Chosen values):", r'<p class="meta"><b>\1:</b>', meta_html)
meta_html = meta_html.replace("<p>", '<p class="meta">')

doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="report_build/style.css"></head><body>
<div class="title-block"><h1>{title}</h1>{meta_html}</div>
{html}
</body></html>"""
(ROOT / "report_raw.html").write_text(doc)

subprocess.run(["node", str(B / "mathjax.js"), str(ROOT / "report_raw.html"), str(ROOT / "report.html")], check=True)
subprocess.run(
    [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        f"--print-to-pdf={ROOT / 'report.pdf'}",
        (ROOT / "report.html").as_uri(),
    ],
    check=True,
    capture_output=True,
)
(ROOT / "report_raw.html").unlink()
print("wrote", ROOT / "report.pdf")
