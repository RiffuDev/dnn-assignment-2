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

kv = {l.split(":", 1)[0]: l.split(":", 1)[1].strip() for l in meta[:2]}
intro = subprocess.run(
    ["pandoc", "-f", "markdown+tex_math_dollars", "-t", "html5", "--mathjax", "--wrap=none"],
    input="\n\n".join(meta[2:]),
    capture_output=True,
    text=True,
    check=True,
).stdout
card = (
    f'<div class="title-card"><h1>{title}</h1><div class="idrow">'
    + "".join(f'<div><span class="lab">{k.upper()}</span><span class="val">{v}</span></div>' for k, v in kv.items())
    + f'</div></div><div class="intro">{intro}</div>'
)
# number the Problem headings with a badge
n = [0]
def badge(m):
    if not m.group(2).startswith("Problem"):
        return m.group(0)
    n[0] += 1
    return f'<h2{m.group(1)}><span class="badge">{n[0]}</span>{m.group(2)}</h2>'
html = re.sub(r"<h2([^>]*)>(.*?)</h2>", badge, html)
running = f'{title}'
who = f'{kv.get("Name", "")} · {kv.get("Student ID", "")}'
doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="report_build/style.css"><style>@page {{ @top-left {{ content: "{running}"; }} @top-right {{ content: "{who}"; }} }} @page :first {{ @top-left {{ content: none; }} @top-right {{ content: none; }} }}</style></head><body>
{card}
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
