"""Render TECHNICAL.md to HTML (black-and-white Mermaid, orthogonal edges) and print it to PDF.

    pip install -r docs/requirements-docs.txt
    python docs/build_pdf.py                     # -> TECHNICAL.html, TECHNICAL.pdf
    python docs/build_pdf.py --out <team_name>.pdf

Needs an installed Chrome or Edge for the headless print, and network access for mermaid.js.
"""

import argparse
import html
import os
import re
import shutil
import subprocess

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.6.0/dist/mermaid.min.js"

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  body {{ font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt;
         line-height: 1.45; color: #000; background: #fff; max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 18pt; margin: 0 0 6px; }}
  h2 {{ font-size: 13pt; margin: 20px 0 6px; border-bottom: 1px solid #000; padding-bottom: 2px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #000; padding: 3px 6px; text-align: left; vertical-align: top; }}
  th {{ background: #eee; }}
  code {{ font-family: Consolas, monospace; font-size: 9pt; white-space: nowrap; }}
  pre code {{ white-space: pre; }}
  pre {{ border: 1px solid #000; padding: 6px 8px; overflow-x: auto; }}
  pre.mermaid svg {{ max-height: 400px; }}
  pre.mermaid {{ border: none; text-align: center; padding: 0; margin: 10px 0 2px; }}
  tr, pre, .mermaid {{ page-break-inside: avoid; }}
  em {{ font-size: 9.5pt; }}
</style>
<script src="{mermaid}"></script>
<script>
  mermaid.initialize({{
    startOnLoad: false,
    theme: "base",
    themeVariables: {{
      background: "#ffffff", primaryColor: "#ffffff", primaryTextColor: "#000000",
      primaryBorderColor: "#000000", lineColor: "#000000", secondaryColor: "#ffffff",
      tertiaryColor: "#ffffff", edgeLabelBackground: "#ffffff",
      fontFamily: "Segoe UI, Helvetica, Arial, sans-serif", fontSize: "15px"
    }},
    flowchart: {{ curve: "step", htmlLabels: true, nodeSpacing: 40, rankSpacing: 40 }}
  }});
  window.addEventListener("load", async () => {{
    await mermaid.run({{ querySelector: "pre.mermaid" }});
    document.body.setAttribute("data-rendered", "1");
  }});
</script>
</head><body>
{body}
</body></html>
"""


def to_html(md_text):
    blocks = []

    def keep(match):
        blocks.append(match.group(1))
        return "\n\nMERMAIDBLOCK{}\n\n".format(len(blocks) - 1)

    md_text = re.sub(r"```mermaid\n(.*?)```", keep, md_text, flags=re.S)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    for i, src in enumerate(blocks):
        body = body.replace("<p>MERMAIDBLOCK{}</p>".format(i),
                            '<pre class="mermaid">{}</pre>'.format(html.escape(src)))
    title = re.search(r"^# (.+)$", md_text, flags=re.M).group(1)
    return TEMPLATE.format(title=html.escape(title), mermaid=MERMAID, body=body)


def find_browser():
    for path in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 shutil.which("google-chrome") or "", shutil.which("chromium") or ""):
        if path and os.path.isfile(path):
            return path
    raise SystemExit("no Chrome or Edge found for the PDF print")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=os.path.join(ROOT, "TECHNICAL.md"))
    parser.add_argument("--out", default=os.path.join(ROOT, "TECHNICAL.pdf"))
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as handle:
        page = to_html(handle.read())
    html_path = os.path.splitext(os.path.abspath(args.out))[0] + ".html"
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    print("wrote", html_path)
    if not args.html_only:
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu",
                        "--no-pdf-header-footer", "--virtual-time-budget=15000",
                        "--print-to-pdf=" + os.path.abspath(args.out),
                        "file:///" + html_path.replace("\\", "/")], check=True)
        print("wrote", os.path.abspath(args.out))
