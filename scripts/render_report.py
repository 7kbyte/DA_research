#!/usr/bin/env python3
"""Render a self-contained Chinese PDF and an HTML view from the report Markdown."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import markdown
from weasyprint import HTML


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("研究报告.md"))
    p.add_argument("--font-file", type=Path, required=True)
    p.add_argument("--pdf", type=Path)
    p.add_argument("--html", type=Path)
    args = p.parse_args()
    source = args.source.resolve()
    font = args.font_file.resolve(strict=True)
    pdf = args.pdf or source.with_suffix(".pdf")
    html_path = args.html or source.with_suffix(".html")
    body = markdown.markdown(source.read_text(), extensions=["tables", "fenced_code", "footnotes", "toc"],
                             output_format="html5")
    html_directory = html_path.resolve().parent
    def rebase_link(match):
        attribute, value = match.groups()
        parts = urlsplit(html.unescape(value))
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        target = (source.parent / unquote(parts.path)).resolve()
        relative = quote(os.path.relpath(target, html_directory))
        value = urlunsplit(("", "", relative, parts.query, parts.fragment))
        return attribute + '="' + html.escape(value, quote=True) + '"'
    body = re.sub(r'(href|src)="([^"]*)"', rebase_link, body)
    css = """
    @font-face { font-family: ReportCJK; src: url(FONT_URL); }
    @page { size: A4; margin: 18mm 17mm 21mm;
      @bottom-left { content: '固定回放预算下的领域适应与遗忘'; font-size: 8pt; color: #666; }
      @bottom-right { content: counter(page) ' / ' counter(pages); font-size: 8pt; color: #666; }
    }
    html { font-family: ReportCJK, 'Noto Serif CJK SC', 'Songti SC', SimSun, serif; color: #20252a; }
    body { font-size: 10.5pt; line-height: 1.6; }
    h1 { font-size: 23pt; line-height: 1.35; margin: 0 0 18pt; color: #183b50; }
    h2 { font-size: 14.5pt; margin-top: 22pt; border-bottom: .6pt solid #b2c4ce;
         padding-bottom: 4pt; color: #183b50; break-after: avoid; }
    h3 { font-size: 11.5pt; margin-top: 15pt; break-after: avoid; }
    p { margin: 7pt 0; orphans: 3; widows: 3; }
    a { color: #155f8b; text-decoration: none; overflow-wrap: anywhere; }
    table { border-collapse: collapse; width: 100%; font-size: 8.5pt; line-height: 1.45; margin: 12pt 0; break-inside: avoid; }
    thead { display: table-header-group; background: #eaf0f3; }
    th, td { border: .5pt solid #b7c4cc; padding: 5pt; vertical-align: top; overflow-wrap: anywhere; }
    tr { break-inside: avoid; }
    img { max-width: 100%; height: auto; }
    pre { white-space: pre-wrap; font-size: 8pt; line-height: 1.4; background: #f2f4f5;
          padding: 8pt; overflow-wrap: anywhere; }
    code { font-family: ReportCJK, monospace; font-size: .88em; overflow-wrap: anywhere; }
    blockquote { border-left: 3pt solid #b2c4ce; padding-left: 10pt; margin-left: 0; }
    li { margin: 3pt 0; }
    """.replace("FONT_URL", "'" + quote(os.path.relpath(font, html_directory)) + "'")
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>研究报告</title><style>'
    document += css + "</style></head><body>" + body + "</body></html>"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document)
    HTML(string=document, base_url=str(html_directory)).write_pdf(pdf)
    import weasyprint
    metadata = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "font": str(font), "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
                "weasyprint_version": weasyprint.__version__, "pdf": str(pdf),
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "html": str(html_path)}
    pdf.with_suffix(".render.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"pdf": str(pdf), "bytes": pdf.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
