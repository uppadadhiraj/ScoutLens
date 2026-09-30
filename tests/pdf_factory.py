"""Build real PDFs for tests (no PDF-writing dependency): hand-written PDF syntax, plus pypdf for
blank/encrypted variants."""
from __future__ import annotations

import io

from pypdf import PdfReader, PdfWriter


def _escape(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return text.replace("•", "\\267")  # bullet in the PDF standard font encoding (not representable in latin-1)


def make_pdf(lines: list[str], *, per_page: int = 45) -> bytes:
    """A valid multi-page text PDF (Helvetica, one line per ``Tj``)."""
    pages = [lines[i : i + per_page] for i in range(0, max(len(lines), 1), per_page)] or [[]]
    objects: list[bytes] = []  # 1-indexed by position + 1

    def add(body: str) -> int:
        objects.append(body.encode("latin-1", "replace"))
        return len(objects)

    catalog = add("<< /Type /Catalog /Pages 2 0 R >>")
    kids_placeholder = add("")  # object 2, patched below
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for page_lines in pages:
        stream = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
        for line in page_lines:
            stream.append(f"({_escape(line)}) Tj T*")
        stream.append("ET")
        content = "\n".join(stream)
        content_id = add(f"<< /Length {len(content.encode('latin-1', 'replace'))} >>\nstream\n{content}\nendstream")
        page_ids.append(add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
        ))
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects[kids_placeholder - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def blank_pdf(pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def encrypted_pdf(lines: list[str], password: str = "secret") -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf(lines))))
    writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
