"""Utilities for compiling LaTeX into PDF with graceful degradation."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency for actual cloud compilation
    import requests
except ImportError:  # pragma: no cover - fallback for environments without requests
    requests = None  # type: ignore[assignment]


class PDFCompiler:
    def __init__(self, *, engine: str = "pdflatex") -> None:
        self.engine = engine
        self.cloud_endpoint = Path.cwd().joinpath(".latex_api").read_text(encoding="utf-8").strip() if Path(".latex_api").exists() else None

    def compile(self, tex_path: Path, output_path: Path) -> Path:
        tex_path = tex_path.resolve()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cloud_endpoint and requests is not None:
            compiled = self._compile_cloud(tex_path, output_path)
            if compiled:
                return compiled

        compiled = self._compile_local(tex_path, output_path)
        if compiled:
            return compiled

        return self._write_minimal_pdf(tex_path, output_path)

    # Cloud compilation ---------------------------------------------------------
    def _compile_cloud(self, tex_path: Path, output_path: Path) -> Optional[Path]:
        if requests is None:
            return None
        try:
            payload = {"source": tex_path.read_text(encoding="utf-8")}
            response = requests.post(self.cloud_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            pdf_bytes = response.content
            if not pdf_bytes.startswith(b"%PDF"):
                return None
            output_path.write_bytes(pdf_bytes)
            return output_path
        except Exception:
            return None

    # Local compilation ---------------------------------------------------------
    def _compile_local(self, tex_path: Path, output_path: Path) -> Optional[Path]:
        engine_path = shutil.which(self.engine)
        if not engine_path:
            return None
        command = [engine_path, "-interaction=nonstopmode", str(tex_path)]
        try:
            subprocess.run(command, check=True, cwd=tex_path.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            return None
        generated_pdf = tex_path.with_suffix(".pdf")
        if generated_pdf.exists():
            generated_pdf.replace(output_path)
            return output_path
        return None

    # Minimal fallback ----------------------------------------------------------
    def _write_minimal_pdf(self, tex_path: Path, output_path: Path) -> Path:
        text = tex_path.read_text(encoding="utf-8")
        content = f"Minimal resume PDF fallback for {tex_path.name}.".encode("utf-8")
        pdf_bytes = _simple_pdf_bytes(content)
        output_path.write_bytes(pdf_bytes)
        return output_path


def _simple_pdf_bytes(body: bytes) -> bytes:
    """Generate a very small PDF document containing the provided body."""

    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    obj2 = b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
    stream = b"BT /F1 12 Tf 72 720 Td (" + body.replace(b"(", b"[").replace(b")", b"]") + b") Tj ET"
    obj3 = b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
    obj4_stream = b"4 0 obj<< /Length " + str(len(stream)).encode() + b" >>stream\n" + stream + b"\nendstream endobj\n"
    obj5 = b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
    xref_offset = len(header + obj1 + obj2 + obj3 + obj4_stream + obj5)
    xref = b"xref\n0 6\n0000000000 65535 f \n" + _xref_entry(len(header)) + _xref_entry(len(header + obj1)) + _xref_entry(len(header + obj1 + obj2)) + _xref_entry(len(header + obj1 + obj2 + obj3)) + _xref_entry(len(header + obj1 + obj2 + obj3 + obj4_stream))
    trailer = b"trailer<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_offset).encode() + b"\n%%EOF"
    return header + obj1 + obj2 + obj3 + obj4_stream + obj5 + xref + trailer


def _xref_entry(offset: int) -> bytes:
    return f"{offset:010} 00000 n \n".encode("ascii")
