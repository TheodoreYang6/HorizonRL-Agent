"""
文档解析器 — PDF/TXT/MD → 纯文本。

支持格式: .pdf (PyPDF2), .txt, .md, .py, .json, .csv
不依赖外部服务，纯离线解析。
"""

from __future__ import annotations

from pathlib import Path

# MIME 类型 → 扩展名映射
_SUPPORTED_TYPES = {
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-python": ".py",
    "application/json": ".json",
    "text/csv": ".csv",
    "application/pdf": ".pdf",
}


def parse_document(file_path: str | Path, filename: str = "") -> str:
    """解析文档文件，返回纯文本。

    根据文件扩展名选择解析器。不支持的格式抛出 ValueError。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not filename:
        filename = path.name

    suffix = path.suffix.lower()
    if not suffix and filename:
        suffix = Path(filename).suffix.lower()

    if suffix in (".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".xml"):
        return _read_text(path)

    if suffix == ".pdf":
        return _parse_pdf(path)

    raise ValueError(f"不支持的文档格式: {suffix}。支持: pdf, txt, md, py, json, csv")


def _read_text(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError(
            "PDF 解析需要 PyPDF2。运行: pip install PyPDF2"
        )

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages)
