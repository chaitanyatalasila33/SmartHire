from io import BytesIO
from pathlib import Path


def extract_resume_text(filename: str, content: bytes) -> str:
    """Extract text from PDF, DOCX, or TXT bytes. Raises clear errors for unsupported/empty files."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        text = content.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        from docx import Document
        doc = Document(BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT resume.")
    if not text.strip():
        raise ValueError("No readable text was found. Try a text-based PDF or upload a TXT/DOCX file.")
    return text.strip()
