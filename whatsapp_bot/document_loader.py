"""
Turns an uploaded/fetched source document (PDF, Word, or a web page) into
KnowledgeBaseEntry rows — the automated replacement for manually retyping
document content into the admin one entry at a time.

Deliberately simple: paragraph-based chunking, no embeddings or retrieval.
The knowledge base is still small enough that services.py just stuffs every
active entry into the prompt wholesale — see
services._build_knowledge_base_context. If the KB ever grows past a few
dozen chunks, that's the point to add real retrieval (embed each chunk,
pull only the top-k relevant ones per reply) rather than raising the limits
here further.
"""
import re

import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

# Target chunk size in characters. Kept deliberately tight — a WhatsApp reply
# needs to paraphrase a chunk down to 1-3 sentences, and a large chunk (legal
# documents especially, with their own numbered sections and bullets) makes
# the model reproduce that structure instead of a short conversational
# answer. Small enough to stay a single focused fact/section; large enough
# to avoid one-sentence fragments that would each need their own noisy title.
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 550


def extract_text(document):
    """Extracts raw text from a KnowledgeBaseDocument's file or source_url."""
    if document.source_url:
        return _extract_from_url(document.source_url)
    if not document.file:
        raise ValueError("This document has neither a file nor a source URL.")

    name = document.file.name.lower()
    if name.endswith(".pdf"):
        return _extract_from_pdf(document.file)
    if name.endswith(".docx"):
        return _extract_from_docx(document.file)
    raise ValueError(f"Unsupported file type: {document.file.name} (only .pdf and .docx are supported)")


def _extract_from_pdf(file_field):
    file_field.open("rb")
    try:
        reader = PdfReader(file_field)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        file_field.close()


def _extract_from_docx(file_field):
    file_field.open("rb")
    try:
        doc = DocxDocument(file_field)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    finally:
        file_field.close()


def _extract_from_url(url):
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return html_to_text(response.text)


def html_to_text(html):
    """Strips a raw HTML page down to its visible text — split out from
    _extract_from_url so it can be tested without a network call."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    return "\n\n".join(lines)


def chunk_text(text):
    """
    Splits text into KB-entry-sized chunks by merging consecutive paragraphs
    up to MAX_CHUNK_CHARS, and breaking any single paragraph that's still too
    long at sentence boundaries. Paragraph-based rather than fixed-size
    windows so each chunk stays topically coherent — a whole FAQ answer or
    policy section per chunk, not a mid-thought cut.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_paragraph(para))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > MAX_CHUNK_CHARS and len(current) >= MIN_CHUNK_CHARS:
            chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _split_long_paragraph(paragraph):
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > MAX_CHUNK_CHARS and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _title_for_chunk(chunk):
    first_line = chunk.splitlines()[0].strip()
    return first_line if len(first_line) <= 80 else f"{first_line[:77]}..."


def process_document(document):
    """
    Extracts + chunks a KnowledgeBaseDocument and (re)creates its
    KnowledgeBaseEntry rows. Idempotent — safe to call again after replacing
    the file/URL, since it deletes previously generated entries first.
    Returns the number of entries created. Raises on failure — the caller
    (the admin action) is responsible for recording processing_error.
    """
    from .models import KnowledgeBaseEntry  # avoid import cycle at module load

    text = extract_text(document)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No extractable text found in this document.")

    document.entries.all().delete()
    for i, chunk in enumerate(chunks):
        KnowledgeBaseEntry.objects.create(
            title=_title_for_chunk(chunk),
            content=chunk,
            source_document=document,
            order=i,
        )
    return len(chunks)
