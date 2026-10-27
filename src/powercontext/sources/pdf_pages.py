# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Complete per-page PDF capture with evidence-page retrieval."""

from __future__ import annotations

import hashlib
import io
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_SOURCE_NAME = "pdf-page"


class PDFDocumentInput(BaseModel):
    """One complete source PDF with its stable document identity."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page of a parsed PDF document."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured PDF page whose identity is ``(document_id, document_sha256, page_number)``."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable, delimiter-safe Source identity for one PDF page."""
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty trimmed string")
    if not isinstance(document_sha256, str) or len(document_sha256) != 64:
        raise ValueError("document_sha256 must be a 64-character lowercase hex digest")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of a PDF into per-page inputs, failing atomically."""
    document_id = value.document_id
    if not isinstance(document_id, str) or not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-empty trimmed string")

    content = value.content
    if not isinstance(content, bytes) or not content:
        raise ValueError("PDF content must be non-empty bytes")

    document_sha256 = hashlib.sha256(content).hexdigest()

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("PDF is encrypted and cannot be read without a password")
        pages = list(reader.pages)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF document could not be parsed") from error

    parsed: list[PDFPageInput] = []
    for index, page in enumerate(pages, start=1):
        try:
            text = page.extract_text()
        except Exception as error:
            raise ValueError("PDF page text could not be extracted") from error
        parsed.append(
            PDFPageInput(
                document_id=document_id,
                document_sha256=document_sha256,
                page_number=index,
                text="" if text is None else text,
            )
        )
    return tuple(parsed)


class PDFPageSourceAdapter:
    """Resolve and read the PDF page Source definition."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose each PDF page as textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(memory, question: str, limit: int = 10):
    """Return up to ``limit`` distinct evidence-page citations for one question.

    The scoped Memory already holds every parsed page as one entry. Because the
    evaluation is offline and lexical, this enumerates the current entries
    through the public Memory API and re-ranks them with a BM25/IDF scorer that
    stems tokens and expands a small finance-domain synonym lexicon, so pages
    whose wording differs from the question (``capex`` vs ``Capital spending``,
    ``legal battles`` vs ``Legal Proceedings``, ...) still surface.
    """
    from powercontext.artifacts import MemoryCitation
    from powercontext.sources._finance_retrieval import rank_pages

    listing = await memory.list()
    if not listing.entries:
        return ()

    scored = rank_pages(listing.entries, question)
    citations: list[MemoryCitation] = []
    seen: set[str] = set()
    for _, record in scored:
        citation: MemoryCitation = record.citation
        if citation.entry_id in seen:
            continue
        seen.add(citation.entry_id)
        citations.append(citation)
        if len(citations) >= limit:
            break
    return tuple(citations)
