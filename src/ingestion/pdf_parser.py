"""
pdf_parser.py

Parses a single financial filing PDF (annual report / quarterly result) into
a list of structured "blocks" — either prose text or tables — each tagged
with page number and metadata so they survive downstream chunking without
getting mangled.

Design choice: we do NOT just dump full-page text. Real filings mix tables
and prose tightly, so naive text extraction (or naive fixed-size chunking
later) will slice a table in half mid-row and produce garbage. Instead we
extract tables and prose as SEPARATE block types per page, and keep tables
whole.

Usage:
    from pdf_parser import parse_filing

    blocks = parse_filing(
        pdf_path="data/raw_pdfs/reliance/reliance_annual_report_2024.pdf",
        company="Reliance Industries",
        doc_type="annual_report",
        period="FY2024",
    )
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber

# OCR fallback only kicks in if a page has ~no extractable text (i.e. it's
# a scanned image, not real text). Threshold is deliberately low — if the
# page has *any* meaningful text layer, trust pdfplumber over OCR.
MIN_CHARS_BEFORE_OCR_FALLBACK = 20


@dataclass
class FilingBlock:
    """A single unit of content extracted from a filing PDF."""

    text: str
    block_type: Literal["prose", "table"]
    company: str
    doc_type: str          # "annual_report" | "quarterly_result"
    period: str            # e.g. "FY2024", "Q2FY25"
    source_file: str
    page_number: int
    table_index: int | None = None   # which table on the page, if block_type == "table"
    extraction_method: Literal["text_layer", "ocr", "docling"] = "text_layer"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "block_type": self.block_type,
            "company": self.company,
            "doc_type": self.doc_type,
            "period": self.period,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "extraction_method": self.extraction_method,
            **self.metadata,
        }


def _table_to_text(table: list[list[str | None]]) -> str:
    """
    Convert a pdfplumber-extracted table (list of rows) into a single
    text block, formatted so the LLM can still read it as a table
    (markdown-ish), while keeping every row intact — this is the
    "don't split a table mid-row" guarantee.
    """
    if not table:
        return ""

    cleaned_rows = []
    for row in table:
        cells = [str(c).strip() if c is not None else "" for c in row]
        cleaned_rows.append(cells)

    header, *body = cleaned_rows
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        # pad/truncate row to header length defensively — real filing
        # tables often have ragged rows due to merged cells
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _ocr_page(page: "pdfplumber.page.Page") -> str:
    """
    Fallback OCR for pages with no usable text layer (scanned filings).
    Requires: pip install pdf2image pytesseract, plus the system
    binaries `poppler-utils` and `tesseract-ocr` installed.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        raise RuntimeError(
            "OCR fallback requires pdf2image + pytesseract "
            "(and system packages poppler-utils, tesseract-ocr)."
        ) from e

    # Render just this page to an image at a reasonable DPI for OCR accuracy
    images = convert_from_path(
        page.page_path if hasattr(page, "page_path") else None,
        first_page=page.page_number,
        last_page=page.page_number,
        dpi=300,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def parse_filing(
    pdf_path: str,
    company: str,
    doc_type: str,
    period: str,
) -> list[FilingBlock]:
    """
    Parse one filing PDF into a list of FilingBlock objects.

    Returns prose blocks (one per page, or per text region) and table
    blocks (one per detected table) in page order. This separation is
    what the chunker downstream relies on to avoid destroying tables.
    """
    pdf_path = str(pdf_path)
    source_file = Path(pdf_path).name
    blocks: list[FilingBlock] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_number = page.page_number

            # --- 1. Extract tables first, and remember their bounding
            # boxes so we can exclude that area from prose extraction
            # (otherwise the same numbers get duplicated as garbled prose).
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            for idx, table in enumerate(tables):
                extracted = table.extract()
                table_text = _table_to_text(extracted)
                if table_text.strip():
                    blocks.append(
                        FilingBlock(
                            text=table_text,
                            block_type="table",
                            company=company,
                            doc_type=doc_type,
                            period=period,
                            source_file=source_file,
                            page_number=page_number,
                            table_index=idx,
                            extraction_method="text_layer",
                        )
                    )

            # --- 2. Extract prose, excluding the table regions
            # Clamp table bboxes to page boundaries first — real-world PDFs
            # often have bboxes that slightly overflow the page, causing errors.
            page_bbox = page.bbox  # (x0, y0, x1, y1)
            page_for_text = page
            for bbox in table_bboxes:
                clamped = (
                    max(bbox[0], page_bbox[0]),
                    max(bbox[1], page_bbox[1]),
                    min(bbox[2], page_bbox[2]),
                    min(bbox[3], page_bbox[3]),
                )
                # skip if clamped bbox has no area
                if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
                    continue
                try:
                    page_for_text = page_for_text.outside_bbox(clamped)
                except ValueError:
                    continue

            prose_text = (page_for_text.extract_text() or "").strip()

            # --- 3. OCR fallback if the page has effectively no text
            # layer (common for scanned annexures / signed certificates).
            extraction_method = "text_layer"
            if len(prose_text) < MIN_CHARS_BEFORE_OCR_FALLBACK and not tables:
                try:
                    ocr_text = _ocr_page(page).strip()
                    if len(ocr_text) > len(prose_text):
                        prose_text = ocr_text
                        extraction_method = "ocr"
                except RuntimeError:
                    # OCR deps not installed — skip silently, page stays
                    # thin on content rather than crashing the whole run.
                    pass

            if prose_text:
                blocks.append(
                    FilingBlock(
                        text=prose_text,
                        block_type="prose",
                        company=company,
                        doc_type=doc_type,
                        period=period,
                        source_file=source_file,
                        page_number=page_number,
                        extraction_method=extraction_method,
                    )
                )

    return blocks


def _compact_markdown_table(md: str) -> str:
    """Docling pads every cell to the widest one in its column, which can
    triple a table's token count for no information gain. Collapse runs of
    spaces and shorten the separator row."""
    lines = []
    for line in md.splitlines():
        if re.fullmatch(r"\|[-|\s:]+\|", line.strip()):
            n_cols = line.count("|") - 1
            lines.append("|" + "---|" * n_cols)
        else:
            lines.append(re.sub(r" {2,}", " ", line).strip())
    return "\n".join(lines)


# OCR misreads a decimal point as a comma ("25108,30" for 25108.30). A comma
# followed by exactly two final digits can't be Indian digit grouping — the
# last group is always three digits ("1,90,845") — so it must be a decimal.
_OCR_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(\d{2})(?![\d,])")


def _fix_ocr_decimal_commas(text: str) -> str:
    return _OCR_DECIMAL_COMMA_RE.sub(r".\1", text)


def parse_filing_docling(
    pdf_path: str,
    company: str,
    doc_type: str,
    period: str,
    ocr: bool = False,
    force_ocr: bool = False,
) -> list[FilingBlock]:
    """
    Same contract as parse_filing(), but uses Docling's layout + table
    structure models instead of pdfplumber.find_tables().

    Why: pdfplumber's ruling-line heuristics put column boundaries through
    the middle of numbers on real filing tables (TCS's P&L came out as
    "Revenue from operati 93" — the tail of 63,973), and the prose pass then
    cut those half-numbers out of the table region. Docling reconstructs
    rows and multi-level column headers ("Three months ended - December 31,
    2024"), so a value always stays attached to its row label and period.

    Each table is also prefixed with the nearest preceding section heading
    (e.g. "Audited Consolidated Interim Statement of Financial Results"),
    which is what disambiguates consolidated vs standalone figures — the
    table body alone doesn't say which one it is.

    Requires: pip install -r requirements-ingest.txt, and the models
    downloaded once via `docling-tools models download layout tableformer`.

    ocr=True turns on Docling's OCR (RapidOCR, torch backend — models from
    `docling-tools models download rapidocr`). Needed for filings whose
    statements are scanned images with no text layer: HDFC Bank's Q1FY27
    results had 21 of 22 pages image-only, L&T's 18 of 32.

    force_ocr=True ignores the embedded text layer entirely. Some filings DO
    have a text layer, but it was produced by poor OCR at the company's end
    (Maruti's Q1FY27 consolidated table reads "<24 60R" for 524,698).
    Docling's force_full_page_ocr option did not help — table cells still
    came from the bad text layer — so each page is rendered to an image and
    converted as an image, which leaves OCR as the only source of text.
    """
    ocr = ocr or force_ocr
    import os
    import tempfile

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
    from docling_core.types.doc import SectionHeaderItem, TableItem, TextItem

    artifacts = os.getenv("DOCLING_ARTIFACTS_PATH", os.path.expanduser("~/.cache/docling/models"))
    opts = PdfPipelineOptions(
        artifacts_path=artifacts,
        do_ocr=ocr,
        ocr_options=RapidOcrOptions(backend="torch"),
        do_table_structure=True,
    )

    # (docling document, page number to use instead of its own) pairs
    docs = []
    if force_ocr:
        import pypdfium2 as pdfium

        converter = DocumentConverter(
            format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)}
        )
        with tempfile.TemporaryDirectory() as tmp:
            for i, page in enumerate(pdfium.PdfDocument(str(pdf_path)), start=1):
                img_path = os.path.join(tmp, f"page_{i}.png")
                page.render(scale=3).to_pil().save(img_path)   # ~216 dpi
                docs.append((converter.convert(img_path).document, i))
    else:
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        docs.append((converter.convert(str(pdf_path)).document, None))

    source_file = Path(pdf_path).name
    blocks: list[FilingBlock] = []
    prose_by_page: dict[int, list[str]] = {}
    table_counts: dict[int, int] = {}
    last_heading = ""

    for doc, page_override in docs:
        for item, _level in doc.iterate_items():
            if not item.prov:
                continue
            page_number = page_override or item.prov[0].page_no

            if isinstance(item, TableItem):
                md = _compact_markdown_table(item.export_to_markdown(doc=doc))
                if not md.strip():
                    continue
                if ocr:
                    md = _fix_ocr_decimal_commas(md)
                caption = item.caption_text(doc) or last_heading
                text = f"{caption}\n{md}" if caption else md
                idx = table_counts.get(page_number, 0)
                table_counts[page_number] = idx + 1
                blocks.append(
                    FilingBlock(
                        text=text,
                        block_type="table",
                        company=company,
                        doc_type=doc_type,
                        period=period,
                        source_file=source_file,
                        page_number=page_number,
                        table_index=idx,
                        extraction_method="docling",
                    )
                )
            elif isinstance(item, TextItem):
                text = item.text.strip()
                if not text:
                    continue
                if isinstance(item, SectionHeaderItem):
                    last_heading = text
                prose_by_page.setdefault(page_number, []).append(text)

    for page_number, parts in prose_by_page.items():
        blocks.append(
            FilingBlock(
                text="\n".join(parts),
                block_type="prose",
                company=company,
                doc_type=doc_type,
                period=period,
                source_file=source_file,
                page_number=page_number,
                extraction_method="docling",
            )
        )

    blocks.sort(key=lambda b: (b.page_number, b.block_type != "table"))
    return blocks


def parse_filing_to_jsonl(
    pdf_path: str,
    company: str,
    doc_type: str,
    period: str,
    output_path: str,
    append: bool = True,
    parser: str = "docling",
    ocr: bool = False,
    force_ocr: bool = False,
) -> int:
    """
    Convenience wrapper: parse a filing and append the resulting blocks
    as JSON lines to output_path (data/processed/chunks.jsonl).
    Returns the number of blocks written.
    """
    import json

    if parser == "docling":
        blocks = parse_filing_docling(
            pdf_path, company, doc_type, period, ocr=ocr, force_ocr=force_ocr
        )
    else:
        blocks = parse_filing(pdf_path, company, doc_type, period)
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for block in blocks:
            f.write(json.dumps(block.to_dict(), ensure_ascii=False) + "\n")
    return len(blocks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a filing PDF into structured blocks.")
    parser.add_argument("pdf_path", help="Path to the filing PDF")
    parser.add_argument("--company", required=True)
    parser.add_argument("--doc-type", required=True, choices=["annual_report", "quarterly_result"])
    parser.add_argument("--period", required=True, help="e.g. FY2024 or Q2FY25")
    parser.add_argument(
        "--output",
        default="data/processed/chunks.jsonl",
        help="Where to append parsed blocks (JSONL)",
    )
    parser.add_argument(
        "--parser",
        default="docling",
        choices=["docling", "pdfplumber"],
        help="docling (default) or the original pdfplumber parser",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of appending")
    parser.add_argument("--ocr", action="store_true", help="Docling OCR for scanned (image-only) pages")
    parser.add_argument("--force-ocr", action="store_true", help="OCR every page, ignoring a bad embedded text layer")
    args = parser.parse_args()

    n = parse_filing_to_jsonl(
        pdf_path=args.pdf_path,
        company=args.company,
        doc_type=args.doc_type,
        period=args.period,
        output_path=args.output,
        append=not args.overwrite,
        parser=args.parser,
        ocr=args.ocr,
        force_ocr=args.force_ocr,
    )
    print(f"Parsed {n} blocks from {args.pdf_path} -> {args.output}")