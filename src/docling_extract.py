"""
docling_extract.py — Convert MedlinePlus PDFs to clean Markdown using Docling.
Handles multi-column layouts and preserves section headings.

Usage:
    python src/docling_extract.py --input data/pdfs/ --output data/markdown/
    python src/docling_extract.py --input data/pdfs/cholesterol.pdf --output data/markdown/
"""

import argparse
import os
from pathlib import Path


def extract_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    """
    Convert a single PDF to Markdown using Docling.
    Returns path to output markdown file.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "Docling not installed. Run: pip install docling"
        )

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (pdf_path.stem + ".md")

    print(f"[docling] Extracting: {pdf_path.name}")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    markdown_text = result.document.export_to_markdown()

    # Clean up common MedlinePlus navigation noise
    cleaned_lines = []
    skip_patterns = [
        "skip to main content",
        "skip navigation",
        "breadcrumb",
        "print this page",
        "email this page",
        "share this page",
        "back to top",
        "a-z index",
    ]
    for line in markdown_text.split("\n"):
        line_lower = line.strip().lower()
        if any(pattern in line_lower for pattern in skip_patterns):
            continue
        cleaned_lines.append(line)

    cleaned_markdown = "\n".join(cleaned_lines)

    # Remove excessive blank lines (more than 2 consecutive)
    import re
    cleaned_markdown = re.sub(r'\n{3,}', '\n\n', cleaned_markdown)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_markdown)

    print(f"[docling] Saved: {output_path} ({len(cleaned_markdown)} chars)")
    return str(output_path)


def extract_directory(input_dir: str, output_dir: str) -> list[str]:
    """Convert all PDFs in a directory to Markdown."""
    input_dir = Path(input_dir)
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[docling] No PDFs found in {input_dir}")
        return []

    print(f"[docling] Found {len(pdf_files)} PDFs in {input_dir}")
    output_paths = []

    for pdf_file in sorted(pdf_files):
        try:
            out_path = extract_pdf_to_markdown(str(pdf_file), output_dir)
            output_paths.append(out_path)
        except Exception as e:
            print(f"[docling] ERROR on {pdf_file.name}: {e}")

    print(f"\n[docling] Extraction complete: {len(output_paths)}/{len(pdf_files)} files converted")
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert MedlinePlus PDFs to Markdown using Docling"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--output", default="data/markdown/",
        help="Output directory for Markdown files (default: data/markdown/)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        extract_pdf_to_markdown(str(input_path), args.output)
    elif input_path.is_dir():
        extract_directory(str(input_path), args.output)
    else:
        print(f"[docling] ERROR: {args.input} is not a PDF file or directory")
        exit(1)


if __name__ == "__main__":
    main()
