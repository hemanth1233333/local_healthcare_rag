from pathlib import Path
from docling.document_converter import DocumentConverter

IN_DIR = Path("data/data/raw/guidelines")
OUT_DIR = Path("data/processed/guidelines_docling")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()

    pdfs = sorted(IN_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {IN_DIR.resolve()}")

    for p in pdfs:
        result = converter.convert(str(p))
        text = result.document.export_to_markdown()
        out = OUT_DIR / (p.stem + ".md")
        out.write_text(text, encoding="utf-8")
        print("Saved", out)

if __name__ == "__main__":
    main()
