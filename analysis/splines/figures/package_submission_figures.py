"""Package Fig. 1 and S1-S7 separately from verified source PDFs.

Run with uv run --locked python and system Poppler (pdftoppm).
Retains original vector graphics; the French/English halves of S4-S6 are
placed side by side on one page, without rasterizing the source PDFs.
"""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/figures"
SOURCES = {
    "Figure_1": ["results/splines/figures/figure_1_spline.pdf"],
    "Figure_S1": ["results/primary/figures/figure_s1_histograms.pdf"],
    "Figure_S2": ["results/splines/figures/figure_romantic_signalling_spline.pdf"],
    "Figure_S3": [
        "results/splines/figures/figure_babel_affiliation_signalling_spline.pdf"
    ],
    "Figure_S4": [
        "results/courtship/doc_assets/figure_s4a.pdf",
        "results/courtship/doc_assets/figure_s4b.pdf",
    ],
    "Figure_S5": [
        "results/courtship/doc_assets/figure_s5a.pdf",
        "results/courtship/doc_assets/figure_s5b.pdf",
    ],
    "Figure_S6": [
        "results/courtship/doc_assets/figure_s6a.pdf",
        "results/courtship/doc_assets/figure_s6b.pdf",
    ],
    "Figure_S7": ["results/splines/robustness/Figure_S7.pdf"],
}
TITLES = {
    "Figure_S4": "Observed occurrence and conditional importance",
    "Figure_S5": "Threshold sensitivity: scores of at least 3",
    "Figure_S6": "Independent later-period models",
}


def combine(name, sources, destination):
    pages = [PdfReader(p).pages[0] for p in sources]
    widths = [float(p.mediabox.width) for p in pages]
    height = max(float(p.mediabox.height) for p in pages)
    gap, header = 18, 50
    width = sum(widths) + gap
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height + header)
    for i, p in enumerate(pages):
        page.merge_transformed_page(
            p, Transformation().translate(tx=0 if i == 0 else widths[0] + gap)
        )
    stream = BytesIO()
    top = canvas.Canvas(stream, pagesize=(width, height + header))
    top.setFont("Helvetica-Bold", 16)
    top.drawCentredString(width / 2, height + 31, TITLES[name])
    top.setFont("Helvetica-Bold", 14)
    top.drawString(18, height + 10, "a  French corpora")
    top.drawString(widths[0] + gap + 18, height + 10, "b  English corpora")
    top.save()
    page.merge_page(PdfReader(stream).pages[0])
    writer.add_metadata(
        {
            "/Title": name.replace("_", " "),
            "/Subject": TITLES[name],
            "/Creator": "Original vector PDF assembly",
        }
    )
    writer.write(destination)


def main():
    for folder in ["PDF", "PNG_300dpi", "previews"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    audit = []
    for name, paths in SOURCES.items():
        sources = [ROOT / p for p in paths]
        pdf = OUT / "PDF" / f"{name}.pdf"
        if len(sources) == 1:
            shutil.copy2(sources[0], pdf)
        else:
            combine(name, sources, pdf)
        reader = PdfReader(pdf)
        assert len(reader.pages) == 1
        assert len(reader.pages[0].extract_text()) > 50
        png = OUT / "PNG_300dpi" / f"{name}.png"
        subprocess.run(
            [
                "pdftoppm",
                "-singlefile",
                "-r",
                "300",
                "-png",
                str(pdf),
                str(png.with_suffix("")),
            ],
            check=True,
            capture_output=True,
        )
        with Image.open(png) as im:
            dpi = im.info.get("dpi")
            assert dpi and min(dpi) >= 299.9
            dims = im.size
            thumb = im.convert("RGB")
            thumb.thumbnail((760, 780))
            tile = Image.new("RGB", (800, 840), "white")
            tile.paste(
                thumb, ((800 - thumb.width) // 2, 45 + (780 - thumb.height) // 2)
            )
            ImageDraw.Draw(tile).text(
                (22, 10),
                name.replace("_", " "),
                fill="black",
                font=ImageFont.truetype(findfont("DejaVu Sans"), 26),
            )
            tile.save(OUT / "previews" / f"{name}.png")
        audit.append(
            dict(
                figure=name,
                pdf_pages=1,
                png_pixels=dims,
                png_dpi=dpi,
                pdf_bytes=pdf.stat().st_size,
                source_files=paths,
                source_sha256={
                    str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
                pdf_sha256=sha256(pdf.read_bytes()).hexdigest(),
                single_page_vector_sources=True,
            )
        )
    overview = Image.new("RGB", (3200, 1680), "white")
    for i, name in enumerate(SOURCES):
        with Image.open(OUT / "previews" / f"{name}.png") as thumb:
            overview.paste(thumb, ((i % 4) * 800, (i // 4) * 840))
    overview.save(OUT / "previews/contact_sheet.png")
    (OUT / "validation.json").write_text(json.dumps(audit, indent=2))
    with ZipFile(OUT / "Figures_1_S1-S7_submission.zip", "w", ZIP_DEFLATED) as z:
        for directory in ["PDF", "PNG_300dpi"]:
            for p in sorted((OUT / directory).iterdir()):
                z.write(p, str(p.relative_to(OUT)))
        for name in ["README.md", "validation.json", "Figure_legends.md"]:
            p = OUT / name
            if p.exists():
                z.write(p, name)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
