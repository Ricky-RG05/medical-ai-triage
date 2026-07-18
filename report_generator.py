import os
from datetime import datetime
# ... rest of reportlab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Color palette ──
DARK_BLUE   = colors.HexColor("#1B3A5C")
MED_BLUE    = colors.HexColor("#2E6DA4")
LIGHT_BLUE  = colors.HexColor("#D6E8F7")
ACCENT      = colors.HexColor("#E8501A")
LIGHT_GRAY  = colors.HexColor("#F5F5F5")
MID_GRAY    = colors.HexColor("#CCCCCC")
WHITE       = colors.white
TEXT_DARK   = colors.HexColor("#1A1A1A")

def generate_pdf(patient_data: dict, analysis_result: str, condition_title: str, selected_folder: str, guide_code: str = "GPC"):    
    section_number_style = ParagraphStyle(
        "SectionNumber",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=MED_BLUE,
        spaceAfter=2,
        spaceBefore=10
    )

    # ✅ Always saves to the correct absolute location
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "reportes_pdfs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"reporte_{timestamp}.pdf")

    # ── Styles ──
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        fontName="Helvetica",
        fontSize=11,
        textColor=LIGHT_BLUE,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=WHITE,
        alignment=TA_LEFT,
        leftIndent=6,
        spaceAfter=0,
    )
    label_style = ParagraphStyle(
        "Label",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=MED_BLUE,
        alignment=TA_LEFT,
    )
    value_style = ParagraphStyle(
        "Value",
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_DARK,
        alignment=TA_LEFT,
    )
    body_style = ParagraphStyle(
        "BodyText",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY,
        leading=14,
        spaceAfter=6,
    )
    footer_style = ParagraphStyle(
        "Footer",
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
    )

    def section_header(text):
        """Returns a colored section header block."""
        header_table = Table(
            [[Paragraph(text, section_header_style)]],
            colWidths=[6.5 * inch]
        )
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), MED_BLUE),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        return header_table

    def build_patient_table(data: dict):
        """Builds a two-column patient data table."""
        items = list(data.items())
        rows = []
        for i in range(0, len(items), 2):
            left_key, left_val = items[i]
            if i + 1 < len(items):
                right_key, right_val = items[i + 1]
            else:
                right_key, right_val = "", ""

            rows.append([
                Paragraph(left_key.replace("_", " ").upper(), label_style),
                Paragraph(str(left_val), value_style),
                Paragraph(right_key.replace("_", " ").upper(), label_style),
                Paragraph(str(right_val), value_style),
            ])

        table = Table(rows, colWidths=[1.35*inch, 1.9*inch, 1.35*inch, 1.9*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GRAY),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID",          (0, 0), (-1, -1), 0.4, MID_GRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return table

    # ── Build story ──
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.6*inch,
        bottomMargin=0.75*inch,
    )
    story = []

    # ── Header banner ──
    header_data = [[
        Paragraph("REPORTE DE EVALUACIÓN CLÍNICA", title_style),
        Paragraph(condition_title, subtitle_style),
        Paragraph(f"{guide_code} · Primer Nivel de Atención", subtitle_style),
    ]]
    header_table = Table(header_data, colWidths=[6.5*inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    # ── Meta info row ──
    meta_date = datetime.now().strftime("%d/%m/%Y  %H:%M")
    meta_data = [[
        Paragraph(f"<b>Fecha:</b> {meta_date}", value_style),
        Paragraph(f"<b>Paciente:</b> {patient_data.get('Nombre', 'N/D')}", value_style),
        Paragraph(f"<b>Edad:</b> {patient_data.get('Edad', 'N/D')}", value_style),
    ]]
    meta_table = Table(meta_data, colWidths=[2.1*inch, 2.8*inch, 1.6*inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.3, MED_BLUE),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    # ── Section: Patient Data ──
    story.append(KeepTogether([
        section_header("📋  DATOS DEL PACIENTE"),
        Spacer(1, 6),
        build_patient_table(patient_data),
    ]))
    story.append(Spacer(1, 16))

    # ── Section: Clinical Analysis ──
    story.append(section_header(f"🩺  ANÁLISIS CLÍNICO (Basado en {guide_code})"))
    story.append(Spacer(1, 8))

    # Parse result into paragraphs
    for line in analysis_result.strip().split("\n"):
        line = line.strip()

        if not line:
            story.append(Spacer(1, 4))
            continue

        # Check if line starts with a section number like "1." or "2."
        if line and line[0].isdigit() and "." in line[:3]:
            # Split into header and body at the first colon
            if ":" in line:
                header_part, body_part = line.split(":", 1)
                story.append(Paragraph(header_part.strip(), section_number_style))
                if body_part.strip():
                    story.append(Paragraph(body_part.strip(), body_style))
            else:
                story.append(Paragraph(line, section_number_style))
        else:
            story.append(Paragraph(line, body_style))

    story.append(Spacer(1, 16))

    # ── Section: Disclaimer ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "⚠️  <b>Aviso importante:</b> Este reporte fue generado por un sistema de inteligencia artificial "
        "con fines de apoyo diagnóstico y NO sustituye el criterio médico profesional. "
        "Toda decisión clínica debe ser validada por un médico certificado.",
        ParagraphStyle("Warning", fontName="Helvetica-Oblique", fontSize=8.5,
                    textColor=colors.HexColor("#8B4000"), leading=13,
                    borderColor=ACCENT, borderWidth=0.5, borderPadding=6,
                    backColor=colors.HexColor("#FFF4ED"))
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Generado automáticamente · Sistema RAG con modelo Qwen2.5 7B · {meta_date}",
        footer_style
    ))

    # ── Build ──
    doc.build(story)
    print(f"📄 Reporte guardado en: {output_path}")