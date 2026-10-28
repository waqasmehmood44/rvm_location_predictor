"""
Generates a 1-page PDF containing ONLY the Re-Balanced Slide Estimates Table (Option 2).
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Palette
COLOR_PRIMARY = colors.HexColor("#0f5132")      # Coripet Forest Green
COLOR_DARK = colors.HexColor("#0f172a")         # Slate 900
COLOR_TEXT = colors.HexColor("#334155")         # Slate 700
COLOR_MUTED = colors.HexColor("#64748b")        # Slate 500
COLOR_BORDER = colors.HexColor("#cbd5e1")       # Slate 300
COLOR_ROW_ALT = colors.HexColor("#f8fafc")      # Soft white/grey alternating
COLOR_ROW_HIGHLIGHT = colors.HexColor("#f0fdf4")# Subtle mint highlight

def build_pdf(filename="reports/rebalanced_slide_estimates_table.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 1 single page with generous margins
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40
    )
    
    usable_width = A4[0] - 72  # 523.27 pt
    styles = getSampleStyleSheet()
    
    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=COLOR_PRIMARY,
        alignment=1
    )
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=COLOR_PRIMARY,
        alignment=1,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=COLOR_MUTED,
        alignment=1,
        spaceAfter=18
    )
    
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    
    th_style_left = ParagraphStyle(
        "TableHeaderLeft",
        parent=th_style,
        alignment=0
    )
    
    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=COLOR_TEXT
    )
    
    td_center = ParagraphStyle(
        "TableCellCenter",
        parent=td_style,
        alignment=1
    )
    
    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=COLOR_DARK
    )
    
    td_bold_center = ParagraphStyle(
        "TableCellBoldCenter",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=COLOR_PRIMARY,
        alignment=1
    )
    
    story = []
    
    # Header Badge
    badge_table = Table(
        [[Paragraph("CORIPET RVM PREDICTOR  •  EXECUTIVE REFERENCE SHEET", badge_style)]],
        colWidths=[usable_width]
    )
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#d1e7dd")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#a3cfbb")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 10))
    
    # Title & Subtitle
    story.append(Paragraph("Factor Clusters: Re-Balanced Slide Estimates", title_style))
    story.append(Paragraph("Normalized Business Proportions (34 Factors across 8 Clusters &bull; Sum = Exactly 100%)", subtitle_style))
    story.append(Spacer(1, 6))
    
    # The Single Standalone Table
    col_w = [
        usable_width * 0.10,  # Cluster
        usable_width * 0.32,  # Cluster Name
        usable_width * 0.10,  # Factors
        usable_width * 0.11,  # Weight
        usable_width * 0.37   # Key Examples
    ]
    
    table_data = [
        [
            Paragraph("Cluster", th_style),
            Paragraph("Cluster Name", th_style_left),
            Paragraph("Factors", th_style),
            Paragraph("Weight", th_style),
            Paragraph("Key Examples & Scope", th_style_left)
        ],
        [
            Paragraph("<b>A</b>", td_center),
            Paragraph("<b>Store Physical Scale & Capacity</b>", td_bold),
            Paragraph("6", td_center),
            Paragraph("<b>23%</b>", td_bold_center),
            Paragraph("Sales area (m²), cash registers, store format archetype", td_style)
        ],
        [
            Paragraph("<b>B</b>", td_center),
            Paragraph("<b>Municipal Demographics & Density</b>", td_bold),
            Paragraph("3", td_center),
            Paragraph("<b>16%</b>", td_bold_center),
            Paragraph("Municipal population, land surface, density per km²", td_style)
        ],
        [
            Paragraph("<b>C</b>", td_center),
            Paragraph("<b>Territorial Wealth & Economics</b>", td_bold),
            Paragraph("2", td_center),
            Paragraph("<b>7%</b>", td_bold_center),
            Paragraph("Municipal GDP, GDP per capita (purchasing power)", td_style)
        ],
        [
            Paragraph("<b>D</b>", td_center),
            Paragraph("<b>Geographic Positioning</b>", td_bold),
            Paragraph("6", td_center),
            Paragraph("<b>14%</b>", td_bold_center),
            Paragraph("Lat/Long, CAP postal code, Provincia, Regione, Nielsen area", td_style)
        ],
        [
            Paragraph("<b>E</b>", td_center),
            Paragraph("<b>Spatial Network & Competitive Density</b>", td_bold),
            Paragraph("6", td_center),
            Paragraph("<b>15%</b>", td_bold_center),
            Paragraph("Nearest Coripet RVM (km), 5km count, competing grocery stores", td_style)
        ],
        [
            Paragraph("<b>F</b>", td_center),
            Paragraph("<b>Hardware Technology</b>", td_bold),
            Paragraph("2", td_center),
            Paragraph("<b>8%</b>", td_bold_center),
            Paragraph("Machine model (TipoMacchina), supplier, compaction speed", td_style)
        ],
        [
            Paragraph("<b>G</b>", td_center),
            Paragraph("<b>Retail Brand & Buying Alliance</b>", td_bold),
            Paragraph("2", td_center),
            Paragraph("<b>14%</b>", td_bold_center),
            Paragraph("Standardized banner, commercial buying group", td_style)
        ],
        [
            Paragraph("<b>H</b>", td_center),
            Paragraph("<b>Temporal & Seasonality Cycles</b>", td_bold),
            Paragraph("7", td_center),
            Paragraph("<b>3%</b>", td_bold_center),
            Paragraph("Weekend/holiday flags, DOW & month sin/cos, quarter", td_style)
        ],
        [
            Paragraph("<b>Total</b>", td_bold_center),
            Paragraph("<b>Operational Framework Total</b>", td_bold),
            Paragraph("<b>34</b>", td_bold_center),
            Paragraph("<b>100%</b>", td_bold_center),
            Paragraph("<b>Exact Mathematical Check (23+16+7+14+15+8+14+3 = 100%)</b>", td_bold)
        ]
    ]
    
    t = Table(table_data, colWidths=col_w)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('BACKGROUND', (0, 2), (-1, 2), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 3), (-1, 3), colors.white),
        ('BACKGROUND', (0, 4), (-1, 4), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 5), (-1, 5), colors.white),
        ('BACKGROUND', (0, 6), (-1, 6), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 7), (-1, 7), colors.white),
        ('BACKGROUND', (0, 8), (-1, 8), COLOR_ROW_ALT),
        
        # Total row
        ('BACKGROUND', (0, 9), (-1, 9), colors.HexColor("#e2e8f0")),
        ('LINEABOVE', (0, 9), (-1, 9), 1.5, COLOR_PRIMARY),
        ('TOPPADDING', (0, 9), (-1, 9), 9),
        ('BOTTOMPADDING', (0, 9), (-1, 9), 9),
        
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('BOX', (0, 0), (-1, -1), 1.2, COLOR_PRIMARY),
        ('TOPPADDING', (0, 1), (-1, 8), 7),
        ('BOTTOMPADDING', (0, 1), (-1, 8), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Simple clean footnote banner
    note_text = Paragraph(
        "<font color='#64748b'><b>Note:</b> Normalized from original presentation draft (~127% raw estimates) "
        "by incorporating temporal seasonality (3%) and re-scaling all clusters proportionally to sum to exactly 100%.</font>",
        td_style
    )
    note_table = Table([[note_text]], colWidths=[usable_width])
    note_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_ROW_ALT),
        ('BOX', (0, 0), (-1, -1), 0.6, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(note_table)
    
    # Custom Single Page Canvas with Footer
    def draw_footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setStrokeColor(COLOR_BORDER)
        canvas_obj.setLineWidth(0.6)
        canvas_obj.line(36, 32, A4[0] - 36, 32)
        
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(COLOR_MUTED)
        canvas_obj.drawString(36, 21, "CONFIDENTIAL  •  CORIPET REVERSE VENDING MACHINE (RVM) PREDICTOR")
        canvas_obj.drawRightString(A4[0] - 36, 21, "1-Page Executive Reference")
        canvas_obj.restoreState()
        
    doc.build(story, onFirstPage=draw_footer)
    print(f"[OK] Successfully built {filename}")

if __name__ == "__main__":
    build_pdf()
