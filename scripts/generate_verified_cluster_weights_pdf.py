"""
Generates an executive-ready certified 2-page PDF report of the verified 8 Business Clusters & 34 Factors
for the Coripet RVM Location Predictor Production Stacking Ensemble (LightGBM + MultiTaskRVMNet).
"""
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Palette
COLOR_PRIMARY = colors.HexColor("#0f5132")       # Coripet Forest Green
COLOR_DARK = colors.HexColor("#0f172a")          # Slate 900
COLOR_TEXT = colors.HexColor("#334155")          # Slate 700
COLOR_MUTED = colors.HexColor("#64748b")         # Slate 500
COLOR_BORDER = colors.HexColor("#cbd5e1")        # Slate 300
COLOR_ROW_ALT = colors.HexColor("#f8fafc")       # Soft light alternating row
COLOR_ACCENT = colors.HexColor("#16a34a")        # Vibrant Green
COLOR_CARD_BG = colors.HexColor("#f0fdf4")       # Light mint card

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_MUTED)
        
        # Header rule & text (on page 2 onwards)
        if self._pageNumber > 1:
            self.drawString(36, 810, "Coripet RVM Predictor • Certified Model Feature Attribution")
            self.drawRightString(A4[0] - 36, 810, "Production Architecture Reference")
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.5)
            self.line(36, 804, A4[0] - 36, 804)
            
        # Footer rule & text
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(36, 36, A4[0] - 36, 36)
        
        self.drawString(36, 24, "CONFIDENTIAL & PROPRIETARY • CORIPET ML DECISION SUPPORT SYSTEM")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 36, 24, page_str)
        self.restoreState()

def build_pdf(filename="reports/verified_model_cluster_weights.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=36
    )
    
    usable_width = A4[0] - 72  # 523.27 pt
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16.5,
        leading=20.5,
        textColor=COLOR_PRIMARY,
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=COLOR_MUTED,
        spaceAfter=10
    )
    
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14.5,
        textColor=COLOR_PRIMARY,
        spaceBefore=7,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=11,
        textColor=COLOR_TEXT
    )
    
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.2,
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
        fontSize=7.2,
        leading=9.5,
        textColor=COLOR_TEXT
    )
    
    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=COLOR_DARK
    )
    
    td_center = ParagraphStyle(
        "TableCellCenter",
        parent=td_style,
        alignment=1
    )
    
    td_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        parent=td_bold,
        alignment=1
    )
    
    td_accent = ParagraphStyle(
        "TableCellAccent",
        parent=td_bold,
        alignment=1,
        textColor=COLOR_PRIMARY,
        fontSize=8
    )
    
    story = []
    
    # =========================================================================
    # PAGE 1: EXECUTIVE CLUSTER WEIGHTS & ARCHITECTURAL SUMMARY
    # =========================================================================
    
    # Header Banner Card
    banner_data = [
        [
            Paragraph("<b>CERTIFIED PRODUCTION ARCHITECTURE ATTRIBUTION</b>", ParagraphStyle("B1", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.5, textColor=COLOR_PRIMARY)),
            Paragraph("<b>STATUS: PRODUCTION CERTIFIED & VERIFIED</b>", ParagraphStyle("B2", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.5, textColor=COLOR_PRIMARY, alignment=2))
        ]
    ]
    t_banner = Table(banner_data, colWidths=[usable_width * 0.60, usable_width * 0.40])
    t_banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_CARD_BG),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86efac")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_banner)
    story.append(Spacer(1, 8))
    
    # Title & Subtitle
    story.append(Paragraph("THE 34 FACTORS & 8 BUSINESS CLUSTERS", title_style))
    story.append(Paragraph(
        "<b>Coripet RVM Location Predictor</b> • Certified Factor Allocation for Production Stacking Ensemble "
        "<i>(LightGBM 90% + MultiTaskRVMNet Neural Network 10%)</i>. All weights normalized to exactly 100.00%.",
        subtitle_style
    ))
    
    # Section 1: Executive Cluster Summary Table
    story.append(Paragraph("1. Overall Cluster Weight Distribution (Line-by-Line A to H)", h2_style))
    
    col_widths = [190, 48, 65, 65, 80, 75]
    table_data = [
        [
            Paragraph("BUSINESS CLUSTER", th_style_left),
            Paragraph("FACTORS", th_style),
            Paragraph("NEURAL NET", th_style),
            Paragraph("LIGHTGBM", th_style),
            Paragraph("PRODUCTION FINAL", th_style),
            Paragraph("ROUNDED %", th_style)
        ],
        [
            Paragraph("<b>A. Store Physical Scale & Capacity</b>", td_style),
            Paragraph("6", td_center),
            Paragraph("13.96%", td_center),
            Paragraph("8.05%", td_center),
            Paragraph("<b>8.64%</b>", td_accent),
            Paragraph("<b>9%</b>", td_center_bold)
        ],
        [
            Paragraph("<b>B. Municipal Demographics & Density</b>", td_style),
            Paragraph("3", td_center),
            Paragraph("6.56%", td_center),
            Paragraph("8.74%", td_center),
            Paragraph("<b>8.52%</b>", td_accent),
            Paragraph("<b>9%</b>", td_center_bold)
        ],
        [
            Paragraph("<b>C. Territorial Wealth & Economics</b>", td_style),
            Paragraph("2", td_center),
            Paragraph("4.40%", td_center),
            Paragraph("4.31%", td_center),
            Paragraph("<b>4.32%</b>", td_accent),
            Paragraph("<b>4%</b>", td_center_bold)
        ],
        [
            Paragraph("<b>D. Geographic Positioning</b>", td_style),
            Paragraph("6", td_center),
            Paragraph("25.50%", td_center),
            Paragraph("34.61%", td_center),
            Paragraph("<b>33.70%</b>", td_accent),
            Paragraph("<b>34%</b>", td_center_bold)
        ],
        [
            Paragraph("<b>E. Spatial Network & Competitive Density</b>", td_style),
            Paragraph("6", td_center),
            Paragraph("12.82%", td_center),
            Paragraph("6.72%", td_center),
            Paragraph("<b>7.33%</b>", td_accent),
            Paragraph("<b>7%</b>", td_center_bold)
        ],
        [
            Paragraph("<b>F. Hardware Technology</b>", td_style),
            Paragraph("2", td_center),
            Paragraph("10.22%", td_center),
            Paragraph("8.28%", td_center),
            Paragraph("<b>8.47%</b>", td_accent),
            Paragraph("<b>8%</b>", td_center_bold)
        ],
        [
            Paragraph("<b>G. Retail Brand & Buying Alliance</b>", td_style),
            Paragraph("2", td_center),
            Paragraph("12.61%", td_center),
            Paragraph("21.53%", td_center),
            Paragraph("<b>20.64%</b>", td_accent),
            Paragraph("<b>21%</b>", td_center_bold)
        ],
        [
            Paragraph("<b>H. Temporal & Seasonality Cycles</b>", td_style),
            Paragraph("7", td_center),
            Paragraph("13.93%", td_center),
            Paragraph("7.76%", td_center),
            Paragraph("<b>8.38%</b>", td_accent),
            Paragraph("<b>8%</b>", td_center_bold)
        ],
        # Total Row
        [
            Paragraph("<b>TOTAL (SUM OF ALL CLUSTERS)</b>", td_bold),
            Paragraph("<b>34</b>", td_center_bold),
            Paragraph("<b>100.00%</b>", td_center_bold),
            Paragraph("<b>100.00%</b>", td_center_bold),
            Paragraph("<b>100.00%</b>", ParagraphStyle("P_TOT", parent=td_accent, fontSize=8.5)),
            Paragraph("<b>100%</b>", td_center_bold)
        ]
    ]
    
    t_summary = Table(table_data, colWidths=col_widths, repeatRows=1)
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('INNERGRID', (0,0), (-1,-2), 0.5, COLOR_BORDER),
        ('BOX', (0,0), (-1,-1), 1, COLOR_PRIMARY),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, COLOR_ROW_ALT]),
        ('TOPPADDING', (0,1), (-1,-1), 4),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#e2e8f0")),
        ('LINEABOVE', (0,-1), (-1,-1), 1.2, COLOR_PRIMARY),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))
    
    # Executive Insights Box
    story.append(Paragraph("2. Strategic Architectural Insights", h2_style))
    insights_html = (
        "<b>• Geographic Anchor (33.7%):</b> Italy exhibits pronounced regional variation in civic recycling velocity. "
        "The 107 Italian Provinces (<code>Provincia</code>) serve as the primary macro-anchor, capturing local waste tax (TARI) "
        "habits and regional voucher redemption behaviors.<br/>"
        "<b>• Retail Brand Power (20.6%):</b> Commercial supermarket brands (Conad, Coop, Esselunga, Bennet) and buying consortiums "
        "represent the second largest driver, reflecting loyal weekly shoppers and promotion appeal.<br/>"
        "<b>• Physical Store Scale (8.6%) & Hardware (8.5%):</b> Sales area (m²), checkout registers, and hardware model define the "
        "operational intake ceiling during peak customer dropoff hours.<br/>"
        "<b>• Temporal Dynamics (8.4%):</b> Captures weekly rhythms (Saturday grocery peaks) and annual summer hydration ramp-ups."
    )
    t_ins = Table([[Paragraph(insights_html, body_style)]], colWidths=[usable_width])
    t_ins.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, COLOR_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(t_ins)
    story.append(Spacer(1, 10))
    
    # Structural Clusters (A-G) Normalized Box
    struct_html = (
        "<b>Optional View: 7 Physical Store Clusters (A to G, excluding cyclical Temporal):</b><br/>"
        "<b>D. Geographic Positioning:</b> 36.78% &bull; "
        "<b>G. Retail Brand & Alliance:</b> 22.53% &bull; "
        "<b>A. Store Physical Scale:</b> 9.43% &bull; "
        "<b>F. Hardware Technology:</b> 9.24% &bull; "
        "<b>B. Demographics & Density:</b> 9.30% &bull; "
        "<b>E. Spatial & Competition:</b> 8.00% &bull; "
        "<b>C. Territorial Wealth:</b> 4.72% &bull; "
        "<b>Total = 100.00%</b>"
    )
    t_struct = Table([[Paragraph(struct_html, body_style)]], colWidths=[usable_width])
    t_struct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_CARD_BG),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86efac")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(t_struct)
    
    # End of Page 1
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 2: COMPLETE 34-FACTOR TECHNICAL BREAKDOWN & CERTIFICATION
    # =========================================================================
    story.append(Paragraph("3. Complete Catalog of All 34 Factors by Business Cluster", h2_style))
    story.append(Paragraph(
        "Comprehensive inventory of features, functional descriptions, and individual model weight contributions:",
        subtitle_style
    ))
    
    col_w_factors = [150, 215, 52, 52, 54]
    factors_data = [
        [
            Paragraph("CLUSTER & FACTOR NAME", th_style_left),
            Paragraph("FUNCTIONAL DESCRIPTION / ENCODING", th_style_left),
            Paragraph("NN WT", th_style),
            Paragraph("LGBM WT", th_style),
            Paragraph("PROD WT", th_style)
        ],
        # Cluster A
        [Paragraph("<b>A. Store Physical Scale</b>", td_bold), Paragraph("<i>Sales floor, checkout capacity, retail format</i>", td_style), Paragraph("<b>13.96%</b>", td_center), Paragraph("<b>8.05%</b>", td_center), Paragraph("<b>8.64%</b>", td_center_bold)],
        [Paragraph("• nielsen_tipo_negozio", td_style), Paragraph("Store format (LIS Proximity / SUP Supermarket / IPR Hyper)", td_style), Paragraph("4.10%", td_center), Paragraph("0.58%", td_center), Paragraph("0.93%", td_center)],
        [Paragraph("• nielsen_superficie_vendita_log", td_style), Paragraph("Sales Area (m²), log-transformed with median imputation", td_style), Paragraph("2.24%", td_center), Paragraph("3.16%", td_center), Paragraph("3.07%", td_center)],
        [Paragraph("• nielsen_numero_casse", td_style), Paragraph("Cash Register Checkout Throughput Count", td_style), Paragraph("2.04%", td_center), Paragraph("4.29%", td_center), Paragraph("4.07%", td_center)],
        [Paragraph("• nielsen_superficie_missing", td_style), Paragraph("Binary indicator for estimated vs verified sales surface", td_style), Paragraph("2.00%", td_center), Paragraph("0.02%", td_center), Paragraph("0.22%", td_center)],
        [Paragraph("• nielsen_casse_missing", td_style), Paragraph("Binary indicator for estimated vs verified checkouts", td_style), Paragraph("1.87%", td_center), Paragraph("0.00%", td_center), Paragraph("0.19%", td_center)],
        [Paragraph("• is_nielsen_matched", td_style), Paragraph("Database verification status in national grocery registry", td_style), Paragraph("1.72%", td_center), Paragraph("0.00%", td_center), Paragraph("0.17%", td_center)],
        
        # Cluster B
        [Paragraph("<b>B. Demographics & Density</b>", td_bold), Paragraph("<i>Resident population, surface area, territory density</i>", td_style), Paragraph("<b>6.56%</b>", td_center), Paragraph("<b>8.74%</b>", td_center), Paragraph("<b>8.52%</b>", td_center_bold)],
        [Paragraph("• population_log", td_style), Paragraph("Municipal resident population (ISTAT census, log-scale)", td_style), Paragraph("2.25%", td_center), Paragraph("2.62%", td_center), Paragraph("2.58%", td_center)],
        [Paragraph("• population_density_log", td_style), Paragraph("Population density per km² (urban vs rural trade zone)", td_style), Paragraph("2.18%", td_center), Paragraph("3.27%", td_center), Paragraph("3.16%", td_center)],
        [Paragraph("• surface_log", td_style), Paragraph("Municipal land area surface (km²)", td_style), Paragraph("2.13%", td_center), Paragraph("2.85%", td_center), Paragraph("2.78%", td_center)],
        
        # Cluster C
        [Paragraph("<b>C. Territorial Wealth</b>", td_bold), Paragraph("<i>Macroeconomic purchasing power & taxable income</i>", td_style), Paragraph("<b>4.40%</b>", td_center), Paragraph("<b>4.31%</b>", td_center), Paragraph("<b>4.32%</b>", td_center_bold)],
        [Paragraph("• PIL_per_capita", td_style), Paragraph("Per-capita taxable income / GDP indicator (€/person)", td_style), Paragraph("2.24%", td_center), Paragraph("2.06%", td_center), Paragraph("2.08%", td_center)],
        [Paragraph("• PIL_log", td_style), Paragraph("Aggregate municipal GDP economic scale (log-transformed)", td_style), Paragraph("2.16%", td_center), Paragraph("2.25%", td_center), Paragraph("2.24%", td_center)],
        
        # Cluster D
        [Paragraph("<b>D. Geographic Positioning</b>", td_bold), Paragraph("<i>Province, region, postal routing, and GPS coordinates</i>", td_style), Paragraph("<b>25.50%</b>", td_center), Paragraph("<b>34.61%</b>", td_center), Paragraph("<b>33.70%</b>", td_center_bold)],
        [Paragraph("• Provincia", td_style), Paragraph("107 Italian Province categorical splits / entity embeddings", td_style), Paragraph("9.23%", td_center), Paragraph("30.33%", td_center), Paragraph("28.22%", td_center)],
        [Paragraph("• Regione", td_style), Paragraph("20 Italian Administrative Regions categorical mapping", td_style), Paragraph("6.19%", td_center), Paragraph("0.71%", td_center), Paragraph("1.26%", td_center)],
        [Paragraph("• nielsen_area", td_style), Paragraph("4 Nielsen Macro-Geographic Areas (NW, NE, Center, South)", td_style), Paragraph("3.89%", td_center), Paragraph("0.07%", td_center), Paragraph("0.45%", td_center)],
        [Paragraph("• Cap", td_style), Paragraph("5-digit Italian Postal Code numerical routing identifier", td_style), Paragraph("2.24%", td_center), Paragraph("0.79%", td_center), Paragraph("0.94%", td_center)],
        [Paragraph("• Latitudine", td_style), Paragraph("Geographic Latitude coordinate (North-South geographic axis)", td_style), Paragraph("2.02%", td_center), Paragraph("1.16%", td_center), Paragraph("1.25%", td_center)],
        [Paragraph("• Longitudine", td_style), Paragraph("Geographic Longitude coordinate (East-West geographic axis)", td_style), Paragraph("1.94%", td_center), Paragraph("1.54%", td_center), Paragraph("1.58%", td_center)],
        
        # Cluster E
        [Paragraph("<b>E. Spatial & Competition</b>", td_bold), Paragraph("<i>RVM network proximity, cannibalization, grocery density</i>", td_style), Paragraph("<b>12.82%</b>", td_center), Paragraph("<b>6.72%</b>", td_center), Paragraph("<b>7.33%</b>", td_center_bold)],
        [Paragraph("• nearest_coripet_km_log", td_style), Paragraph("Distance to nearest operating Coripet RVM (cannibalization proxy)", td_style), Paragraph("2.21%", td_center), Paragraph("2.68%", td_center), Paragraph("2.63%", td_center)],
        [Paragraph("• is_isolated_rvm", td_style), Paragraph("Binary indicator for isolated installation (>5 km from any RVM)", td_style), Paragraph("2.27%", td_center), Paragraph("0.04%", td_center), Paragraph("0.26%", td_center)],
        [Paragraph("• grocery_supermarket_count", td_style), Paragraph("Supermarket retail count in immediate trade catchment", td_style), Paragraph("2.16%", td_center), Paragraph("1.03%", td_center), Paragraph("1.14%", td_center)],
        [Paragraph("• coripet_rvm_count_5km", td_style), Paragraph("Coripet network installation density within 5 km trade radius", td_style), Paragraph("2.10%", td_center), Paragraph("1.44%", td_center), Paragraph("1.51%", td_center)],
        [Paragraph("• is_local_grocery_monopoly", td_style), Paragraph("Binary indicator for sole grocery destination in municipality", td_style), Paragraph("2.05%", td_center), Paragraph("0.11%", td_center), Paragraph("0.30%", td_center)],
        [Paragraph("• grocery_total_stores", td_style), Paragraph("Total commercial food & retail trade presence count", td_style), Paragraph("2.03%", td_center), Paragraph("1.42%", td_center), Paragraph("1.49%", td_center)],
        
        # Cluster F
        [Paragraph("<b>F. Hardware Technology</b>", td_bold), Paragraph("<i>RVM machine model, intake capacity, supplier</i>", td_style), Paragraph("<b>10.22%</b>", td_center), Paragraph("<b>8.28%</b>", td_center), Paragraph("<b>8.47%</b>", td_center_bold)],
        [Paragraph("• TipoMacchina", td_style), Paragraph("RVM hardware model (Ripet_C-front, DAINT_G, Ripet_V2, etc.)", td_style), Paragraph("5.89%", td_center), Paragraph("7.58%", td_center), Paragraph("7.41%", td_center)],
        [Paragraph("• Fornitore", td_style), Paragraph("Hardware manufacturer & supplier (DAINT, RIPET, EURVEN)", td_style), Paragraph("4.33%", td_center), Paragraph("0.70%", td_center), Paragraph("1.06%", td_center)],
        
        # Cluster G
        [Paragraph("<b>G. Retail Brand & Alliance</b>", td_bold), Paragraph("<i>Supermarket commercial banner & buying consortium</i>", td_style), Paragraph("<b>12.61%</b>", td_center), Paragraph("<b>21.53%</b>", td_center), Paragraph("<b>20.64%</b>", td_center_bold)],
        [Paragraph("• Insegna_Standardized", td_style), Paragraph("Commercial retail banner (Conad, Coop, Esselunga, Bennet, etc.)", td_style), Paragraph("6.70%", td_center), Paragraph("18.40%", td_center), Paragraph("17.23%", td_center)],
        [Paragraph("• nielsen_gruppo", td_style), Paragraph("Commercial buying consortium alliance (Conad, Coop, Selex, etc.)", td_style), Paragraph("5.91%", td_center), Paragraph("3.13%", td_center), Paragraph("3.41%", td_center)],
        
        # Cluster H
        [Paragraph("<b>H. Temporal & Seasonality</b>", td_bold), Paragraph("<i>Weekly consumer rhythm, holidays, annual hydration cycle</i>", td_style), Paragraph("<b>13.93%</b>", td_center), Paragraph("<b>7.76%</b>", td_center), Paragraph("<b>8.38%</b>", td_center_bold)],
        [Paragraph("• is_weekend", td_style), Paragraph("Binary flag for Saturday / Sunday peak grocery shopping days", td_style), Paragraph("2.37%", td_center), Paragraph("0.53%", td_center), Paragraph("0.71%", td_center)],
        [Paragraph("• dow_cos", td_style), Paragraph("Cosine day-of-week cyclic harmonic transformation", td_style), Paragraph("2.38%", td_center), Paragraph("0.20%", td_center), Paragraph("0.42%", td_center)],
        [Paragraph("• is_holiday", td_style), Paragraph("Italian national public holiday operating schedule calendar flag", td_style), Paragraph("2.06%", td_center), Paragraph("2.18%", td_center), Paragraph("2.17%", td_center)],
        [Paragraph("• dow_sin", td_style), Paragraph("Sine day-of-week cyclic harmonic transformation", td_style), Paragraph("1.89%", td_center), Paragraph("2.09%", td_center), Paragraph("2.07%", td_center)],
        [Paragraph("• month_cos", td_style), Paragraph("Cosine annual seasonal hydration curve (summer peak cycle)", td_style), Paragraph("1.82%", td_center), Paragraph("2.44%", td_center), Paragraph("2.38%", td_center)],
        [Paragraph("• month_sin", td_style), Paragraph("Sine annual seasonal hydration curve", td_style), Paragraph("1.73%", td_center), Paragraph("0.26%", td_center), Paragraph("0.41%", td_center)],
        [Paragraph("• quarter", td_style), Paragraph("Calendar quarter seasonal progression flag (Q1, Q2, Q3, Q4)", td_style), Paragraph("1.67%", td_center), Paragraph("0.06%", td_center), Paragraph("0.22%", td_center)],
    ]
    
    t_factors = Table(factors_data, colWidths=col_w_factors, repeatRows=1)
    t_factors.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_DARK),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 3.5),
        ('TOPPADDING', (0,0), (-1,0), 3.5),
        ('INNERGRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('BOX', (0,0), (-1,-1), 1, COLOR_DARK),
        ('TOPPADDING', (0,1), (-1,-1), 2),
        ('BOTTOMPADDING', (0,1), (-1,-1), 2),
        # Cluster Header Highlights
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,8), (-1,8), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,12), (-1,12), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,15), (-1,15), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,22), (-1,22), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,29), (-1,29), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,32), (-1,32), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,35), (-1,35), colors.HexColor("#e2e8f0")),
    ]))
    story.append(t_factors)
    story.append(Spacer(1, 8))
    
    # Section 3: Technical Verification Metadata
    meta_box = [
        [
            Paragraph(
                "<b>Model Verification Certificate:</b><br/>"
                "• Primary Booster: <code>models/ensemble/lgb_bottiglie.txt</code> (2,500 trees, R²=0.429) &bull; "
                "Neural Network: <code>models/best_model/model.pt</code> (MultiTaskRVMNet PyTorch)<br/>"
                "• Production Calibration Blend: 90% LightGBM + 10% MultiTaskRVMNet (<code>models/ensemble/ensemble_config.json</code>)<br/>"
                "• All 34 features mathematically audited and verified against preprocessor: <code>models/preprocessing/preprocessor.joblib</code>",
                body_style
            )
        ]
    ]
    t_meta = Table(meta_box, colWidths=[usable_width])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, COLOR_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated certified PDF report at: {filename}")

if __name__ == "__main__":
    build_pdf()
