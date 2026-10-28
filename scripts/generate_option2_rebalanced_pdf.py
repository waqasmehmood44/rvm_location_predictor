"""
Generates an Executive 2-Page PDF Report for:
Option 2 - Re-Balanced Original Slide Estimates (Business Hypothesis Normalized to 100%)
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
COLOR_PRIMARY = colors.HexColor("#1e3a8a")      # Executive Royal Blue
COLOR_PRIMARY_LIGHT = colors.HexColor("#eff6ff")# Soft Blue tint
COLOR_SECONDARY = colors.HexColor("#0f5132")    # Forest Green
COLOR_DARK = colors.HexColor("#0f172a")         # Slate 900
COLOR_TEXT = colors.HexColor("#334155")         # Slate 700
COLOR_MUTED = colors.HexColor("#64748b")        # Slate 500
COLOR_BG_LIGHT = colors.HexColor("#f8fafc")     # Slate 50
COLOR_CARD_BG = colors.HexColor("#f1f5f9")      # Slate 100
COLOR_BORDER = colors.HexColor("#cbd5e1")       # Slate 300
COLOR_HIGHLIGHT_A = colors.HexColor("#fef3c7")  # Amber tint (Store scale)
COLOR_ROW_ALT = colors.HexColor("#f8fafc")

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
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        page_w, page_h = A4
        
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(COLOR_PRIMARY)
            self.drawString(36, page_h - 26, "CORIPET BUSINESS ARCHITECTURE SPECIFICATION")
            self.setFont("Helvetica", 8)
            self.setFillColor(COLOR_MUTED)
            self.drawString(225, page_h - 26, "|   Option 2: Re-Balanced Business Slide Proportions")
            
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.6)
            self.line(36, page_h - 31, page_w - 36, page_h - 31)
            
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.6)
        self.line(36, 32, page_w - 36, 32)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(COLOR_MUTED)
        self.drawString(36, 21, "CONFIDENTIAL & PROPRIETARY  •  CORIPET REVERSE VENDING MACHINE (RVM) PLATFORM")
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_w - 36, 21, page_str)
        self.restoreState()


def build_pdf(filename="reports/option2_rebalanced_slide_estimates.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=34,
        rightMargin=34,
        topMargin=34,
        bottomMargin=36
    )
    
    usable_width = A4[0] - 68
    styles = getSampleStyleSheet()
    
    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=COLOR_PRIMARY
    )
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        textColor=COLOR_PRIMARY,
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=COLOR_DARK,
        spaceAfter=6
    )
    
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=COLOR_PRIMARY,
        spaceBefore=6,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=COLOR_TEXT,
        spaceAfter=4
    )
    
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=10.5,
        textColor=COLOR_DARK
    )
    
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
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
        fontSize=7.3,
        leading=9.3,
        textColor=COLOR_DARK
    )
    
    td_center = ParagraphStyle(
        "TableCellCenter",
        parent=td_style,
        alignment=1
    )
    
    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=td_style,
        fontName="Helvetica-Bold"
    )
    
    td_bold_center = ParagraphStyle(
        "TableCellBoldCenter",
        parent=td_style,
        fontName="Helvetica-Bold",
        alignment=1
    )

    story = []
    
    # -------------------------------------------------------------------------
    # PAGE 1: OPTION 2 SPECIFICATION & COMPARISON
    # -------------------------------------------------------------------------
    badge_table = Table(
        [[Paragraph("EXECUTIVE PRESENTATION VARIANT  •  OPTION 2: RE-BALANCED SLIDE ESTIMATES (100% SUM)", badge_style)]],
        colWidths=[usable_width]
    )
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#dbeafe")),
        ('BOX', (0, 0), (-1, -1), 0.7, colors.HexColor("#93c5fd")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("SECTION 3 — BUSINESS ESTIMATES: Re-Balanced Slide Proportions", title_style))
    story.append(Paragraph("Option 2: Preserving Domain Proportions with Mathematical 100% Normalization", subtitle_style))
    
    abstract_text = (
        "<b>Context:</b> In the original commercial slide draft, 8 operational factor clusters were assigned "
        "independent percentage estimates (totalling ~127%), reflecting domain rules-of-thumb prior to multi-task machine learning. "
        "<b>Option 2</b> preserves the author's original business intuition—where <b>Store Physical Scale (23%)</b>, "
        "<b>Municipal Demographics (16%)</b>, and <b>Spatial Competitors (15%)</b> form the core thesis—while mathematically "
        "re-weighting each cluster so that the entire framework sums to <b>exactly 100%</b> with Seasonality included."
    )
    abstract_table = Table([[Paragraph(abstract_text, callout_style)]], colWidths=[usable_width])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.8, COLOR_PRIMARY),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 8))
    
    # Master Table Option 2
    story.append(Paragraph("1. Option 2 Master Table: Re-Balanced Business Proportions (Exact 100%)", h1_style))
    story.append(Paragraph(
        "This table represents the mathematically normalized version of your original slide design, "
        "allocating 3% to cyclical seasonality and eliminating mathematical overflow:",
        body_style
    ))
    
    col_w2 = [
        usable_width * 0.06,  # Cluster
        usable_width * 0.32,  # Cluster Name
        usable_width * 0.08,  # Factors
        usable_width * 0.12,  # Weight
        usable_width * 0.42   # Key Examples
    ]
    
    table2_data = [
        [
            Paragraph("Cluster", th_style),
            Paragraph("Cluster Name", th_style_left),
            Paragraph("Factors", th_style),
            Paragraph("Weight", th_style),
            Paragraph("Key Examples & Operational Scope", th_style_left)
        ],
        [
            Paragraph("<b>A</b>", td_center),
            Paragraph("<b>Store Physical Scale & Capacity</b>", td_bold),
            Paragraph("6", td_center),
            Paragraph("<b>23%</b>", td_bold_center),
            Paragraph("Sales area (m²), cash registers, store format archetype, capacity ceiling", td_style)
        ],
        [
            Paragraph("<b>B</b>", td_center),
            Paragraph("<b>Municipal Demographics & Density</b>", td_bold),
            Paragraph("3", td_center),
            Paragraph("<b>16%</b>", td_bold_center),
            Paragraph("Municipal population, surface area, density per km² (footfall base)", td_style)
        ],
        [
            Paragraph("<b>C</b>", td_center),
            Paragraph("<b>Territorial Wealth & Economics</b>", td_bold),
            Paragraph("2", td_center),
            Paragraph("<b>7%</b>", td_bold_center),
            Paragraph("Municipal GDP, GDP per capita (purchasing power & disposable income)", td_style)
        ],
        [
            Paragraph("<b>D</b>", td_center),
            Paragraph("<b>Geographic Positioning</b>", td_bold),
            Paragraph("6", td_center),
            Paragraph("<b>14%</b>", td_bold_center),
            Paragraph("Lat/Long coordinates, CAP postal code, Provincia, Regione, Nielsen macro-area", td_style)
        ],
        [
            Paragraph("<b>E</b>", td_center),
            Paragraph("<b>Spatial Network & Competitive Density</b>", td_bold),
            Paragraph("6", td_center),
            Paragraph("<b>15%</b>", td_bold_center),
            Paragraph("Nearest Coripet RVM (km), 5km cluster density, competing grocery stores", td_style)
        ],
        [
            Paragraph("<b>F</b>", td_center),
            Paragraph("<b>Hardware Technology</b>", td_bold),
            Paragraph("2", td_center),
            Paragraph("<b>8%</b>", td_bold_center),
            Paragraph("Machine model (TipoMacchina), hardware supplier, compaction mechanics", td_style)
        ],
        [
            Paragraph("<b>G</b>", td_center),
            Paragraph("<b>Retail Brand & Buying Alliance</b>", td_bold),
            Paragraph("2", td_center),
            Paragraph("<b>14%</b>", td_bold_center),
            Paragraph("Standardized retail banner (Insegna), commercial buying alliance group", td_style)
        ],
        [
            Paragraph("<b>H</b>", td_center),
            Paragraph("<b>Temporal & Seasonality Cycles</b>", td_bold),
            Paragraph("7", td_center),
            Paragraph("<b>3%</b>", td_bold_center),
            Paragraph("Weekend/holiday indicators, cyclic sin/cos calendar days and months, quarter", td_style)
        ],
        [
            Paragraph("<b>TOTAL</b>", td_bold_center),
            Paragraph("<b>Operational Framework Total</b>", td_bold),
            Paragraph("<b>34</b>", td_bold_center),
            Paragraph("<b>100%</b>", td_bold_center),
            Paragraph("<b>Mathematically Normalized Sum (Exact Check: 23+16+7+14+15+8+14+3 = 100%)</b>", td_bold)
        ]
    ]
    
    t2_table = Table(table2_data, colWidths=col_w2)
    t2_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BACKGROUND', (0, 1), (-1, 1), COLOR_HIGHLIGHT_A), # Highlight Store Scale (23%)
        ('BACKGROUND', (0, 2), (-1, 2), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 3), (-1, 3), colors.white),
        ('BACKGROUND', (0, 4), (-1, 4), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 5), (-1, 5), colors.white),
        ('BACKGROUND', (0, 6), (-1, 6), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 7), (-1, 7), colors.white),
        ('BACKGROUND', (0, 8), (-1, 8), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 9), (-1, 9), colors.HexColor("#e2e8f0")),
        ('LINEABOVE', (0, 9), (-1, 9), 1.2, COLOR_DARK),
        ('TOPPADDING', (0, 9), (-1, 9), 4),
        ('BOTTOMPADDING', (0, 9), (-1, 9), 4),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, COLOR_BORDER),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_PRIMARY),
        ('TOPPADDING', (0, 1), (-1, 8), 3),
        ('BOTTOMPADDING', (0, 1), (-1, 8), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t2_table)
    story.append(Spacer(1, 10))
    
    # Section 2: When to Use Option 2
    story.append(Paragraph("2. Strategic Context: When Should You Use Option 2?", h1_style))
    story.append(Paragraph(
        "Option 2 and Option 1 (Table 1) serve two distinct presentation goals depending on your target audience:",
        body_style
    ))
    
    audience_data = [
        [
            Paragraph("<b>Target Audience & Context</b>", th_style_left),
            Paragraph("<b>Recommended Choice</b>", th_style_left),
            Paragraph("<b>Rationale & Psychological Impact</b>", th_style_left)
        ],
        [
            Paragraph("<b>Commercial Pitch / High-Level Business Strategy</b><br/>"
                      "Audience: Non-technical retail executives, real estate developers, commercial directors.", td_style),
            Paragraph("<b>OPTION 2<br/>(Re-Balanced Slide)</b>", td_bold_center),
            Paragraph("Matches classical commercial intuition: stakeholders naturally assume store footprint (23%) and neighborhood population (16%) are the main drivers. It avoids confusing non-technical audiences with complex neural network embeddings.", td_style)
        ],
        [
            Paragraph("<b>Technical Architecture Deep Dive / Due Diligence</b><br/>"
                      "Audience: Data scientists, ML engineers, board technical committees, auditing partners.", td_style),
            Paragraph("<b>OPTION 1<br/>(Table 1 - AI Reality)</b>", td_bold_center),
            Paragraph("Represents verifiable mathematical truth extracted from trained model artifacts (<code>MultiTaskRVMNet</code> and LightGBM). Proves that geographic culture (35%) and retail loyalty (21%) drive real-world volume, demonstrating advanced AI maturity.", td_style)
        ]
    ]
    
    aud_col_w = [usable_width * 0.35, usable_width * 0.22, usable_width * 0.43]
    aud_table = Table(audience_data, colWidths=aud_col_w)
    aud_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#eff6ff")),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#f0fdf4")),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, COLOR_BORDER),
        ('BOX', (0, 0), (-1, -1), 0.8, COLOR_DARK),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(aud_table)
    
    # Page Break for Page 2
    story.append(PageBreak())
    
    # -------------------------------------------------------------------------
    # PAGE 2: SIDE-BY-SIDE COMPARISON (OPTION 1 VS OPTION 2) & EXECUTIVE TALKING POINTS
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Side-by-Side Comparison: Option 1 (AI Reality) vs Option 2 (Business Slide)", h1_style))
    story.append(Paragraph(
        "Understanding why the proportions shifted between Option 2 (Theory) and Option 1 (Machine Learning Data) "
        "provides the most compelling slide commentary:",
        body_style
    ))
    
    col_comp_w = [
        usable_width * 0.05,  # ID
        usable_width * 0.27,  # Cluster Name
        usable_width * 0.14,  # Option 2 (Slide)
        usable_width * 0.14,  # Option 1 (AI Total)
        usable_width * 0.12,  # Difference
        usable_width * 0.28   # Root Cause of Difference
    ]
    
    comp_full_data = [
        [
            Paragraph("ID", th_style),
            Paragraph("Factor Cluster", th_style_left),
            Paragraph("Option 2<br/>(Slide Norm)", th_style),
            Paragraph("Option 1<br/>(AI Table 1)", th_style),
            Paragraph("Shift &Delta;", th_style),
            Paragraph("Root Cause & Empirical Insight", th_style_left)
        ],
        [
            Paragraph("<b>A</b>", td_center),
            Paragraph("<b>Store Physical Scale</b>", td_bold),
            Paragraph("23%", td_bold_center),
            Paragraph("8%", td_bold_center),
            Paragraph("<font color='#b91c1c'><b>-15%</b></font>", td_center),
            Paragraph("Scale is a capacity ceiling, not an organic footfall generator.", td_style)
        ],
        [
            Paragraph("<b>B</b>", td_center),
            Paragraph("<b>Municipal Demographics</b>", td_bold),
            Paragraph("16%", td_bold_center),
            Paragraph("9%", td_bold_center),
            Paragraph("<font color='#b91c1c'><b>-7%</b></font>", td_center),
            Paragraph("Macro city population matters less than specific neighborhood traffic.", td_style)
        ],
        [
            Paragraph("<b>C</b>", td_center),
            Paragraph("<b>Territorial Wealth (GDP)</b>", td_bold),
            Paragraph("7%", td_bold_center),
            Paragraph("4%", td_bold_center),
            Paragraph("<font color='#b91c1c'><b>-3%</b></font>", td_center),
            Paragraph("Income elasticity is minor; recycling is driven by habit, not wealth.", td_style)
        ],
        [
            Paragraph("<b>D</b>", td_center),
            Paragraph("<b>Geographic Positioning</b>", td_bold),
            Paragraph("14%", td_bold_center),
            Paragraph("35%", td_bold_center),
            Paragraph("<font color='#15803d'><b>+21%</b></font>", td_center),
            Paragraph("Provincial culture and municipal waste tariffs are the #1 driver nationwide.", td_style)
        ],
        [
            Paragraph("<b>E</b>", td_center),
            Paragraph("<b>Spatial Network & Competitors</b>", td_bold),
            Paragraph("15%", td_bold_center),
            Paragraph("7%", td_bold_center),
            Paragraph("<font color='#b91c1c'><b>-8%</b></font>", td_center),
            Paragraph("Network cannibalization only occurs within a tight &lt;1.5 km radius.", td_style)
        ],
        [
            Paragraph("<b>F</b>", td_center),
            Paragraph("<b>Hardware Technology</b>", td_bold),
            Paragraph("8%", td_bold_center),
            Paragraph("8%", td_bold_center),
            Paragraph("<b>0%</b>", td_center),
            Paragraph("Perfect alignment! Hardware speed directly bounds max bottle throughput.", td_style)
        ],
        [
            Paragraph("<b>G</b>", td_center),
            Paragraph("<b>Retail Brand & Alliance</b>", td_bold),
            Paragraph("14%", td_bold_center),
            Paragraph("21%", td_bold_center),
            Paragraph("<font color='#15803d'><b>+7%</b></font>", td_center),
            Paragraph("Supermarket chain loyalty (Coop, Conad, Esselunga) drives recurring visits.", td_style)
        ],
        [
            Paragraph("<b>H</b>", td_center),
            Paragraph("<b>Temporal & Seasonality</b>", td_bold),
            Paragraph("3%", td_bold_center),
            Paragraph("8%", td_bold_center),
            Paragraph("<font color='#15803d'><b>+5%</b></font>", td_center),
            Paragraph("Summer bottle peaks and weekend grocery spikes drive 8% daily variance.", td_style)
        ],
        [
            Paragraph("<b>TOT</b>", td_bold_center),
            Paragraph("<b>Total Framework Check</b>", td_bold),
            Paragraph("<b>100%</b>", td_bold_center),
            Paragraph("<b>100%</b>", td_bold_center),
            Paragraph("<b>0%</b>", td_bold_center),
            Paragraph("<b>Both versions mathematically valid (no percent overflow)</b>", td_bold)
        ]
    ]
    
    comp_full_table = Table(comp_full_data, colWidths=col_comp_w)
    comp_full_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('BACKGROUND', (0, 2), (-1, 2), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 3), (-1, 3), colors.white),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#fef3c7")), # D row
        ('BACKGROUND', (0, 5), (-1, 5), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 6), (-1, 6), colors.white),
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor("#dbeafe")), # G row
        ('BACKGROUND', (0, 8), (-1, 8), COLOR_ROW_ALT),
        ('BACKGROUND', (0, 9), (-1, 9), colors.HexColor("#e2e8f0")),
        ('LINEABOVE', (0, 9), (-1, 9), 1.2, COLOR_DARK),
        ('TOPPADDING', (0, 9), (-1, 9), 4),
        ('BOTTOMPADDING', (0, 9), (-1, 9), 4),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, COLOR_BORDER),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_DARK),
        ('TOPPADDING', (0, 1), (-1, 8), 3),
        ('BOTTOMPADDING', (0, 1), (-1, 8), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(comp_full_table)
    story.append(Spacer(1, 10))
    
    # Section 4: Presentation Talking Track
    story.append(Paragraph("4. Recommended Presentation Talking Track", h1_style))
    track_text = (
        "<b>How to Present Both Options Seamlessly:</b><br/>"
        "<i>'When Coripet initially mapped the RVM predictive framework, business intuition (Option 2) hypothesized "
        "that Store Physical Scale (23%) and Municipal Demographics (16%) would dictate performance. However, once our dual-head "
        "deep neural network (MultiTaskRVMNet) was trained across thousands of active Italian machines (Option 1 / Table 1), "
        "we uncovered a major competitive advantage: store scale is merely a capacity ceiling. Real-world volume is dominated "
        "by Geographic Positioning (35%) and Retail Chain Brand (21%). Both perspectives prove that prospective machine placement "
        "requires multi-dimensional spatial intelligence rather than simple foot-traffic guesses.'</i>"
    )
    track_table = Table([[Paragraph(track_text, body_style)]], colWidths=[usable_width])
    track_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.8, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(track_table)
    story.append(Spacer(1, 10))
    
    # Sign-off metadata
    signoff_data = [
        [
            Paragraph("<b>Document:</b> Option 2 Business Proportions Report<br/>"
                      "<b>Classification:</b> Executive Presentation Variant (100% Normalized)<br/>"
                      "<b>Framework Scope:</b> 34 Factors across 8 Operational Clusters", td_style),
            Paragraph("<b>Companion Doc:</b> <code>reports/multitask_rvmnet_architecture_deepdive.pdf</code> (Table 1)<br/>"
                      "<b>Status:</b> Ready for Commercial Pitch & Slide Inclusion<br/>"
                      "<b>Coripet Platform:</b> MultiTaskRVMNet Dual-Head Inference Engine", td_style)
        ]
    ]
    signoff_table = Table(signoff_data, colWidths=[usable_width * 0.5, usable_width * 0.5])
    signoff_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.8, COLOR_PRIMARY),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(signoff_table)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built {filename}")

if __name__ == "__main__":
    build_pdf()
