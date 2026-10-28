"""
Generates an executive-ready, certified PDF report detailing all Defect and Outlier Detection
Conditions, Mathematical Formulations, Operational Protocols, and Model Training Governance
for the Coripet RVM Location Predictor & Fleet Management System.
"""
import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_OUTPUT = ROOT_DIR / "reports" / "Coripet_RVM_Defect_and_Outlier_Detection_Specification.pdf"
PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Corporate Palette matching Coripet Brand
COLOR_PRIMARY = colors.HexColor("#0f5132")       # Coripet Deep Forest Green
COLOR_SECONDARY = colors.HexColor("#198754")     # Vibrant Green
COLOR_DARK = colors.HexColor("#0f172a")          # Slate 900
COLOR_TEXT = colors.HexColor("#334155")          # Slate 700
COLOR_MUTED = colors.HexColor("#64748b")         # Slate 500
COLOR_BORDER = colors.HexColor("#cbd5e1")        # Slate 300
COLOR_ROW_ALT = colors.HexColor("#f8fafc")       # Soft light gray
COLOR_CARD_BG = colors.HexColor("#f0fdf4")       # Light mint background
COLOR_ALERT_BG = colors.HexColor("#fef2f2")      # Soft red alert background
COLOR_ALERT_TEXT = colors.HexColor("#991b1b")    # Deep red alert
COLOR_WARN_BG = colors.HexColor("#fffbeb")       # Soft amber warning background
COLOR_WARN_TEXT = colors.HexColor("#92400e")     # Deep amber warning


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
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_MUTED)

        # Running Header (Page 2 onwards)
        if self._pageNumber > 1:
            self.drawString(36, 810, "Coripet RVM Predictor • Defect & Outlier Detection Engineering Specification")
            self.drawRightString(A4[0] - 36, 810, "Fleet Diagnostics & Training Governance")
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.5)
            self.line(36, 804, A4[0] - 36, 804)

        # Running Footer
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(36, 36, A4[0] - 36, 36)

        self.drawString(36, 24, "CONFIDENTIAL & PROPRIETARY • CORIPET REVERSE VENDING MACHINE INTELLIGENCE")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 36, 24, page_str)
        self.restoreState()


def build_pdf(filename=str(PDF_OUTPUT)):
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
        fontSize=15.5,
        leading=19.5,
        textColor=COLOR_PRIMARY,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=COLOR_MUTED,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=COLOR_DARK,
        spaceBefore=10,
        spaceAfter=5
    )

    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=COLOR_PRIMARY,
        spaceBefore=6,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.8,
        textColor=COLOR_TEXT,
        spaceAfter=5
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=COLOR_DARK
    )

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.white
    )

    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.8,
        textColor=COLOR_TEXT
    )

    td_code = ParagraphStyle(
        "TableCode",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=6.5,
        leading=8.2,
        textColor=COLOR_PRIMARY
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE & EXECUTIVE ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("CORIPET RVM FLEET • DEFECT & OUTLIER DETECTION SPECIFICATION", title_style))
    story.append(Paragraph("Comprehensive Mathematical Formulations, 8 Certified Anomaly Regimes, Model Governance & Operational Action Protocols", subtitle_style))

    # Executive Overview Callout Box
    overview_text = (
        "<b>Executive Purpose:</b> This engineering standard establishes the definitive criteria for detecting, classifying, "
        "and handling anomalous machine behavior across Coripet's reverse vending machine (RVM) network (1,681+ machines). "
        "The system operates on a <b>Dual-Track Architecture</b>: (1) <b>AI Training Governance</b> to filter out hardware failure "
        "artifacts and avoid gradient corruption, and (2) <b>Operational Maintenance Dispatch</b> to provide field engineers "
        "and retail account managers with actionable diagnostic alerts."
    )
    overview_table = Table([[Paragraph(overview_text, callout_style)]], colWidths=[usable_width])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_CARD_BG),
        ('BOX', (0, 0), (-1, -1), 0.75, COLOR_SECONDARY),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 10))

    # Section 1: The Dual-Track Framework
    story.append(Paragraph("1. DUAL-TRACK ARCHITECTURAL FRAMEWORK", h1_style))
    framework_intro = (
        "In smart reverse vending networks, anomalies must never be treated simply as 'bad rows' to blindly delete. "
        "A sudden collapse from 1,000 to below 400 bottles indicates either a <i>hardware failure</i>, a <i>store staff bin bottleneck</i>, "
        "or a <i>supermarket promo expiration</i>. Discarding the machine would lose valuable commercial intelligence, while blindly training on "
        "hardware breakdowns distorts neural network predictions across the entire national supermarket network."
    )
    story.append(Paragraph(framework_intro, body_style))

    track_data = [
        [
            Paragraph("<b>Track A: Machine Learning Training Governance</b>", ParagraphStyle("TH_A", parent=th_style, textColor=COLOR_DARK)),
            Paragraph("<b>Track B: Coripet Operational & Field Dispatch</b>", ParagraphStyle("TH_B", parent=th_style, textColor=COLOR_DARK))
        ],
        [
            Paragraph(
                "• <b>Loss Masking (Zero-Mask)</b>: Prolonged breakdown periods (consecutive zeros) are masked out from Huber loss.<br/>"
                "• <b>Target Winsorization</b>: Single-day extreme spikes (>7,500 bottles) are clipped to 6,500 to prevent squared-error gradient explosions.<br/>"
                "• <b>Sample Downweighting</b>: Capacity-jammed and staff-congested days receive sample weight <i>w = 0.25</i>.<br/>"
                "• <b>Huber Loss δ=1.0</b>: Provides linear penalty transitions for heavy-tailed operational residuals.",
                td_style
            ),
            Paragraph(
                "• <b>Defective Machine Dispatch</b>: Automatically tags machines with hardware outages for urgent technician maintenance.<br/>"
                "• <b>Store Manager Alert</b>: Identifies full bin choke-points where staff fail to empty compactors twice daily.<br/>"
                "• <b>Commercial Optimization</b>: Identifies chronic underperformers hidden in basements or parking lots for store entrance relocation.<br/>"
                "• <b>Expansion Planning</b>: Identifies Super-Hubs (>3,000 bottles/day) requiring a 2nd RVM unit.",
                td_style
            )
        ]
    ]
    t_track = Table(track_data, colWidths=[usable_width * 0.5, usable_width * 0.5])
    t_track.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0, 1), (-1, -1), COLOR_ROW_ALT),
        ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_track)
    story.append(Spacer(1, 10))

    # Section 2: Complete 8-Condition Taxonomy Table
    story.append(Paragraph("2. COMPREHENSIVE OUTLIER & DEFECT TAXONOMY (8 CERTIFIED REGIMES)", h1_style))
    story.append(Paragraph("The system enforces 8 distinct mathematical and operational detection rules during data ingestion and model training:", body_style))

    taxonomy_rows = [
        [
            Paragraph("Regime & Code", th_style),
            Paragraph("Mathematical Detection Condition", th_style),
            Paragraph("Root Cause & Profile", th_style),
            Paragraph("AI Training Action", th_style),
            Paragraph("Field Ops Protocol", th_style)
        ],
        [
            Paragraph("<b>1. Hardware Defect / Offline</b><br/><font color='#b91c1c'>DEFECTIVE_OFFLINE</font>", td_code),
            Paragraph("Consecutive zeros ≥ 7 days<br/><i>OR</i> >20% zero-collection days across last 60 operational days (Mature only).", td_style),
            Paragraph("Compactor motor failure, optical board burnt, power switch turned off by store.", td_style),
            Paragraph("<b>Zero-Mask</b>:<br/>Exclude zero days from Huber loss calculation.", td_style),
            Paragraph("<b>P1 Dispatch</b>:<br/>Send technician to replace compactor/sensor.", td_style)
        ],
        [
            Paragraph("<b>2. High Volatility Swing</b><br/><font color='#c2410c'>VOLATILITY_SWING</font>", td_code),
            Paragraph("Base: V<sub>30d</sub> ∈ [400, 800]<br/>Peak: V<sub>7d</sub> ≥ 1.5×V<sub>30d</sub> (1000+)<br/>Drop: Next V<sub>7d</sub> ≤ 0.65×V<sub>30d</sub> (&lt;400)<br/>with CV > 0.85 within 60 days.", td_style),
            Paragraph("Retail campaign coupon expired; holiday tourist boom/bust; staff bin fatigue.", td_style),
            Paragraph("<b>Regime Segment</b>:<br/>Condition on cyclical calendar; evaluate Presenze drop.", td_style),
            Paragraph("<b>Campaign Audit</b>:<br/>Verify banner promo schedule vs staff bin emptying.", td_style)
        ],
        [
            Paragraph("<b>3. Capacity Jam Bottleneck</b><br/><font color='#d97706'>CAPACITY_BOTTLENECK</font>", td_code),
            Paragraph("Presenze ≥ 0.80×Median(P)<br/>AND Bottiglie ≤ 0.60×Median(B)<br/><i>OR</i> Bottles/Visit drops from ≥35 to &lt;15.", td_style),
            Paragraph("Storage bin full by midday; compactor halted; staff fails to replace collection bag.", td_style),
            Paragraph("<b>Downweight</b>:<br/>Set sample weight w = 0.25 on congested days.", td_style),
            Paragraph("<b>Store Alert</b>:<br/>Mandate twice-daily bag changes (12:00 & 18:00).", td_style)
        ],
        [
            Paragraph("<b>4. Extreme Bulk Spike</b><br/><font color='#7c3aed'>EXTREME_SPIKE_EVENT</font>", td_code),
            Paragraph("Single-day collection > 7,500 bottles (>99.8th percentile)<br/><i>OR</i> Z-score > 4.5 vs machine mean.", td_style),
            Paragraph("School recycling contest, restaurant bulk deposit, technician test loop.", td_style),
            Paragraph("<b>Winsorize</b>:<br/>Cap target label at 6,500 bottles; Huber δ=1.0.", td_style),
            Paragraph("<b>Commercial Check</b>:<br/>Verify if bulk partner deposited commercial bottles.", td_style)
        ],
        [
            Paragraph("<b>5. Chronic Underperformer</b><br/><font color='#dc2626'>CHRONIC_UNDERPERFORMER</font>", td_code),
            Paragraph("Store Area > 1,500 m²<br/>Density > 1,000/km²<br/>Median &lt; 250 bottles for ≥60d<br/>Residual &lt; -2.5×MAE.", td_style),
            Paragraph("Hidden machine placement (underground parking, behind carts, zero customer visibility).", td_style),
            Paragraph("<b>Downweight</b>:<br/>Sample weight w = 0.30; prevent lowering baseline expectations.", td_style),
            Paragraph("<b>Relocation Notice</b>:<br/>Request store manager move machine to main entrance.", td_style)
        ],
        [
            Paragraph("<b>6. Super-Hub Overperformer</b><br/><font color='#059669'>SUPER_HUB_OUTLIER</font>", td_code),
            Paragraph("Median volume > 3,000 bottles/day<br/>Residual > +2.5×MAE consistently across mature operating months.", td_style),
            Paragraph("Sole recycling point in municipality; local neighborhood green hub.", td_style),
            Paragraph("<b>Preserve</b>:<br/>Keep in training with full weight w = 1.0 (valid demand).", td_style),
            Paragraph("<b>Capital Expansion</b>:<br/>Deploy 2nd RVM unit to double store recycling capacity.", td_style)
        ],
        [
            Paragraph("<b>7. Zombie Sensor Trickle</b><br/><font color='#475569'>SENSOR_DRIFT_TRICKLE</font>", td_code),
            Paragraph("Daily volume 1 ≤ B ≤ 5 bottles every day with 0 or 1 user visits for ≥14 consecutive operational days.", td_style),
            Paragraph("Optical sensor dust trigger, misaligned laser barrier, or phantom counting loop while store closed.", td_style),
            Paragraph("<b>Cleanse</b>:<br/>Zero out ghost readings; mask as offline period.", td_style),
            Paragraph("<b>Hardware Recalibration</b>:<br/>Clean optical lenses & recalibrate laser gate.", td_style)
        ],
        [
            Paragraph("<b>8. Schedule Inconsistency</b><br/><font color='#0284c7'>SCHEDULE_ANOMALY</font>", td_code),
            Paragraph("Active collection > 500 bottles on verified national closing days (Easter, Christmas, closed Sundays).", td_style),
            Paragraph("External outdoor machine accessed while store closed, or internal clock date/time desynchronization.", td_style),
            Paragraph("<b>Temporal Harmonize</b>:<br/>Verify external machine attribute; resync NTP timestamp.", td_style),
            Paragraph("<b>Clock Sync</b>:<br/>Push automated NTP time update to machine onboard PC.", td_style)
        ]
    ]

    col_widths = [usable_width * 0.18, usable_width * 0.23, usable_width * 0.21, usable_width * 0.18, usable_width * 0.20]
    t_tax = Table(taxonomy_rows, colWidths=col_widths, repeatRows=1)
    t_tax.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT])
    ]))
    story.append(t_tax)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: DEEP DIVE: VOLATILITY SWINGS & MATHEMATICAL FORMULATIONS
    # =========================================================================
    story.append(Paragraph("3. DEEP DIVE: THE 600 → 1,000+ → &lt;400 VOLATILITY PATTERN", h1_style))
    deep_dive_intro = (
        "The pattern where an RVM performs at <b>600 bottles/day</b>, surges to <b>1,000+ bottles/day</b>, and subsequently crashes "
        "to <b>below 400 bottles/day</b> is the single most informative operational signature in retail recycling. "
        "The system runs a multi-stage diagnostic tree to determine the precise root cause:"
    )
    story.append(Paragraph(deep_dive_intro, body_style))

    # Diagnostic Flowchart Table
    flow_data = [
        [
            Paragraph("<b>Diagnostic Test</b>", th_style),
            Paragraph("<b>Evidence Condition</b>", th_style),
            Paragraph("<b>Diagnosed Root Cause</b>", th_style),
            Paragraph("<b>Corrective Action</b>", th_style)
        ],
        [
            Paragraph("<b>Test 1: Footfall Correlation</b>", td_code),
            Paragraph("PRESENZE (User visits) crashed by ≥50% concurrently with BOTTIGLIE.", td_style),
            Paragraph("<b>Customer Demand Shock</b><br/>Supermarket promotional coupon campaign ended, or summer tourist season concluded.", td_style),
            Paragraph("<b>Marketing</b>: Retain machine data in training; condition on calendar month features. Inform retailer of promo drop-off.", td_style)
        ],
        [
            Paragraph("<b>Test 2: Capacity Choke Check</b>", td_code),
            Paragraph("PRESENZE remained steady (≥80%), but BOTTIGLIE collapsed (&lt;400). Bottles/user dropped from 45 to &lt;15.", td_style),
            Paragraph("<b>Bin Full / Staff Burnout Jam</b><br/>High volume overwhelmed internal bins; staff delayed emptying compactor; machine sat in 'Out of Service' lock.", td_style),
            Paragraph("<b>Operations</b>: Downweight jammed days in training (w=0.25). Mandate scheduled bag emptying every 4 hours during peak retail.", td_style)
        ],
        [
            Paragraph("<b>Test 3: Mechanical Degradation</b>", td_code),
            Paragraph("Zero-collection hours spiked during prime hours (10:00-12:00, 17:00-19:00) with erratic sensor reads.", td_style),
            Paragraph("<b>Compactor / Belt Failure</b><br/>Heavy 1,000+ surge caused mechanical wear or optical sensor misalignment.", td_style),
            Paragraph("<b>Field Tech</b>: Dispatch field technician to service crusher teeth, replace drive belt, and clean barcode optics.", td_style)
        ],
        [
            Paragraph("<b>Test 4: Cannibalization Check</b>", td_code),
            Paragraph("Nearby RVM within 3 km was activated during the surge period.", td_style),
            Paragraph("<b>Network Cannibalization</b><br/>A competitor supermarket or new Coripet machine absorbed local recycling volume.", td_style),
            Paragraph("<b>Spatial ML</b>: Update <code>nearest_coripet_km</code> and <code>coripet_rvm_count_5km</code> spatial density features.", td_style)
        ]
    ]

    t_flow = Table(flow_data, colWidths=[usable_width * 0.22, usable_width * 0.26, usable_width * 0.26, usable_width * 0.26])
    t_flow.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT])
    ]))
    story.append(t_flow)
    story.append(Spacer(1, 10))

    # Section 4: Mathematical Formulations & Implementations
    story.append(Paragraph("4. MATHEMATICAL FORMULATIONS & TRAINING INTEGRATION", h1_style))

    math_p1 = (
        "<b>1. Machine Volatility Metric:</b> To capture non-stationary regime shifts without noise overfitting, we compute the "
        "moving Coefficient of Variation (CV) over a 60-day rolling window:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>CV<sub>t</sub> = &sigma;<sub>60d</sub> / &mu;<sub>60d</sub></b>, where machines with <b>CV<sub>t</sub> &gt; 0.85</b> are flagged for regime shift inspection.<br/>"
        "<b>2. MultiTask Huber Loss Formulation:</b> To shield the neural network from outlier distortion while maintaining sub-bottle precision, "
        "the objective function optimizes the Huber loss on log-transformed targets <i>z = log(1 + y)</i>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>L<sub>&delta;</sub>(z, z_pred) = 0.5(z - z_pred)<sup>2</sup></b> for |z - z_pred| &le; &delta;, and <b>&delta;(|z - z_pred| - 0.5&delta;)</b> for |z - z_pred| &gt; &delta; (with &delta; = 1.0).<br/>"
        "<b>3. Loss Weight Masking Formula:</b> Each observation <i>i</i> receives an automated sample weight <i>w<sub>i</sub></i> prior to gradient backpropagation:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>w<sub>i</sub> = 0.0</i> if <i>i &isin; DEFECTIVE_OFFLINE</i> (consecutive zeros in mature stores).<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>w<sub>i</sub> = 0.25</i> if <i>i &isin; CAPACITY_BOTTLENECK</i> (high visits, choked volumes).<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>w<sub>i</sub> = 0.30</i> if <i>i &isin; CHRONIC_UNDERPERFORMER</i> (obstructed physical placement).<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>w<sub>i</sub> = 1.00</i> for standard mature operations and verified super-hubs."
    )
    story.append(Paragraph(math_p1, body_style))
    story.append(Spacer(1, 6))

    # Section 5: Automated Pipeline & Output Artifacts
    story.append(Paragraph("5. AUTOMATED PIPELINE EXECUTION & ARTIFACT SPECIFICATION", h1_style))
    pipeline_text = (
        "The outlier detection engine is deployed in two synchronized execution modes:<br/>"
        "• <b>Automated Pre-Training Gate (Monthly at 05:00 AM)</b>: Integrated directly into <code>scripts/monthly_automated_retrain.py</code>. "
        "Before neural network weights are updated, the pipeline scans the active database, applies loss masking, clips extreme spikes, "
        "and archives the verified audit snapshot to <code>models/archive/outlier_machines_report.json</code>.<br/>"
        "• <b>Standalone Field Maintenance Auditor (On-Demand CLI)</b>: Available at any time via <code>python scripts/audit_machine_health.py</code>. "
        "Generates an operational spreadsheet <code>data/reports/defected_and_outlier_machines.csv</code> for retail operations and field engineering teams."
    )
    story.append(Paragraph(pipeline_text, body_style))
    story.append(Spacer(1, 6))

    # Output Schema Table
    schema_data = [
        [
            Paragraph("Field Name", th_style),
            Paragraph("Data Type", th_style),
            Paragraph("Description & Content Example", th_style),
            Paragraph("Target Audience", th_style)
        ],
        [
            Paragraph("<code>PF</code>", td_code),
            Paragraph("String (12)", td_style),
            Paragraph("Unique Physical Point ID (e.g., <code>MC069OUNP53U</code>)", td_style),
            Paragraph("All Teams", td_style)
        ],
        [
            Paragraph("<code>Store_Banner</code>", td_code),
            Paragraph("String", td_style),
            Paragraph("Retail Supermarket Banner (e.g., <code>ESSELUNGA</code>, <code>CONAD</code>)", td_style),
            Paragraph("Account Managers", td_style)
        ],
        [
            Paragraph("<code>Municipality_Region</code>", td_code),
            Paragraph("String", td_style),
            Paragraph("Geographic location (e.g., <code>Milano (Lombardia)</code>)", td_style),
            Paragraph("Field Logistics", td_style)
        ],
        [
            Paragraph("<code>Anomaly_Classification</code>", td_code),
            Paragraph("Enum", td_style),
            Paragraph("<code>DEFECTIVE_OFFLINE</code>, <code>VOLATILITY_SWING</code>, <code>CAPACITY_BOTTLENECK</code>, etc.", td_style),
            Paragraph("Maintenance Leads", td_style)
        ],
        [
            Paragraph("<code>Downtime_Days</code>", td_code),
            Paragraph("Integer", td_style),
            Paragraph("Total consecutive days with zero collections (e.g., <code>14</code>)", td_style),
            Paragraph("Field Engineers", td_style)
        ],
        [
            Paragraph("<code>Median_Actual_Bottles</code>", td_code),
            Paragraph("Float", td_style),
            Paragraph("Recent 30-day operating median collection volume (e.g., <code>215.0</code>)", td_style),
            Paragraph("Data Scientists", td_style)
        ],
        [
            Paragraph("<code>Predicted_Expected_Bottles</code>", td_code),
            Paragraph("Float", td_style),
            Paragraph("AI baseline expected volume based on demographics & store scale (e.g., <code>850.0</code>)", td_style),
            Paragraph("Commercial Team", td_style)
        ],
        [
            Paragraph("<code>Recommended_Action</code>", td_code),
            Paragraph("Text", td_style),
            Paragraph("Actionable directive: <i>'Dispatch technician to unjam compactor motor; check power cable'</i>", td_style),
            Paragraph("Store Managers & Techs", td_style)
        ]
    ]

    t_schema = Table(schema_data, colWidths=[usable_width * 0.22, usable_width * 0.12, usable_width * 0.46, usable_width * 0.20])
    t_schema.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT])
    ]))
    story.append(t_schema)

    # Build document with running headers/footers
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF specification: {filename} ({round(os.path.getsize(filename)/1024, 1)} KB)")


if __name__ == "__main__":
    build_pdf()
