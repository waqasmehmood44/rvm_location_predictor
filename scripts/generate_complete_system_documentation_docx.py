"""
Master Word Document Generator for Coripet RVM Prospective Predictor & Fleet Intelligence System.
Generates an exhaustive, production-grade technical and business manual in .docx format,
strictly aligned with the current live application features in src/ui/app.py,
src/ui/outlier_dashboard.py, src/ui/historical_vs_prediction.py, and the neural network in
src/model/network.py and src/model/predict.py.
"""

import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DOCX = ROOT_DIR / "docs" / "Coripet_RVM_Predictor_Complete_System_Manual.docx"
OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)

# Corporate Color Palette
HEX_PRIMARY = "0f5132"      # Deep Coripet Forest Green
HEX_SECONDARY = "198754"    # Vibrant Green
HEX_DARK = "0f172a"         # Dark Slate / Text Dark
HEX_MUTED = "64748b"        # Slate 500 / Subtitles
HEX_LIGHT_BG = "f8fafc"     # Off-white / light slate alternating row
HEX_CARD_BG = "f0fdf4"      # Mint / Soft Green card background
HEX_BORDER = "cbd5e1"       # Light gray border
HEX_ALERT_BG = "fff1f2"     # Red / rose alert background
HEX_WARN_BG = "fffbeb"      # Amber alert background
HEX_INFO_BG = "eff6ff"      # Soft blue background

COLOR_PRIMARY = RGBColor(15, 81, 50)
COLOR_SECONDARY = RGBColor(25, 135, 84)
COLOR_DARK = RGBColor(15, 23, 42)
COLOR_MUTED = RGBColor(100, 116, 139)


def set_cell_background(cell, hex_color):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding in twips (1/20th of a point)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def set_cell_border(cell, **kwargs):
    """Sets cell borders."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn(f'w:{key}'), str(val))


def add_callout(doc, text_content, title="NOTE", hex_bg=HEX_CARD_BG, hex_border=HEX_SECONDARY):
    """Creates an executive callout box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, hex_bg)
    set_cell_margins(cell, top=130, bottom=130, left=180, right=160)
    set_cell_border(cell, 
                    left={"sz": 24, "val": "single", "color": hex_border},
                    top={"sz": 4, "val": "single", "color": HEX_BORDER},
                    right={"sz": 4, "val": "single", "color": HEX_BORDER},
                    bottom={"sz": 4, "val": "single", "color": HEX_BORDER})
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    
    run_title = p.add_run(f"📌 {title}: ")
    run_title.bold = True
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(10)
    run_title.font.color.rgb = COLOR_PRIMARY
    
    run_text = p.add_run(text_content)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = COLOR_DARK
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_code_block(doc, code_text):
    """Creates a styled monospaced code/architecture block."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, "0f172a")  # Dark slate background
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    set_cell_border(cell, 
                    left={"sz": 4, "val": "single", "color": "334155"},
                    top={"sz": 4, "val": "single", "color": "334155"},
                    right={"sz": 4, "val": "single", "color": "334155"},
                    bottom={"sz": 4, "val": "single", "color": "334155"})
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    
    run = p.add_run(code_text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(241, 245, 249)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_heading_1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(18)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    run = h.runs[0]
    run.font.name = "Calibri"
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return h


def add_heading_2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    run = h.runs[0]
    run.font.name = "Calibri"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return h


def add_heading_3(doc, text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(2)
    h.paragraph_format.keep_with_next = True
    run = h.runs[0]
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = COLOR_DARK
    return h


def add_body_p(doc, text, bold_prefix=None, space_after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10)
        r_pre.font.color.rgb = COLOR_DARK
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = COLOR_DARK
    return p


def add_bullet_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10)
        r_pre.font.color.rgb = COLOR_DARK
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = COLOR_DARK
    return p


def create_styled_table(doc, headers, data, col_widths=None):
    """Creates a corporate styled data table."""
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=110, bottom=110, left=130, right=130)
        set_cell_border(hdr_cells[i], 
                        bottom={"sz": 12, "val": "single", "color": HEX_DARK},
                        top={"sz": 4, "val": "single", "color": HEX_PRIMARY},
                        left={"sz": 4, "val": "single", "color": HEX_BORDER},
                        right={"sz": 4, "val": "single", "color": HEX_BORDER})
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    for row_idx, row_data in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        bg_color = HEX_LIGHT_BG if row_idx % 2 == 1 else "ffffff"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = str(cell_value)
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=75, bottom=75, left=110, right=110)
            set_cell_border(row_cells[col_idx],
                            bottom={"sz": 4, "val": "single", "color": HEX_BORDER},
                            top={"sz": 4, "val": "single", "color": HEX_BORDER},
                            left={"sz": 4, "val": "single", "color": HEX_BORDER},
                            right={"sz": 4, "val": "single", "color": HEX_BORDER})
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(9)
                r.font.color.rgb = COLOR_DARK

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


print("Building Comprehensive Master Manual strictly matching live application features...")
doc = Document()

# Configure Standard Margins
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)
    
    # Configure Header & Footer
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("Coripet RVM Prospective Predictor & Fleet Intelligence • Production Manual")
    hrun.font.name = "Calibri"
    hrun.font.size = Pt(8)
    hrun.font.color.rgb = COLOR_MUTED
    
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    frun = fp.add_run("CONFIDENTIAL • CORIPET REVERSE VENDING MACHINE INTELLIGENCE SYSTEM")
    frun.font.name = "Calibri"
    frun.font.size = Pt(8)
    frun.font.color.rgb = COLOR_MUTED

# =========================================================================
# COVER / TITLE BLOCK
# =========================================================================
p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(24)
p_title.paragraph_format.space_after = Pt(4)
r_t = p_title.add_run("CORIPET RVM PROSPECTIVE PREDICTOR")
r_t.font.name = "Calibri"
r_t.font.size = Pt(24)
r_t.font.bold = True
r_t.font.color.rgb = COLOR_PRIMARY

p_sub = doc.add_paragraph()
p_sub.paragraph_format.space_after = Pt(12)
r_s = p_sub.add_run("Complete Operational, Technical & Feature Manual for the Production Decision Support Dashboard")
r_s.font.name = "Calibri"
r_s.font.size = Pt(13)
r_s.font.color.rgb = COLOR_SECONDARY

# Meta Info Table
meta_headers = ["System Attribute", "Live Production Specification"]
meta_data = [
    ["Application Core", "Streamlit 4-Tab Decision Support Suite (Port 8501) powered by MultiTaskRVMNet & LightGBM"],
    ["Active Navigation Tabs", "🎯 Prospective Predictor | 📋 SPL Machines | 💡 POT Machines | 🔍 Outlier & Defect Simulator"],
    ["Neural Network Core", "MultiTaskRVMNet (PyTorch): Shared Trunk [256->128->64] + Dual Heads (PRESENZE & BOTTIGLIE)"],
    ["Stacking Ensemble", "LightGBM Gradient Boosted Trees (90% Bottles / 65% Presenze) + MultiTaskRVMNet (10% / 35%)"],
    ["Loss Formulation", "MultiTaskHuberLoss (delta=1.0) with Task-Balancing & Anomaly Sample Weights"],
    ["Certified Hardware Suite", "4 Certified Models Evaluated Simultaneously: Daint A-bis, Daint B-Bis, Daint G, Ripet C-Front"],
    ["Geographic Coverage", "Republic of Italy: 107 Provinces, 7,904 Municipalities, 25,104 Nielsen Store Universe"],
    ["Database Repository", "Universal MariaDB 10.6+ / PostgreSQL 16 Architecture (1,849 Active Installed Machines, 1.77M Telemetry Rows in master_daily_rvm)"],
    ["Inspection Ingestion", "Live Coripet REST API (macchine.coripet.it) with PUNTOVENDITA Commercial Grocery Filter"],
    ["Clearance Engine", "6 Core Operational Clearance Gates with 250 b/d Breakeven Economic Threshold"],
    ["Anomaly Fleet Audit", "10-Option Rule Filtering Dropdown covering 8 Certified Hardware & Operational Anomaly Regimes"]
]
create_styled_table(doc, meta_headers, meta_data, col_widths=[2.0, 4.8])

add_callout(
    doc,
    "This document serves as the authoritative, comprehensive system manual for the live Coripet RVM Prospective Predictor. "
    "Every section directly documents the active features, live UI screens, operational calculators, data pipelines, "
    "deep neural network architecture, and mathematical models running in production, strictly reflecting the current codebase architecture.",
    title="PRODUCTION SYSTEM SPECIFICATION"
)

# =========================================================================
# SECTION 1: APPLICATION NAVIGATION & CORE ARCHITECTURE
# =========================================================================
add_heading_1(doc, "1. Application Navigation & High-Level Architecture")

add_body_p(
    doc,
    "The Coripet RVM Prospective Predictor is an enterprise spatial intelligence and decision-support application. "
    "It enables Coripet commercial directors, network development managers, and retail partners to evaluate prospective supermarket "
    "locations, manage active candidate pipelines, and monitor overall machine fleet health. "
    "The application is accessible via an adaptive, dark glassmorphic interface divided into four primary navigation views:"
)

nav_headers = ["Navigation Tab", "Icon / Label", "Primary Function", "Operational Target User"]
nav_data = [
    ["Tab 1", "🎯 Prospective Predictor", "Guided 3-step decision support wizard with real-time scenario modeling, certified 4-model capacity matrix, 12-month intake profile, and 6-gate clearance verdict.", "Business Development & Commercial Underwriters"],
    ["Tab 2", "📋 SPL Machines", "Live directory of inspected candidate locations (Sopralluogo Completato, 61 machines) with 5-month recency filtering, volume tiers, and one-click 'Results 📊' prediction handoff.", "Network Operations & Deployment Planners"],
    ["Tab 3", "💡 POT Machines", "Pipeline candidate site exploration (Potenziale status, 331 machines) for preliminary economic feasibility screening prior to on-site technical inspection dispatch.", "Commercial Strategy & Partnership Teams"],
    ["Tab 4", "🔍 Outlier & Defect Simulator", "Continuous fleet intelligence auditing 1,849 machines in database with 10-option rule filtering, 2-year timeline outlier markers, and root-cause diagnostics.", "Field Maintenance & Data Governance Teams"]
]
create_styled_table(doc, nav_headers, nav_data, col_widths=[0.8, 1.8, 2.7, 1.5])

add_heading_2(doc, "1.1 Verified National Fleet Census & Pipeline Breakdown")
add_body_p(
    doc,
    "The Coripet intelligence architecture strictly filters and categorizes the national network into active operational installations "
    "and prospective candidate pipeline stages across Italy, removing all decommissioned or inactive machines:"
)

census_headers = ["Fleet Census Metric", "Exact Count", "Database Scope & Architectural Definition", "Application & Modeling Role"]
census_data = [
    [
        "Active Installed Fleet (INSTALLAZIONI)",
        "1,849 machines",
        "Official Coripet active operational installation universe (PowerBI Installazioni baseline). Exactly 1,849 live, actively collecting RVMs operating across Italy (1,217 Cestone, 425 Cassonetto, 207 Sacco; 986 DAINT_G, 165 DAINT_A-bis, 19 DAINT_B-bis, 6 DAINT_B, 2 DAINT_A, Ripet C-Front/Bis).",
        "Active commercial footprint and operational telemetry universe for fleet monitoring."
    ],
    [
        "Certified AI Training Cohort",
        "1,681 stores",
        "1,481,324 certified daily collection records meeting 4-gate maturity rules (supermarket grocery retail, active installation, standard reward attribute, mature operating baseline >= 90 days).",
        "Authoritative baseline training and cross-validation population for MultiTaskRVMNet & LightGBM."
    ],
    [
        "Inspected Candidate Pipeline (SPL)",
        "61 machines",
        "Candidate supermarket store locations in Tab 2 with completed on-site technical inspection ('Sopralluogo Completato').",
        "Immediate deployment decision, capital allocation, and underwriting handoff pipeline."
    ],
    [
        "Prospective Candidate Pipeline (POT)",
        "331 machines",
        "Prospective candidate supermarket locations in Tab 3 undergoing preliminary feasibility screening ('Potenziale').",
        "Early-stage commercial opportunity exploration and screening prior to field dispatch."
    ]
]
create_styled_table(doc, census_headers, census_data, col_widths=[1.7, 1.1, 2.3, 1.7])

add_callout(
    doc,
    "The operational network consists strictly of 1,849 active installed machines across Italy, directly matching Coripet's official PowerBI Installazioni dashboard. "
    "All decommissioned, inactive, or removed machines have been purged from the active operational scope. "
    "For predictive machine learning modeling, exactly 1,681 mature active stores satisfy all 4 certification gates (PUNTOVENDITA retail classification, active status, standard reward attribute, and >= 90-day post-activation maturity). "
    "Candidate uninstalled locations are tracked separately in the SPL (61 inspected) and POT (331 prospective) development pipelines.",
    title="ACTIVE FLEET & PIPELINE CENSUS CERTIFICATION"
)

# =========================================================================
# SECTION 2: TAB 1 - GUIDED 3-STEP DECISION SUPPORT WIZARD
# =========================================================================
add_heading_1(doc, "2. Tab 1: Guided 3-Step Prospective Analysis Wizard")

add_body_p(
    doc,
    "The Prospective Predictor utilizes a 3-step guided wizard located in the left sidebar to collect site information, "
    "resolve census data, auto-detect commercial reward agreements, and underwrite prospective machine collection volume. "
    "Machine model selection has been deliberately removed from Step 1 inputs because the system evaluates all four certified "
    "hardware models simultaneously in the Step 4 Results Dashboard."
)

add_heading_2(doc, "2.1 Step 1: Location & Site Setup")
add_body_p(
    doc,
    "Step 1 captures the minimal geographic and brand information required to identify the prospective retail site:"
)
add_bullet_p(doc, "Offers two operational modes: (1) Select from official Coripet directory with intelligent autocomplete, or (2) Click '➕ Custom' to enter a custom supermarket brand (e.g., 'Esselunga', 'Decò', 'Ipercoop').", bold_prefix="• Store Name / Insegna: ")
add_bullet_p(doc, "Physical street address of the retail facility (e.g., 'Via delle Forze Armate 12', 'Via Emilia Ponente 140').", bold_prefix="• Street Address: ")
add_bullet_p(doc, "Official 5-digit numeric Italian postal code (e.g., '20124', '40133', '80121'). Validated to prevent invalid non-numeric submissions.", bold_prefix="• Postal Code (CAP): ")
add_bullet_p(doc, "Italian municipality name (e.g., 'Milano', 'Busto Arsizio', 'Napoli').", bold_prefix="• Municipality (Comune): ")
add_bullet_p(doc, "Initiates spatial geocoding, ISTAT demographic join, and Nielsen store scale matching, transitioning the user to Step 2.", bold_prefix="• '🔍 Step 1: Detect Location & Nielsen Scale' Button: ")

add_heading_2(doc, "2.2 Step 2: Store Scale & Nielsen Census Confirmation")
add_body_p(
    doc,
    "Step 2 connects to the Nielsen Store Universe of 25,104 Italian grocery stores to establish physical retail scale. "
    "The system displays one of two verified states:"
)
add_bullet_p(doc, "Displays a green confirmation banner showing census store name, sales square meters (m²), checkout count, and store format. The operator can accept verified census values or adjust them.", bold_prefix="• Exact Nielsen Census Match [Verified]: ")
add_bullet_p(doc, "Displays an amber advisory indicating the store was not found in the census. The operator selects a standard archetype or enters Custom sales area (m²) and checkouts (casse).", bold_prefix="• No Exact Match (Manual Archetype Selection): ")

format_headers = ["Store Format / Profile", "Format Code", "Sales Floor Area (m²)", "Active Checkouts", "Representative Retail Banners"]
format_data = [
    ["1: Small Supermarket / Proximity", "LIS / VIC", "350 m²", "2 Casse", "Carrefour Express, Conad City, Crai Local, Coop Voce"],
    ["2: Standard Supermarket", "SUP", "750 m²", "4 Casse", "Conad, Coop, Despar, Tigros, Sigma, Pam"],
    ["3: Superstore / Large Supermarket", "SUP", "1,850 m²", "9 Casse", "Esselunga, Conad Superstore, Famila Superstore"],
    ["4: Hypermarket / Mall Anchor", "IPER", "3,750 m²", "21 Casse", "Ipercoop, Spazio Conad, Carrefour Iper, Il Gigante"],
    ["5: Discount Store", "DIS", "750 m²", "4 Casse", "Eurospin, Lidl, MD, Penny Market, ALDI"],
    ["Custom Retail Footprint", "custom", "50 – 25,000 m² (user entered)", "1 – 100 Casse (user entered)", "Flagship stores, mixed-use transit hubs, multi-level retail"]
]
create_styled_table(doc, format_headers, format_data, col_widths=[1.8, 0.9, 1.4, 1.1, 1.6])

add_bullet_p(doc, "Locks in sales area and checkouts, auto-detects commercial reward contracts, and advances to Step 3.", bold_prefix="• '➡️ Step 2: Confirm Scale & Fetch Rewards' Button: ")

add_heading_2(doc, "2.3 Step 3: Commercial Rewards Policy & Go-Live Scheduling")
add_body_p(
    doc,
    "Step 3 resolves the commercial partnership and customer incentive agreement between Coripet and the supermarket chain. "
    "The system displays the auto-detected circuit (Filiera), incentive scheme, typical promotional voucher rule, and incentive tier. "
    "The user can keep the auto-detected contract or select an alternative commercial policy:"
)
add_bullet_p(doc, "Converts Coripet points into store loyalty card points (Punti Spesa / Carta Socio). Yields highest recurring habituation (1.35x–1.45x footfall multiplier).", bold_prefix="1. Conversione a punti fedeltà: ")
add_bullet_p(doc, "Direct supermarket checkout discount (e.g., €3 discount on €25 minimum spend per 200 bottles). Standard national supermarket incentive (1.15x–1.25x volume multiplier).", bold_prefix="2. Emissione Buono sconto: ")
add_bullet_p(doc, "Direct promotional gift or cash prize with no minimum spend required. High promotional excitement (1.10x–1.18x multiplier).", bold_prefix="3. Omaggio (Free Gift / Direct Voucher): ")
add_bullet_p(doc, "Enables precise custom parameter configuration: discount amount (€), minimum checkout spend (€), points/bottles required, and base scheme archetype.", bold_prefix="4. 🛠️ Custom Reward Policy: ")

add_body_p(
    doc,
    "Step 3 also includes an expandable Go-Live Scheduling tool that allows underwriters to select the Planned Go-Live Month "
    "(January through December), Go-Live Year (2026, 2027, 2028), and toggle the 'Include 3-Month Adoption Ramp-Up' checkbox "
    "(modeling Month 1-3 customer discovery and Coripet app onboarding)."
)
add_bullet_p(doc, "Executes the full end-to-end multi-layer evaluation pipeline and opens Step 4 in the main dashboard.", bold_prefix="• '🚀 Step 3: Run Full Prediction Analysis' Button: ")

# =========================================================================
# SECTION 3: TAB 1 - STEP 4 LIVE RESULTS DASHBOARD ANATOMY
# =========================================================================
add_heading_1(doc, "3. Tab 1: Step 4 Live Results Dashboard Structure")

add_body_p(
    doc,
    "Upon running an analysis (or clicking 'Results 📊' on any candidate site), the main screen transitions to the Step 4 Live Dashboard. "
    "The layout consists of structured executive sections:"
)

add_heading_2(doc, "3.1 Interactive Top Scenario Switchers")
add_body_p(
    doc,
    "Positioned at the very top of Step 4 are two interactive dropdown switchers that allow instant scenario recalculation without resetting the wizard:"
)
add_bullet_p(doc, "Allows switching the active view among the 4 certified models (Daint A-bis, Daint B-Bis, Daint G, Ripet C-Front). Recalculates utilization, overflow risk, and logistics rhythm.", bold_prefix="• ⚙️ Active Machine Model (Option Split): ")
add_bullet_p(doc, "Allows switching between Auto-Detected Filiera promotion, Direct Grocery Voucher, Store Loyalty Points, Free Gift, or Custom User-Defined Policy. Updates volume and footfall immediately.", bold_prefix="• 🎁 Commercial Reward Scheme: ")

add_heading_2(doc, "3.2 Certified Hardware Sizing & Capacity Split (All 4 Models Table)")
add_body_p(
    doc,
    "Rather than forcing the user to re-run simulations for each machine, the dashboard renders a comparative split matrix "
    "evaluating all four recognized Coripet machines simultaneously for the target location:"
)

hw_headers = ["Machine Model", "Hardware Class", "Bin Storage", "Daily Vol", "Monthly Vol", "Util. %", "Sizing Verdict", "Service", "Status Chip"]
hw_data = [
    ["Daint A-bis", "Compact (Single-Stream)", "300 b/d (~8 kg)", "Realized b/d", "Daily x 30.4", "Daily / 300", "⚠️ Overload OR ✅ Optimal", "Daily", "Available / INSTALLED"],
    ["Daint B-Bis", "Compact (Dual-Stream)", "500 b/d (~14 kg)", "Realized b/d", "Daily x 30.4", "Daily / 500", "⚠️ Overload OR ✅ Optimal", "Twice/Wk", "Available / INSTALLED"],
    ["Daint G", "Heavy-Duty (Dual-Stream)", "4,500 b/d (~120 kg)", "Realized b/d", "Daily x 30.4", "Daily / 4,500", "✅ Certified OR ℹ️ Buffer", "Bi-Weekly", "Available / INSTALLED / AI BEST FIT"],
    ["Ripet C-Front", "Industrial Compactor", "4,500 b/d (~120 kg)", "Realized b/d", "Daily x 30.4", "Daily / 4,500", "✅ Certified OR ℹ️ Buffer", "Bi-Weekly", "Available / INSTALLED / AI BEST FIT"]
]
create_styled_table(doc, hw_headers, hw_data, col_widths=[1.1, 1.4, 0.9, 0.7, 0.7, 0.5, 1.1, 0.6, 0.8])

add_body_p(
    doc,
    "Directly beneath the table, the 'AI Automated Sizing Strategy Advisory Box' explains the automated hardware selection logic: "
    "for new sites, daily throughput <= 260 b/d recommends Daint A-bis (70–85% utilization); <= 450 b/d recommends Daint B-Bis (safe buffer); "
    "<= 900 b/d recommends Daint G; and > 900 b/d recommends Ripet C-Front industrial compactor. "
    "For existing network installations, the box confirms the installed machine ID and database status."
)

add_heading_2(doc, "3.3 Executive Status Header Cards")
add_body_p(
    doc,
    "Three high-level executive cards summarize the investment case:"
)
add_bullet_p(doc, "Displays Verdict Title (e.g. 'Commercially Viable', 'Conditional', 'Review'), Action Chip ('Proceed', 'Review'), Active Model indicator dot, and Potential Volume Tier ('Super Prendente', 'Alto Prendente', 'Medio Prendente', 'Basso Prendente').", bold_prefix="1. 💼 Investment Verdict: ")
add_bullet_p(doc, "Displays Stacking Ensemble Confidence ('High Confidence'), model architecture subtitle ('Calibrated Stacking Ensemble'), and Clearance Gate status ('Suitable', 'Conditional', 'Unfavorable').", bold_prefix="2. 🎯 Model Confidence: ")
add_bullet_p(doc, "Displays Comune, Province, Region, 5-digit CAP pill, verified retail store name & street address, Nielsen Geographic Area (Area 1, 2, 3, or 4), active Machine Type, and Sales Area / Casse chip.", bold_prefix="3. 📍 Target Location: ")

add_heading_2(doc, "3.4 Key Operational Metrics Row")
add_body_p(
    doc,
    "Five dedicated metric cards present the primary operational figures:"
)
add_bullet_p(doc, "Expected unique daily recycling visitors visiting the RVM unit.", bold_prefix="• Daily Footfall (users/d): ")
add_bullet_p(doc, "Underwritten daily PET container intake under mature operating conditions.", bold_prefix="• Daily PET Yield (b/day): ")
add_bullet_p(doc, "30-day operational collection run (Daily Yield x 30.4).", bold_prefix="• Monthly Volume (b/mo): ")
add_bullet_p(doc, "Recycle intensity per visit (Daily Bottles / Daily Footfall). Standard national range is 22 to 45 b/user.", bold_prefix="• Bottles / User (b/user): ")
add_bullet_p(doc, "Network benchmark percentile ranking this store against the 1,849 active installed Coripet machines across Italy.", bold_prefix="• Suitability Rank (% score): ")

add_heading_2(doc, "3.5 Two-Column Visuals & Verbatim Location Analysis")
add_body_p(
    doc,
    "The dashboard presents a dual-column analysis section:"
)
add_bullet_p(doc, "Visual semi-circular Plotly gauge displaying the Suitability Score (0 to 100%) with color-graded arc. Accompanied by three scenario cards: Floor P25 (conservative floor, 82% of baseline), Steady Base P50 (steady-state baseline), and Peak Ceiling P75 (peak ceiling, 125% of baseline).", bold_prefix="• Left Column (📊 Network Suitability Gauge): ")
add_bullet_p(doc, "Monospaced terminal console providing the verbatim audit record of all key metrics, suitability score, volume recommendation tier, and confidence rating for instant export into underwriting memos.", bold_prefix="• Right Column (📋 Verbatim Location Analysis): ")

# =========================================================================
# SECTION 4: MULTI-TASK NEURAL NETWORK (MultiTaskRVMNet) & STACKING ENSEMBLE
# =========================================================================
add_heading_1(doc, "4. Multi-Task Neural Network (MultiTaskRVMNet) & Stacking Ensemble")

add_body_p(
    doc,
    "At the computational heart of the Coripet Prospective Predictor is a custom PyTorch multi-task neural network "
    "(MultiTaskRVMNet) combined with a LightGBM gradient-boosted decision tree stacking ensemble. "
    "Unlike generic regression models that treat container intake and store footfall independently, MultiTaskRVMNet explicitly "
    "models the strong co-dependence between unique recycling shoppers (PRESENZE) and collected PET containers (BOTTIGLIE)."
)

add_heading_2(doc, "4.1 MultiTaskRVMNet Network Topology")
add_body_p(
    doc,
    "The neural network architecture consists of three coordinated processing stages: "
    "(1) Categorical Entity Embedding Layers, (2) Shared Deep Representation Backbone, and (3) Specialized Regression Output Heads."
)

add_code_block(
    doc,
    "INPUT TENSORS:\n"
    "  ├── Numerical Features [x_num: continuous variables, scaled with StandardScaler]\n"
    "  └── Categorical Features [x_cat: discrete tokens for brand, model, region, format...]\n"
    "            │\n"
    "            ▼\n"
    "ENTITY EMBEDDING DICTIONARY (ModuleDict):\n"
    "  ├── Insegna_Standardized Embedding  [Vocab -> Dim d_1]\n"
    "  ├── TipoMacchina Embedding          [Vocab -> Dim d_2]\n"
    "  ├── nielsen_gruppo Embedding        [Vocab -> Dim d_3]\n"
    "  ├── nielsen_tipo_negozio Embedding  [Vocab -> Dim d_4]\n"
    "  ├── nielsen_area Embedding          [Vocab -> Dim d_5]\n"
    "  ├── Regione Embedding               [Vocab -> Dim d_6]\n"
    "  ├── Provincia Embedding             [Vocab -> Dim d_7]\n"
    "  └── Fornitore Embedding             [Vocab -> Dim d_8]\n"
    "            │\n"
    "            ▼ Concatenate [x_num, emb_1, emb_2, ... emb_8] (Total In Dim: N_in)\n"
    "SHARED DENSE TRUNK (Feature Representation Backbone):\n"
    "  ├── Linear(N_in, 256) ──► GELU() ──► LayerNorm(256) ──► Dropout(p=0.20)\n"
    "  ├── Linear(256, 128)  ──► GELU() ──► LayerNorm(128) ──► Dropout(p=0.20)\n"
    "  └── Linear(128, 64)   ──► GELU() ──► LayerNorm(64)  ──► Dropout(p=0.10)\n"
    "            │\n"
    "            ├─────────────────────────────────────────┐\n"
    "            ▼                                         ▼\n"
    "HEAD 1: PRESENZE HEAD                     HEAD 2: BOTTIGLIE HEAD\n"
    "  ├── Linear(64, 32)                        ├── Linear(64, 32)\n"
    "  ├── GELU()                                ├── GELU()\n"
    "  └── Linear(32, 1)                         └── Linear(32, 1)\n"
    "            │                                         │\n"
    "            ▼                                         ▼\n"
    "  log1p(PRESENZE / day)                     log1p(BOTTIGLIE / day)"
)

add_heading_2(doc, "4.2 Entity Embedding Sizing Formula & Categorical Vocabularies")
add_body_p(
    doc,
    "Categorical attributes are mapped to dense continuous embedding vectors rather than sparse one-hot vectors. "
    "The embedding dimensionality d for each categorical variable is computed dynamically using an adaptive square-root heuristic:"
)
add_bullet_p(doc, "d_emb = min(32, max(4, round(sqrt(V) * 2))), where V is the vocabulary size.", bold_prefix="• Embedding Sizing Rule: ")

emb_headers = ["Categorical Feature", "Vocabulary Domain", "Estimated Vocab (V)", "Embedding Dim (d)", "Semantic Representation"]
emb_data = [
    ["Insegna_Standardized", "Retail Brands (Esselunga, Conad, Coop, Decò...)", "22 Banners", "10 Dim", "Captures chain-specific customer loyalty and foot-traffic density."],
    ["TipoMacchina", "Hardware Models (Daint A-bis, B-Bis, G, Ripet C-Front)", "4 Certified", "4 Dim", "Encodes compaction stream mechanism, intake speed, and bin limits."],
    ["nielsen_gruppo", "Buying Alliances (Selex, Conad, Coop, Agorà...)", "15 Groups", "8 Dim", "Captures promotional buying group power and regional market share."],
    ["nielsen_tipo_negozio", "Store Formats (IPER, SUP, DIS, LIS, VIC)", "6 Formats", "5 Dim", "Encodes physical shopper browsing duration and shopping basket scale."],
    ["nielsen_area", "Macro Geographic Areas (Area 1, 2, 3, 4)", "4 Areas", "4 Dim", "Represents macro-regional recycling culture and seasonal beverage volume."],
    ["Regione", "20 Italian Administrative Regions", "20 Regions", "9 Dim", "Captures regional waste disposal tariffs and regional municipal regulations."],
    ["Provincia", "107 Italian Administrative Provinces", "107 Prov", "21 Dim", "Fine-grained demographic density, income distribution, and logistics."],
    ["Fornitore", "Machine Hardware Vendors & Manufacturers", "6 Vendors", "5 Dim", "Encodes vendor telematics telemetry calibration and hardware reliability."]
]
create_styled_table(doc, emb_headers, emb_data, col_widths=[1.5, 1.8, 1.0, 0.9, 1.6])

add_heading_2(doc, "4.3 Continuous Feature Normalization & Skewness Mitigation")
add_body_p(
    doc,
    "Continuous demographic, spatial, and store variables exhibit extreme positive skewness (e.g. population ranges from 500 inhabitants "
    "in rural mountain communes to 1,400,000 in Rome). To stabilize neural network gradients, skewed variables undergo a log1p transform "
    "prior to standard z-score normalization:"
)
add_bullet_p(doc, "x_log = log(1 + x) applied to: municipal population, land surface km², population density (ab/km²), municipal GDP (PIL), store sales area (m²), and distance to nearest Coripet machine (km).", bold_prefix="• Log-Transformed Skewed Features: ")
add_bullet_p(doc, "Standardized using training set medians and standard deviations: latitude, longitude, postal code numeric prefix, PIL per capita wealth index, checkout counts, competitive grocery store density, and machines within 5 km.", bold_prefix="• Standard Continuous Features: ")
add_bullet_p(doc, "Encoded as sinusoidal harmonic pairs: sin(2π * DOW / 7) and cos(2π * DOW / 7) for weekly rhythm; sin(2π * Month / 12) and cos(2π * Month / 12) for annual seasonality, preserving calendar continuity across Sunday/Monday and December/January.", bold_prefix="• Cyclical Calendar Encodings: ")

add_heading_2(doc, "4.4 Multi-Task Huber Loss Function & Outlier Sample Weighting")
add_body_p(
    doc,
    "Because retail collection data contains occasional anomalous spikes (e.g. commercial restaurant bulk drops) and zero-volume "
    "downtime records, training with standard Mean Squared Error (MSE) would cause severe gradient explosion. "
    "The network is trained with a dual-head Multi-Task Huber Loss with threshold delta = 1.0:"
)
add_bullet_p(doc, "Huber(e) = 0.5 * e² if |e| <= 1.0; else |e| - 0.5. Becomes quadratic for small errors and strictly linear for large deviations, making the model inherently robust to extreme outlier spikes.", bold_prefix="• Huber Loss Definition: ")
add_bullet_p(doc, "L_total = w_bottiglie * L_huber(pred_b, target_b) + w_presenze * L_huber(pred_p, target_p), where target values are expressed in log1p space: y = log(1 + actual).", bold_prefix="• Multi-Task Objective: ")
add_bullet_p(doc, "During training, machines flagged as chronic capacity jams (Rule 2) or sensor drift (Rule 7) receive reduced sample weights (w = 0.25 to 0.30), preventing defective hardware from corrupting prospective weights.", bold_prefix="• Anomaly Sample Weighting: ")

add_heading_2(doc, "4.5 Stacking Ensemble Architecture (LightGBM + MultiTaskRVMNet)")
add_body_p(
    doc,
    "In production inference, the neural network is blended with a trained LightGBM gradient boosted tree ensemble. "
    "This dual-engine architecture combines the tabular pattern recognition of gradient boosted trees with the deep non-linear "
    "representation and entity embeddings of the PyTorch neural network. The ensemble weights are calibrated via out-of-fold cross-validation:"
)
add_bullet_p(doc, "Forecast_Bottles = 0.90 * LightGBM_Bottles + 0.10 * NeuralNet_Bottles", bold_prefix="• BOTTIGLIE Ensemble Formula: ")
add_bullet_p(doc, "Forecast_Presenze = 0.65 * LightGBM_Presenze + 0.35 * NeuralNet_Presenze", bold_prefix="• PRESENZE Ensemble Formula: ")
add_body_p(
    doc,
    "The 90% LightGBM weighting on collection volume provides exceptional stability against localized grocery volume swings, "
    "while the 35% neural net weighting on footfall ensures customer adoption curves reflect subtle demographic interaction effects."
)

add_heading_2(doc, "4.6 84-Day Calendar Simulation Grid (Seasonal & Weekly Marginalization)")
add_body_p(
    doc,
    "Rather than executing a single static point prediction, the model runs a dynamic marginalization over an 84-day representative annual calendar. "
    "The simulation grid evaluates 12 complete 7-day Monday-to-Sunday cycles centered on the second Monday of each month across the forecast year. "
    "This generates 84 individual forward inference passes per candidate location, allowing the engine to report both Arithmetic Mean "
    "(strictly additive: Mean x 365 = Annual Volume) and Median P50 (steady-state operational baseline immune to holiday closures)."
)

# =========================================================================
# SECTION 5: RETROSPECTIVE BENCHMARK & STATISTICAL PROBABILITY PROFILE
# =========================================================================
add_heading_1(doc, "5. Retrospective Benchmark & Statistical Probability Profile")

add_body_p(
    doc,
    "The platform features advanced retrospective benchmarking and parametric statistical modeling."
)

add_heading_2(doc, "5.1 Section 3.25: Historical Actuals vs. Prospective AI Comparison")
add_body_p(
    doc,
    "For active installed machines, this section connects to `master_daily_rvm` in PostgreSQL to evaluate 24 past months of "
    "actual telemetry collections against the 12 forward-looking prospective AI forecast months."
)
add_callout(
    doc,
    "CANDIDATE SITE SUPPRESSION RULE: For uninstalled candidate locations (SPL and POT status), no physical machine is operating on site. "
    "The retrospective historical comparison engine is strictly suppressed for SPL and POT sites to prevent misleading flat zero-graphs, "
    "ensuring candidate sites are evaluated purely on forward-looking prospective AI simulation.",
    title="LIFECYCLE SEPARATION PRINCIPLE",
    hex_bg=HEX_INFO_BG,
    hex_border="0284c7"
)

add_heading_2(doc, "5.2 Section 3.5: Statistical Probability & Future Performance Profile")
add_body_p(
    doc,
    "Calibrated to empirical network variance, this section provides parametric risk modeling and volume distribution analysis:"
)

add_heading_3(doc, "5.2.1 Capital Safety & Value-at-Risk 80% Band")
add_bullet_p(doc, "Evaluates probability of exceeding the 250 b/d operational breakeven threshold. Rated as Very Low Risk, Low Risk, Moderate Risk, or Elevated Risk.", bold_prefix="• 🛡️ Capital Safety (%): ")
add_bullet_p(doc, "90% statistical exceedance floor (P10). Daily bottles and monthly volume that the machine is virtually guaranteed to exceed 9 days out of 10.", bold_prefix="• 📉 Downside Floor (P10): ")
add_bullet_p(doc, "Top 10% peak upside potential (P90). Daily bottles and monthly volume achievable during peak seasonal and marketing campaigns.", bold_prefix="• 📈 Upside Ceiling (P90): ")

add_heading_3(doc, "5.2.2 Official Volume Performance Classification Table")
add_body_p(
    doc,
    "Presents the single authoritative probability distribution table across Coripet's four official volume performance tiers (sums to 100%):"
)

prob_tier_headers = ["Volume Tier & Strategic Role", "Daily Throughput", "Annual Output (365d)", "Probability %", "Dominant Badge"]
prob_tier_data = [
    ["Basso Prendente (Low Yield) • Cost review advised", "< 300 b/d", "< 109,500 bottles/year", "P(Basso) %", "DOMINANT if highest"],
    ["Medio Prendente (Standard Yield) • Core sustainable baseline", "300 – 1,000 b/d", "109,500 – 365,000 bottles/year", "P(Medio) %", "DOMINANT if highest"],
    ["Alto Prendente (High Yield) • Top commercial performer", "1,000 – 2,000 b/d", "365,000 – 730,000 bottles/year", "P(Alto) %", "DOMINANT if highest"],
    ["Super Prendente (Super Anchor) • Strategic hub of excellence", "> 2,000 b/d", "> 730,000 bottles/year", "P(Super) %", "DOMINANT if highest"]
]
create_styled_table(doc, prob_tier_headers, prob_tier_data, col_widths=[2.2, 1.2, 1.6, 1.0, 1.0])

add_heading_3(doc, "5.2.3 AI Hardware Fit & Capacity Certification")
add_body_p(
    doc,
    "Compares the physical compactor storage capacity of the selected machine against projected volume: "
    "storage capacity (bottles and kg PET), daily saturation limit (b/d), daily overflow risk percentage, "
    "autonomous storage buffer in days, and recommended servicing rhythm. "
    "If overflow risk exceeds 20% or projected volume exceeds saturation limit, the system triggers a 'CAPACITY BOTTLENECK DETECTED' banner "
    "and advises upgrading to Daint G or Ripet C-Front (4,500 bottles)."
)

# =========================================================================
# SECTION 6: COMMERCIAL SENSITIVITY, CLEARANCE GATES & METHODOLOGY
# =========================================================================
add_heading_1(doc, "6. Commercial Sensitivity, Clearance Gates & Growth Levers")

add_body_p(
    doc,
    "The remaining sections of Step 4 provide deep commercial analytics, operational gatekeeping, and volume growth strategies."
)

add_heading_2(doc, "6.1 Section 4 & 4.5: Commercial Rewards & Sensitivity Matrix")
add_body_p(
    doc,
    "Section 4 displays the active Commercial Rewards & Filiera Intelligence (circuit, incentive tier, incentive strength, voucher terms) "
    "alongside the Spatial Catchment & Local Network Density card (nearest Coripet distance, network position, machines in 3km, competitive grocery stores). "
    "Section 4.5 renders the Promotional Reward Sensitivity & Impact Matrix table, comparing all reward policy options "
    "(Daily bottles, Monthly volume, Active footfall, and Variance vs Selected %) with the currently selected promotion highlighted in green."
)

add_heading_2(doc, "6.2 Section 5: Operational Clearance Gates (Core 6 Gates)")
add_body_p(
    doc,
    "The clearance engine evaluates candidate sites across the six core operational gates:"
)

core_gate_headers = ["Gate Key", "Clearance Gate Name", "Evaluation Metric & Threshold", "Gate Statuses"]
core_gate_data = [
    ["cannibalization_gate", "Spatial Cannibalization", "Evaluates distance to nearest active Coripet RVM and catchment dilution penalty (<750m warns, <250m fails).", "PASS | WARN | FAIL"],
    ["economic_volume_gate", "Economic Breakeven", "Strictly anchored to 250 b/d operational breakeven floor. Calculates exact volume deficit or surplus in b/d and b/mo.", "PASS | WARN | FAIL"],
    ["store_format_gate", "Store Format & Scale", "Validates sales floor area (>=300 m²) and active cash registers (>=3 checkouts).", "PASS | WARN | FAIL"],
    ["demographic_gate", "Demographic Catchment", "Evaluates municipal resident population (>=5,000) and population density (>=150 ab/km²).", "PASS | WARN | FAIL"],
    ["hardware_sizing_gate", "Hardware Sizing Match", "Ensures machine capacity matches projected volume (eliminates oversized heavy units on low volume or compact units on high volume).", "PASS | WARN | FAIL"],
    ["logistics_feasibility_gate", "Logistics Servicing Route", "Validates distance from regional logistics depot to ensure economic collection servicing (<25 km standard).", "PASS | WARN | FAIL"]
]
create_styled_table(doc, core_gate_headers, core_gate_data, col_widths=[1.5, 1.6, 2.7, 1.2])

add_body_p(
    doc,
    "An Underwriting Verdict Banner synthesizes the gate outcomes into an executive verdict: "
    "'UNCONDITIONAL CLEARANCE: Site Fully Approved For Deployment' (all gates passed), "
    "'CONDITIONAL CLEARANCE: Operational & Sizing Optimization Advised' (warnings present), or "
    "'DEPLOYMENT CONDITIONAL / REJECTED: Capital Protection Gate Failed' (one or more critical gates failed)."
)

add_heading_2(doc, "6.3 Section 6: 12-Month Rolling Operational Intake Profile")
add_body_p(
    doc,
    "Renders an interactive Plotly monthly collection trajectory starting from the confirmed Go-Live Month and Year. "
    "Incorporates a 3-month adoption ramp-up curve (Month 1: 50%, Month 2: 75%, Month 3: 90%, Month 4+: 100% steady state) "
    "multiplied by monthly seasonal hydration indices (summer peak vs. winter baseline). "
    "Includes an expandable data table detailing daily and monthly targets, maturity percentage, lifecycle stage, and seasonal context."
)

add_heading_2(doc, "6.4 Section 7: Prediction Methodology & Strategic Growth Levers")
add_body_p(
    doc,
    "The final section of Step 4 is split into two deep intelligence cards:"
)
add_bullet_p(doc, "Details the 4-stage pipeline: (1) Geospatial & Nielsen Retail Scale Driver (m², casse, ISTAT population, density, PIL), (2) Dual AI Ensemble & Brand Embeddings (LightGBM + NeuralNet weights and brand equity), (3) Spatial Catchment & Network Dilution (nearest distance, network density, grocery competition, dilution haircut), and (4) Promotional Multiplier & Operational Underwriting.", bold_prefix="• 🧠 How This Prediction Was Calculated: ")
add_bullet_p(doc, "Presents four dynamic operational levers to shift actual store throughput toward the P90 Upside Ceiling: (1) Promotional Policy Strategy & Gamification, (2) Hardware Capacity & Pre-Weekend Bag-Swap Protocol, (3) Spatial Catchment Defense & Exclusive Rewards, and (4) Store Format Lobby Placement & Wayfinding Directionals.", bold_prefix="• 🚀 How to Increase This Prediction Number: ")

# =========================================================================
# SECTION 7: TABS 2 & 3 - SPL AND POT CANDIDATE DIRECTORIES
# =========================================================================
add_heading_1(doc, "7. Tabs 2 & 3: SPL and POT Candidate Site Directories")

add_body_p(
    doc,
    "Tabs 2 and 3 provide live enterprise directories of Coripet candidate locations across Italy. "
    "Tab 2 (`📋 SPL Machines`) displays approved candidate sites that have completed on-site technical inspection (Sopralluogo Completato). "
    "Tab 3 (`💡 POT Machines`) houses early-stage prospective pipeline locations (Potenziale status) undergoing pre-inspection economic screening."
)

add_heading_2(doc, "7.1 Live Coripet REST API Ingestion & Commercial Grocery Filtering")
add_body_p(
    doc,
    "The directories ingest live records from the official Coripet REST API (macchine.coripet.it) through two coordinated endpoints: "
    "`mt_CaricaMacchine` (master site registry) and `mt_CaricaSopralluoghi` (inspection dates and statuses). "
    "To ensure commercial relevance, records are strictly filtered to `TipoPunto == 'PUNTOVENDITA'`, eliminating non-retail and administrative testing locations. "
    "If the central API is unreachable, the engine seamlessly falls back to local backup caches."
)

add_heading_2(doc, "7.2 Survey Recency Filtering (Last 5 Months Prioritization)")
add_body_p(
    doc,
    "To enable operations teams to immediately focus on newly surveyed locations, the Survey Recency filter is positioned prominently with recent periods prioritized:"
)
add_bullet_p(doc, "Positioned at the top of the selector. Isolates candidate sites surveyed within the last 150 days for immediate hardware deployment scheduling.", bold_prefix="• ⏱️ Last 5 Months (Top Priority): ")
add_bullet_p(doc, "Isolates sites surveyed within the past calendar year.", bold_prefix="• 📅 Last 12 Months: ")
add_bullet_p(doc, "Displays multi-year prospective pipeline candidates.", bold_prefix="• 🗓️ Last 24 Months: ")
add_bullet_p(doc, "Displays the entire cumulative candidate registry.", bold_prefix="• 🌐 All Sites: ")

add_heading_2(doc, "7.3 Batch Predictive AI Ranking & Volume Tiers")
add_body_p(
    doc,
    "Every candidate site in the directory is processed through the full AI predictive engine and ranked in descending order of daily collection yield. "
    "Each store is assigned an official Volume Classification Tier:"
)
add_bullet_p(doc, "> 2,000 b/d (> 60,000 b/mo). Tier 1 Strategic Anchor. Priority allocation for Daint G or Ripet C-Front.", bold_prefix="• 🟣 Super Prendente: ")
add_bullet_p(doc, "1,000 – 2,000 b/d (30,000 – 60,000 b/mo). Tier 2 High Yield supermarket location.", bold_prefix="• 🟢 Alto Prendente: ")
add_bullet_p(doc, "300 – 1,000 b/d (9,000 – 30,000 b/mo). Tier 3 Solid Contributor standard neighborhood baseline.", bold_prefix="• 🔵 Medio Prendente: ")
add_bullet_p(doc, "< 300 b/d (< 9,000 b/mo). Tier 4 Constrained yield requiring compact machine (Daint A-bis) or co-financing review.", bold_prefix="• 🔴 Basso Prendente: ")

add_heading_2(doc, "7.4 Candidate Card Anatomy & One-Click Handoff ('Results 📊' Action)")
add_body_p(
    doc,
    "Each store is displayed as an interactive card containing: rank badge (#1 🏆, #2 🥈, #3 🥉), Store Name and PuntoFisico ID, "
    "Retail Filiera, color-coded inspection badge (Definitivo, In Corso, Riaperto), exact inspection date, complete address, "
    "auto-inferred Nielsen scale (m² and checkouts), predicted daily and monthly volume, volume tier badge, and voucher multiplier. "
    "Clicking the red 'Results 📊' button populates the store data into session state, automatically transitions the user to Tab 1, "
    "and renders the complete Step 4 Executive Dashboard without requiring manual re-entry."
)

# =========================================================================
# SECTION 8: TAB 4 - OUTLIER & DEFECT DETECTION SIMULATOR
# =========================================================================
add_heading_1(doc, "8. Tab 4: Fleet Outlier & Hardware Defect Detection Simulator")

add_body_p(
    doc,
    "Tab 4 (`🔍 Outlier & Defect Simulator`) provides a complete fleet intelligence and predictive maintenance engine. "
    "It continuously audits all 1,849 active installed machines in the live census to identify hardware breakdowns, capacity jams, "
    "commercial spikes, and erratic operating regimes."
)

add_heading_2(doc, "8.1 Database Performance Architecture (3-Way Indexed UNION)")
add_body_p(
    doc,
    "Historical collection telemetry is stored in the `master_daily_rvm` table. "
    "To ensure instant UI responsiveness and prevent database timeouts, the system utilizes three dedicated B-tree indexes: "
    "`idx_master_daily_rvm_pf` (on pf), `idx_master_daily_rvm_macchina` (on macchina), and `idx_master_daily_rvm_codprod` (on codicemacchinaprod). "
    "Queries execute via a 3-way indexed UNION structure that avoids multi-column OR sequential scans, reducing query execution time from 3,437 ms down to 7.9 ms."
)

add_heading_2(doc, "8.2 10-Option Outlier Rule Filtering Bar")
add_body_p(
    doc,
    "The Outlier Dashboard features a prominent 10-option dropdown filter that allows maintenance teams to isolate stores by anomaly regime:"
)

rule_headers = ["Filter Option", "Regime Code", "Mathematical Trigger Condition", "Operational Field Protocol"]
rule_data = [
    ["1. All Outliers", "ALL_ANOMALIES", "Filters fleet strictly to machines triggering at least one anomaly rule (Rules 1 to 7).", "Executive triage of entire abnormal fleet."],
    ["2. Rule 1: DEFECTIVE_OFFLINE", "DEFECTIVE_OFFLINE", "Consecutive Zeros >= 7 days OR Recent 60-day Zero Rate >= 20%.", "P1 Field Dispatch: Inspect compactor motor, laser scanner, or verify power switch."],
    ["3. Rule 2: CAPACITY_BOTTLENECK", "CAPACITY_BOTTLENECK", "Footfall >= 75% normal, but Bottles < 60% baseline & < 15 b/visit.", "Store Alert: Compactor full by midday. Mandate scheduled bag change twice daily."],
    ["4. Rule 3: VOLATILITY_SWING", "VOLATILITY_SWING", "Baseline surges to 1.5x (1,000+ b/d) then crashes to <400 b/d with CV > 0.85.", "Campaign Audit: Supermarket reward campaign expired or staff clearing fatigue."],
    ["5. Rule 4: EXTREME_SPIKE_EVENT", "EXTREME_SPIKE_EVENT", "Single-day collection > 7,500 bottles OR Z-Score > 4.5 vs machine mean.", "Commercial Review: Bulk commercial drop (bar/restaurant) or school contest."],
    ["6. Rule 5: CHRONIC_UNDERPERFORMER", "CHRONIC_UNDERPERFORMER", "Mature store volume < 250 b/d despite high AI forecast (gap >= 400 b/d).", "Store Repositioning: Relocate RVM from hidden basement/parking to main store entrance."],
    ["7. Rule 6: SUPER_HUB_OUTLIER", "SUPER_HUB_OUTLIER", "Lifetime median collection >= 3,000 bottles/day consistently.", "Capital Expansion: High-priority strategic site to deploy a 2nd Coripet machine."],
    ["8. Rule 7: SENSOR_DRIFT_TRICKLE", "SENSOR_DRIFT_TRICKLE", "Daily volume 1 <= B <= 5 bottles with 0 users for >= 14 days.", "Optical Recalibration: Clean contaminated optical lenses and recalibrate sensor gate."],
    ["9. Rule 8: HEALTHY_NORMAL", "HEALTHY_NORMAL", "Machine operates within nominal parametric distribution (zero rule triggers).", "Nominal Operation: Continue routine standard service schedule."],
    ["10. Full Fleet", "FULL_FLEET", "Displays all 1,849 machines in the audited diagnostic cohort including healthy units.", "Comprehensive network overview and exploratory search."]
]
create_styled_table(doc, rule_headers, rule_data, col_widths=[1.5, 1.4, 2.3, 1.8])

add_heading_2(doc, "8.3 2-Year Plotly Daily Trajectory & Point-Outlier Markers")
add_body_p(
    doc,
    "When selecting any machine from the fleet table, the dashboard renders an interactive 2-year daily bottles collection chart. "
    "To isolate specific operational incidents, daily collection points are tagged with three distinct point-outlier markers:"
)
add_bullet_p(doc, "Collection = 0 bottles. Pinpoints hardware breakdown or store power cutoff.", bold_prefix="🔴 Zero-Collection Downtime Outlier: ")
add_bullet_p(doc, "Bottles >= 97th percentile, Bottles > 1.4x 30d rolling baseline, and Bottles > 500. Pinpoints commercial bulk dumps or school recycling contests.", bold_prefix="⚡ Volume Spike Outlier: ")
add_bullet_p(doc, "0 < Bottles < 0.30x 30d baseline and 30d baseline > 150 b/d. Pinpoints partial-day jams and midday full-bin lockouts.", bold_prefix="⚠️ Severe Drop Outlier: ")
add_bullet_p(doc, "30-day moving average curve depicting true baseline momentum, smoothed against weekend fluctuations.", bold_prefix="📈 Rolling 30-Day Baseline: ")

add_body_p(
    doc,
    "The dashboard includes a Root-Cause Diagnostic narrative card outlining operational findings, maintenance history, "
    "and a '📥 Download Filtered Fleet Report (CSV)' button allowing immediate export of the filtered machine registry."
)

# =========================================================================
# SECTION 9: TECHNICAL ARCHITECTURE & DATABASE INFRASTRUCTURE
# =========================================================================
add_heading_1(doc, "9. Technical Architecture, Database & Deployment Infrastructure")

add_body_p(
    doc,
    "The underlying technical stack is engineered for production stability, low latency, and high mathematical precision:"
)
add_bullet_p(doc, "Universal MariaDB 10.6+ and PostgreSQL 16 relational database with master table `master_daily_rvm` containing 1.77M+ (1,770,755) certified collection records across the 1,849 active installed machines. Optimized with sub-10ms indexed UNION queries and connection pooling.", bold_prefix="• Database Layer: ")
add_bullet_p(doc, "Multi-task tabular neural network built in PyTorch (MultiTaskRVMNet) with entity embeddings for categorical features, GELU activations, and LayerNorm. Simulates an 84-day seasonal calendar grid.", bold_prefix="• Neural Network Core: ")
add_bullet_p(doc, "Calibrated gradient-boosted decision trees (LightGBM) blended with neural network predictions using optimal cross-validated ensemble weights.", bold_prefix="• Stacking Ensemble: ")
add_bullet_p(doc, "Haversine spatial calculations, concentric buffer rings (1km, 3km, 5km), and demographic census integration from ISTAT municipal records.", bold_prefix="• Spatial Intelligence: ")
add_bullet_p(doc, "Streamlit web application running on port 8501 with custom CSS glassmorphism, responsive column flexboxes, and interactive Plotly visualization components.", bold_prefix="• User Interface: ")

# Save Document strictly to single authoritative master file
try:
    doc.save(str(OUTPUT_DOCX))
    print(f"\nAuthoritative Master Document successfully generated at: {OUTPUT_DOCX.name}")
except PermissionError:
    print(f"\nWarning: {OUTPUT_DOCX.name} is currently open in MS Word. Please close MS Word and re-run.")

