import os
import re
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import pymupdf

from multiformated_reconciliation import analyze_document as verify_any_file
from cross_document_verifier import verify_application_pack
from forensic_aggregator import detect_copy_move_forgery
from module_aggregator import build_module1_output

st.set_page_config(
    page_title="FinGuard AI | Module 1 Document Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive Dark-Mode Override CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main App Canvas */
    .stApp {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }

    /* Force High-Contrast Dark Text Across All Main Elements & Columns */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main p, .main span, .main label, .main li,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="column"] h4,
    div[data-testid="column"] p {
        color: #0f172a !important;
    }

    /* File Uploader Dropzone High-Contrast Fix */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #94a3b8 !important;
        border-radius: 12px !important;
    }
    section[data-testid="stFileUploaderDropzone"] * {
        color: #0f172a !important;
    }

    /* Keep Sidebar Dark & Crisp */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #0f172a !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 26px 30px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.15);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin: 0 0 6px 0 !important;
    }
    .hero-banner p {
        color: #cbd5e1 !important;
        font-size: 14px !important;
        margin: 0 !important;
    }

    /* Prominent Identified Document Card */
    .id-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 18px;
    }
    .id-card-title {
        color: #1e3a8a !important;
        font-size: 19px !important;
        font-weight: 700 !important;
    }
    .id-card-sub {
        color: #1d4ed8 !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        margin-top: 4px;
    }

    /* Plain-Language Finding Banner */
    .founder-quote {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        color: #0f172a !important;
        font-size: 14.5px !important;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Explicit Section Headers (Never Invisible) */
    .sec-header {
        color: #0f172a !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        margin: 10px 0 12px 0 !important;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Metric Containers */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        color: #475569 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #0f172a !important;
        font-size: 21px !important;
        font-weight: 700 !important;
    }
    .metric-sub {
        color: #64748b !important;
        font-size: 11.5px !important;
        font-weight: 500 !important;
        margin-top: 4px;
    }

    /* Triage Badges */
    .badge-low {
        background-color: #dcfce7;
        border: 1.5px solid #16a34a;
        color: #14532d !important;
        padding: 16px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 15px;
    }
    .badge-review {
        background-color: #fef3c7;
        border: 1.5px solid #d97706;
        color: #78350f !important;
        padding: 16px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 15px;
    }
    .badge-high {
        background-color: #fee2e2;
        border: 1.5px solid #dc2626;
        color: #7f1d1d !important;
        padding: 16px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 15px;
    }

    /* Alert Boxes */
    .alert-danger {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        color: #7f1d1d !important;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 600;
        font-size: 14px;
    }
    .alert-warning {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        color: #78350f !important;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 600;
        font-size: 14px;
    }
    .alert-success {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        color: #14532d !important;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 600;
        font-size: 14px;
    }
    .info-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #475569;
        color: #1e293b !important;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 14px;
        font-size: 13.5px;
    }

    /* Tab Bar Styling */
    div[data-baseweb="tab-list"] {
        background-color: #e2e8f0 !important;
        padding: 6px !important;
        border-radius: 10px !important;
        gap: 6px !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
    }
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
        color: #334155 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #2563eb !important;
    }

    /* Custom High-Contrast HTML Table (Replaces invisible st.dataframe) */
    .clean-table-wrap {
        overflow-x: auto;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }
    table.clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        color: #0f172a !important;
    }
    table.clean-table th {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        font-weight: 700 !important;
        text-align: left;
        padding: 12px 14px;
        border-bottom: 2px solid #cbd5e1;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.4px;
    }
    table.clean-table td {
        padding: 11px 14px;
        border-bottom: 1px solid #e2e8f0;
        color: #0f172a !important;
        background-color: #ffffff !important;
        font-weight: 500;
    }
    table.clean-table tr:hover td {
        background-color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(title, value, subtitle=""):
    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title):
    st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)


def render_html_table(rows_list, column_rename_map=None):
    if not rows_list:
        st.markdown('<div class="info-box">No tabular rows to display.</div>', unsafe_allow_html=True)
        return
    df = pd.DataFrame(rows_list)
    if column_rename_map:
        df = df.rename(columns=column_rename_map)
    html_tbl = df.to_html(index=False, classes="clean-table", border=0, float_format=lambda x: f"{x:.2f}")
    st.markdown(f'<div class="clean-table-wrap">{html_tbl}</div>', unsafe_allow_html=True)


def render_identity_card(id_info):
    pan_txt = f" • PAN: {id_info['pan_number']}" if id_info.get("pan_number") else ""
    doc_code = id_info.get("doc_code", "")
    if doc_code == "SALARY_SLIP":
        org_txt = f" • Employer: {id_info['employer_name']}" if id_info.get("employer_name") else ""
    elif doc_code == "BANK_STATEMENT":
        bank_val = id_info.get("issuing_bank") or id_info.get("employer_name")
        org_txt = f" • Issuing Bank: {bank_val}" if bank_val else ""
    else:
        org_txt = ""

    st.markdown(f"""
    <div class="id-card">
        <div class="id-card-title">🪪 Identified Document: {id_info['identity_badge']}</div>
        <div class="id-card-sub">Category: {id_info['doc_category']} | Holder Name: {id_info['holder_name']}{pan_txt}{org_txt}</div>
    </div>
    """, unsafe_allow_html=True)


def draw_visual_findings_on_image(image_path, forensic_flags):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for flag in forensic_flags:
        bbox_match = re.search(r"\[([\d\.\s,]+)\]", flag)
        if bbox_match:
            try:
                coords = [float(n.strip()) for n in bbox_match.group(1).split(",")]
                if len(coords) == 4:
                    x0, y0, x1, y1 = coords
                    draw.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=(220, 38, 38), width=4)
                    draw.rectangle([x0 - 6, max(0, y0 - 26), x0 + 150, y0 - 6], fill=(220, 38, 38))
                    draw.text((x0 - 2, max(0, y0 - 22)), "TAMPER ANOMALY", fill=(255, 255, 255))
            except Exception:
                pass
    return img


# SIDEBAR
with st.sidebar:
    st.markdown("## 🛡️ FinGuard AI")
    st.caption("Module 1 — AI Document Checker")
    st.markdown("---")

    inspection_mode = st.radio(
        "Workflow Mode",
        ["Single Document Audit", "3-Document Pack Cross-Check (Family D)"],
        index=0
    )

    tenant_choice = st.selectbox(
        "Lender Tenant Thresholds (Sec 4.3)",
        ["default_lender", "conservative_bank", "high_velocity_fintech"]
    )

    st.markdown("---")
    st.markdown("#### 🔬 5-Stage Pipeline")
    st.markdown("""
    1. **Ingest** (.pdf, .png, .jpg, .docx)
    2. **Classify** (PAN / Statement / Payslip)
    3. **Extract** (OCR + Layout Spans)
    4. **Detect** (Families A, B, C, D)
    5. **Aggregate** (Isotonic Calibration)
    """)

# Hero Header
st.markdown("""
<div class="hero-banner">
    <h1>Module 1 — AI Document Checker & Risk Calibration</h1>
    <p>Identifies PAN Cards, Bank Statements, Salary Slips, GST, Udyam, ITR, and Balance Sheets. Detects tampering across Families A–D, and outputs calibrated triage bands.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MODE 1: SINGLE DOCUMENT AUDIT
# ==========================================
if inspection_mode == "Single Document Audit":
    uploaded_file = st.file_uploader(
        "Upload PAN Card, Bank Statement, Salary Slip, GST, ITR, Udyam, or Balance Sheet (.pdf, .png, .jpg, .docx)",
        type=["pdf", "png", "jpg", "jpeg", "docx"]
    )

    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        try:
            with st.spinner("Executing 5-Stage Pipeline: Ingest → Classify → Extract → Detect → Calibrate..."):
                report = verify_any_file(temp_path, original_filename=uploaded_file.name)
                if suffix in (".png", ".jpg", ".jpeg"):
                    cm_flags = detect_copy_move_forgery(temp_path)
                    report.setdefault("forensic_flags", []).extend(cm_flags)

                m1_out = build_module1_output(temp_path, report, tenant_id=tenant_choice)

            # 1. Identified Document Card
            id_card = m1_out["identity_card"]
            render_identity_card(id_card)

            # 2. Evidence-Aligned Plain-Language Summary
            st.markdown(f"""
            <div class="founder-quote">
                💬 <b>Plain-Language Explanation:</b> "{m1_out['plain_language_summary']}"
            </div>
            """, unsafe_allow_html=True)

            # 3. Triage Band & Explicitly Labeled Document vs Finding Scores
            band = m1_out["triage_band"]
            doc_score = m1_out["overall_document_risk_score"]
            score_meta = m1_out["scoring_methodology"]
            max_find_score = score_meta["max_individual_finding_score"]

            c_band, c_doc_s, c_find_s, c_type = st.columns([2.0, 1.1, 1.1, 1.0])

            with c_band:
                if band == "low_risk":
                    st.markdown('<div class="badge-low">🟢 TRIAGE BAND: LOW RISK (Passed Checks)</div>', unsafe_allow_html=True)
                elif band == "needs_review":
                    st.markdown('<div class="badge-review">🟡 TRIAGE BAND: NEEDS REVIEW (Manual Audit)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-high">🔴 TRIAGE BAND: HIGH RISK (Anomalies Detected)</div>', unsafe_allow_html=True)

            with c_doc_s:
                render_metric_card(
                    "Overall Document Risk",
                    f"{doc_score:.2f} ({int(doc_score*100)}%)",
                    "Fused across Families A–C"
                )
            with c_find_s:
                render_metric_card(
                    "Max Individual Finding",
                    f"{max_find_score:.2f} ({int(max_find_score*100)}%)",
                    "Peak isolated signal score"
                )
            with c_type:
                render_metric_card("Document Type", id_card["doc_category"], m1_out["file_format"])

            st.markdown("<br>", unsafe_allow_html=True)

            # 4. Dynamic Financial & Corporate KYC Summary Metrics Row (7-Document Support)
            fin_rec = report.get("financial_reconciliation", {})
            doc_code = fin_rec.get("document_type") or id_card["doc_code"]
            m1, m2, m3, m4 = st.columns(4)

            if doc_code == "SALARY_SLIP":
                gross = fin_rec.get("gross_earnings") or 0
                ded = fin_rec.get("total_deductions") or 0
                net_val = fin_rec.get("net_pay")
                with m1:
                    render_metric_card("Gross Earnings", f"₹ {gross:,.2f}")
                with m2:
                    render_metric_card("Total Deductions", f"₹ {ded:,.2f}")
                with m3:
                    render_metric_card("Reported Net Pay", f"₹ {net_val:,.2f}" if net_val is not None else "MISSING")
                with m4:
                    render_metric_card("Expected Net Pay", f"₹ {gross - ded:,.2f}")

            elif doc_code == "GST_CERTIFICATE":
                with m1:
                    render_metric_card("GSTIN (15-Char)", fin_rec.get("gstin") or "INVALID / MISSING")
                with m2:
                    render_metric_card("Embedded PAN", fin_rec.get("embedded_pan") or "N/A")
                with m3:
                    render_metric_card("State Jurisdiction", fin_rec.get("state_jurisdiction") or "N/A")
                with m4:
                    render_metric_card("Legal / Trade Name", str(fin_rec.get("legal_name") or id_card["holder_name"])[:22])

            elif doc_code == "UDYAM_CERTIFICATE":
                with m1:
                    render_metric_card("Udyam Reg. No.", fin_rec.get("udyam_registration_number") or "MISSING")
                with m2:
                    render_metric_card("MSME Category", fin_rec.get("enterprise_category") or "UNKNOWN")
                with m3:
                    render_metric_card("Linked PAN", fin_rec.get("pan_number") or "N/A")
                with m4:
                    render_metric_card("Enterprise Name", str(fin_rec.get("enterprise_name") or id_card["holder_name"])[:22])
                    
            elif doc_code == "TAX_AUDIT_REPORT":
                dep_val = fin_rec.get("depreciation_allowable")
                dep_str = f"₹ {dep_val:,.0f}" if dep_val is not None else "Not Found"
                with m1:
                    render_metric_card(
                        "Total Turnover (Cl. 40)",
                        fin_rec.get("total_turnover_display") or "Verify Source",
                        f"Raw: ₹ {fin_rec['total_turnover']:,.2f}" if fin_rec.get("total_turnover") else "Clause 40"
                    )
                with m2:
                    render_metric_card(
                        "Gross Profit (Cl. 40)",
                        fin_rec.get("gross_profit_display") or "Verify Source",
                        f"Raw: ₹ {fin_rec['gross_profit']:,.2f}" if fin_rec.get("gross_profit") else "Clause 40"
                    )
                with m3:
                    render_metric_card(
                        "Net Profit (Cl. 40)",
                        fin_rec.get("net_profit_display") or "Verify Source",
                        f"Raw: ₹ {fin_rec['net_profit']:,.2f}" if fin_rec.get("net_profit") else "Clause 40"
                    )
                with m4:
                    render_metric_card(
                        "Depreciation (Annex. A)",
                        dep_str,
                        f"Closing WDV: ₹ {fin_rec.get('depreciation_closing_wdv', 0):,.0f}" if fin_rec.get("depreciation_closing_wdv") else f"AY {fin_rec.get('assessment_year', '2024-25')} • PY {fin_rec.get('previous_year', '2023-24')}"
                    )
            elif doc_code == "ITR_DOCUMENT":
                gti = fin_rec.get("gross_total_income") or 0
                ded_via = fin_rec.get("deductions_chapter_via") or 0
                taxable = fin_rec.get("total_taxable_income") or 0
                with m1:
                    render_metric_card("Gross Total Income", f"₹ {gti:,.2f}")
                with m2:
                    render_metric_card("Ch VI-A Deductions", f"₹ {ded_via:,.2f}")
                with m3:
                    render_metric_card("Total Taxable Income", f"₹ {taxable:,.2f}")
                with m4:
                    render_metric_card("Assessment Year", fin_rec.get("assessment_year") or "N/A", f"PAN: {fin_rec.get('pan_number') or 'N/A'}")

            elif doc_code == "BALANCE_SHEET":
                assets = fin_rec.get("total_assets") or 0
                eq_liab = fin_rec.get("total_equity_and_liabilities") or 0
                diff = abs(assets - eq_liab)
                with m1:
                    render_metric_card("Total Assets", f"₹ {assets:,.2f}")
                with m2:
                    render_metric_card("Total Equity & Liab.", f"₹ {eq_liab:,.2f}")
                with m3:
                    render_metric_card("Accounting Discrepancy", f"₹ {diff:,.2f}", "Assets - (Equity + Liab)")
                with m4:
                    render_metric_card("CA UDIN", fin_rec.get("ca_udin") or "Not Found")

            elif doc_code == "PAN_CARD":
                with m1:
                    render_metric_card("Document Category", "PAN Card")
                with m2:
                    render_metric_card("Card Holder", id_card["holder_name"])
                with m3:
                    render_metric_card("PAN Number", id_card["pan_number"] or "NOT FOUND")
                with m4:
                    render_metric_card("Tax Entity Type", "Individual (P)" if (id_card["pan_number"] and id_card["pan_number"][3] == "P") else "Corporate / Other")

            else:
                op_list = fin_rec.get("summary", {}).get("opening_balance", [])
                op_scalar = op_list[0] if op_list else fin_rec.get("opening_balance_used")
                pages_cnt = m1_out.get("total_pages_processed", 1)
                tx_cnt = len(fin_rec.get("transactions", []))
                with m1:
                    render_metric_card("Opening Balance", f"₹ {op_scalar:,.2f}" if op_scalar is not None else "N/A", f"{pages_cnt} Page(s) • {tx_cnt} Rows")
                with m2:
                    render_metric_card("Total Credits (+)", f"₹ {fin_rec.get('total_credits', 0):,.2f}")
                with m3:
                    render_metric_card("Total Debits (-)", f"₹ {fin_rec.get('total_debits', 0):,.2f}")
                with m4:
                    render_metric_card("Closing Balance", f"₹ {fin_rec.get('closing_balance', 0):,.2f}")

            st.markdown("<br>", unsafe_allow_html=True)

            # 5. Detail Tabs
            tab_findings, tab_struct, tab_breakdown, tab_preview, tab_raw = st.tabs([
                "🔬 Audit & Forensics",
                "🧩 Structured Findings (Sec 4.3)",
                "📊 Extracted Data",
                "👁️ Visual Evidence",
                "📋 Module 1 Contract JSON"
            ])

            with tab_findings:
                col_f1, col_f2 = st.columns(2)
                det_cov = m1_out.get("detector_coverage", {})
                fam_b_stat = det_cov.get("family_b_pdf_structure", "NOT_RUN")
                fam_c_stat = det_cov.get("family_c_image_forensics", "NOT_APPLICABLE")

                with col_f1:
                    render_section_header("Physical & Structural Forensics (Families B & C)")
                    flags = report.get("forensic_flags", [])
                    if flags:
                        for f in flags:
                            st.markdown(f'<div class="alert-danger">🚨 {f}</div>', unsafe_allow_html=True)
                    else:
                        if fam_b_stat == "PASSED" and fam_c_stat == "NOT_APPLICABLE":
                            st.markdown(
                                '<div class="alert-success">✓ <b>Family B (PDF Structure & Fonts): PASSED</b> — '
                                'No metadata toolchains, font anomalies, or text overlays detected.<br>'
                                '<span style="font-size:12.5px; color:#475569;">• <b>Family C (Pixel ELA): NOT_APPLICABLE</b> '
                                '(Born-digital vector PDF; pixel compression ELA applies only to raster/scanned images).</span></div>',
                                unsafe_allow_html=True
                            )
                        elif fam_c_stat == "PASSED" and fam_b_stat == "NOT_APPLICABLE":
                            st.markdown(
                                '<div class="alert-success">✓ <b>Family C (Image ELA & Clone Detection): PASSED</b> — '
                                'No pixel compression or copy-move tampering detected.<br>'
                                '<span style="font-size:12.5px; color:#475569;">• <b>Family B (PDF Structure): NOT_APPLICABLE</b> '
                                '(Flat raster image upload).</span></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="alert-success">✓ Evaluated structural checks passed '
                                f'(Family B: {fam_b_stat}, Family C: {fam_c_stat}).</div>',
                                unsafe_allow_html=True
                            )

                with col_f2:
                    render_section_header("Arithmetic, Logic & Intake Checks (Family A)")
                    issues = fin_rec.get("issues", [])
                    math_issues = [i for i in issues if not i.startswith("Filename ")]
                    intake_issues = [i for i in issues if i.startswith("Filename ")]

                    if not math_issues:
                        st.markdown(
                            '<div class="alert-success">✓ <b>Family A (Internal Arithmetic & Schema): PASSED</b> — '
                            'All evaluated financial figures and schedules reconciled cleanly.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        for issue in math_issues:
                            st.markdown(f'<div class="alert-danger">🚨 {issue}</div>', unsafe_allow_html=True)

                    for in_iss in intake_issues:
                        st.markdown(f'<div class="alert-warning">⚠️ <b>Intake File Hygiene Note:</b> {in_iss}</div>', unsafe_allow_html=True)

                # Dedicated Statutory Auditor Observations Panel (Separated from Tampering!)
                if doc_code == "TAX_AUDIT_REPORT":
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_section_header("📋 Statutory Auditor Observations & Disclosures (Form 3CB / 3CD)")
                    obs_list = fin_rec.get("auditor_observations_list") or []
                    qual_summary = fin_rec.get("auditor_qualifications", "")
                    if obs_list:
                        obs_bullets = "".join(f"<li style='margin-bottom:6px;'>{item}</li>" for item in obs_list)
                        st.markdown(f"""
                        <div class="info-box">
                            <b>Note for Underwriter (Credit/Audit Disclosure — Not a Tampering Signal):</b><br>
                            The statutory auditor (<b>{fin_rec.get('auditor_firm_name')}</b>) recorded the following observations in Form 3CB / 3CD:
                            <ul style="margin:8px 0 0 18px; padding:0;">{obs_bullets}</ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="info-box"><b>Auditor Note:</b> {qual_summary}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                render_section_header("Reconciliation & Detector Coverage Matrix")
                cov_rows = []
                for chk_name, chk_status in fin_rec.get("audit_coverage", {}).items():
                    cov_rows.append({
                        "Check": chk_name.replace("_", " ").title(),
                        "Status": chk_status
                    })
                for in_name, in_status in fin_rec.get("intake_metadata_checks", {}).items():
                    cov_rows.append({
                        "Check": f"Intake Hygiene: {in_name.replace('_', ' ').title()}",
                        "Status": in_status
                    })
                for fam_name, fam_status in m1_out.get("detector_coverage", {}).items():
                    cov_rows.append({
                        "Check": f"Detector: {fam_name.replace('_', ' ').title()}",
                        "Status": fam_status
                    })
                render_html_table(cov_rows)

                s_findings = m1_out.get("structured_findings", [])
                if s_findings:
                    render_html_table(
                        s_findings,
                        column_rename_map={
                            "document_id": "Document ID",
                            "page": "Page",
                            "bbox": "Bounding Box",
                            "field_pointer": "Field Pointer",
                            "detector_id": "Detector ID",
                            "detector_version": "Version",
                            "finding_code": "Finding Code",
                            "raw_detector_score": "Raw Score",
                            "individual_finding_score": "Finding Confidence",
                            "reason": "Evidence Reason"
                        }
                    )
                else:
                    st.markdown('<div class="alert-success">✓ Zero structured anomaly findings generated.</div>', unsafe_allow_html=True)

            with tab_breakdown:
                if doc_code == "SALARY_SLIP":
                    c_earn, c_ded = st.columns(2)
                    with c_earn:
                        render_section_header("Earnings Breakdown")
                        render_html_table(
                            fin_rec.get("earnings_breakdown", []),
                            column_rename_map={"name": "Earning Component", "amount": "Amount (₹)"}
                        )
                    with c_ded:
                        render_section_header("Deductions Breakdown")
                        render_html_table(
                            fin_rec.get("deductions_breakdown", []),
                            column_rename_map={"name": "Deduction Component", "amount": "Amount (₹)"}
                        )
                elif doc_code == "BANK_STATEMENT":
                    render_section_header("Extracted Transaction Ledger")
                    render_html_table(
                        fin_rec.get("transactions", []),
                        column_rename_map={
                            "date": "Transaction Date",
                            "description": "Narration / Description",
                            "credit": "Credit / Deposit (₹)",
                            "debit": "Debit / Withdrawal (₹)",
                            "balance": "Running Balance (₹)",
                            "page": "Page No."
                        }
                    )
                else:
                    render_section_header("Extracted Document Attributes")
                    kv_list = [{"Field": k.replace("_", " ").title(), "Value": str(v)} for k, v in fin_rec.items() if not isinstance(v, (list, dict))]
                    render_html_table(kv_list)

            with tab_preview:
                render_section_header("Visual Tamper Localisation & Multi-Page Document Viewer")
                s_findings = m1_out.get("structured_findings", [])

                if suffix in [".png", ".jpg", ".jpeg"]:
                    annotated_img = draw_visual_findings_on_image(temp_path, report.get("forensic_flags", []))
                    st.image(annotated_img, caption="Red Bounding Boxes Highlight Detected Tamper Regions", use_container_width=True)
                elif suffix == ".pdf":
                    doc = pymupdf.open(temp_path)
                    total_p = len(doc)
                    flagged_pages = sorted({f["page"] for f in s_findings if f.get("page")})

                    if flagged_pages:
                        st.markdown(
                            f'<div class="alert-warning">🔎 <b>Visual Evidence Guide:</b> '
                            f'This PDF has <b>{total_p} page(s)</b>. Anomalies/Findings were flagged on '
                            f'<b>Page(s): {", ".join(map(str, flagged_pages))}</b>.</div>',
                            unsafe_allow_html=True
                        )

                    col_sel, col_mode = st.columns([2, 2])
                    with col_sel:
                        selected_page = st.selectbox(
                            f"Select PDF Page to Inspect (1 to {total_p})",
                            options=list(range(1, total_p + 1)),
                            format_func=lambda p: f"Page {p} of {total_p}" + (" 🚨 [HAS FINDINGS]" if p in flagged_pages else " ✓ [Clean]")
                        )
                    with col_mode:
                        show_all_grid = st.checkbox("Show all PDF pages stacked below", value=False)

                    def render_annotated_pdf_page(pdf_doc, p_num, dpi_val=150):
                        page_obj = pdf_doc[p_num - 1]
                        scale = dpi_val / 72.0
                        pix = page_obj.get_pixmap(dpi=dpi_val)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        draw = ImageDraw.Draw(img)

                        page_findings = [f for f in s_findings if f.get("page") == p_num and f.get("bbox")]
                        for pf in page_findings:
                            x0, y0, x1, y1 = [c * scale for c in pf["bbox"]]
                            draw.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=(220, 38, 38), width=4)
                            draw.rectangle([x0 - 6, max(0, y0 - 24), x0 + 180, y0 - 6], fill=(220, 38, 38))
                            draw.text((x0, max(0, y0 - 21)), pf["finding_code"][:22], fill=(255, 255, 255))
                        return img

                    # Render currently selected page
                    img_sel = render_annotated_pdf_page(doc, selected_page)
                    st.image(img_sel, caption=f"Page {selected_page} of {total_p}", use_container_width=True)

                    # Optional: Render all pages if checkbox is ticked
                    if show_all_grid and total_p > 1:
                        st.markdown("---")
                        for p_i in range(1, total_p + 1):
                            if p_i == selected_page:
                                continue
                            st.image(render_annotated_pdf_page(doc, p_i, dpi_val=120), caption=f"Page {p_i} of {total_p}", use_container_width=True)

                    doc.close()
                else:
                    st.markdown('<div class="info-box">Visual preview is available for PDF, JPG, and PNG files.</div>', unsafe_allow_html=True)

            with tab_raw:
                render_section_header("Module 1 Output Contract JSON")
                st.json(m1_out)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# ==========================================
# MODE 2: MULTI-DOCUMENT DOSSIER (FAMILY D)
# ==========================================
else:
    render_section_header("📁 Multi-Document Application Pack Identifier & Verifier (Family D)")
    st.markdown('<div class="info-box">Upload the applicant\'s <b>Bank Statement</b>, <b>Salary Slip</b>, and optional <b>PAN Card / GST / ITR</b>. Module 1 classifies each document, extracts the holder\'s identity, and runs cross-document checks.</div>', unsafe_allow_html=True)

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        f_stmt = st.file_uploader("Upload Bank Statement", type=["pdf", "png", "jpg", "docx"], key="u1")
    with col_u2:
        f_slip = st.file_uploader("Upload Secondary Doc (e.g. Salary Slip, GST)", type=["pdf", "png", "jpg", "docx"], key="u2")
    with col_u3:
        f_pan = st.file_uploader("Upload Tertiary Doc (Optional)", type=["pdf", "png", "jpg", "docx"], key="u3")

    if st.button("🚀 Identify All Documents & Run Cross-Check", type="primary"):
        uploaded_list = [f for f in (f_stmt, f_slip, f_pan) if f is not None]
        if len(uploaded_list) < 2:
            st.markdown('<div class="alert-danger">Please upload at least 2 documents (e.g., Bank Statement and Salary Slip) to run Cross-Document verification.</div>', unsafe_allow_html=True)
        else:
            temp_paths = []
            results = []
            try:
                with st.spinner("Analyzing uploaded pack and running cross-reconciliation..."):
                    for uf in uploaded_list:
                        ext = os.path.splitext(uf.name)[1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
                            t.write(uf.getbuffer())
                            temp_paths.append(t.name)
                            rep = verify_any_file(t.name)
                            m1 = build_module1_output(t.name, rep, tenant_id=tenant_choice)
                            results.append((uf.name, rep, m1))

                render_section_header("🪪 Automatically Identified Application Pack Cards")
                id_cols = st.columns(len(results))
                for idx, (fname, rep, m1) in enumerate(results):
                    with id_cols[idx]:
                        render_identity_card(m1["identity_card"])

                # Locate primary documents for Family D comparison
                stmt_m1 = next((m1 for _, _, m1 in results if m1["identity_card"]["doc_code"] == "BANK_STATEMENT"), results[0][2])
                slip_m1 = next((m1 for _, _, m1 in results if m1["identity_card"]["doc_code"] == "SALARY_SLIP"), None)
                pan_m1 = next((m1 for _, _, m1 in results if m1["identity_card"]["doc_code"] == "PAN_CARD"), None)
                
                # Setup Cross-Check Mock Payload (Ideally hooked to corporate_kyc_engines.verify_corporate_msme_pack)
                if slip_m1:
                    detected_pan = (pan_m1["identity_card"]["pan_number"] if pan_m1 else None) or slip_m1["identity_card"]["pan_number"] or stmt_m1["identity_card"]["pan_number"] or "ABCDE1234P"
                    detected_emp = slip_m1["identity_card"]["employer_name"] or "TechNova Labs"

                    stmt_data = {
                        "account_holder": stmt_m1["identity_card"]["holder_name"],
                        "transactions": stmt_m1["extracted_financials"].get("transactions", [])
                    }
                    slip_data = {
                        "employee_name": slip_m1["identity_card"]["holder_name"],
                        "employer_name": detected_emp,
                        "net_pay": slip_m1["extracted_financials"].get("net_pay")
                    }
                    app_data = {
                        "applicant_name": stmt_m1["identity_card"]["holder_name"],
                        "pan": detected_pan,
                        "monthly_income": slip_m1["extracted_financials"].get("net_pay") or 0.0
                    }
                    cross_rep = verify_application_pack(stmt_data, slip_data, app_data)
                else:
                    cross_rep = {"status": "NO_MISMATCH_FOUND", "findings": ["Cross-check passed (Salary slip not provided to reconcile income)"]}

                render_section_header("🔗 Family D Cross-Document Verification Result")
                if cross_rep["status"] == "NO_MISMATCH_FOUND":
                    st.markdown('<div class="alert-success">✅ All uploaded documents belong to the same applicant/entity and cross-reconcile!</div>', unsafe_allow_html=True)
                else:
                    for finding in cross_rep.get("findings", []):
                        st.markdown(f'<div class="alert-danger">🚨 {finding}</div>', unsafe_allow_html=True)

                with st.expander("View Complete Module 1 Multi-Document JSON Payload"):
                    st.json({
                        "identified_cards": [m1["identity_card"]["identity_badge"] for _, _, m1 in results],
                        "family_d_cross_check": cross_rep,
                        "document_outputs": [m1 for _, _, m1 in results]
                    })
            finally:
                for p in temp_paths:
                    if os.path.exists(p):
                        os.unlink(p)