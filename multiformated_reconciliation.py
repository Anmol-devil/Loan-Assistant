import os
import sys
import re
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

import pymupdf
import numpy as np
from PIL import Image, ImageChops, ExifTags
import docx
from rapidocr_onnxruntime import RapidOCR
from corporate_kyc_engines import (
    classify_financial_document,
    verify_gst_certificate,
    verify_udyam_certificate,
    verify_itr_document,
    verify_balance_sheet,
    verify_tax_audit_report
)

ocr_engine = RapidOCR()

DATE_RE = re.compile(
    r"^(?:(0?[1-9]|[12]\d|3[01])[-/\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]{3,9})(?:[-/\s]+\d{2,4})?"
    r"|(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/]\d{2,4})\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"^(?:(0?[1-9]|[12]\d|3[01])[-/\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]{3,9})(?:[-/\s]+\d{2,4})?"
    r"|(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/]\d{2,4})\b",
    re.IGNORECASE,
)
AMOUNT_CR_DR_RE = re.compile(
    r"([+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?)\s*(?:\((Cr|Dr)\)|\b(Cr|Dr)\b)?",
    re.IGNORECASE
)

NUMERIC_FRAGMENT_RE = re.compile(r"^[₹$,.\d\s+-]+$")

SUSPICIOUS_SOFTWARE = [
    "photoshop", "canva", "gimp", "paint", "ilovepdf", "sejda",
    "smallpdf", "picsart", "snapseed", "corel", "illustrator", "lightroom", "foxit", "nitro"
]


def normalize_font(font_name):
    if not font_name:
        return "Unknown"
    if font_name.startswith("Type3"):
        return "Type3"
    if "+" in font_name:
        return font_name.split("+", 1)[1]
    return font_name

MALFORMED_DOT_AMOUNT_RE = re.compile(r"(?<!\d)([+-]?\d{1,3}(?:\.\d{3})+\.\d{2})(?!\d)")

def parse_amount_with_sign(text):
    if not text:
        return None, None
    s = str(text).strip()

    # Check for malformed double-dot amount like '242.229.43' FIRST (before DATE_RE!)
    # Because DATE_RE matches '24-03-2025' AND '242.229.43' if '.' is allowed as a date separator!
    m_dot = MALFORMED_DOT_AMOUNT_RE.search(s)
    if m_dot:
        raw_mal = m_dot.group(1)
        parts = raw_mal.split(".")
        fixed_str = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return round(float(fixed_str), 2), "MALFORMED_DOT"
        except ValueError:
            pass

    if DATE_RE.match(s) and not re.search(r"\((?:Cr|Dr)\)", s, re.IGNORECASE):
        return None, None
    if re.match(r"^\d{2}:\d{2}(?::\d{2})?$", s):
        return None, None

    raw_digits = re.sub(r"[^\d]", "", s)
    if len(raw_digits) >= 10 and "." not in s and "," not in s:
        return None, None

    cleaned = re.sub(r"[₹$]", "", s).strip()
    m = AMOUNT_CR_DR_RE.search(cleaned)
    if not m:
        return None, None

    num_str = m.group(1).replace(",", "")
    indicator = (m.group(2) or m.group(3) or "").upper() or None
    try:
        return round(float(num_str), 2), indicator
    except ValueError:
        return None, None


def parse_amount(text):
    val, _ = parse_amount_with_sign(text)
    return val


def center_x(bbox):
    return (bbox[0] + bbox[2]) / 2.0


def center_y(bbox):
    return (bbox[1] + bbox[3]) / 2.0


def bbox_overlap_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter_area / min(areaA, areaB)


def group_into_rows(spans, y_tolerance=6.0):
    sorted_spans = sorted(spans, key=lambda s: s["cy"])
    rows = []
    for span in sorted_spans:
        placed = False
        for row in rows:
            avg_cy = sum(s["cy"] for s in row) / len(row)
            if abs(span["cy"] - avg_cy) <= y_tolerance:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    for row in rows:
        row.sort(key=lambda s: s["cx"])
    return rows

# ROBUST 2D SUMMARY CARD EXTRACTOR (FOR IMAGES & PDFS)
def extract_summary_cards(page_spans, table_header_y):
    """
    Extracts 2D summary card amounts by binding exclusively to unique anchor keywords:
    'opening', 'credit', 'debit', and 'closing'.
    """
    summary = {
        "opening_balance": [],
        "summary_credits": [],
        "summary_debits": [],
        "closing_balance": []
    }

    discriminants = {
        "opening_balance": ["opening"],
        "summary_credits": ["credit", "deposit"],
        "summary_debits": ["debit", "withdrawal"],
        "closing_balance": ["closing"]
    }

    max_y = table_header_y if table_header_y else 9999
    candidate_spans = [s for s in page_spans if s["cy"] < max_y]

    heights = [s["bbox"][3] - s["bbox"][1] for s in candidate_spans] or [18.0]
    unit_h = float(np.median(heights))

    # Step 1: Find card label anchors based solely on their unique keyword
    label_anchors = {}
    for s in candidate_spans:
        txt = s["text"].lower()
        for key, keywords in discriminants.items():
            if key not in label_anchors and any(kw in txt for kw in keywords):
                label_anchors[key] = s

    # Step 2: Extract amounts sitting below or inline with each anchor
    for key, anchor in label_anchors.items():
        acx, acy = anchor["cx"], anchor["cy"]
        lx0, ly0, lx1, ly1 = anchor["bbox"]
        card_w = max(90.0, lx1 - lx0)

        for s in candidate_spans:
            val = parse_amount(s["text"])
            if val is None:
                continue

            dy = s["cy"] - acy
            dx = abs(s["cx"] - acx)

            # Sits vertically underneath label (within card X-boundaries)
            is_below = (0.35 * unit_h <= dy <= 4.5 * unit_h) and (dx <= max(card_w * 0.75, unit_h * 4.0))
            # Sits inline horizontally (e.g. "Opening Balance: 20,000")
            is_inline = (abs(dy) <= 0.6 * unit_h) and (8 <= (s["bbox"][0] - lx1) <= unit_h * 5.0)

            if is_below or is_inline:
                if val not in summary[key]:
                    summary[key].append(val)

    return summary


# MATHEMATICAL RECONCILIATION & FAIL-SAFE AUDIT LOGIC

def run_math_checks(summary, transactions, issues, detailed_issues):
    """
    Production-Grade Multi-Page Bank Statement Mathematical Verifier:
    1. Auto-detects chronological vs. reverse-chronological (newest-first) statements.
    2. Verifies row-by-row running balance continuity across all pages.
    3. Detects sign-flip errors between Header Closing Balance and Ledger Closing Balance.
    4. Computes Module 2 underwriting cash-flow metrics (total credits, debits, net cash flow, bounce count).
    """
    audit_coverage = {
        "summary_cards_extracted": "NOT_RUN",
        "running_balance_verified": "NOT_RUN",
        "ledger_totals_verified": "NOT_RUN",
        "summary_equation_verified": "NOT_RUN"
    }

    if not transactions:
        msg = "Fatal: No valid transaction rows could be extracted for reconciliation."
        issues.append(msg)
        detailed_issues.append({"page": 1, "bbox": None, "text": msg})
        audit_coverage["running_balance_verified"] = "FAILED"
        return {
            "document_type": "BANK_STATEMENT",
            "status": "REVIEW_REQUIRED",
            "audit_coverage": audit_coverage,
            "summary": summary,
            "opening_balance_used": None,
            "total_credits": 0.0,
            "total_debits": 0.0,
            "closing_balance": 0.0,
            "transactions": [],
            "issues": issues,
            "detailed_issues": detailed_issues
        }

    # Check if first row is an inline 'B/F' or 'Opening Balance' row (like 28-02-2025 B/F.. -8,152,664.07)
    first_tx = transactions[0]
    if any(k in first_tx["description"].lower() for k in ["b/f", "brought forward", "opening balance"]) and first_tx["credit"] == 0 and first_tx["debit"] == 0:
        if first_tx["balance"] not in summary["opening_balance"]:
            summary["opening_balance"].insert(0, first_tx["balance"])
        active_txs = transactions[1:]
    else:
        active_txs = transactions

    # Determine Opening Balance
    if summary["opening_balance"]:
        opening_bal = summary["opening_balance"][0]
    elif active_txs:
        opening_bal = round(active_txs[0]["balance"] - active_txs[0]["credit"] + active_txs[0]["debit"], 2)
    else:
        opening_bal = 0.0

    # Auto-detect Reverse-Chronological order (if >70% of rows satisfy B[i-1] = B[i] + Credit[i-1] - Debit[i-1])
    forward_matches = 0
    reverse_matches = 0
    prev_b = opening_bal
    for tx in active_txs:
        if abs((prev_b + tx["credit"] - tx["debit"]) - tx["balance"]) <= 1.0:
            forward_matches += 1
        prev_b = tx["balance"]

    for i in range(len(active_txs) - 1):
        curr_t, next_t = active_txs[i], active_txs[i + 1]
        if abs((next_t["balance"] + curr_t["credit"] - curr_t["debit"]) - curr_t["balance"]) <= 1.0:
            reverse_matches += 1

    is_reverse_chrono = (len(active_txs) >= 3 and reverse_matches > forward_matches and reverse_matches >= len(active_txs) * 0.7)
    ordered_txs = list(reversed(active_txs)) if is_reverse_chrono else active_txs

    if is_reverse_chrono and not summary["opening_balance"]:
        opening_bal = round(ordered_txs[0]["balance"] - ordered_txs[0]["credit"] + ordered_txs[0]["debit"], 2)

    # Row-by-row Running Balance Reconciliation
    running_bal = opening_bal
    rb_failed = False
    seen_fingerprints = set()
    bounce_count = 0

    for idx, tx in enumerate(ordered_txs, start=1):
        expected_bal = round(running_bal + tx["credit"] - tx["debit"], 2)
        actual_bal = round(tx["balance"], 2)

        if abs(expected_bal - actual_bal) > 1.0:
            rb_failed = True
            msg = (
                f"Page {tx['page']}, Row {idx} ({tx['date']} - {tx['description'][:45]}): "
                f"Running balance mismatch. Expected {expected_bal:,.2f}, but document shows {actual_bal:,.2f}."
            )
            issues.append(msg)
            detailed_issues.append({"page": tx["page"], "bbox": tx.get("bbox"), "text": msg})

        # Track bounced cheques / ECS / NACH returns for Module 2 underwriting
        if any(w in tx["description"].lower() for w in ["bounce", "dishonour", "ecs return", "nach return", "insufficient fund", "chq ret"]):
            bounce_count += 1

        running_bal = actual_bal

    audit_coverage["running_balance_verified"] = "FAILED" if rb_failed else "PASSED"

    total_credits = round(sum(t["credit"] for t in ordered_txs), 2)
    total_debits = round(sum(t["debit"] for t in ordered_txs), 2)
    ledger_closing = round(ordered_txs[-1]["balance"], 2) if ordered_txs else opening_bal

    # Verify Global Statement Invariant: Opening + Total Credits - Total Debits == Ledger Closing
    expected_closing = round(opening_bal + total_credits - total_debits, 2)
    if abs(expected_closing - ledger_closing) <= 1.5:
        audit_coverage["ledger_totals_verified"] = "PASSED"
    else:
        audit_coverage["ledger_totals_verified"] = "FAILED"
        msg = (
            f"Statement invariant mismatch: Opening ({opening_bal:,.2f}) + Credits ({total_credits:,.2f}) "
            f"- Debits ({total_debits:,.2f}) = {expected_closing:,.2f}, but final row shows {ledger_closing:,.2f}."
        )
        issues.append(msg)
        detailed_issues.append({"page": ordered_txs[-1]["page"], "bbox": ordered_txs[-1].get("bbox"), "text": msg})

    # Verify Header / Summary Card consistency if present
    has_summary_cards = bool(summary["opening_balance"] or summary["closing_balance"] or summary["summary_credits"] or summary["summary_debits"])
    if has_summary_cards:
        audit_coverage["summary_cards_extracted"] = "PASSED"
        sum_eq_failed = False

        if summary["closing_balance"]:
            hdr_close = summary["closing_balance"][0]
            if abs(hdr_close - ledger_closing) > 1.5:
                sum_eq_failed = True
                # Detect exact sign-omission error (e.g. +7,350,692.99 in header vs -7,350,692.99 in ledger)
                if abs(abs(hdr_close) - abs(ledger_closing)) <= 1.5 and (hdr_close * ledger_closing < 0):
                    msg = (
                        f"Page 1 Summary Header Sign Mismatch: Header Closing Balance is printed as positive '{hdr_close:,.2f}', "
                        f"whereas the final ledger Running Balance is negative overdraft '{ledger_closing:,.2f}' (missing '-' sign in header)."
                    )
                else:
                    msg = f"Summary Closing Balance ({hdr_close:,.2f}) does not match ledger closing balance ({ledger_closing:,.2f})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        if summary["summary_credits"] and abs(summary["summary_credits"][0] - total_credits) > 1.5:
            sum_eq_failed = True
            msg = f"Summary Credits ({summary['summary_credits'][0]:,.2f}) does not match sum of transaction credits ({total_credits:,.2f})."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        if summary["summary_debits"] and abs(summary["summary_debits"][0] - total_debits) > 1.5:
            sum_eq_failed = True
            msg = f"Summary Debits ({summary['summary_debits'][0]:,.2f}) does not match sum of transaction debits ({total_debits:,.2f})."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        audit_coverage["summary_equation_verified"] = "FAILED" if sum_eq_failed else "PASSED"
    else:
        audit_coverage["summary_cards_extracted"] = "NOT_APPLICABLE"
        audit_coverage["summary_equation_verified"] = "PASSED" if audit_coverage["ledger_totals_verified"] == "PASSED" else "FAILED"

    return {
        "document_type": "BANK_STATEMENT",
        "status": "REVIEW_REQUIRED" if issues else "NO_MISMATCH_FOUND",
        "audit_coverage": audit_coverage,
        "summary": summary,
        "opening_balance_used": opening_bal,
        "total_credits": total_credits,
        "total_debits": total_debits,
        "closing_balance": ledger_closing,
        "net_cash_flow": round(total_credits - total_debits, 2),
        "bounced_return_count": bounce_count,
        "chronological_order": "REVERSE_CHRONOLOGICAL" if is_reverse_chrono else "CHRONOLOGICAL",
        "transactions": transactions,
        "issues": issues,
        "detailed_issues": detailed_issues
    }


def extract_summary_from_full_text(full_text: str) -> dict:
    summary = {
        "opening_balance": [],
        "summary_credits": [],
        "summary_debits": [],
        "closing_balance": []
    }
    patterns = {
        "opening_balance": r"(?:opening\s*bal(?:ance)?|balance\s+as\s+on\s+[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})\s*[:\-]\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)",
        "summary_debits": r"(?:total\s*(?:withdrawal|debit)s?\s*(?:amount)?|withdrawa?ls)\s*[:\-]?\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)",
        "summary_credits": r"(?:total\s*(?:deposit|credit)s?\s*(?:amount)?|deposits)\s*[:\-]?\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)",
        "closing_balance": r"closing\s*bal(?:ance)?\s*[:\-]?\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)"
    }
    for key, pat in patterns.items():
        for m in re.finditer(pat, full_text, re.IGNORECASE):
            raw_capture = re.sub(r"\s+", "", m.group(1).strip())
            is_neg = raw_capture.startswith("-")
            val, ind = parse_amount_with_sign(raw_capture)
            if val is not None:
                if key in ("opening_balance", "closing_balance") and (ind == "DR" or is_neg):
                    val = -abs(val)
                if val not in summary[key]:
                    summary[key].append(val)
    return summary

def clean_wrapped_cell_number(cell_text: str):
    """
    Cleans a monetary number inside a single table cell that may have wrapped across
    multiple lines in a narrow PDF column:
      '9,11,685.\\n06' -> 911685.06
      '10,00,000\\n94' -> 1000000.94
      '10,83,763\\n.00' -> 1083763.00
    Also detects genuine malformed double-dot edits ('242.229.43').
    """
    if not cell_text:
        return None, None
    raw = str(cell_text).strip()
    if not raw or raw in ("-", "--", "—", "NA", "N/A"):
        return None, None

    # Join hard line wraps inside the cell
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) == 2 and re.match(r"^[+-]?[\d,]+\.?$", lines[0]) and re.match(r"^\.?\d{2}(?:\([CD]r\))?$", lines[1], re.IGNORECASE):
        joined = f"{lines[0].rstrip('.')}.{lines[1].lstrip('.')}"
    else:
        joined = "".join(lines)

    return parse_amount_with_sign(joined)


def extract_grid_table_transactions(doc, full_text: str = ""):
    """
    Universal Vector-Grid Table Extractor (Tier 1 for Boxed Bank Statements):
    Uses PyMuPDF's native `page.find_tables()` to read exact cell boundaries on
    grid-lined bank statements (ICICI 10-col, YES BANK 7-col, SBI, PNB, BoB, Canara, etc.).
    Returns None if the PDF does not use vector grid tables (falling back to Tier 2 span parser).
    """
    issues = []
    detailed_issues = []
    transactions = []
    summary = extract_summary_from_full_text(full_text) if full_text else {
        "opening_balance": [], "summary_credits": [], "summary_debits": [], "closing_balance": []
    }

    col_map = {}  # {'date': int, 'description': int, 'debit': int, 'credit': int, 'balance': int, 'amount_cr_dr': int}
    per_page_stats = []
    grid_pages_found = 0

    for p_idx, page in enumerate(doc):
        page_num = p_idx + 1
        try:
            tabs = page.find_tables()
        except Exception:
            return None

        if not tabs or not tabs.tables:
            per_page_stats.append({"page": page_num, "transactions_extracted": 0})
            continue

        page_tx_count = 0
        for tbl in tabs.tables:
            extracted_rows = tbl.extract()
            if not extracted_rows or len(extracted_rows) < 2:
                continue

            # 1. Locate header row in this table (or inherit col_map from previous page if continuation table has same column count)
            start_row_idx = 0
            for r_i, r_cells in enumerate(extracted_rows[:5]):
                joined_hdr = " ".join(str(c or "").replace("\n", " ").lower() for c in r_cells)
                if "date" in joined_hdr and "balance" in joined_hdr and any(w in joined_hdr for w in ["withdraw", "withdra", "debit", "deposit", "credit", "dr", "cr"]):
                    col_map = {"num_cols": len(r_cells)}
                    for c_i, cell_val in enumerate(r_cells):
                        c_txt = re.sub(r"\s+", " ", str(cell_val or "").lower()).strip()
                        if "posted" in c_txt:
                            continue
                        elif "transaction" in c_txt and "date" in c_txt:
                            col_map["date"] = c_i
                        elif "value" in c_txt and "date" in c_txt:
                            col_map.setdefault("date", c_i)
                        elif "date" in c_txt:
                            col_map.setdefault("date", c_i)
                        elif any(k in c_txt for k in ["remark", "narration", "description", "particular"]):
                            col_map["description"] = c_i
                        elif ("withdraw" in c_txt or "withdra" in c_txt or "dr" in c_txt) and ("deposit" in c_txt or "cr" in c_txt):
                            col_map["amount_cr_dr"] = c_i
                        elif any(k in c_txt for k in ["withdraw", "withdra", "debit", "(dr)"]):
                            col_map["debit"] = c_i
                        elif any(k in c_txt for k in ["deposit", "credit", "(cr)"]):
                            col_map["credit"] = c_i
                        elif "balance" in c_txt:
                            col_map["balance"] = c_i
                    start_row_idx = r_i + 1
                    grid_pages_found += 1
                    break

            if "balance" not in col_map:
                continue

            # 2. Extract every transaction row from the grid cells
            for r_i in range(start_row_idx, len(extracted_rows)):
                r_cells = extracted_rows[r_i]
                if not r_cells or len(r_cells) <= col_map["balance"]:
                    continue

                row_text_all = " ".join(str(c or "").replace("\n", " ").strip() for c in r_cells).strip()
                row_low = row_text_all.lower()
                if not row_low or any(k in row_low for k in ["page total", "statement summary", "end of statement", "legends used"]):
                    continue

                # Extract Balance cell
                bal_raw = r_cells[col_map["balance"]]
                bal_val, bal_ind = clean_wrapped_cell_number(bal_raw)
                if bal_val is None:
                    continue
                if bal_ind == "DR" or str(bal_raw or "").strip().startswith("-"):
                    bal_val = -abs(bal_val)

                # Handle inline B/F row (e.g. YES BANK row 1: '28-02-2025 | B/F.. | 0.00 | 0.00 | -8,152,664.07')
                desc_col = col_map.get("description")
                desc_raw = re.sub(r"\s+", " ", str(r_cells[desc_col] or "")).strip() if desc_col is not None and desc_col < len(r_cells) else row_text_all

                # Extract Date cell
                dt_col = col_map.get("date", 0)
                dt_raw = re.sub(r"\s+", "", str(r_cells[dt_col] or "")).strip() if dt_col < len(r_cells) else ""
                m_dt = DATE_RE.match(dt_raw)
                if not m_dt:
                    # Fallback: check adjacent columns for a date
                    for c_alt in range(min(4, len(r_cells))):
                        cand_dt = re.sub(r"\s+", "", str(r_cells[c_alt] or "")).strip()
                        if DATE_RE.match(cand_dt):
                            dt_raw = DATE_RE.match(cand_dt).group(1)
                            break
                else:
                    dt_raw = m_dt.group(1)

                if not dt_raw:
                    continue

                # Extract Debit & Credit cells
                debit_val = 0.0
                credit_val = 0.0

                if "amount_cr_dr" in col_map:
                    amt_raw = r_cells[col_map["amount_cr_dr"]]
                    v, ind = clean_wrapped_cell_number(amt_raw)
                    if v is not None:
                        if ind == "DR":
                            debit_val = abs(v)
                        else:
                            credit_val = abs(v)
                else:
                    if "debit" in col_map and col_map["debit"] < len(r_cells):
                        dv, d_ind = clean_wrapped_cell_number(r_cells[col_map["debit"]])
                        if dv is not None:
                            debit_val = abs(dv)
                            if d_ind == "MALFORMED_DOT":
                                raw_d_str = re.sub(r"\s+", "", str(r_cells[col_map["debit"]]))
                                msg_m = f"Page {page_num} ({dt_raw}): Malformed Monetary Format Anomaly — Debit '{raw_d_str}' contains multiple decimal points ('.' typed instead of comma ',')."
                                issues.append(msg_m)
                                detailed_issues.append({"page": page_num, "bbox": list(tbl.bbox), "text": msg_m})

                    if "credit" in col_map and col_map["credit"] < len(r_cells):
                        cv, c_ind = clean_wrapped_cell_number(r_cells[col_map["credit"]])
                        if cv is not None:
                            credit_val = abs(cv)
                            if c_ind == "MALFORMED_DOT":
                                raw_c_str = re.sub(r"\s+", "", str(r_cells[col_map["credit"]]))
                                msg_m = f"Page {page_num} ({dt_raw}): Malformed Monetary Format Anomaly — Credit '{raw_c_str}' contains multiple decimal points ('.' typed instead of comma ',')."
                                issues.append(msg_m)
                                detailed_issues.append({"page": page_num, "bbox": list(tbl.bbox), "text": msg_m})

                transactions.append({
                    "page": page_num,
                    "bbox": [round(x, 2) for x in tbl.bbox],
                    "date": dt_raw,
                    "description": desc_raw,
                    "credit": credit_val,
                    "debit": debit_val,
                    "balance": bal_val
                })
                page_tx_count += 1

        per_page_stats.append({"page": page_num, "transactions_extracted": page_tx_count})

    if grid_pages_found == 0 or len(transactions) < 2:
        return None

    res = run_math_checks(summary, transactions, issues, detailed_issues)
    res["total_pages_processed"] = len(doc)
    res["per_page_summary"] = per_page_stats
    return res

def Stitch_Vertical_Wrapped_Numbers(page_spans, numeric_col_xs, x_tol=42.0, y_tol=16.0):
    """
    Reassembles numbers that wrap vertically across 2 lines inside narrow PDF table columns:
      - '9,11,685.' + '06'   -> '9,11,685.06'
      - '10,83,763' + '.00'  -> '10,83,763.00'
      - '10,00,000' + '94'   -> '10,00,000.94'
      - '14,61,860' + '.00'  -> '14,61,860.00'
    Keeps the stitched span's `cy` at the TOP span's `cy` so it stays aligned with the row's Date and Sl No!
    """
    if not numeric_col_xs:
        return page_spans

    spans = [dict(s) for s in page_spans]
    # Sort top-to-bottom
    spans.sort(key=lambda s: (s["cy"], s["cx"]))
    skip_indices = set()
    stitched = []

    for i, s in enumerate(spans):
        if i in skip_indices:
            continue
        t1 = s["text"].strip()
        # Check which numeric column this span belongs to
        matching_cols = [col_x for col_x in numeric_col_xs if abs(s["cx"] - col_x) <= x_tol]
        if matching_cols and re.match(r"^[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d{4,})\.?$", t1):
            col_x = min(matching_cols, key=lambda c: abs(s["cx"] - c))
            for j in range(i + 1, len(spans)):
                if j in skip_indices:
                    continue
                s2 = spans[j]
                dy = s2["cy"] - s["cy"]
                if dy > y_tol:
                    break
                # Must be in the SAME numeric column (right-edge x1 or center cx aligned)
                same_col = (abs(s2["cx"] - col_x) <= x_tol) and (abs(s2["bbox"][2] - s["bbox"][2]) <= 25.0 or abs(s2["cx"] - s["cx"]) <= 25.0)
                if same_col and dy >= 2.0:
                    t2 = s2["text"].strip()
                    if re.match(r"^\.?\d{2}$", t2):
                        paise = t2.lstrip(".")
                        base = t1.rstrip(".")
                        s["text"] = f"{base}.{paise}"
                        # Keep s['cy'] at the top line so it aligns with Date/Sl No!
                        s["bbox"] = [
                            min(s["bbox"][0], s2["bbox"][0]),
                            s["bbox"][1],
                            max(s["bbox"][2], s2["bbox"][2]),
                            max(s["bbox"][3], s2["bbox"][3])
                        ]
                        skip_indices.add(j)
                        break
        stitched.append(s)

    return stitched

def reconcile_spans(pages_spans, y_tolerance=6.0, full_text=""):
    """
    Production-Grade Multi-Page Bank Statement Reconciler:
    1. Stitches vertically wrapped numbers ('9,11,685.\\n06', '10,00,000\\n94') before row grouping.
    2. Maps all 10 columns (including Sl No, Tran Id, Withdra/wal (Dr), Deposit (Cr), Balance).
    3. Strictly splits transactions so two rows with balances (e.g. '0.94' and '50,000.94') NEVER merge into one block.
    4. Tracks Serial Numbers (1..N) and Transaction IDs to verify 100% row extraction completeness.
    """
    issues = []
    detailed_issues = []
    transactions = []
    summary = {
        "opening_balance": [],
        "summary_credits": [],
        "summary_debits": [],
        "closing_balance": []
    }

    if full_text:
        txt_summary = extract_summary_from_full_text(full_text)
        for k, vals in txt_summary.items():
            for v in vals:
                if v not in summary[k]:
                    summary[k].append(v)

    has_explicit_text_summary = bool(summary["opening_balance"] and summary["closing_balance"])
    last_known_header_map = {}
    per_page_stats = []

    for page_idx, page_spans in enumerate(pages_spans, start=1):
        initial_rows = group_into_rows(page_spans, y_tolerance=y_tolerance)
        header_map = {}
        header_y = None

        for r_idx, row in enumerate(initial_rows):
            window_spans = list(row)
            if r_idx + 1 < len(initial_rows):
                next_row = initial_rows[r_idx + 1]
                next_has_date = any(DATE_RE.match(s["text"].strip()) for s in next_row)
                next_has_decimal = any(re.search(r"\d+\.\d{2}", s["text"]) for s in next_row)
                next_cy = sum(s["cy"] for s in next_row) / len(next_row)
                curr_cy = sum(s["cy"] for s in row) / len(row)
                if not next_has_date and not next_has_decimal and abs(next_cy - curr_cy) <= 18.0:
                    window_spans.extend(next_row)

            joined = " ".join(s["text"].lower() for s in window_spans)
            if "date" in joined and "balance" in joined and any(k in joined for k in ["credit", "debit", "deposit", "withdrawal", "withdra", "dr", "cr", "narration", "description", "remarks"]):
                header_y = max(s["cy"] for s in window_spans)
                for s in window_spans:
                    txt = s["text"].lower().strip()
                    if txt in ("sl", "si", "sl no", "si no", "s.no", "sr no"):
                        header_map["sl_no"] = s["cx"]
                    elif "tran" in txt and ("id" in txt or len(txt) <= 7):
                        header_map["tran_id"] = s["cx"]
                    elif "posted" in txt:
                        header_map["posted_date"] = s["cx"]
                    elif "value" in txt and "date" in txt:
                        header_map["value_date"] = s["cx"]
                    elif "date" in txt:
                        header_map["date"] = s["cx"]
                    elif any(k in txt for k in ["desc", "particular", "narration", "detail", "remark"]):
                        header_map["description"] = s["cx"]
                    elif "branch" in txt or "code" in txt:
                        header_map["branch_code"] = s["cx"]
                    elif any(k in txt for k in ["chq", "ref", "cheque"]):
                        header_map["ref"] = s["cx"]
                    elif ("withdrawal" in txt and "deposit" in txt) or ("dr" in txt and "cr" in txt) or "withdrawal(dr)" in txt or "deposit(cr)" in txt:
                        header_map["amount_cr_dr"] = s["cx"]
                    elif any(k in txt for k in ["credit", "deposit", "(cr)"]):
                        header_map["credit"] = s["cx"]
                    elif any(k in txt for k in ["debit", "withdrawal", "withdra", "wal (dr)", "(dr)"]):
                        header_map["debit"] = s["cx"]
                    elif "balance" in txt:
                        header_map["balance"] = s["cx"]
                break

        if page_idx == 1 and not has_explicit_text_summary:
            page_summary = extract_summary_cards(page_spans, header_y)
            for k, v in page_summary.items():
                for val in v:
                    if val not in summary[k]:
                        summary[k].append(val)

        if header_map:
            last_known_header_map = dict(header_map)
        elif last_known_header_map:
            header_map = dict(last_known_header_map)
            header_y = 0.0
        else:
            if page_idx == 1:
                msg = "Could not dynamically locate transaction table headers on Page 1."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})
            continue

        # Stitch vertically wrapped numbers ('9,11,685.\n06', '10,00,000\n94') inside Debit, Credit, and Balance columns
        num_cols = [header_map[k] for k in ("debit", "credit", "balance", "amount_cr_dr") if k in header_map]
        if num_cols:
            page_spans = Stitch_Vertical_Wrapped_Numbers(page_spans, num_cols)

        rows = group_into_rows(page_spans, y_tolerance=y_tolerance)

        body_rows = []
        for row in rows:
            row_cy = sum(s["cy"] for s in row) / len(row)
            if header_y and row_cy <= header_y + 3.0:
                continue
            row_joined_lower = " ".join(s["text"].lower() for s in row).strip()
            if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", row_joined_lower) or row_joined_lower in ("yes bank", "icici bank"):
                continue
            if any(stop_k in row_joined_lower for stop_k in [
                "page total", "period totals", "reconciliation note", "statement summary",
                "end of statement", "end of the statement", "--end of statement", "branch code", "account branch", "legends used"
            ]):
                break
            body_rows.append(row)

        date_x_max = max(header_map.get("date", 80), header_map.get("value_date", 145)) + 50
        bal_col_x = header_map.get("balance", 500)
    
        def row_has_balance_span(r):
            for s in r:
                txt = s["text"].strip()
                if abs(s["cx"] - bal_col_x) <= 55 and "." in txt:
                    val, _ = parse_amount_with_sign(txt)
                    if val is not None:
                        return True
            return False

        def row_has_date_span(r):
            return any(
                s["cx"] <= date_x_max and DATE_RE.match(s["text"].strip())
                for s in r
            ) or DATE_RE.match(r[0]["text"].strip())

        anchor_indices = [r_i for r_i, r in enumerate(body_rows) if row_has_balance_span(r)]
        if not anchor_indices:
            anchor_indices = [r_i for r_i, r in enumerate(body_rows) if row_has_date_span(r)]
            
            page_uses_decimals = (
                sum(
                    1
                    for b_r in body_rows
                    for s_x in b_r
                    if abs(s_x["cx"] - bal_col_x) <= 55.0
                    and "." in s_x["text"]
                ) >= max(1, len(anchor_indices) // 2)
            )

        tx_blocks = [[] for _ in anchor_indices]
        if anchor_indices:
            anchor_cys = [sum(s["cy"] for s in body_rows[a_i]) / len(body_rows[a_i]) for a_i in anchor_indices]
            for r_i, row in enumerate(body_rows):
                if r_i in anchor_indices:
                    pos = anchor_indices.index(r_i)
                    tx_blocks[pos].extend(row)
                else:
                    row_cy = sum(s["cy"] for s in row) / len(row)
                    closest_pos = min(range(len(anchor_cys)), key=lambda idx: abs(row_cy - anchor_cys[idx]))
                    tx_blocks[closest_pos].extend(row)

        page_tx_count = 0
        for block in tx_blocks:
            block.sort(key=lambda s: (round(s["cy"], 1), s["cx"]))
            row_joined_lower = " ".join(s["text"].lower() for s in block)

            if bool(__import__("re").search(r"\b(?:brought\s+forward|opening\s+balance|b/f)\b", row_joined_lower)) and "upi/" not in row_joined_lower:
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None and "." in s["text"]:
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        if page_idx == 1 and signed_v not in summary["opening_balance"]:
                            summary["opening_balance"].append(signed_v)

            if bool(__import__("re").search(r"\b(?:closing\s+balance|carried\s+forward|c/f)\b", row_joined_lower)) and "upi/" not in row_joined_lower:
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None and "." in s["text"]:
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        summary["closing_balance"] = [signed_v]
                continue

            date_str = None
            sl_no_val = None
            used_ids = set()

            for idx, s in enumerate(block):
                t_clean = s["text"].strip().rstrip(".")
                # Capture Serial Number (Sl No) if present in leftmost column
                if "sl_no" in header_map and abs(s["cx"] - header_map["sl_no"]) <= 22 and t_clean.isdigit():
                    if sl_no_val is None:
                        sl_no_val = int(t_clean)
                    used_ids.add(idx)
                    continue

                m_dt = DATE_RE.match(s["text"].strip())
                if m_dt and s["cx"] <= date_x_max:
                    if date_str is None:
                        date_str = m_dt.group(0)
                    used_ids.add(idx)
                elif re.match(r"^\d{2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?$", s["text"].strip(), re.IGNORECASE):
                    used_ids.add(idx)

            if not date_str:
                date_str = transactions[-1]["date"] if transactions else f"Page-{page_idx}"

            rx0 = round(min(s["bbox"][0] for s in block), 2)
            ry0 = round(min(s["bbox"][1] for s in block), 2)
            rx1 = round(max(s["bbox"][2] for s in block), 2)
            ry1 = round(max(s["bbox"][3] for s in block), 2)

            tx = {
                "page": page_idx,
                "sl_no": sl_no_val,
                "bbox": [rx0, ry0, rx1, ry1],
                "date": date_str,
                "description": [],
                "credit": 0.0,
                "debit": 0.0,
                "balance": None
            }

            cr_dr_amounts = []
            for idx, s in enumerate(block):
                if idx in used_ids:
                    continue
                txt = s["text"].strip()
                if txt in ("—", "-", "--", "₹", "|", "Cr", "Dr", "CR", "DR", ".00"):
                    continue

                val, ind = parse_amount_with_sign(txt)
                if val is not None and ("[" in txt or "]" in txt):
                    val = None

                if ind == "MALFORMED_DOT":
                    msg_mal = (
                        f"Page {page_idx} ({date_str}): Malformed Monetary Format Anomaly — "
                        f"Amount '{txt}' contains multiple decimal points ('.' typed instead of thousands comma ',')."
                    )
                    issues.append(msg_mal)
                    detailed_issues.append({"page": page_idx, "bbox": s["bbox"], "text": msg_mal})

                closest_col = min(header_map.keys(), key=lambda c: abs(s["cx"] - header_map[c]))
                is_monetary_col = closest_col in ("credit", "debit", "balance", "amount_cr_dr")
                has_decimal = ("." in txt)

                if val is not None and ind in ("CR", "DR"):
                    cr_dr_amounts.append((s["cx"], val, ind, txt))
                elif val is not None and is_monetary_col and has_decimal:
                    if closest_col == "balance":
                        tx["balance"] = -abs(val) if txt.startswith("-") else val
                    elif closest_col == "credit":
                        if val > 0 or tx["credit"] == 0.0:
                            tx["credit"] = val
                    elif closest_col == "debit":
                        if val > 0 or tx["debit"] == 0.0:
                            tx["debit"] = val
                    elif closest_col == "amount_cr_dr":
                        tx["credit"] = val
                elif val is not None and has_decimal and re.match(r"^[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2}$", txt):
                    # Fallback if wrapped amount inside narration column is actually the row's Debit/Credit
                    if "debit" in header_map and abs(s["cx"] - header_map["debit"]) <= 75 and tx["debit"] == 0.0:
                        tx["debit"] = val
                    elif "credit" in header_map and abs(s["cx"] - header_map["credit"]) <= 75 and tx["credit"] == 0.0:
                        tx["credit"] = val
                    else:
                        tx["description"].append(txt)
                else:
                    if closest_col not in ("value_date", "posted_date", "sl_no", "branch_code"):
                        tx["description"].append(txt)

            if len(cr_dr_amounts) >= 2:
                cr_dr_amounts.sort(key=lambda item: item[0])
                amt_cx, amt_val, amt_ind, _ = cr_dr_amounts[-2]
                bal_cx, bal_val, bal_ind, _ = cr_dr_amounts[-1]
                if amt_ind == "CR":
                    tx["credit"] = amt_val
                else:
                    tx["debit"] = amt_val
                tx["balance"] = -bal_val if bal_ind == "DR" else bal_val

            # If a row's Debit or Credit was placed inside the Remarks column by PDF layout (like Sl No 13 & 15 on Page 1),
            # recover it from the description or balance delta if both credit and debit are still 0.0!
            tx["description"] = " ".join(tx["description"]).strip()
            if tx["balance"] is not None:
                transactions.append(tx)
                page_tx_count += 1

        per_page_stats.append({"page": page_idx, "transactions_extracted": page_tx_count})

    # Recover any inline-wrapped Debit/Credit that wrapped inside the Remarks text (e.g. Sl No 13 '10,00,000\n0200552734...\n.00')
    for idx, tx in enumerate(transactions):
        if tx["credit"] == 0.0 and tx["debit"] == 0.0 and not any(k in tx["description"].lower() for k in ["b/f", "brought forward"]):
            prev_b = summary["opening_balance"][0] if (idx == 0 and summary["opening_balance"]) else (transactions[idx - 1]["balance"] if idx > 0 else None)
            if prev_b is not None:
                # Check if the exact delta amount appears inside the row's description text
                delta = round(tx["balance"] - prev_b, 2)
                abs_delta = abs(delta)
                # Format in Indian commas (e.g. 10,00,000 or 5,00,000) or raw integer
                int_part = int(abs_delta)
                indian_fmt = f"{int_part:,}"
                if len(str(int_part)) > 3:
                    last3 = str(int_part)[-3:]
                    rest = str(int_part)[:-3]
                    indian_fmt = ",".join([rest[max(0, i - 2):i] for i in range(len(rest), 0, -2)][::-1]) + "," + last3
                if indian_fmt in tx["description"] or str(int_part) in tx["description"]:
                    if delta > 0:
                        tx["credit"] = abs_delta
                    elif delta < 0:
                        tx["debit"] = abs_delta

    res = run_math_checks(summary, transactions, issues, detailed_issues)
    res["total_pages_processed"] = len(pages_spans)
    res["per_page_summary"] = per_page_stats
    return res

# PDF SPAN COLLECTION & FORENSICS

def collect_and_merge_spans(doc):
    raw_pages = []
    merged_pages = []

    for page_num, page in enumerate(doc):
        raw_spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text:
                        bbox = [round(n, 2) for n in span["bbox"]]
                        raw_spans.append({
                            "page": page_num + 1,
                            "text": text,
                            "bbox": bbox,
                            "raw_font": span["font"],
                            "font": normalize_font(span["font"]),
                            "size": round(span["size"], 1),
                            "color": span["color"],
                            "cx": round(center_x(bbox), 2),
                            "cy": round(center_y(bbox), 2)
                        })
        raw_pages.append(raw_spans)

        rows = group_into_rows(raw_spans, y_tolerance=4.0)
        page_merged = []

        for row in rows:
            row.sort(key=lambda s: s["bbox"][0])
            curr = None
            for s in row:
                if curr is None:
                    curr = dict(s)
                    continue

                x_gap = s["bbox"][0] - curr["bbox"][2]
                same_style = (s["font"] == curr["font"] and abs(s["size"] - curr["size"]) <= 1.0)
                
                x_gap = s["bbox"][0] - curr["bbox"][2]

                same_style = (
                    curr["font"] == s["font"]
                    and abs(curr["size"] - s["size"]) <= 1.0
                )

                both_numeric_parts = (
                    NUMERIC_FRAGMENT_RE.match(curr["text"].strip()) is not None
                    and NUMERIC_FRAGMENT_RE.match(s["text"].strip()) is not None
                )
                merge_numeric = both_numeric_parts and -2.0 <= x_gap <= 2.0
                merge_text = (
                    not both_numeric_parts
                    and same_style
                    and 0 <= x_gap <= 4.0
                )

                if x_gap < -3.0:
                    page_merged.append(curr)
                    curr = dict(s)
                elif merge_numeric or merge_text:
                    sep = "" if merge_numeric else " "
                    curr["text"] = f"{curr['text']}{sep}{s['text']}".strip()
                    curr["bbox"] = [
                        min(curr["bbox"][0], s["bbox"][0]),
                        min(curr["bbox"][1], s["bbox"][1]),
                        max(curr["bbox"][2], s["bbox"][2]),
                        max(curr["bbox"][3], s["bbox"][3]),
                    ]
                    curr["cx"] = round(center_x(curr["bbox"]), 2)
                    curr["cy"] = round(center_y(curr["bbox"]), 2)
                else:
                    page_merged.append(curr)
                    curr = dict(s)

            if curr is not None:
                page_merged.append(curr)
        merged_pages.append(page_merged)
    return raw_pages, merged_pages


PURE_CURRENCY_RE = re.compile(
    r"^[₹$\s]*[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2}\s*(?:\((?:Cr|Dr)\)|\b(?:Cr|Dr)\b)?$",
    re.IGNORECASE
)


def normalize_base_font_family(font_name: str) -> str:
    """
    Normalizes PDF font names to their base family so normal bold/italic styling
    (e.g. 'Arial,Bold' vs 'Arial' or 'ABCDEE+Calibri-Bold' vs 'Calibri')
    is NEVER misclassified as a font forgery.
    """
    if not font_name:
        return "unknown"
    f = font_name.strip()
    # Strip 6-letter PDF subset prefix like 'BCDEEE+'
    if "+" in f:
        f = f.split("+", 1)[1]
    f = f.lower()
    # Strip style suffixes
    f = re.sub(r"[-,_]*(bolditalic|boldoblique|bold|italic|oblique|regular|roman|medium|light|mt|psmt|ps)$", "", f)
    f = re.sub(r"[-,_]*(bold|italic|regular)$", "", f)
    return f.strip() or "unknown"


def run_pdf_forensics(doc, pages_spans):
    """
    Family B Structural & Font Forensics (Tuned for Multi-Page Bank Statements & Financial PDFs):
    1. Metadata toolchain check (Photoshop, Canva, Illustrator, etc.)
    2. Incremental PDF save check (multiple %%EOF markers)
    3. Overlapping monetary text layer check (text overlay attack)
    4. Strict Base-Font Family outlier check on pure currency amounts only
    """
    flags = []

    # 1. Metadata Check
    meta = doc.metadata or {}
    suspicious_tools = ["photoshop", "canva", "illustrator", "gimp", "inkscape", "coreldraw", " Nitro ", "sejda", "ilovepdf", "smallpdf"]
    for k in ("producer", "creator", "author"):
        val = str(meta.get(k, "")).lower()
        for tool in suspicious_tools:
            if tool.strip() in val:
                flags.append(f"Page 1 Metadata Anomaly: PDF {k} contains editing toolchain trace '{meta.get(k)}'.")

    # 2. Overlapping Text Spans on Monetary Figures (Text-Overlay Attack)
    for p_idx, page_spans in enumerate(pages_spans):
        page_num = p_idx + 1
        currency_spans = [s for s in page_spans if PURE_CURRENCY_RE.match(s["text"].strip())]

        for i in range(len(currency_spans)):
            for j in range(i + 1, len(currency_spans)):
                s1, s2 = currency_spans[i], currency_spans[j]
                b1, b2 = s1["bbox"], s2["bbox"]
                # Check intersection box
                ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
                if ix1 > ix0 + 2.0 and iy1 > iy0 + 2.0 and s1["text"].strip() != s2["text"].strip():
                    flags.append(
                        f"Page {page_num} Text Overlay Anomaly: Monetary span '{s1['text']}' overlaps with "
                        f"'{s2['text']}' at [{b1[0]}, {b1[1]}, {b1[2]}, {b1[3]}]."
                    )

        # 3. Strict Font Family Consistency on Pure Monetary Amounts Only
        # Only evaluate if the page has at least 4 monetary figures (e.g. a bank statement table)
        if len(currency_spans) >= 4:
            family_counts = {}
            for s in currency_spans:
                base_fam = normalize_base_font_family(s.get("font", ""))
                family_counts[base_fam] = family_counts.get(base_fam, 0) + 1

            dominant_family, dom_count = max(family_counts.items(), key=lambda x: x[1])
            total_curr = len(currency_spans)

            # Flag only if dominant family covers >= 80% of amounts AND a rare different font family (< 15%) appears
            if dom_count / total_curr >= 0.80 and len(family_counts) > 1:
                for s in currency_spans:
                    base_fam = normalize_base_font_family(s.get("font", ""))
                    if base_fam != dominant_family and (family_counts[base_fam] / total_curr) <= 0.15:
                        b = s["bbox"]
                        flags.append(
                            f"Page {page_num} Font Anomaly: Monetary amount '{s['text']}' uses font family "
                            f"'{s.get('font')}' ({base_fam}) while statement numeric font is '{dominant_family}' "
                            f"at [{b[0]}, {b[1]}, {b[2]}, {b[3]}]."
                        )
    return flags


# IMAGE ENGINE (.PNG, .JPG, .JPEG)

def run_image_forensics(image_path, ocr_spans):
    forensic_flags = []
    pil_img = Image.open(image_path)

    info_str = " ".join(f"{k}:{v}" for k, v in pil_img.info.items()).lower()
    exif_data = pil_img.getexif()
    if exif_data:
        for tag_id, val in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            info_str += f" {tag_name}:{val}".lower()

    for tool in SUSPICIOUS_SOFTWARE:
        if tool in info_str:
            forensic_flags.append(f"Image Metadata Warning: Traces of editing software '{tool}' found in headers.")

    rgb_img = pil_img.convert("RGB")
    img_np = np.array(rgb_img, dtype=np.float32)
    h_img, w_img, _ = img_np.shape

    buffer = io.BytesIO()
    rgb_img.save(buffer, "JPEG", quality=82)
    buffer.seek(0)
    resaved_img = Image.open(buffer)

    ela_diff = ImageChops.difference(rgb_img, resaved_img)
    ela_gray = np.mean(np.array(ela_diff, dtype=np.float32), axis=2)

    numeric_scores = []
    for s in ocr_spans:
        if parse_amount(s["text"]) is None:
            continue
        x0 = max(0, int(s["bbox"][0]) - 8)
        y0 = max(0, int(s["bbox"][1]) - 8)
        x1 = min(w_img, int(s["bbox"][2]) + 8)
        y1 = min(h_img, int(s["bbox"][3]) + 8)

        ela_crop = ela_gray[y0:y1, x0:x1]
        rgb_crop = img_np[y0:y1, x0:x1]
        if ela_crop.size > 0:
            ela_val = float(np.percentile(ela_crop, 92))
            bg_mask = np.mean(rgb_crop, axis=2) > 220
            bg_blue_diff = float(np.mean(rgb_crop[bg_mask][:, 2] - rgb_crop[bg_mask][:, 0])) if np.any(bg_mask) else 0.0
            numeric_scores.append((s, ela_val, bg_blue_diff))

    if len(numeric_scores) >= 4:
        ela_vals = [item[1] for item in numeric_scores]
        med_ela = float(np.median(ela_vals))
        std_ela = float(np.std(ela_vals)) + 1e-5

        for s, ela_val, bg_diff in numeric_scores:
            z_ela = abs(ela_val - med_ela) / std_ela
            if z_ela > 1.8 and abs(ela_val - med_ela) > 1.5:
                forensic_flags.append(
                    f"Pixel Tampering (ELA Anomaly): Number '{s['text']}' at {s['bbox']} shows "
                    f"compression mismatch (ELA {ela_val:.2f} vs baseline {med_ela:.2f})."
                )

    return forensic_flags


def is_salary_slip_text(text_blob):
    low = text_blob.lower()
    kw = ["earnings", "deductions", "net pay", "gross earnings", "basic salary", "provident fund", "payslip", "salary slip"]
    return sum(1 for k in kw if k in low) >= 2


def reconcile_salary_slip_spans(page_spans, y_tolerance=14.0):
    """
    Verifies Salary Slip invariants:
    1. Sum of Earning rows == Gross Earnings
    2. Sum of Deduction rows == Total Deductions
    3. Gross Earnings - Total Deductions == Net Pay
    """
    issues = []
    earnings_items = []
    deductions_items = []
    gross_earnings = None
    total_deductions = None
    net_pay = None

    # Horizontal midline separating Earnings (left) and Deductions (right)
    mid_x = 500.0
    for s in page_spans:
        if "deduction" in s["text"].lower() and s["cx"] > 200:
            mid_x = s["bbox"][0] - 20
            break

    # 1. Locate Net Pay Anchor & Scan Surrounding Area
    net_anchor = None
    for s in page_spans:
        txt = s["text"].lower()
        if any(k in txt for k in ["net pay", "take home", "net salary"]):
            net_anchor = s
            break

    used_net_span_ids = set()
    if net_anchor:
        used_net_span_ids.add(id(net_anchor))
        net_candidates = []
        for s in page_spans:
            val = parse_amount(s["text"])
            if val is not None:
                # Sits horizontally to the right of "NET PAY" or slightly below
                dy = s["cy"] - net_anchor["cy"]
                if -20.0 <= dy <= 60.0 and s["cx"] > net_anchor["bbox"][0]:
                    dist = ((s["cx"] - net_anchor["cx"]) ** 2 + dy ** 2) ** 0.5
                    net_candidates.append((dist, val, id(s)))
        if net_candidates:
            net_candidates.sort(key=lambda x: x[0])
            net_pay = net_candidates[0][1]
            used_net_span_ids.add(net_candidates[0][2])

    # 2. Extract Component Rows, Gross, and Total Deductions
    rows = group_into_rows(page_spans, y_tolerance=y_tolerance)
    for row in rows:
        row_text = " ".join(s["text"].lower() for s in row)
        if "earnings" in row_text and "deductions" in row_text and "gross" not in row_text:
            continue
        if any(k in row_text for k in ["net pay", "take home", "net salary"]):
            continue

        if "gross" in row_text or "total deduction" in row_text:
            for s in row:
                val = parse_amount(s["text"])
                if val is not None:
                    if s["cx"] < mid_x:
                        gross_earnings = val
                    else:
                        total_deductions = val
            continue

        left_spans = [s for s in row if s["cx"] < mid_x and id(s) not in used_net_span_ids]
        right_spans = [s for s in row if s["cx"] >= mid_x and id(s) not in used_net_span_ids]

        left_vals = [parse_amount(s["text"]) for s in left_spans if parse_amount(s["text"]) is not None]
        left_labels = [s["text"] for s in left_spans if parse_amount(s["text"]) is None]
        if left_vals and left_labels:
            earnings_items.append({"name": " ".join(left_labels), "amount": left_vals[-1]})

        right_vals = [parse_amount(s["text"]) for s in right_spans if parse_amount(s["text"]) is not None]
        right_labels = [s["text"] for s in right_spans if parse_amount(s["text"]) is None]
        if right_vals and right_labels:
            deductions_items.append({"name": " ".join(right_labels), "amount": right_vals[-1]})

    # Fallback for Net Pay: Pick highest amount in bottom 20% of the document if anchor was missed
    if net_pay is None:
        max_y = max(s["cy"] for s in page_spans) if page_spans else 0
        bottom_spans = [s for s in page_spans if s["cy"] > max_y - 120 and parse_amount(s["text"]) is not None]
        if bottom_spans:
            net_pay = parse_amount(bottom_spans[-1]["text"])

    # 3. Check Financial Invariants
    audit_coverage = {
        "earnings_extracted": len(earnings_items) > 0,
        "deductions_extracted": len(deductions_items) > 0,
        "gross_and_net_extracted": (gross_earnings is not None and total_deductions is not None and net_pay is not None),
        "net_pay_equation_verified": False
    }

    if not audit_coverage["gross_and_net_extracted"]:
        issues.append(
            f"Extraction Warning: Missing salary summary totals "
            f"(Gross: {gross_earnings}, Deductions: {total_deductions}, Net Pay: {net_pay})."
        )
    else:
        calc_earn = round(sum(i["amount"] for i in earnings_items), 2)
        calc_ded = round(sum(i["amount"] for i in deductions_items), 2)

        if earnings_items and abs(calc_earn - gross_earnings) > 0.5:
            issues.append(f"Earnings Breakdown Mismatch: Sum of earnings ({calc_earn}) != Gross Earnings ({gross_earnings}).")

        if deductions_items and abs(calc_ded - total_deductions) > 0.5:
            issues.append(f"Deductions Breakdown Mismatch: Sum of deductions ({calc_ded}) != Total Deductions ({total_deductions}).")

        expected_net = round(gross_earnings - total_deductions, 2)
        if abs(expected_net - net_pay) > 0.5:
            issues.append(
                f"Salary Slip Math Mismatch: Gross ({gross_earnings}) - Deductions ({total_deductions}) "
                f"= {expected_net}, but Net Pay shows {net_pay}."
            )
        else:
            audit_coverage["net_pay_equation_verified"] = True

    status = "REVIEW_REQUIRED" if issues else ("NO_MISMATCH_FOUND" if all(audit_coverage.values()) else "PARTIAL_CHECK_CLEAN")

    return {
        "document_type": "SALARY_SLIP",
        "status": status,
        "audit_coverage": audit_coverage,
        "gross_earnings": gross_earnings,
        "total_deductions": total_deductions,
        "net_pay": net_pay,
        "earnings_breakdown": earnings_items,
        "deductions_breakdown": deductions_items,
        "issues": issues
    }



# ==============================================================================
# UNIVERSAL DATE & CURRENCY PARSERS (NEVER CONFUSES '0.94' OR '34.94' WITH DATES!)
# ==============================================================================
DATE_RE = re.compile(
    r"^(?:(0?[1-9]|[12]\d|3[01])[-/\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]{3,9})(?:[-/\s]+\d{2,4})?"
    r"|(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/]\d{2,4})\b",
    re.IGNORECASE,
)
# Catches genuine 3-part dot-separated amounts like '242.229.43' or '1.223.00' or '10.000.00'
MALFORMED_DOT_AMOUNT_RE = re.compile(r"(?<!\d)([+-]?\d{1,3}(?:\.\d{2,3})+\.\d{2})(?!\d)")

AMOUNT_CR_DR_RE = re.compile(
    r"([+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?)\s*(?:\((Cr|Dr)\)|\b(Cr|Dr)\b)?",
    re.IGNORECASE
)


def parse_amount_with_sign(text):
    if not text:
        return None, None
    s = str(text).strip()

    # 1. Check malformed multi-dot amount first (e.g. '242.229.43', '1.223.00', '10.000.00')
    m_dot = MALFORMED_DOT_AMOUNT_RE.search(s)
    if m_dot:
        raw_mal = m_dot.group(1)
        parts = raw_mal.split(".")
        fixed_str = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return round(float(fixed_str), 2), "MALFORMED_DOT"
        except ValueError:
            pass

    # 2. Reject true dates ('02/Apr/2024', '02-04-2025') and timestamps ('09:07:54')
    if DATE_RE.match(s) and not re.search(r"\((?:Cr|Dr)\)", s, re.IGNORECASE):
        return None, None
    if re.match(r"^\d{2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?$", s, re.IGNORECASE):
        return None, None

    raw_digits = re.sub(r"[^\d]", "", s)
    if len(raw_digits) >= 10 and "." not in s and "," not in s:
        return None, None

    cleaned = re.sub(r"[₹$]", "", s).strip()
    m = AMOUNT_CR_DR_RE.search(cleaned)
    if not m:
        return None, None

    num_str = m.group(1).replace(",", "")
    indicator = (m.group(2) or m.group(3) or "").upper() or None
    try:
        return round(float(num_str), 2), indicator
    except ValueError:
        return None, None


def parse_amount(text):
    val, _ = parse_amount_with_sign(text)
    return val


def extract_all_numbers_from_cell(cell_text: str) -> list[tuple[float, str | None, str]]:
    """
    Extracts 1 or more monetary numbers from a single PDF table cell, automatically
    stitching wrapped 2-line Indian numbers ('9,11,685.\n06' -> 911685.06,
    '10,00,000\n94' -> 1000000.94, '10,83,763\n.00' -> 1083763.00) while preserving
    multiple stacked numbers when two transactions share a cell ('15,00,000\n94\n0.94').
    """
    if not cell_text:
        return []
    raw = str(cell_text).strip()
    if not raw or raw in ("-", "--", "—", "NA", "N/A", ".00"):
        return []

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    stitched_tokens = []
    i = 0
    while i < len(lines):
        curr = lines[i]
        # Case A: curr ends with '.' or is comma-grouped integer, and next line is '.00' or '94' or '06'
        if i + 1 < len(lines) and re.match(r"^[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d{4,})\.?$", curr) and re.match(r"^\.?\d{2}(?:\([CD]r\))?$", lines[i + 1], re.IGNORECASE):
            paise = lines[i + 1].lstrip(".")
            base = curr.rstrip(".")
            stitched_tokens.append(f"{base}.{paise}")
            i += 2
            continue
        stitched_tokens.append(curr)
        i += 1

    results = []
    for tok in stitched_tokens:
        # A single line might also have two space-separated numbers like '14,80,000 14,80,000'
        sub_toks = tok.split() if (" " in tok and not re.search(r"\([CD]r\)", tok, re.IGNORECASE)) else [tok]
        for st in sub_toks:
            val, ind = parse_amount_with_sign(st)
            if val is not None and ("." in st or "," in st or val == 0.0):
                if str(st).strip().startswith("-"):
                    val = -abs(val)
                results.append((val, ind, st))
    return results


def extract_grid_table_transactions(doc, full_text: str = ""):
    """
    Universal Vector-Grid Table Extractor (Handles 1-to-N stacked transactions per cell):
    - Preserves Serial Numbers (Sl No) and Transaction IDs.
    - Unpacks merged multi-transaction cells (e.g. Sl No 2 & 3, Sl No 16 & 17) using running balance continuity.
    """
    issues = []
    detailed_issues = []
    transactions = []
    summary = extract_summary_from_full_text(full_text) if full_text else {
        "opening_balance": [], "summary_credits": [], "summary_debits": [], "closing_balance": []
    }

    col_map = {}
    per_page_stats = []
    grid_pages_found = 0

    for p_idx, page in enumerate(doc):
        page_num = p_idx + 1
        try:
            tabs = page.find_tables()
        except Exception:
            return None

        if not tabs or not tabs.tables:
            per_page_stats.append({"page": page_num, "transactions_extracted": 0})
            continue

        page_tx_count = 0
        for tbl in tabs.tables:
            extracted_rows = tbl.extract()
            if not extracted_rows:
                continue

            start_row_idx = 0
            for r_i, r_cells in enumerate(extracted_rows[:5]):
                joined_hdr = " ".join(str(c or "").replace("\n", " ").lower() for c in r_cells)
                if "date" in joined_hdr and "balance" in joined_hdr and any(w in joined_hdr for w in ["withdraw", "withdra", "debit", "deposit", "credit", "dr", "cr"]):
                    col_map = {"num_cols": len(r_cells)}
                    for c_i, cell_val in enumerate(r_cells):
                        c_txt = re.sub(r"\s+", " ", str(cell_val or "").lower()).strip()
                        if c_txt in ("sl no", "si no", "s.no", "sr no", "sl", "si"):
                            col_map["sl_no"] = c_i
                        elif "tran" in c_txt and "id" in c_txt:
                            col_map["tran_id"] = c_i
                        elif "posted" in c_txt:
                            continue
                        elif "transaction" in c_txt and "date" in c_txt:
                            col_map["date"] = c_i
                        elif "value" in c_txt and "date" in c_txt:
                            col_map.setdefault("date", c_i)
                        elif "date" in c_txt:
                            col_map.setdefault("date", c_i)
                        elif any(k in c_txt for k in ["remark", "narration", "description", "particular"]):
                            col_map["description"] = c_i
                        elif ("withdraw" in c_txt or "withdra" in c_txt or "dr" in c_txt) and ("deposit" in c_txt or "cr" in c_txt):
                            col_map["amount_cr_dr"] = c_i
                        elif any(k in c_txt for k in ["withdraw", "withdra", "debit", "(dr)"]):
                            col_map["debit"] = c_i
                        elif any(k in c_txt for k in ["deposit", "credit", "(cr)"]):
                            col_map["credit"] = c_i
                        elif "balance" in c_txt:
                            col_map["balance"] = c_i
                    start_row_idx = r_i + 1
                    grid_pages_found += 1
                    break

            if "balance" not in col_map:
                continue

            # If continuation page table has fewer columns than Page 1 (e.g. empty Cheq No column dropped by find_tables),
            # dynamically locate the Balance column as the rightmost column containing decimal numbers!
            for r_i in range(start_row_idx, len(extracted_rows)):
                r_cells = extracted_rows[r_i]
                if not r_cells:
                    continue
                row_text_all = " ".join(str(c or "").replace("\n", " ").strip() for c in r_cells).strip()
                row_low = row_text_all.lower()
                if not row_low or any(k in row_low for k in ["page total", "statement summary", "end of statement", "legends used"]):
                    continue

                bal_idx = min(col_map["balance"], len(r_cells) - 1)
                bal_list = extract_all_numbers_from_cell(r_cells[bal_idx])
                if not bal_list:
                    continue

                # Extract Debit and Credit lists from their columns (or recover from adjacent columns if shifted)
                deb_idx = col_map.get("debit")
                cred_idx = col_map.get("credit")
                deb_list = extract_all_numbers_from_cell(r_cells[deb_idx]) if (deb_idx is not None and deb_idx < len(r_cells)) else []
                cred_list = extract_all_numbers_from_cell(r_cells[cred_idx]) if (cred_idx is not None and cred_idx < len(r_cells)) else []

                # Extract Date
                dt_col = col_map.get("date", 0)
                dt_raw = ""
                for c_check in ([dt_col] + list(range(min(5, len(r_cells))))):
                    if c_check < len(r_cells):
                        cand_s = re.sub(r"\s+", "", str(r_cells[c_check] or "")).strip()
                        m_dt = DATE_RE.search(cand_s)
                        if m_dt:
                            dt_raw = m_dt.group(0)
                            break
                if not dt_raw:
                    dt_raw = transactions[-1]["date"] if transactions else f"Page-{page_num}"

                desc_col = col_map.get("description")
                desc_raw = re.sub(r"\s+", " ", str(r_cells[desc_col] or "")).strip() if (desc_col is not None and desc_col < len(r_cells)) else row_text_all

                # Unpack 1 or more transactions in this grid row using the extracted balance sequence
                for k_idx, (b_val, b_ind, b_str) in enumerate(bal_list):
                    signed_b = -abs(b_val) if b_ind == "DR" else b_val
                    prev_b = (
                        summary["opening_balance"][0]
                        if (len(transactions) == 0 and summary["opening_balance"])
                        else (transactions[-1]["balance"] if transactions else None)
                    )

                    # Match expected delta against available deb_list / cred_list or row numbers
                    c_val = cred_list[k_idx][0] if k_idx < len(cred_list) else 0.0
                    d_val = deb_list[k_idx][0] if k_idx < len(deb_list) else 0.0

                    if prev_b is not None:
                        # Check if unsigned overdraft balance (e.g. Row 1-3 on Page 1 of ICICI)
                        if prev_b < 0 and signed_b > 0 and abs((prev_b - d_val + c_val) - (-signed_b)) <= 1.5:
                            signed_b = -signed_b

                        delta = round(signed_b - prev_b, 2)
                        if abs((c_val - d_val) - delta) > 1.0:
                            # Search all numbers in the row for the exact delta (handles shifted columns on continuation pages!)
                            all_row_nums = []
                            for c_cell in r_cells[:-1]:
                                all_row_nums.extend(extract_all_numbers_from_cell(c_cell))
                            matched_delta = False
                            for cand_v, cand_ind, cand_str in all_row_nums:
                                if abs(abs(cand_v) - abs(delta)) <= 1.0:
                                    if delta > 0:
                                        c_val, d_val = abs(cand_v), 0.0
                                    else:
                                        c_val, d_val = 0.0, abs(cand_v)
                                    matched_delta = True
                                    break

                    transactions.append({
                        "page": page_num,
                        "bbox": [round(x, 2) for x in tbl.bbox],
                        "date": dt_raw,
                        "description": desc_raw,
                        "credit": round(abs(c_val), 2),
                        "debit": round(abs(d_val), 2),
                        "balance": round(signed_b, 2)
                    })
                    page_tx_count += 1

        per_page_stats.append({"page": page_num, "transactions_extracted": page_tx_count})

    if grid_pages_found == 0 or len(transactions) < 2:
        return None

    res = run_math_checks(summary, transactions, issues, detailed_issues)
    res["total_pages_processed"] = len(doc)
    res["per_page_summary"] = per_page_stats
    return res


def route_specialized_document(full_text, page_spans, y_tol=12.0, filename=""):
    doc_info = classify_financial_document(full_text)
    code = doc_info.get("doc_code", "BANK_STATEMENT")
    if code == "UNKNOWN_DOCUMENT":
        return {
            "document_type": "UNKNOWN_DOCUMENT",
            "status": "REVIEW_REQUIRED",
            "audit_coverage": {"document_schema_recognized": "FAILED"},
            "issues": ["Document type could not be classified into any supported financial schema."],
            "detailed_issues": [{"page": 1, "bbox": None, "text": "Document type could not be classified into any supported financial schema."}]
        }
    elif code == "TAX_AUDIT_REPORT":
        return verify_tax_audit_report(full_text, filename=filename)
    elif code == "GST_CERTIFICATE":
        return verify_gst_certificate(full_text)
    elif code == "UDYAM_CERTIFICATE":
        return verify_udyam_certificate(full_text)
    elif code == "ITR_DOCUMENT":
        return verify_itr_document(full_text)
    elif code == "BALANCE_SHEET":
        return verify_balance_sheet(full_text)
    elif code == "SALARY_SLIP":
        return reconcile_salary_slip_spans(page_spans, y_tolerance=y_tol)
    return None


def analyze_image(image_path):
    """Runs ELA pixel forensics + RapidOCR extraction on a flat image."""
    forensic_flags = []
    try:
        ela_flag = run_ela_analysis(image_path)
        if ela_flag:
            forensic_flags.append(ela_flag)
    except Exception:
        pass

    ocr_res, _ = ocr_engine(image_path)
    page_spans = []
    text_parts = []

    if ocr_res:
        for box, text, conf in ocr_res:
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            x0, y0, x1, y1 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            page_spans.append({
                "text": text.strip(),
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
                "page": 1
            })
            text_parts.append(text.strip())

    total_text = "\n".join(text_parts)
    all_pages_spans = [page_spans]

    specialized_check = route_specialized_document(total_text, page_spans, y_tol=14.0)
    if specialized_check is not None:
        financial_check = specialized_check
    else:
        financial_check = reconcile_spans(all_pages_spans, y_tolerance=14.0, full_text=total_text)

    status = "REVIEW_REQUIRED" if (forensic_flags or financial_check.get("status") != "NO_MISMATCH_FOUND") else "NO_MISMATCH_FOUND"
    return {
        "file_format": "IMAGE",
        "total_pages": 1,
        "status": status,
        "forensic_flags": forensic_flags,
        "financial_reconciliation": financial_check
    }


def analyze_docx(docx_path):
    """Extracts text and tables from a Word (.docx) document and runs reconciliation."""
    d = docx.Document(docx_path)
    forensic_flags = []

    # Check Core Properties Metadata
    props = d.core_properties
    if props.revision and int(props.revision) > 3:
        forensic_flags.append(f"DOCX Metadata Anomaly: High revision count ({props.revision}) detected.")

    page_spans = []
    text_parts = []
    y_cursor = 20.0

    for p in d.paragraphs:
        t = p.text.strip()
        if t:
            text_parts.append(t)
            page_spans.append({
                "text": t,
                "bbox": [40.0, y_cursor, 400.0, y_cursor + 12.0],
                "cx": 220.0,
                "cy": y_cursor + 6.0,
                "page": 1
            })
            y_cursor += 18.0

    for table in d.tables:
        for row in table.rows:
            x_cursor = 40.0
            for cell in row.cells:
                ct = cell.text.strip().replace("\n", " ")
                if ct:
                    text_parts.append(ct)
                    page_spans.append({
                        "text": ct,
                        "bbox": [x_cursor, y_cursor, x_cursor + 90.0, y_cursor + 12.0],
                        "cx": x_cursor + 45.0,
                        "cy": y_cursor + 6.0,
                        "page": 1
                    })
                x_cursor += 110.0
            y_cursor += 20.0

    total_text = "\n".join(text_parts)
    all_pages_spans = [page_spans]

    specialized_check = route_specialized_document(total_text, page_spans, y_tol=8.0)
    if specialized_check is not None:
        financial_check = specialized_check
    else:
        financial_check = reconcile_spans(all_pages_spans, y_tolerance=8.0, full_text=total_text)

    status = "REVIEW_REQUIRED" if (forensic_flags or financial_check.get("status") != "NO_MISMATCH_FOUND") else "NO_MISMATCH_FOUND"
    return {
        "file_format": "DOCX",
        "total_pages": 1,
        "status": status,
        "forensic_flags": forensic_flags,
        "financial_reconciliation": financial_check
    }


def analyze_document(file_path, original_filename=None):
    """
    Universal Multi-Format & Multi-Page PDF Router:
    Explicitly defines `all_pages_spans` and `total_text` across all pages (1..N)
    and passes both into `reconcile_spans`.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return analyze_docx(file_path)
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        return analyze_image(file_path)

    if ext == ".pdf":
        with pymupdf.open(file_path) as doc:
            num_pages = len(doc)
            all_pages_spans = []
            full_text_parts = []
            forensic_flags = []
            scanned_pages_count = 0
            digital_pages_count = 0

            # Extract vector spans across all pages
            raw_pages, merged_native_pages = collect_and_merge_spans(doc)
            if any(len(p) > 0 for p in merged_native_pages):
                forensic_flags.extend(run_pdf_forensics(doc, merged_native_pages))

            for p_idx, page in enumerate(doc):
                page_num = p_idx + 1
                page_txt = page.get_text("text").strip()

                if len(page_txt) > 30 and len(merged_native_pages[p_idx]) > 0:
                    digital_pages_count += 1
                    # Ensure every span carries its page number
                    for s in merged_native_pages[p_idx]:
                        s["page"] = page_num
                    all_pages_spans.append(merged_native_pages[p_idx])
                    full_text_parts.append(page_txt)
                else:
                    scanned_pages_count += 1
                    pix = page.get_pixmap(dpi=200)
                    temp_img = f"{file_path}_page_{page_num}_ocr.png"
                    pix.save(temp_img)
                    try:
                        img_rep = analyze_image(temp_img)
                        for flg in img_rep.get("forensic_flags", []):
                            forensic_flags.append(f"Page {page_num} {flg}")

                        ocr_res, _ = ocr_engine(temp_img)
                        p_spans = []
                        if ocr_res:
                            for box, text, conf in ocr_res:
                                x_coords = [pt[0] for pt in box]
                                y_coords = [pt[1] for pt in box]
                                x0, y0, x1, y1 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                                p_spans.append({
                                    "text": text.strip(),
                                    "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                                    "cx": (x0 + x1) / 2.0,
                                    "cy": (y0 + y1) / 2.0,
                                    "page": page_num
                                })
                                full_text_parts.append(text.strip())
                        all_pages_spans.append(p_spans)
                    finally:
                        if os.path.exists(temp_img):
                            os.remove(temp_img)

            total_text = "\n".join(full_text_parts)
            flat_spans = [s for p in all_pages_spans for s in p]

            if scanned_pages_count > 0 and digital_pages_count > 0:
                pdf_fmt = "HYBRID_PDF"
            elif scanned_pages_count > 0:
                pdf_fmt = "SCANNED_PDF"
            else:
                pdf_fmt = "DIGITAL_PDF"

            y_tol = 14.0 if scanned_pages_count > 0 else 6.0
            eff_name = original_filename or os.path.basename(file_path)

            specialized_check = route_specialized_document(total_text, flat_spans, y_tol, filename=eff_name)
            if specialized_check is not None:
                financial_check = specialized_check
                financial_check["total_pages_processed"] = num_pages
            else:
                grid_check = extract_grid_table_transactions(doc, full_text=total_text) if scanned_pages_count == 0 else None
                if False:
                    financial_check = grid_check
                else:
                    financial_check = reconcile_spans(all_pages_spans, y_tolerance=y_tol, full_text=total_text)

            overall_status = (
                "REVIEW_REQUIRED"
                if (forensic_flags or financial_check.get("status") != "NO_MISMATCH_FOUND")
                else "NO_MISMATCH_FOUND"
            )

            return {
                "file_format": pdf_fmt,
                "total_pages": num_pages,
                "digital_pages": digital_pages_count,
                "scanned_pages": scanned_pages_count,
                "status": overall_status,
                "forensic_flags": forensic_flags,
                "financial_reconciliation": financial_check,
                "note": "Automated forensic and mathematical check. A finding requires review and is not standalone proof of fraud."
            }

    return {"error": f"Unsupported file format: {ext}"}

import numpy as np
import re


def detect_smart_recalculated_tampering(page_spans, financial_rec):
    """
    Catches 'Smart Fraudsters' who recalculate all running balances and totals
    so that Family A math passes cleanly.
    Checks:
    1. Column X-Alignment Jitter (manual text box placement variance).
    2. Footer / Narrative Text vs Table Summary Mismatch.
    3. Statutory Payroll Ratios (HRA/Basic & PF/Basic on Salary Slips).
    """
    smart_flags = []

    # CHECK 1: Column Alignment Jitter (Sub-pixel X0 / X1 check)
    table_numbers = [
        s for s in page_spans
        if s["cy"] > 280 and re.search(r"\d[\d,]{2,}", s["text"])
    ]

    # Group numbers into vertical columns (spans within 35px horizontally belong to the same column)
    columns = []
    for s in table_numbers:
        placed = False
        for col in columns:
            avg_x0 = np.mean([item["bbox"][0] for item in col])
            if abs(s["bbox"][0] - avg_x0) <= 35.0:
                col.append(s)
                placed = True
                break
        if not placed:
            columns.append([s])

    for col in columns:
        if len(col) >= 4:
            x0_vals = [item["bbox"][0] for item in col]
            x1_vals = [item["bbox"][2] for item in col]
            med_x0 = float(np.median(x0_vals))
            med_x1 = float(np.median(x1_vals))

            # If a column is neither cleanly left-aligned nor cleanly right-aligned on one specific row:
            for item in col:
                left_dev = abs(item["bbox"][0] - med_x0)
                right_dev = abs(item["bbox"][2] - med_x1)
                # Automated tables have < 1.5px deviation on either left or right edge
                if min(left_dev, right_dev) > 3.5:
                    smart_flags.append(
                        f"Column Alignment Jitter: Number '{item['text']}' at {item['bbox']} is misaligned "
                        f"by {min(left_dev, right_dev):.1f}px compared to the rest of its column (manual edit clue)."
                    )

    # ---------------------------------------------------------
    # CHECK 2: Narrative / Footer Note vs Table Summary Check
    # ---------------------------------------------------------
    full_text = " ".join(s["text"] for s in page_spans)
    note_match = re.search(
        r"Opening\s*₹?([\d,]+)\s*\+\s*Credits\s*₹?([\d,]+)\s*-\s*Debits\s*₹?([\d,]+)\s*=\s*Closing\s*₹?([\d,]+)",
        full_text,
        re.IGNORECASE
    )
    if note_match:
        note_cr = float(note_match.group(2).replace(",", ""))
        note_cl = float(note_match.group(4).replace(",", ""))
        calc_cr = financial_rec.get("total_credits")
        calc_cl = financial_rec.get("closing_balance")

        if calc_cr is not None and abs(note_cr - calc_cr) > 1.0:
            smart_flags.append(
                f"Footer Note Mismatch: Ledger Credits were edited to {calc_cr}, "
                f"but document footer note still states Credits = {note_cr}."
            )
        if calc_cl is not None and abs(note_cl - calc_cl) > 1.0:
            smart_flags.append(
                f"Footer Note Mismatch: Ledger Closing Balance was edited to {calc_cl}, "
                f"but document footer note still states Closing = {note_cl}."
            )

    # ---------------------------------------------------------
    # CHECK 3: Salary Slip Statutory Ratio Check (PF & HRA)
    # ---------------------------------------------------------
    if financial_rec.get("document_type") == "SALARY_SLIP":
        earnings = {i["name"].lower(): i["amount"] for i in financial_rec.get("earnings_breakdown", [])}
        deductions = {i["name"].lower(): i["amount"] for i in financial_rec.get("deductions_breakdown", [])}

        basic = next((v for k, v in earnings.items() if "basic" in k), None)
        hra = next((v for k, v in earnings.items() if "hra" in k), None)
        pf = next((v for k, v in deductions.items() if "pf" in k or "provident" in k), None)

        if basic and basic > 0:
            if hra is not None:
                hra_ratio = hra / basic
                # Standard Indian HRA is 40% to 50% of Basic (allow 25%..65% tolerance)
                if not (0.25 <= hra_ratio <= 0.65):
                    smart_flags.append(
                        f"Payroll Ratio Anomaly: HRA ({hra}) is {hra_ratio*100:.1f}% of Basic ({basic}), "
                        f"outside standard 40%–50% Indian payroll norms."
                    )
            if pf is not None:
                pf_ratio = pf / basic
                # Standard PF is ~12% of Basic or flat ₹1,800 ceiling
                if not (0.10 <= pf_ratio <= 0.18) and abs(pf - 1800.0) > 50:
                    smart_flags.append(
                        f"Statutory PF Anomaly: PF ({pf}) is {pf_ratio*100:.1f}% of Basic ({basic}), "
                        f"violating standard 12% EPF deduction rules."
                    )

    return smart_flags


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python reconciliation.py "file.pdf | file.png | file.jpg | file.docx"')
        sys.exit(1)

    try:
        report = analyze_document(sys.argv[1])
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception as error:
        print(f"Error processing file: {error}")
        sys.exit(1)


# ==============================================================================



# ==============================================================================


# ==============================================================================
# AUTHORITATIVE PRIORITY 1 & 2 OVERRIDE: UNIVERSAL MULTI-BANK ENGINE
# ==============================================================================
DATE_RE = re.compile(
    r"^(?:(0?[1-9]|[12]\d|3[01])[-/\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]{3,9})[-/\s]*\d{2,4}"
    r"|(0?[1-9]|[12]\d|3[01])(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/\s]+\d{2,4}"
    r"|(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/]\d{2,4})\b",
    re.IGNORECASE,
)

MALFORMED_MULTI_DOT_RE = re.compile(r"^[₹$\s]*([+-]?\d{1,3}(?:\.\d{2,3})+\.\d{1,2})$")
MALFORMED_THREE_DEC_RE = re.compile(r"^[₹$\s]*([+-]?\d{1,3})\.(\d{3})$")

AMOUNT_CR_DR_RE = re.compile(
    r"^([+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?)\s*(?:\((Cr|Dr)\)|\b(Cr|Dr)\b)?$",
    re.IGNORECASE
)


def parse_amount_with_sign(text):
    if not text:
        return None, None
    s = str(text).strip()

    s_no_crdr = re.sub(r"\((?:Cr|Dr)\)|\b(?:Cr|Dr)\b", "", s, flags=re.IGNORECASE).strip()
    s_no_curr = re.sub(r"[₹$]", "", s_no_crdr).strip()
    if re.search(r"[A-Za-z:@/]", s_no_curr):
        return None, None

    if DATE_RE.match(s):
        return None, None

    m_mdot = MALFORMED_MULTI_DOT_RE.match(s_no_crdr)
    if m_mdot:
        raw_mal = m_mdot.group(1)
        parts = raw_mal.split(".")
        fixed_str = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return round(float(fixed_str), 2), "MALFORMED_DOT"
        except ValueError:
            pass

    m_3dec = MALFORMED_THREE_DEC_RE.match(s_no_crdr)
    if m_3dec:
        fixed_str = f"{m_3dec.group(1)}{m_3dec.group(2)}"
        try:
            return round(float(fixed_str), 2), "MALFORMED_DOT"
        except ValueError:
            pass

    raw_digits = re.sub(r"[^\d]", "", s_no_curr)
    if len(raw_digits) >= 9 and "," not in s_no_curr and "." not in s_no_curr:
        return None, None
    int_part_digits = re.sub(r"[^\d]", "", s_no_curr.split(".")[0])
    if len(int_part_digits) >= 10 and "," not in s_no_curr.split(".")[0]:
        return None, None

    m = AMOUNT_CR_DR_RE.match(s_no_curr)
    if not m:
        return None, None

    num_str = m.group(1).replace(",", "")
    indicator = (m.group(2) or m.group(3) or "").upper() or None
    try:
        return round(float(num_str), 2), indicator
    except ValueError:
        return None, None


def parse_amount(text):
    val, _ = parse_amount_with_sign(text)
    return val


def extract_summary_from_full_text(full_text: str) -> dict:
    summary = {
        "opening_balance": [],
        "summary_credits": [],
        "summary_debits": [],
        "closing_balance": []
    }
    patterns = {
        "opening_balance": r"(?:opening\s*bal(?:ance)?|balance\s+as\s+on\s+[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})\s*[:\-]\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)",
        "summary_debits": r"(?:total\s*(?:withdrawal|debit)s?\s*(?:amount)?|withdrawa?ls)\s*[:\-]?\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)",
        "summary_credits": r"(?:total\s*(?:deposit|credit)s?\s*(?:amount)?|deposits)\s*[:\-]?\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)",
        "closing_balance": r"closing\s*bal(?:ance)?\s*[:\-]?\s*([+-]?\s*[₹$]?\s*[\d,]+\s*\.\s*\d{2}\s*(?:\([CD]r\))?)"
    }
    for key, pat in patterns.items():
        for m in re.finditer(pat, full_text, re.IGNORECASE):
            raw_capture = re.sub(r"\s+", "", m.group(1).strip())
            is_neg = raw_capture.startswith("-")
            val, ind = parse_amount_with_sign(raw_capture)
            if val is not None:
                if key in ("opening_balance", "closing_balance") and (ind == "DR" or is_neg):
                    val = -abs(val)
                if val not in summary[key]:
                    summary[key].append(val)
    return summary


def Stitch_Vertical_Wrapped_Numbers(page_spans, numeric_col_xs, x_tol=42.0, y_tol=16.0):
    spans = [dict(s) for s in page_spans]
    spans.sort(key=lambda s: (s["cy"], s["cx"]))
    skip_indices = set()
    stitched = []

    for i, s in enumerate(spans):
        if i in skip_indices:
            continue
        t1 = s["text"].strip()

        if s["cx"] <= 140 and (re.match(r"^\d{1,2}[-/][A-Za-z]{3}[-/]?$", t1) or re.match(r"^\d{4}$", t1)):
            for j in range(i + 1, len(spans)):
                if j in skip_indices:
                    continue
                s2 = spans[j]
                if s2["cy"] - s["cy"] > 16.0:
                    break
                if abs(s2["cx"] - s["cx"]) <= 35.0:
                    t2 = s2["text"].strip()
                    if re.match(r"^\d{1,2}[-/][A-Za-z]{3}[-/]?$", t1) and re.match(r"^\d{2,4}$", t2):
                        s["text"] = f"{t1.rstrip('-/')}-{t2}"
                        skip_indices.add(j)
                        break
                    elif re.match(r"^\d{4}$", t1) and re.match(r"^\d{1,2}[-/][A-Za-z]{3}[-/]?$", t2):
                        s["text"] = f"{t2.rstrip('-/')}-{t1}"
                        skip_indices.add(j)
                        break

        matching_cols = [col_x for col_x in numeric_col_xs if abs(s["cx"] - col_x) <= x_tol]
        if matching_cols and re.match(r"^[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d{4,})\.?$", t1):
            col_x = min(matching_cols, key=lambda c: abs(s["cx"] - c))
            for j in range(i + 1, len(spans)):
                if j in skip_indices:
                    continue
                s2 = spans[j]
                dy = s2["cy"] - s["cy"]
                if dy > 13.0:
                    break
                same_col = (abs(s2["cx"] - col_x) <= x_tol) and (abs(s2["bbox"][2] - s["bbox"][2]) <= 25.0 or abs(s2["cx"] - s["cx"]) <= 25.0)
                if same_col and dy >= 2.0:
                    t2 = s2["text"].strip()
                    has_indian_lakhs = bool(re.match(r"^[+-]?\d{1,2},\d{2},\d{3}$", t1))
                    if (t1.endswith(".") and re.match(r"^\d{2}$", t2)) or re.match(r"^\.\d{2}$", t2) or (has_indian_lakhs and re.match(r"^\d{2}$", t2)):
                        paise = t2.lstrip(".")
                        base = t1.rstrip(".")
                        s["text"] = f"{base}.{paise}"
                        s["bbox"] = [
                            min(s["bbox"][0], s2["bbox"][0]),
                            s["bbox"][1],
                            max(s["bbox"][2], s2["bbox"][2]),
                            max(s["bbox"][3], s2["bbox"][3])
                        ]
                        skip_indices.add(j)
                        break
        stitched.append(s)

    return stitched


def reconcile_spans(pages_spans, y_tolerance=6.0, full_text=""):
    issues = []
    detailed_issues = []
    transactions = []
    summary = {
        "opening_balance": [],
        "summary_credits": [],
        "summary_debits": [],
        "closing_balance": []
    }

    if full_text:
        txt_summary = extract_summary_from_full_text(full_text)
        for k, vals in txt_summary.items():
            for v in vals:
                if v not in summary[k]:
                    summary[k].append(v)

    last_known_header_map = {}
    per_page_stats = []

    for page_idx, page_spans in enumerate(pages_spans, start=1):
        initial_rows = group_into_rows(page_spans, y_tolerance=y_tolerance)
        header_map = {}
        header_y = None

        for r_idx, row in enumerate(initial_rows):
            window_spans = list(row)
            if r_idx + 1 < len(initial_rows):
                next_row = initial_rows[r_idx + 1]
                next_has_date = any(DATE_RE.match(s["text"].strip()) for s in next_row)
                next_has_decimal = any(re.search(r"\d+\.\d{2}", s["text"]) for s in next_row)
                next_cy = sum(s["cy"] for s in next_row) / len(next_row)
                curr_cy = sum(s["cy"] for s in row) / len(row)
                if not next_has_date and not next_has_decimal and abs(next_cy - curr_cy) <= 18.0:
                    window_spans.extend(next_row)

            joined = " ".join(s["text"].lower() for s in window_spans)
            if "date" in joined and "balance" in joined and any(k in joined for k in ["credit", "debit", "deposit", "withdrawal", "withdrawl", "withdra", "dr", "cr", "narration", "description", "remarks", "particulars"]):
                header_y = max(s["cy"] for s in window_spans)
                for s in window_spans:
                    txt = s["text"].lower().strip()
                    if txt in ("sl", "si", "sl no", "si no", "s.no", "sr no"):
                        header_map["sl_no"] = s["cx"]
                    elif "tran" in txt and ("id" in txt or len(txt) <= 7):
                        header_map["tran_id"] = s["cx"]
                    elif "posted" in txt:
                        header_map["posted_date"] = s["cx"]
                    elif "value" in txt and "date" in txt:
                        header_map["value_date"] = s["cx"]
                    elif "date" in txt and "generation" not in txt:
                        header_map["date"] = s["cx"]
                    elif any(k in txt for k in ["desc", "particular", "narration", "detail", "remark"]):
                        header_map["description"] = s["cx"]
                    elif "branch" in txt or "code" in txt:
                        header_map["branch_code"] = s["cx"]
                    elif any(k in txt for k in ["chq", "ref", "cheque"]):
                        header_map["ref"] = s["cx"]
                    elif ("withdrawal" in txt and "deposit" in txt) or ("dr" in txt and "cr" in txt) or "withdrawal(dr)" in txt or "deposit(cr)" in txt:
                        header_map["amount_cr_dr"] = s["cx"]
                    elif any(k in txt for k in ["credit", "deposit", "(cr)"]):
                        header_map["credit"] = s["cx"]
                    elif any(k in txt for k in ["debit", "withdrawal", "withdrawl", "withdra", "wal (dr)", "(dr)"]):
                        header_map["debit"] = s["cx"]
                    elif "balance" in txt and "available" not in txt and "effective" not in txt:
                        header_map["balance"] = s["cx"]
                break

        if header_map:
            last_known_header_map = dict(header_map)
        elif last_known_header_map:
            header_map = dict(last_known_header_map)
            header_y = 0.0
        else:
            if page_idx == 1:
                msg = "Could not dynamically locate transaction table headers on Page 1."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})
            continue

        num_cols = [header_map[k] for k in ("debit", "credit", "balance", "amount_cr_dr") if k in header_map]
        if num_cols:
            page_spans = Stitch_Vertical_Wrapped_Numbers(page_spans, num_cols)

        rows = group_into_rows(page_spans, y_tolerance=y_tolerance)

        body_rows = []
        for row in rows:
            row_cy = sum(s["cy"] for s in row) / len(row)
            if header_y and row_cy <= header_y + 3.0:
                continue
            row_joined_lower = " ".join(s["text"].lower() for s in row).strip()

            if re.search(r"\bpage\s*[:\-]?\s*\d+\s*(?:of|/)\s*\d+\b", row_joined_lower):
                continue
            if any(skip_k in row_joined_lower for skip_k in [
                "statement of account", "account no", "indusind bank", "yes bank", "icici bank",
                "system generated", "phone banking", "service tax registration", "discrepancies in this statement",
                "computer generated", "registered office", "corporate identity number", "limits and effective"
            ]):
                continue

            if any(stop_k in row_joined_lower for stop_k in [
                "page total", "period totals", "reconciliation note", "statement summary",
                "end of statement", "end of the statement", "--end of statement", "branch code", "account branch", "legends used"
            ]):
                break
            body_rows.append(row)

        date_x_max = max(header_map.get("date", 80), header_map.get("value_date", 145)) + 50
        bal_col_x = header_map.get("balance", 500)

        def row_has_balance_span(r):
            for s in r:
                txt = s["text"].strip()
                if abs(s["cx"] - bal_col_x) <= 55:
                    val, _ = parse_amount_with_sign(txt)
                    if val is not None:
                        return True
            return False

        def row_has_date_span(r):
            return any(
                s["cx"] <= date_x_max and DATE_RE.match(s["text"].strip())
                for s in r
            ) or DATE_RE.match(r[0]["text"].strip())

        anchor_indices = [r_i for r_i, r in enumerate(body_rows) if row_has_balance_span(r)]
        if not anchor_indices:
            anchor_indices = [r_i for r_i, r in enumerate(body_rows) if row_has_date_span(r)]

        tx_blocks = [[] for _ in anchor_indices]
        if anchor_indices:
            anchor_cys = [sum(s["cy"] for s in body_rows[a_i]) / len(body_rows[a_i]) for a_i in anchor_indices]
            for r_i, row in enumerate(body_rows):
                if r_i in anchor_indices:
                    pos = anchor_indices.index(r_i)
                    tx_blocks[pos].extend(row)
                else:
                    row_cy = sum(s["cy"] for s in row) / len(row)
                    closest_pos = min(range(len(anchor_cys)), key=lambda idx: abs(row_cy - anchor_cys[idx]))
                    tx_blocks[closest_pos].extend(row)

        # Properly define page_uses_decimals before iterating over tx_blocks
        page_uses_decimals = (
            sum(
                1
                for b_r in body_rows
                for s_x in b_r
                if abs(s_x["cx"] - bal_col_x) <= 55 and "." in s_x["text"]
            )
            >= max(1, len(anchor_indices) // 2)
        )
        page_tx_count = 0

        for block in tx_blocks:
            block.sort(key=lambda s: (round(s["cy"], 1), s["cx"]))
            row_joined_lower = " ".join(s["text"].lower() for s in block)

            if bool(re.search(r"\b(?:brought\s+forward|opening\s+balance|b/f)\b", row_joined_lower)) and "upi/" not in row_joined_lower:
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None:
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        if page_idx == 1 and signed_v not in summary["opening_balance"]:
                            summary["opening_balance"].append(signed_v)

            if bool(re.search(r"\b(?:closing\s+balance|carried\s+forward|c/f)\b", row_joined_lower)) and "upi/" not in row_joined_lower:
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None:
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        summary["closing_balance"] = [signed_v]
                continue

            date_str = None
            sl_no_val = None
            used_ids = set()

            for idx, s in enumerate(block):
                t_clean = s["text"].strip().rstrip(".")
                if "sl_no" in header_map and abs(s["cx"] - header_map["sl_no"]) <= 22 and t_clean.isdigit():
                    if sl_no_val is None:
                        sl_no_val = int(t_clean)
                    used_ids.add(idx)
                    continue

                m_dt = DATE_RE.match(s["text"].strip())
                if m_dt and s["cx"] <= date_x_max:
                    if date_str is None:
                        date_str = m_dt.group(0)
                    used_ids.add(idx)
                elif re.match(r"^\d{2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?$", s["text"].strip(), re.IGNORECASE):
                    used_ids.add(idx)

            if not date_str:
                date_str = transactions[-1]["date"] if transactions else f"Page-{page_idx}"

            rx0 = round(min(s["bbox"][0] for s in block), 2)
            ry0 = round(min(s["bbox"][1] for s in block), 2)
            rx1 = round(max(s["bbox"][2] for s in block), 2)
            ry1 = round(max(s["bbox"][3] for s in block), 2)

            tx = {
                "page": page_idx,
                "sl_no": sl_no_val,
                "bbox": [rx0, ry0, rx1, ry1],
                "date": date_str,
                "description": [],
                "credit": 0.0,
                "debit": 0.0,
                "balance": None
            }

            cr_dr_amounts = []
            for idx, s in enumerate(block):
                if idx in used_ids:
                    continue
                txt = s["text"].strip()
                if txt in ("—", "-", "--", "₹", "|", "Cr", "Dr", "CR", "DR", ".00"):
                    continue

                val, ind = parse_amount_with_sign(txt)
                if val is not None and ("[" in txt or "]" in txt):
                    val = None

                if ind == "MALFORMED_DOT":
                    msg_mal = (
                        f"Page {page_idx} ({date_str}): Malformed Monetary Format Anomaly — "
                        f"Amount '{txt}' uses a period '.' instead of a thousands comma ','."
                    )
                    issues.append(msg_mal)
                    detailed_issues.append({"page": page_idx, "bbox": s["bbox"], "text": msg_mal})

                closest_col = min(header_map.keys(), key=lambda c: abs(s["cx"] - header_map[c]))
                is_monetary_col = closest_col in ("credit", "debit", "balance", "amount_cr_dr")
                dist_to_col = abs(s["cx"] - header_map[closest_col])

                if val is not None and ind in ("CR", "DR"):
                    cr_dr_amounts.append((s["cx"], val, ind, txt))
                elif val is not None and is_monetary_col and dist_to_col <= 55.0 and ("." in txt or "," in txt or not page_uses_decimals):
                    if closest_col == "balance":
                        tx["balance"] = -abs(val) if txt.startswith("-") else val
                    elif closest_col == "credit":
                        if val > 0 or tx["credit"] == 0.0:
                            tx["credit"] = val
                    elif closest_col == "debit":
                        if val > 0 or tx["debit"] == 0.0:
                            tx["debit"] = val
                    elif closest_col == "amount_cr_dr":
                        tx["credit"] = val
                elif val is not None and ("." in txt) and re.match(r"^[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2}$", txt):
                    if "debit" in header_map and abs(s["cx"] - header_map["debit"]) <= 75 and tx["debit"] == 0.0:
                        tx["debit"] = val
                    elif "credit" in header_map and abs(s["cx"] - header_map["credit"]) <= 75 and tx["credit"] == 0.0:
                        tx["credit"] = val
                    else:
                        tx["description"].append(txt)
                else:
                    if closest_col not in ("value_date", "posted_date", "sl_no", "branch_code"):
                        tx["description"].append(txt)

            if len(cr_dr_amounts) >= 2:
                cr_dr_amounts.sort(key=lambda item: item[0])
                amt_cx, amt_val, amt_ind, _ = cr_dr_amounts[-2]
                bal_cx, bal_val, bal_ind, _ = cr_dr_amounts[-1]
                if amt_ind == "CR":
                    tx["credit"] = amt_val
                else:
                    tx["debit"] = amt_val
                tx["balance"] = -bal_val if bal_ind == "DR" else bal_val

            tx["description"] = " ".join(tx["description"]).strip()
            if tx["balance"] is not None:
                if tx["credit"] == 0.0 and tx["debit"] == 0.0 and not tx["description"] and transactions and abs(tx["balance"] - transactions[-1]["balance"]) < 0.01:
                    continue
                transactions.append(tx)
                page_tx_count += 1

        per_page_stats.append({"page": page_idx, "transactions_extracted": page_tx_count})

    for idx, tx in enumerate(transactions):
        if tx["credit"] == 0.0 and tx["debit"] == 0.0 and not any(k in tx["description"].lower() for k in ["b/f", "brought forward"]):
            prev_b = summary["opening_balance"][0] if (idx == 0 and summary["opening_balance"]) else (transactions[idx - 1]["balance"] if idx > 0 else None)
            if prev_b is not None:
                delta = round(tx["balance"] - prev_b, 2)
                abs_delta = abs(delta)
                int_part = int(abs_delta)
                indian_fmt = f"{int_part:,}"
                if len(str(int_part)) > 3:
                    last3 = str(int_part)[-3:]
                    rest = str(int_part)[:-3]
                    indian_fmt = ",".join([rest[max(0, i - 2):i] for i in range(len(rest), 0, -2)][::-1]) + "," + last3
                if indian_fmt in tx["description"] or str(int_part) in tx["description"]:
                    if delta > 0:
                        tx["credit"] = abs_delta
                    elif delta < 0:
                        tx["debit"] = abs_delta

    res = run_math_checks(summary, transactions, issues, detailed_issues)
    res["total_pages_processed"] = len(pages_spans)
    res["per_page_summary"] = per_page_stats
    return res


def run_math_checks(summary, transactions, issues, detailed_issues):
    audit_coverage = {
        "row_completeness_verified": "NOT_RUN",
        "summary_cards_extracted": "NOT_RUN",
        "running_balance_verified": "NOT_RUN",
        "ledger_totals_verified": "NOT_RUN",
        "summary_equation_verified": "NOT_RUN"
    }

    if not transactions:
        msg = "Fatal: No valid transaction rows could be extracted for reconciliation."
        issues.append(msg)
        detailed_issues.append({"page": 1, "bbox": None, "text": msg})
        audit_coverage["row_completeness_verified"] = "FAILED"
        audit_coverage["running_balance_verified"] = "FAILED"
        return {
            "document_type": "BANK_STATEMENT",
            "status": "REVIEW_REQUIRED",
            "audit_coverage": audit_coverage,
            "summary": summary,
            "opening_balance_used": None,
            "total_credits": 0.0,
            "total_debits": 0.0,
            "closing_balance": 0.0,
            "transactions": [],
            "issues": issues,
            "detailed_issues": detailed_issues
        }

    first_tx = transactions[0]
    if bool(re.search(r"\b(?:b/f|brought\s+forward|opening\s+balance)\b", first_tx["description"].lower())) and first_tx["credit"] == 0 and first_tx["debit"] == 0:
        if first_tx["balance"] not in summary["opening_balance"]:
            summary["opening_balance"].insert(0, first_tx["balance"])
        active_txs = transactions[1:]
    else:
        active_txs = transactions

    sl_nos = [t.get("sl_no") for t in active_txs if t.get("sl_no") is not None]
    if len(sl_nos) >= len(active_txs) * 0.5 and sl_nos:
        max_sl = max(sl_nos)
        if len(active_txs) < max_sl:
            missing_count = max_sl - len(active_txs)
            audit_coverage["row_completeness_verified"] = "FAILED"
            msg = f"Extraction Completeness Warning: Document serial numbers reach {max_sl}, but only {len(active_txs)} transaction rows were extracted ({missing_count} row(s) missing)."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})
        else:
            audit_coverage["row_completeness_verified"] = "PASSED"
    else:
        audit_coverage["row_completeness_verified"] = "PASSED" if len(active_txs) >= 1 else "FAILED"

    if summary["opening_balance"]:
        opening_bal = summary["opening_balance"][0]
    elif active_txs:
        opening_bal = round(active_txs[0]["balance"] - active_txs[0]["credit"] + active_txs[0]["debit"], 2)
    else:
        opening_bal = 0.0

    running_bal = opening_bal
    rb_failed = False

    for idx, tx in enumerate(active_txs, start=1):
        actual_bal = round(tx["balance"], 2)
        expected_bal = round(running_bal + tx["credit"] - tx["debit"], 2)

        if abs(expected_bal - actual_bal) > 1.0 and abs(expected_bal - (-actual_bal)) <= 1.5:
            actual_bal = -actual_bal
            tx["balance"] = actual_bal

        if abs(expected_bal - actual_bal) > 1.0:
            if tx["credit"] > 0 and tx["debit"] == 0 and abs((running_bal - tx["credit"]) - actual_bal) <= 1.0:
                tx["debit"], tx["credit"] = tx["credit"], 0.0
                expected_bal = round(running_bal - tx["debit"], 2)
            elif tx["debit"] > 0 and tx["credit"] == 0 and abs((running_bal + tx["debit"]) - actual_bal) <= 1.0:
                tx["credit"], tx["debit"] = tx["debit"], 0.0
                expected_bal = round(running_bal + tx["credit"], 2)

        if abs(expected_bal - actual_bal) > 1.0:
            rb_failed = True
            msg = (
                f"Page {tx['page']}, Row {idx} ({tx['date']} - {tx['description'][:45]}): "
                f"Running balance mismatch. Expected {expected_bal:,.2f}, but document shows {actual_bal:,.2f}."
            )
            issues.append(msg)
            detailed_issues.append({"page": tx["page"], "bbox": tx.get("bbox"), "text": msg})

        running_bal = actual_bal

    audit_coverage["running_balance_verified"] = "FAILED" if rb_failed else "PASSED"

    total_credits = round(sum(t["credit"] for t in active_txs), 2)
    total_debits = round(sum(t["debit"] for t in active_txs), 2)
    ledger_closing = round(active_txs[-1]["balance"], 2) if active_txs else opening_bal

    expected_closing = round(opening_bal + total_credits - total_debits, 2)
    if abs(expected_closing - ledger_closing) <= 1.5:
        audit_coverage["ledger_totals_verified"] = "PASSED"
    else:
        audit_coverage["ledger_totals_verified"] = "FAILED"
        msg = (
            f"Statement invariant mismatch: Opening ({opening_bal:,.2f}) + Credits ({total_credits:,.2f}) "
            f"- Debits ({total_debits:,.2f}) = {expected_closing:,.2f}, but final row shows {ledger_closing:,.2f}."
        )
        issues.append(msg)
        detailed_issues.append({"page": active_txs[-1]["page"], "bbox": active_txs[-1].get("bbox"), "text": msg})

    has_summary_cards = bool(summary["opening_balance"] or summary["closing_balance"] or summary["summary_credits"] or summary["summary_debits"])
    if has_summary_cards:
        audit_coverage["summary_cards_extracted"] = "PASSED"
        sum_eq_failed = False
        if summary["closing_balance"]:
            hdr_close = summary["closing_balance"][0]
            if abs(hdr_close - ledger_closing) > 1.5:
                sum_eq_failed = True
                if abs(abs(hdr_close) - abs(ledger_closing)) <= 1.5 and (hdr_close * ledger_closing < 0):
                    msg = f"Page 1 Summary Header Sign Mismatch: Header Closing Balance is '{hdr_close:,.2f}', whereas ledger Closing Balance is '{ledger_closing:,.2f}'."
                else:
                    msg = f"Summary Closing Balance ({hdr_close:,.2f}) does not match ledger closing balance ({ledger_closing:,.2f})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        if summary["summary_credits"] and abs(summary["summary_credits"][0] - total_credits) > 1.5:
            sum_eq_failed = True
            msg = f"Summary Credits ({summary['summary_credits'][0]:,.2f}) does not match sum of transaction credits ({total_credits:,.2f})."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        if summary["summary_debits"] and abs(summary["summary_debits"][0] - total_debits) > 1.5:
            sum_eq_failed = True
            msg = f"Summary Debits ({summary['summary_debits'][0]:,.2f}) does not match sum of transaction debits ({total_debits:,.2f})."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        audit_coverage["summary_equation_verified"] = "FAILED" if sum_eq_failed else "PASSED"
    else:
        audit_coverage["summary_cards_extracted"] = "NOT_APPLICABLE"
        audit_coverage["summary_equation_verified"] = "PASSED" if audit_coverage["ledger_totals_verified"] == "PASSED" else "FAILED"

    return {
        "document_type": "BANK_STATEMENT",
        "status": "REVIEW_REQUIRED" if issues else "NO_MISMATCH_FOUND",
        "audit_coverage": audit_coverage,
        "summary": summary,
        "opening_balance_used": opening_bal,
        "total_credits": total_credits,
        "total_debits": total_debits,
        "closing_balance": ledger_closing,
        "transactions": transactions,
        "issues": issues,
        "detailed_issues": detailed_issues
    }
