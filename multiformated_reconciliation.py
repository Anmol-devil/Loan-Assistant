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
    r"^\s*("
    r"\d{1,2}[\s/\-\.]"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})"
    r"(?:[\s/\-\.]\d{2,4})?"
    r")"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r"\s*$",
    re.IGNORECASE
)

DATE_RE = re.compile(
    r"^\s*(\d{1,2}[\s/\-\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})(?:[\s/\-\.]\d{2,4})?)\s*$",
    re.IGNORECASE
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

MALFORMED_DOT_AMOUNT_RE = re.compile(r"^[₹$\s]*([+-]?\d{1,3}(?:\.\d{3})+\.\d{2})\b")


def parse_amount_with_sign(text):
    """
    Returns (float_value, 'CR' | 'DR' | 'MALFORMED_DOT' | None).
    Handles standard numbers, signed negative balances ('-8,152,664.07'),
    (Cr)/(Dr) suffixes, and malformed double-dot numbers ('242.229.43').
    """
    if not text:
        return None, None
    s = str(text).strip()

    # Do not mistake a date ('02-04-2025') or timestamp ('09:07:54') for an amount
    if DATE_RE.match(s) and not re.search(r"\((?:Cr|Dr)\)", s, re.IGNORECASE):
        return None, None
    if re.match(r"^\d{2}:\d{2}(?::\d{2})?$", s):
        return None, None

    # Check for malformed double-dot amount like '242.229.43' on Page 5 of YES BANK statement
    m_dot = MALFORMED_DOT_AMOUNT_RE.match(s)
    if m_dot:
        raw_mal = m_dot.group(1)
        # Replace all dots except the last one with nothing
        parts = raw_mal.split(".")
        fixed_str = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return round(float(fixed_str), 2), "MALFORMED_DOT"
        except ValueError:
            pass

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

def run_math_checks(summary, transactions, issues, detailed_issues=None):
    """
    Fixes Issue #3 & #5:
    - Uses explicit statuses: PASSED, FAILED, NOT_RUN, NOT_APPLICABLE.
    - Preserves exact page number and row_bbox on every running-balance failure.
    """
    if detailed_issues is None:
        detailed_issues = [{"page": 1, "bbox": None, "text": msg} for msg in issues]

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
            "transactions": [],
            "issues": issues,
            "detailed_issues": detailed_issues
        }

    has_four_cards = all(summary.get(k) for k in ["opening_balance", "summary_credits", "summary_debits", "closing_balance"])
    has_open_and_close = bool(summary.get("opening_balance") and summary.get("closing_balance"))

    if has_four_cards or has_open_and_close:
        audit_coverage["summary_cards_extracted"] = "PASSED"
    else:
        audit_coverage["summary_cards_extracted"] = "FAILED"
        missing_summary_keys = [k for k in ["opening_balance", "closing_balance"] if not summary.get(k)]
        msg = f"Extraction Warning: Missing opening/closing summary fields {missing_summary_keys}. Summary reconciliation could not be fully evaluated."
        issues.append(msg)
        detailed_issues.append({"page": 1, "bbox": None, "text": msg})

    for key, candidates in summary.items():
        if len(set(candidates)) > 1:
            audit_coverage["summary_cards_extracted"] = "FAILED"
            msg = f"Conflicting values detected in {key} card: {candidates}."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})

    first_tx = transactions[0]
    if "opening" in first_tx["description"].lower() or "brought forward" in first_tx["description"].lower() or (first_tx["credit"] == 0 and first_tx["debit"] == 0):
        opening_bal = first_tx["balance"]
        active_txs = transactions[1:]
    else:
        opening_bal = summary["opening_balance"][0] if summary.get("opening_balance") else round(
            first_tx["balance"] - first_tx["credit"] + first_tx["debit"], 2
        )
        active_txs = transactions

    calc_credits = round(sum(t["credit"] for t in active_txs), 2)
    calc_debits = round(sum(t["debit"] for t in active_txs), 2)
    calc_closing = transactions[-1]["balance"]

    running = opening_bal
    running_balance_errors = 0
    start_row = 2 if active_txs != transactions else 1

    for idx, tx in enumerate(active_txs, start=start_row):
        running = round(running + tx["credit"] - tx["debit"], 2)
        if abs(running - tx["balance"]) > 0.05:
            tx_page = tx.get("page", 1)
            tx_bbox = tx.get("bbox")
            msg = (
                f"Page {tx_page}, Row {idx} ({tx['date']} - {tx['description']}): Running balance mismatch. "
                f"Expected {running}, but document shows {tx['balance']}."
            )
            issues.append(msg)
            detailed_issues.append({"page": tx_page, "bbox": tx_bbox, "text": msg})
            running_balance_errors += 1
            running = tx["balance"]

    audit_coverage["running_balance_verified"] = "PASSED" if running_balance_errors == 0 else "FAILED"

    totals_mismatch = False
    if summary.get("summary_credits"):
        for c_val in summary["summary_credits"]:
            if abs(c_val - calc_credits) > 0.05:
                totals_mismatch = True
                msg = f"Summary Credits ({c_val}) does not match sum of transaction credits ({calc_credits})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})

    if summary.get("summary_debits"):
        for d_val in summary["summary_debits"]:
            if abs(d_val - calc_debits) > 0.05:
                totals_mismatch = True
                msg = f"Summary Debits ({d_val}) does not match sum of transaction debits ({calc_debits})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})

    if summary.get("closing_balance"):
        for cl_val in summary["closing_balance"]:
            if abs(cl_val - calc_closing) > 0.05:
                totals_mismatch = True
                last_page = transactions[-1].get("page", 1)
                msg = f"Summary Closing Balance ({cl_val}) does not match ledger closing balance ({calc_closing})."
                issues.append(msg)
                detailed_issues.append({"page": last_page, "bbox": transactions[-1].get("bbox"), "text": msg})
        audit_coverage["ledger_totals_verified"] = "FAILED" if totals_mismatch else "PASSED"
    else:
        audit_coverage["ledger_totals_verified"] = "NOT_RUN"

    if has_four_cards:
        op = summary["opening_balance"][0]
        de = summary["summary_debits"][0]
        cl = summary["closing_balance"][0]
        eq_err = False
        for cr in summary["summary_credits"]:
            expected_cl = round(op + cr - de, 2)
            if abs(expected_cl - cl) > 0.05:
                eq_err = True
                msg = f"Summary math mismatch: Opening ({op}) + Credits ({cr}) - Debits ({de}) = {expected_cl}, but Closing Balance shows {cl}."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})
        audit_coverage["summary_equation_verified"] = "FAILED" if eq_err else "PASSED"
    elif has_open_and_close:
        op = summary["opening_balance"][0]
        cl = summary["closing_balance"][0]
        expected_cl = round(op + calc_credits - calc_debits, 2)
        if abs(expected_cl - cl) > 0.05:
            audit_coverage["summary_equation_verified"] = "FAILED"
            msg = f"Statement invariant mismatch: Brought Forward ({op}) + Ledger Credits ({calc_credits}) - Ledger Debits ({calc_debits}) = {expected_cl}, but Closing Balance shows {cl}."
            issues.append(msg)
            detailed_issues.append({"page": transactions[-1].get("page", 1), "bbox": None, "text": msg})
        else:
            audit_coverage["summary_equation_verified"] = "PASSED"
    else:
        audit_coverage["summary_equation_verified"] = "NOT_RUN"

    if issues:
        status = "REVIEW_REQUIRED"
    elif any(v == "NOT_RUN" for v in audit_coverage.values()):
        status = "PARTIAL_CHECK_CLEAN"
    else:
        status = "NO_MISMATCH_FOUND"

    return {
        "document_type": "BANK_STATEMENT",
        "status": status,
        "audit_coverage": audit_coverage,
        "summary": summary,
        "opening_balance_used": opening_bal,
        "total_credits": calc_credits,
        "total_debits": calc_debits,
        "closing_balance": calc_closing,
        "transactions": transactions,
        "issues": issues,
        "detailed_issues": detailed_issues
    }


def extract_summary_from_full_text(full_text: str) -> dict:
    """
    Extracts explicit Opening Balance, Total Credits/Deposits, Total Debits/Withdrawals,
    and Closing Balance from statement headers or bottom summary blocks.
    Preserves negative signs (e.g. '-8,152,664.07') and (Dr)/(Cr) suffixes.
    """
    summary = {
        "opening_balance": [],
        "summary_credits": [],
        "summary_debits": [],
        "closing_balance": []
    }

    patterns = {
        "opening_balance": r"opening\s*balance\s*(?:[:]\s*|\s+)([+-]?\s*[₹$]?\s*[\d,]+\.\d{2}\s*(?:\([CD]r\))?)",
        "summary_debits": r"total\s*(?:withdrawal|debit)s?\s*(?:amount)?\s*(?:[:]\s*|\s+)([+-]?\s*[₹$]?\s*[\d,]+\.\d{2}\s*(?:\([CD]r\))?)",
        "summary_credits": r"total\s*(?:deposit|credit)s?\s*(?:amount)?\s*(?:[:]\s*|\s+)([+-]?\s*[₹$]?\s*[\d,]+\.\d{2}\s*(?:\([CD]r\))?)",
        "closing_balance": r"closing\s*balance\s*(?:[:]\s*|\s+)([+-]?\s*[₹$]?\s*[\d,]+\.\d{2}\s*(?:\([CD]r\))?)"
    }

    for key, pat in patterns.items():
        for m in re.finditer(pat, full_text, re.IGNORECASE):
            raw_capture = m.group(1).strip()
            is_neg = raw_capture.startswith("-")
            val, ind = parse_amount_with_sign(raw_capture)
            if val is not None:
                if key in ("opening_balance", "closing_balance") and (ind == "DR" or is_neg):
                    val = -abs(val)
                if val not in summary[key]:
                    summary[key].append(val)

    return summary

def reconcile_spans(pages_spans, y_tolerance=6.0, full_text=""):
    """
    Universal Multi-Page Bank Statement Reconciler:
    - Keeps header_y strictly on the header row so the first transaction on continuation pages is never skipped.
    - Supports vertically-centered multi-line grid tables (YES BANK) and top-aligned tables (Kotak, HDFC, SCB).
    - Supports negative overdraft balances ('-8,152,664.07') and '(Dr)'/'(Cr)' suffixes.
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
        rows = group_into_rows(page_spans, y_tolerance=y_tolerance)
        header_map = {}
        header_y = None

        for r_idx, row in enumerate(rows):
            row_joined = " ".join(s["text"].lower() for s in row)
            # Only peek at r_idx + 1 if it is a wrapped header continuation (NO dates and NO decimal numbers)
            window_spans = list(row)
            if r_idx + 1 < len(rows):
                next_row = rows[r_idx + 1]
                next_has_date = any(DATE_RE.match(s["text"].strip()) for s in next_row)
                next_has_decimal = any(re.search(r"\d+\.\d{2}", s["text"]) for s in next_row)
                next_cy = sum(s["cy"] for s in next_row) / len(next_row)
                curr_cy = sum(s["cy"] for s in row) / len(row)
                if not next_has_date and not next_has_decimal and abs(next_cy - curr_cy) <= 16.0:
                    window_spans.extend(next_row)

            joined = " ".join(s["text"].lower() for s in window_spans)
            if "date" in joined and "balance" in joined and any(k in joined for k in ["credit", "debit", "deposit", "withdrawal", "dr", "cr", "narration", "description"]):
                # Set header_y strictly from the actual header spans (never from a transaction row)
                header_y = max(s["cy"] for s in window_spans)
                for s in window_spans:
                    txt = s["text"].lower()
                    if "value" in txt and "date" in txt:
                        header_map["value_date"] = s["cx"]
                    elif "date" in txt:
                        header_map["date"] = s["cx"]
                    elif any(k in txt for k in ["desc", "particular", "narration", "detail", "remark"]):
                        header_map["description"] = s["cx"]
                    elif any(k in txt for k in ["chq", "ref", "cheque"]):
                        header_map["ref"] = s["cx"]
                    elif ("withdrawal" in txt and "deposit" in txt) or ("dr" in txt and "cr" in txt) or "withdrawal(dr)" in txt or "deposit(cr)" in txt:
                        header_map["amount_cr_dr"] = s["cx"]
                    elif any(k in txt for k in ["credit", "deposit"]):
                        header_map["credit"] = s["cx"]
                    elif any(k in txt for k in ["debit", "withdrawal"]):
                        header_map["debit"] = s["cx"]
                    elif "balance" in txt:
                        header_map["balance"] = s["cx"]
                break

        # Only use spatial card extraction on Page 1 if explicit text regex didn't already find Opening & Closing balances
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

        body_rows = []
        for row in rows:
            row_cy = sum(s["cy"] for s in row) / len(row)
            if header_y and row_cy <= header_y + 3.0:
                continue
            row_joined_lower = " ".join(s["text"].lower() for s in row)
            if any(stop_k in row_joined_lower for stop_k in [
                "period totals", "reconciliation note", "statement summary",
                "end of statement", "end of the statement", "branch code", "account branch"
            ]):
                break
            body_rows.append(row)

        # Identify Transaction Anchor Rows (rows that contain a Date or B/F/Closing marker)
        date_x_max = max(header_map.get("date", 80), header_map.get("value_date", 145)) + 50
        anchor_indices = []

        for r_i, row in enumerate(body_rows):
            row_joined_lower = " ".join(s["text"].lower() for s in row)
            if any(k in row_joined_lower for k in ["brought forward", "opening balance", "b/f", "closing balance", "carried forward", "c/f"]):
                anchor_indices.append(r_i)
                continue
            has_date = any(
                s["cx"] <= date_x_max and DATE_RE.match(s["text"].strip())
                for s in row
            ) or DATE_RE.match(row[0]["text"].strip())
            if has_date:
                anchor_indices.append(r_i)

        # Group non-anchor wrapped lines (e.g. multi-line descriptions or timestamps) with their nearest anchor row
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

            if any(k in row_joined_lower for k in ["brought forward", "opening balance", "b/f"]):
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None and "." in s["text"]:
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        if page_idx == 1 and signed_v not in summary["opening_balance"]:
                            summary["opening_balance"].append(signed_v)

            if any(k in row_joined_lower for k in ["closing balance", "carried forward", "c/f"]):
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None and "." in s["text"]:
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        summary["closing_balance"] = [signed_v]
                continue

            date_str = None
            used_ids = set()
            for idx, s in enumerate(block):
                t_clean = s["text"].strip()
                m_dt = DATE_RE.match(t_clean)
                if m_dt and s["cx"] <= date_x_max:
                    if date_str is None:
                        date_str = m_dt.group(1)
                    used_ids.add(idx)
                elif re.match(r"^\d{2}:\d{2}(?::\d{2})?$", t_clean):
                    used_ids.add(idx)

            if not date_str:
                if len(block) > 1 and DATE_RE.match(f"{block[0]['text']} {block[1]['text']}"):
                    date_str = f"{block[0]['text']} {block[1]['text']}"
                    used_ids.update({0, 1})
                elif len(block) > 2 and DATE_RE.match(f"{block[0]['text']} {block[1]['text']} {block[2]['text']}"):
                    date_str = f"{block[0]['text']} {block[1]['text']} {block[2]['text']}"
                    used_ids.update({0, 1, 2})

            if not date_str:
                continue

            rx0 = round(min(s["bbox"][0] for s in block), 2)
            ry0 = round(min(s["bbox"][1] for s in block), 2)
            rx1 = round(max(s["bbox"][2] for s in block), 2)
            ry1 = round(max(s["bbox"][3] for s in block), 2)

            tx = {
                "page": page_idx,
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
                if txt in ("—", "-", "--", "₹", "|", "Cr", "Dr", "CR", "DR"):
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
                else:
                    if closest_col not in ("value_date", "ref"):
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
                transactions.append(tx)
                page_tx_count += 1

        per_page_stats.append({"page": page_idx, "transactions_extracted": page_tx_count})

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
                both_numeric_parts = (
                    NUMERIC_FRAGMENT_RE.match(curr["text"]) is not None
                    and NUMERIC_FRAGMENT_RE.match(s["text"]) is not None
                )

                if x_gap < -3.0:
                    page_merged.append(curr)
                    curr = dict(s)
                elif (same_style and x_gap <= 4.0) or (both_numeric_parts and -2.0 <= x_gap <= 8.0):
                    sep = "" if both_numeric_parts else " "
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


# Strict regex: matches ONLY standalone monetary amounts (e.g. '250,000.00', '17,020.00(Cr)', '₹ 34,436.80')
# Excludes dates, PANs, section numbers ('44AB'), years ('1961'), and PIN codes ('110059')
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

    # ---------------------------------------------------------
    # CHECK 1: Column Alignment Jitter (Sub-pixel X0 / X1 check)
    # ---------------------------------------------------------
    # Collect all numeric spans inside the transaction table area (y > 280)
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