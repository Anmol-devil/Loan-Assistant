import re
from typing import Dict, Any, Optional

PAN_RE = re.compile(r"\b([A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z])\b")
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")
UDYAM_RE = re.compile(r"\b(UDYAM-[A-Z]{2}-\d{2}-\d{7})\b", re.IGNORECASE)
ITR_ACK_RE = re.compile(r"\b(\d{15})\b")
UDIN_RE = re.compile(r"\b(\d{2}\d{6}[A-Z0-9]{10})\b")

GST_STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra & Nagar Haveli and Daman & Diu", "27": "Maharashtra", "29": "Karnataka", "30": "Goa",
    "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman & Nicobar Islands", "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
    "97": "Other Territory",
}
INDIAN_STATE_CODES = GST_STATE_CODES


def parse_indian_number(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = re.sub(r"[₹$,\s]", "", str(text))
    m = re.search(r"([+-]?\d+(?:\.\d{1,2})?)", cleaned)
    return round(float(m.group(1)), 2) if m else None


def _parse_gst_float(s: str) -> float:
    return round(float(str(s).replace(",", "").strip()), 2)


def _verify_gstin_checksum(gstin: str) -> bool:
    """Validates the official Indian GSTIN 15th-character Mod-36 checksum."""
    if not gstin or len(gstin) != 15:
        return False
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total = 0
    for idx, ch in enumerate(gstin[:14]):
        val = chars.find(ch)
        if val == -1:
            return False
        factor = 1 if idx % 2 == 0 else 2
        prod = val * factor
        total += (prod // 36) + (prod % 36)
    expected_char = chars[(36 - (total % 36)) % 36]
    return gstin[14] == expected_char


def _extract_gst_field_value(
    lines: list[str],
    label_patterns: tuple[str, ...],
    bad_values: tuple[str, ...],
    value_filter_re: Optional[str] = None,
) -> Optional[str]:
    """
    Tries each label pattern in priority order across all lines.
    Captures either inline values or values on the next 1-3 lines.
    """
    for pat in label_patterns:
        for idx, ln in enumerate(lines):
            m = re.search(pat, ln, re.IGNORECASE)
            if not m:
                continue

            # 1. Check remainder of the same line
            rem = ln[m.end():].strip(" :|-.\t()")
            if rem and len(rem) >= 2 and not any(b in rem.lower() for b in bad_values):
                cand = re.sub(r"\s+", " ", rem).strip()
                if not value_filter_re or re.search(value_filter_re, cand, re.IGNORECASE):
                    return cand

            # 2. Check next 1-3 lines
            for k in range(idx + 1, min(len(lines), idx + 4)):
                nxt = lines[k].strip(" :|-\t")
                if not nxt or re.match(r"^\([a-z0-9]\)$", nxt, re.I):
                    continue
                if any(b in nxt.lower() for b in bad_values):
                    continue
                cand = re.sub(r"\s+", " ", nxt).strip()
                if not value_filter_re or re.search(value_filter_re, cand, re.IGNORECASE):
                    return cand
    return None


def classify_financial_document(full_text: str) -> Dict[str, Any]:
    """
    Strictly classifies financial/KYC documents based on explicit evidence.
    Checks Form 3CB/3CD Tax Audit Reports and all 4 GST Portal documents (REG-06,
    GSTR-1, GSTR-3B, GSTR-2B) before ITR or Bank Statements.
    """
    clean_txt = re.sub(r"---\s*PAGE\s*\d+\s*---", "", str(full_text or ""), flags=re.IGNORECASE)
    low = clean_txt.lower()
    up = clean_txt.upper()

    # 1. Tax Audit Report (Form 3CB / 3CD under Section 44AB)
    if any(k in low for k in ["form no. 3cb", "form no. 3cd", "section 44ab", "form 3cb", "form 3cd", "rule 6g"]):
        return {
            "doc_code": "TAX_AUDIT_REPORT",
            "doc_category": "Tax Audit Report (Form 3CB/3CD)",
            "classification_confidence": 0.96,
        }

    # 2. All 4 Official GST Portal Documents (REG-06, GSTR-1, GSTR-3B, GSTR-2B)
    if re.search(r"\bform\s+gstr[\-\s]*3b\b|\bgstr[\-\s]*3b\b", low):
        return {
            "doc_code": "GST_CERTIFICATE",
            "doc_category": "GST Summary Return (GSTR-3B)",
            "classification_confidence": 0.97,
        }
    if re.search(r"\bform\s+gstr[\-\s]*2b\b|\bgstr[\-\s]*2b\b|auto[\-\s]*drafted\s+itc\s+statement", low):
        return {
            "doc_code": "GST_CERTIFICATE",
            "doc_category": "GST ITC Statement (GSTR-2B)",
            "classification_confidence": 0.97,
        }
    if re.search(r"\bform\s+gstr[\-\s]*1\b|\bgstr[\-\s]*1\b|details\s+of\s+outward\s+supplies", low):
        return {
            "doc_code": "GST_CERTIFICATE",
            "doc_category": "GST Sales Return (GSTR-1)",
            "classification_confidence": 0.97,
        }
    if (
        "gst reg-06" in low
        or "goods and services tax" in low
        or ("registration certificate" in low and GSTIN_RE.search(up))
    ):
        return {
            "doc_code": "GST_CERTIFICATE",
            "doc_category": "GST Registration Certificate (REG-06)",
            "classification_confidence": 0.96,
        }

    # 3. Udyam Registration Certificate
    if "udyam registration" in low or "ministry of micro, small and medium" in low or UDYAM_RE.search(clean_txt):
        return {
            "doc_code": "UDYAM_CERTIFICATE",
            "doc_category": "Udyam Certificate",
            "classification_confidence": 0.95,
        }

    # 4. Income Tax Return (ITR-V / Tax Computation)
    if any(k in low for k in ["indian income tax return", "itr-v", "itr-", "acknowledgement number", "gross total income"]):
        return {
            "doc_code": "ITR_DOCUMENT",
            "doc_category": "ITR / Tax Return",
            "classification_confidence": 0.92,
        }

    # 5. Balance Sheet
    if any(k in low for k in ["balance sheet as at", "equity and liabilities", "shareholders' funds", "non-current assets", "sundry debtors"]):
        return {
            "doc_code": "BALANCE_SHEET",
            "doc_category": "Balance Sheet",
            "classification_confidence": 0.92,
        }

    # 6. PAN Card
    if any(k in low for k in ["income tax department", "permanent account number"]) and "statement" not in low and "itr" not in low:
        return {
            "doc_code": "PAN_CARD",
            "doc_category": "PAN Card",
            "classification_confidence": 0.95,
        }

    # 7. Salary Slip
    salary_keywords = ["earnings", "deductions", "net pay", "gross earnings", "basic salary", "payslip", "salary slip", "provident fund"]
    if sum(1 for k in salary_keywords if k in low) >= 2:
        return {
            "doc_code": "SALARY_SLIP",
            "doc_category": "Salary Slip",
            "classification_confidence": 0.90,
        }

    # 8. Bank Statement / Ledger
    bank_keywords = [
        "opening balance", "closing balance", "brought forward", "statement of account",
        "bank statement", "withdrawal", "deposit", "credit", "debit", "balance", "neft", "rtgs", "imps",
    ]
    if sum(1 for k in bank_keywords if k in low) >= 2:
        return {
            "doc_code": "BANK_STATEMENT",
            "doc_category": "Bank Statement",
            "classification_confidence": 0.90,
        }

    return {
        "doc_code": "UNKNOWN_DOCUMENT",
        "doc_category": "Unknown Document",
        "classification_confidence": 0.0,
    }


def verify_gst_certificate(full_text: str, filename: str = "") -> Dict[str, Any]:
    """
    Unified Family A verifier for all 4 Indian GST Portal documents:
      1. FORM GST REG-06 (Registration Certificate)
      2. FORM GSTR-1 (Outward Supplies Return)
      3. FORM GSTR-3B (Monthly/Quarterly Summary Return - Liability & ITC)
      4. FORM GSTR-2B (Auto-drafted ITC Statement)
    """
    txt = full_text or ""
    up = txt.upper()
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    amt_pat = r"([+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2})"

    # 1. Detect exact GST Form subtype
    if re.search(r"\bFORM\s+GSTR[\-\s]*3B\b|\bGSTR[\-\s]*3B\b", txt, re.I):
        gst_form_type = "FORM GSTR-3B"
        form_display_label = "GST Summary Return (GSTR-3B)"
    elif re.search(r"\bFORM\s+GSTR[\-\s]*2B\b|\bGSTR[\-\s]*2B\b|Auto[\-\s]*drafted\s+ITC\s+Statement", txt, re.I):
        gst_form_type = "FORM GSTR-2B"
        form_display_label = "GST ITC Statement (GSTR-2B)"
    elif re.search(r"\bFORM\s+GSTR[\-\s]*1\b|\bGSTR[\-\s]*1\b|Details\s+of\s+outward\s+supplies", txt, re.I):
        gst_form_type = "FORM GSTR-1"
        form_display_label = "GST Sales Return (GSTR-1)"
    elif re.search(r"\bFORM\s+GST\s+REG[\-\s]*06\b|Registration\s+Certificate", txt, re.I):
        gst_form_type = "FORM GST REG-06"
        form_display_label = "GST Registration Certificate (REG-06)"
    else:
        gst_form_type = "GST_DOCUMENT"
        form_display_label = "GST Document"

    # 2. Extract & validate 15-character GSTIN, State Code, Embedded PAN, and Mod-36 Checksum
    m_gstin = GSTIN_RE.search(up)
    gstin = m_gstin.group(1) if m_gstin else None

    state_code = gstin[:2] if gstin else None
    state_name = GST_STATE_CODES.get(state_code) if state_code else None
    state_ok = bool(state_name)

    embedded_pan = gstin[2:12] if gstin else None
    pan_ok = bool(embedded_pan and PAN_RE.fullmatch(embedded_pan))
    checksum_ok = _verify_gstin_checksum(gstin) if gstin else False

    # 3. Extract Common Entity & Filing Metadata
    bad_name_tokens = (
        "of the registered", "of the", "registered person", "trade name", "if any", "arn",
        "gstin", "financial year", "tax period", "constitution", "address", "description",
        "not applicable", "n/a", "legal name", "no. of records", "document type",
    )
    legal_name = _extract_gst_field_value(
        lines,
        (r"Legal\s+Name\s+of\s+the\s+registered\s+person", r"Legal\s+Name"),
        bad_name_tokens,
    )
    trade_name = _extract_gst_field_value(
        lines,
        (r"Trade\s+Name\s*,?\s*if\s+any", r"Trade\s+Name"),
        bad_name_tokens,
    )
    arn = _extract_gst_field_value(
        lines,
        (r"\bARN\b(?!\s*date)",),
        ("arn date", "date"),
        value_filter_re=r"^[A-Z0-9]{10,20}$",
    )
    arn_date = _extract_gst_field_value(
        lines,
        (r"\b(?:ARN\s+date|Date\s+of\s+ARN|Date\s+of\s+Filing)\b",),
        ("description",),
        value_filter_re=r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}",
    )
    fin_year = _extract_gst_field_value(
        lines,
        (r"Financial\s+year", r"\bYear\b"),
        ("tax period", "return period"),
        value_filter_re=r"^\d{4}\s*-\s*\d{2,4}$",
    )
    tax_period = _extract_gst_field_value(
        lines,
        (r"Tax\s+period", r"Return\s+period", r"\bPeriod\b", r"\bMonth\b"),
        ("description", "year", "earlier tax"),
        value_filter_re=r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Q[1-4])",
    )
    signatory = _extract_gst_field_value(
        lines,
        (r"Name\s+of\s+Authorized\s+Signatory",),
        ("designation", "signature", "status"),
    )

    # REG-06 specific metadata
    constitution = _extract_gst_field_value(lines, (r"Constitution\s+of\s+Business",), ("address", "type"))
    reg_type = _extract_gst_field_value(lines, (r"Type\s+of\s+Registration",), ("date", "period"))
    validity_from = _extract_gst_field_value(
        lines,
        (r"Date\s+of\s+Liability", r"Period\s+of\s+Validity\s*[:\-]?\s*From"),
        ("to", "not applicable"),
        value_filter_re=r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}",
    )

    if legal_name:
        legal_name = legal_name.title()
    if trade_name:
        trade_name = trade_name.title()

    issues = []
    detailed_issues = []
    financial_summary: Dict[str, Any] = {}
    reconciliation_status = "NOT_APPLICABLE"

    # =========================================================================
    # 4A. FORM GSTR-1 ARITHMETIC RECONCILIATION
    # =========================================================================
    if gst_form_type == "FORM GSTR-1":
        m_tot = re.search(
            rf"Total\s+Liability\s*\([^\)]*\)[\s\S]{{0,80}}?{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
            txt,
            re.IGNORECASE,
        )
        if m_tot:
            tot_val = _parse_gst_float(m_tot.group(1))
            tot_igst = _parse_gst_float(m_tot.group(2))
            tot_cgst = _parse_gst_float(m_tot.group(3))
            tot_sgst = _parse_gst_float(m_tot.group(4))
            tot_cess = _parse_gst_float(m_tot.group(5))
            financial_summary = {
                "total_taxable_value": tot_val,
                "total_igst": tot_igst,
                "total_cgst": tot_cgst,
                "total_sgst": tot_sgst,
                "total_cess": tot_cess,
                "total_tax": round(tot_igst + tot_cgst + tot_sgst + tot_cess, 2),
            }

            cgst_sgst_ok = abs(tot_cgst - tot_sgst) <= 1.0
            if not cgst_sgst_ok:
                msg = f"GSTR-1 Tax Invariant Mismatch: Total CGST ({tot_cgst:,.2f}) != Total SGST ({tot_sgst:,.2f})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})

            # Handle cross-page section headers and flexible count/doc-type span order
            sec_specs = [
                r"4A\s*-\s*Taxable\s+outward[\s\S]{0,450}?Total\s+(?:\d+\s+Invoice|Invoice\s+\d+)\s+" + rf"{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
                r"5\s*-\s*Taxable\s+outward\s+inter-state[\s\S]{0,450}?Total\s+(?:\d+\s+Invoice|Invoice\s+\d+)\s+" + rf"{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
                r"7\s*-\s*Taxable\s+supplies[\s\S]{0,650}?Total\s+(?:\d+\s+Net\s+Value|Net\s+Value\s+\d+)\s+" + rf"{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
                r"9B\s*-\s*Credit/Debit\s+Notes\s*\(Registered\)[\s\S]{0,650}?Total\s*-\s*Net\s+off[\s\S]{0,120}?(?:\d+\s+Note|Note\s+\d+)\s+" + rf"{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
            ]
            sum_v = sum_i = sum_c = sum_s = 0.0
            matched_secs = 0
            for pat in sec_specs:
                ms = re.search(pat, txt, re.IGNORECASE)
                if ms:
                    matched_secs += 1
                    sum_v = round(sum_v + _parse_gst_float(ms.group(1)), 2)
                    sum_i = round(sum_i + _parse_gst_float(ms.group(2)), 2)
                    sum_c = round(sum_c + _parse_gst_float(ms.group(3)), 2)
                    sum_s = round(sum_s + _parse_gst_float(ms.group(4)), 2)

            if matched_secs > 0:
                val_ok = abs(sum_v - tot_val) <= 2.0
                tax_ok = (
                    abs(sum_i - tot_igst) <= 2.0
                    and abs(sum_c - tot_cgst) <= 2.0
                    and abs(sum_s - tot_sgst) <= 2.0
                )
                if val_ok and tax_ok and cgst_sgst_ok:
                    reconciliation_status = "PASSED"
                elif not val_ok or not tax_ok:
                    reconciliation_status = "FAILED"
                    msg = (
                        f"GSTR-1 Table Summation Mismatch: Sum of outward tables (Value {sum_v:,.2f}, "
                        f"IGST {sum_i:,.2f}, CGST {sum_c:,.2f}, SGST {sum_s:,.2f}) != Total Liability "
                        f"(Value {tot_val:,.2f}, IGST {tot_igst:,.2f}, CGST {tot_cgst:,.2f}, SGST {tot_sgst:,.2f})."
                    )
                    issues.append(msg)
                    detailed_issues.append({"page": 1, "bbox": None, "text": msg})
            else:
                reconciliation_status = "PASSED" if cgst_sgst_ok else "FAILED"
        else:
            reconciliation_status = "NOT_RUN"

    # =========================================================================
    # 4B. FORM GSTR-3B LIABILITY & NET ITC RECONCILIATION
    # =========================================================================
    elif gst_form_type == "FORM GSTR-3B":
        checks_passed = []

        # Table 3.1(a): Outward taxable supplies
        m_31a = re.search(
            rf"\(a\)\s*Outward\s+taxable\s+supplies[\s\S]{{0,140}}?{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
            txt,
            re.IGNORECASE,
        )
        if m_31a:
            out_val = _parse_gst_float(m_31a.group(1))
            out_igst = _parse_gst_float(m_31a.group(2))
            out_cgst = _parse_gst_float(m_31a.group(3))
            out_sgst = _parse_gst_float(m_31a.group(4))
            out_cess = _parse_gst_float(m_31a.group(5))
            financial_summary["outward_taxable_supplies"] = {
                "taxable_value": out_val,
                "igst": out_igst,
                "cgst": out_cgst,
                "sgst": out_sgst,
                "cess": out_cess,
            }
            if abs(out_cgst - out_sgst) <= 1.0:
                checks_passed.append(True)
            else:
                checks_passed.append(False)
                msg = f"GSTR-3B Table 3.1(a) Tax Mismatch: CGST ({out_cgst:,.2f}) != SGST ({out_sgst:,.2f})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        # Table 4(C): Net ITC Available (IGST, CGST, SGST, Cess)
        m_4c = re.search(
            rf"\(C\)\s*Net\s+ITC\s+Available[\s\S]{{0,100}}?{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
            txt,
            re.IGNORECASE,
        )
        if m_4c:
            net_itc = {
                "igst": _parse_gst_float(m_4c.group(1)),
                "cgst": _parse_gst_float(m_4c.group(2)),
                "sgst": _parse_gst_float(m_4c.group(3)),
                "cess": _parse_gst_float(m_4c.group(4)),
            }
            financial_summary["net_itc_available"] = net_itc
            if abs(net_itc["cgst"] - net_itc["sgst"]) <= 2.0:
                checks_passed.append(True)
            else:
                checks_passed.append(False)
                msg = f"GSTR-3B Table 4(C) Net ITC Mismatch: CGST ({net_itc['cgst']:,.2f}) != SGST ({net_itc['sgst']:,.2f})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})

        reconciliation_status = (
            "FAILED" if (checks_passed and not all(checks_passed))
            else ("PASSED" if checks_passed else "NOT_RUN")
        )

    # =========================================================================
    # 4C. FORM GSTR-2B ELIGIBLE ITC STATEMENT VERIFICATION
    # =========================================================================
    elif gst_form_type == "FORM GSTR-2B":
        m_2b_itc = re.search(
            rf"(?:All\s+other\s+ITC|Part\s*A[\s\S]{{0,160}}?Total)[\s\S]{{0,140}}?{amt_pat}\s+{amt_pat}\s+{amt_pat}\s+{amt_pat}",
            txt,
            re.IGNORECASE,
        )
        if m_2b_itc:
            itc_igst = _parse_gst_float(m_2b_itc.group(1))
            itc_cgst = _parse_gst_float(m_2b_itc.group(2))
            itc_sgst = _parse_gst_float(m_2b_itc.group(3))
            itc_cess = _parse_gst_float(m_2b_itc.group(4))
            financial_summary["eligible_itc_part_a"] = {
                "igst": itc_igst,
                "cgst": itc_cgst,
                "sgst": itc_sgst,
                "cess": itc_cess,
                "total_eligible_itc": round(itc_igst + itc_cgst + itc_sgst + itc_cess, 2),
            }
            if abs(itc_cgst - itc_sgst) <= 2.0:
                reconciliation_status = "PASSED"
            else:
                reconciliation_status = "FAILED"
                msg = f"GSTR-2B Part A ITC Mismatch: Eligible CGST ({itc_cgst:,.2f}) != Eligible SGST ({itc_sgst:,.2f})."
                issues.append(msg)
                detailed_issues.append({"page": 1, "bbox": None, "text": msg})
        else:
            reconciliation_status = "NOT_RUN"

    # =========================================================================
    # 4D. FORM GST REG-06 REGISTRATION CERTIFICATE VERIFICATION
    # =========================================================================
    elif gst_form_type == "FORM GST REG-06":
        financial_summary["registration_details"] = {
            "constitution_of_business": constitution,
            "registration_type": reg_type,
            "validity_from": validity_from,
        }
        reconciliation_status = "PASSED" if (gstin and checksum_ok and legal_name) else "FAILED"

    # 5. Core GSTIN Integrity Checks (Applicable to all 4 forms)
    if not gstin:
        msg = "GST Warning: Valid 15-character GSTIN could not be extracted."
        issues.append(msg)
        detailed_issues.append({"page": 1, "bbox": None, "text": msg})
    else:
        if not state_ok:
            msg = f"GST State Code Anomaly: Prefix '{state_code}' is not a valid Indian GST state code."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})
        if not pan_ok:
            msg = f"GST Embedded PAN Anomaly: Characters 3-12 ('{embedded_pan}') do not form a valid Indian PAN."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})
        if not checksum_ok:
            msg = f"GSTIN Checksum Failure: 15th character of '{gstin}' fails the official Mod-36 GSTIN checksum."
            issues.append(msg)
            detailed_issues.append({"page": 1, "bbox": None, "text": msg})

    audit_coverage = {
        "gstin_extracted": "PASSED" if gstin else "FAILED",
        "gstin_checksum_verified": ("PASSED" if checksum_ok else "FAILED") if gstin else "NOT_RUN",
        "state_code_valid": ("PASSED" if state_ok else "FAILED") if gstin else "NOT_RUN",
        "embedded_pan_valid": ("PASSED" if pan_ok else "FAILED") if gstin else "NOT_RUN",
        "legal_name_extracted": "PASSED" if legal_name else "FAILED",
        "gst_form_reconciled": reconciliation_status,
    }

    return {
        "document_type": "GST_CERTIFICATE",
        "gst_form_type": gst_form_type,
        "form_display_label": form_display_label,
        "status": "REVIEW_REQUIRED" if issues else "NO_MISMATCH_FOUND",
        "gstin": gstin,
        "embedded_pan": embedded_pan,
        "pan_number": embedded_pan,
        "state_code": state_code,
        "jurisdiction": f"{state_code} - {state_name}" if state_name else "Unknown",
        "legal_name": legal_name or "Unknown",
        "trade_name": trade_name,
        "arn": arn,
        "arn_date": arn_date,
        "financial_year": fin_year,
        "tax_period": tax_period,
        "authorized_signatory": signatory.title() if signatory else None,
        "constitution_of_business": constitution,
        "gst_financial_summary": financial_summary,
        "audit_coverage": audit_coverage,
        "issues": issues,
        "detailed_issues": detailed_issues,
    }


def verify_udyam_certificate(full_text: str) -> Dict[str, Any]:
    txt = full_text or ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    issues = []

    udyam_match = UDYAM_RE.search(txt.upper())
    udyam_no = udyam_match.group(1) if udyam_match else None

    pan_match = PAN_RE.search(txt.upper())
    pan_no = pan_match.group(1) if pan_match else None

    ent_type = None
    for t in ["MICRO", "SMALL", "MEDIUM"]:
        if re.search(rf"\b{t}\b", txt, re.IGNORECASE):
            ent_type = t.title()
            break

    ent_name = _extract_gst_field_value(
        lines,
        (r"NAME\s+OF\s+ENTERPRISE",),
        ("major activity", "type of enterprise", "organisation", "udyam", "social category"),
    )
    if ent_name:
        ent_name = ent_name.title()

    audit_coverage = {
        "udyam_number_valid": "PASSED" if udyam_no is not None else "FAILED",
        "enterprise_type_identified": "PASSED" if ent_type is not None else "FAILED",
        "enterprise_name_extracted": "PASSED" if ent_name is not None else "FAILED",
    }

    if not udyam_no:
        issues.append({"page": 1, "bbox": None, "text": "Udyam Certificate Error: Valid UDYAM-XX-00-0000000 registration number not found."})
    if not ent_type:
        issues.append({"page": 1, "bbox": None, "text": "Udyam Extraction Warning: Enterprise classification (Micro/Small/Medium) could not be verified."})

    return {
        "document_type": "UDYAM_CERTIFICATE",
        "status": "REVIEW_REQUIRED" if issues else "NO_MISMATCH_FOUND",
        "audit_coverage": audit_coverage,
        "udyam_registration_number": udyam_no,
        "enterprise_name": ent_name,
        "legal_name": ent_name,
        "enterprise_category": ent_type,
        "pan_number": pan_no,
        "issues": [i["text"] for i in issues],
        "detailed_issues": issues,
    }


def verify_itr_document(full_text: str) -> Dict[str, Any]:
    txt = full_text or ""
    issues = []

    pan_match = PAN_RE.search(txt.upper())
    pan_no = pan_match.group(1) if pan_match else None

    m_ack_lbl = re.search(r"acknowledgement\s*(?:number|no\.?)\s*[:\-]?\s*(\d{15})\b", txt, re.IGNORECASE)
    ack_match = m_ack_lbl or ITR_ACK_RE.search(txt)
    ack_no = ack_match.group(1) if ack_match else None

    ay_match = re.search(r"assessment\s*year\s*[:\-]?\s*(\d{4}\s*-\s*\d{2,4})", txt, re.IGNORECASE)
    assessment_year = ay_match.group(1).replace(" ", "") if ay_match else None

    def extract_field_val(pattern: str) -> Optional[float]:
        m = re.search(pattern, txt, re.IGNORECASE)
        return parse_indian_number(m.group(1)) if m else None

    gross_total_income = extract_field_val(r"gross\s*total\s*income[^\n\d]*([\d,]+(?:\.\d{1,2})?)")
    deductions_via = extract_field_val(r"(?:deductions?\s*under\s*chapter\s*vi-?a|total\s*deductions)[^\n\d]*([\d,]+(?:\.\d{1,2})?)") or 0.0
    total_taxable_income = extract_field_val(r"(?:total\s*taxable\s*income|total\s*income)[^\n\d]*([\d,]+(?:\.\d{1,2})?)")
    net_tax_payable = extract_field_val(r"(?:net\s*tax\s*payable|total\s*tax\s*payable)[^\n\d]*([\d,]+(?:\.\d{1,2})?)")

    has_income = (gross_total_income is not None and total_taxable_income is not None)
    audit_coverage = {
        "pan_extracted": "PASSED" if pan_no is not None else "FAILED",
        "income_figures_extracted": "PASSED" if has_income else "FAILED",
        "tax_equation_verified": "NOT_RUN",
    }

    if not pan_no:
        issues.append({"page": 1, "bbox": None, "text": "ITR Error: Valid 10-character PAN not found on ITR document."})

    if has_income:
        expected_taxable = max(0.0, round(gross_total_income - deductions_via, 2))
        if abs(expected_taxable - total_taxable_income) > 15.0:
            audit_coverage["tax_equation_verified"] = "FAILED"
            issues.append({
                "page": 1,
                "bbox": None,
                "text": (
                    f"ITR Computation Mismatch: Gross Total Income ({gross_total_income:,.2f}) - "
                    f"Chapter VI-A Deductions ({deductions_via:,.2f}) = {expected_taxable:,.2f}, "
                    f"but Total Taxable Income shows {total_taxable_income:,.2f}."
                ),
            })
        else:
            audit_coverage["tax_equation_verified"] = "PASSED"
    else:
        issues.append({
            "page": 1,
            "bbox": None,
            "text": f"ITR Extraction Warning: Could not extract both Gross Total Income ({gross_total_income}) and Total Taxable Income ({total_taxable_income}).",
        })

    return {
        "document_type": "ITR_DOCUMENT",
        "status": "REVIEW_REQUIRED" if issues else "NO_MISMATCH_FOUND",
        "audit_coverage": audit_coverage,
        "pan_number": pan_no,
        "acknowledgement_number": ack_no,
        "assessment_year": assessment_year,
        "gross_total_income": gross_total_income,
        "deductions_chapter_via": deductions_via,
        "total_taxable_income": total_taxable_income,
        "net_tax_payable": net_tax_payable,
        "issues": [i["text"] for i in issues],
        "detailed_issues": issues,
    }


def verify_balance_sheet(full_text: str) -> Dict[str, Any]:
    txt = full_text or ""
    issues = []

    def extract_bs_line(pattern: str) -> Optional[float]:
        m = re.search(pattern, txt, re.IGNORECASE)
        return parse_indian_number(m.group(1)) if m else None

    equity = extract_bs_line(r"(?:shareholders'?\s*funds|total\s*equity|capital\s*account)[^\n\d]*([\d,]+(?:\.\d{1,2})?)")
    liabilities = extract_bs_line(r"(?:total\s*liabilities|current\s*and\s*non-current\s*liabilities)[^\n\d]*([\d,]+(?:\.\d{1,2})?)")
    total_eq_and_liab = (
        extract_bs_line(r"total\s*(?:equity|capital)\s*(?:and|&)\s*liabilities[^\n\d]*([\d,]+(?:\.\d{1,2})?)")
        or extract_bs_line(r"total\s*liabilities[^\n\d]*([\d,]+(?:\.\d{1,2})?)")
    )
    total_assets = extract_bs_line(r"total\s*assets[^\n\d]*([\d,]+(?:\.\d{1,2})?)")

    udin_match = UDIN_RE.search(txt.upper())
    udin = udin_match.group(1) if udin_match else None

    if total_eq_and_liab is None and equity is not None and liabilities is not None:
        total_eq_and_liab = round(equity + liabilities, 2)

    both_extracted = (total_assets is not None and total_eq_and_liab is not None)
    audit_coverage = {
        "total_assets_extracted": "PASSED" if total_assets is not None else "FAILED",
        "total_liabilities_equity_extracted": "PASSED" if total_eq_and_liab is not None else "FAILED",
        "accounting_equation_verified": "NOT_RUN",
    }

    if both_extracted:
        eq_failed = False
        if equity is not None and liabilities is not None and liabilities != total_eq_and_liab:
            if abs((equity + liabilities) - total_eq_and_liab) > 1.0:
                eq_failed = True
                issues.append({
                    "page": 1,
                    "bbox": None,
                    "text": f"Balance Sheet Liability Mismatch: Equity ({equity:,.2f}) + Liabilities ({liabilities:,.2f}) != Total Equity & Liabilities ({total_eq_and_liab:,.2f}).",
                })

        if abs(total_assets - total_eq_and_liab) > 1.0:
            eq_failed = True
            issues.append({
                "page": 1,
                "bbox": None,
                "text": f"Balance Sheet Fundamental Equation Broken: Total Assets ({total_assets:,.2f}) != Total Equity & Liabilities ({total_eq_and_liab:,.2f}).",
            })
        audit_coverage["accounting_equation_verified"] = "FAILED" if eq_failed else "PASSED"
    else:
        issues.append({
            "page": 1,
            "bbox": None,
            "text": f"Balance Sheet Extraction Warning: Could not locate both Total Assets ({total_assets}) and Total Equity & Liabilities ({total_eq_and_liab}).",
        })

    return {
        "document_type": "BALANCE_SHEET",
        "status": "REVIEW_REQUIRED" if issues else "NO_MISMATCH_FOUND",
        "audit_coverage": audit_coverage,
        "shareholders_equity": equity,
        "total_liabilities": liabilities,
        "total_equity_and_liabilities": total_eq_and_liab,
        "total_assets": total_assets,
        "ca_udin": udin,
        "issues": [i["text"] for i in issues],
        "detailed_issues": issues,
    }


def format_inr_compact(val: Optional[float]) -> str:
    """Formats raw rupees into compact Indian notation (Cr / L / ₹) for UI cards."""
    if val is None:
        return "Not Found"
    abs_v = abs(val)
    if abs_v >= 1e7:
        return f"₹ {val / 1e7:.2f} Cr"
    if abs_v >= 1e5:
        return f"₹ {val / 1e5:.2f} L"
    return f"₹ {val:,.0f}"


def extract_complete_auditor_observations(full_text: str) -> list[str]:
    """
    Extracts complete, full-sentence statutory auditor observations from Form 3CB
    without mid-sentence line-wrap clipping or Form 3CD boilerplate noise.
    """
    form_3cd_split = re.split(r"FORM\s*NO\.?\s*3CD", full_text, maxsplit=1, flags=re.IGNORECASE)
    form_3cb_text = form_3cd_split[0] if form_3cd_split else full_text[:4000]

    reflowed_3cb = re.sub(r"[ \t]+", " ", form_3cb_text)
    reflowed_3cb = re.sub(r"(?<![.:\n])\n+(?=[a-z0-9(,\-])", " ", reflowed_3cb)
    reflowed_all = re.sub(r"[ \t]+", " ", full_text)
    reflowed_all = re.sub(r"(?<![.:\n])\n+(?=[a-z0-9(,\-])", " ", reflowed_all)

    boilerplate_markers = [
        "account payee bank draft",
        "account payee cheque",
        "section 44ab",
        "whether the assessee",
        "particulars of each loan",
        "in our opinion and to the best of our information",
        "proper books of account",
        "true and fair view",
    ]

    def is_valid_observation_sentence(sent: str) -> bool:
        s_low = sent.lower().strip()
        if len(s_low) < 35:
            return False
        if re.match(r"^(?:and|or|bank|to|from|of|in|for|with|by)\b", s_low):
            return False
        if any(bp in s_low for bp in boilerplate_markers):
            return False
        return True

    observations = []
    topic_rules = [
        (
            "Supporting Documents & Vouchers",
            r"([A-Z][^.\n]{20,260}?(?:supporting|vouchers?|petty\s+expenses|self[\s\-]*made|bills\s+or\s+vouchers)[^.\n]{0,160}?\.)",
        ),
        (
            "Balance Confirmations (Debtors / Creditors / Loans)",
            r"([A-Z][^.\n]{20,260}?(?:balances?\s+of|subject\s+to\s+confirmation|sundry\s+debtors?|sundry\s+creditors?|unsecured\s+loans?|confirmations?\s+were\s+not)[^.\n]{0,160}?\.)",
        ),
        (
            "Inventory & Stock Valuation",
            r"([A-Z][^.\n]{20,260}?(?:closing\s+stock|inventory|physically\s+verified|valued\s+and\s+verified|relied\s+upon)[^.\n]{0,160}?\.)",
        ),
        (
            "Loans / Deposits & Repayment Verification (Sec 269SS / 269T)",
            r"([A-Z][^.\n]{20,260}?(?:loans?\s+or\s+deposits?|accepted\s+or\s+repaid|269ss|269t|squared\s+up|entries\s+in\s+the\s+books)[^.\n]{0,160}?\.)",
        ),
    ]

    for label, pattern in topic_rules:
        found_sent = None
        for source_txt in (reflowed_3cb, reflowed_all):
            for m in re.finditer(pattern, source_txt):
                candidate = re.sub(r"\s+", " ", m.group(1)).strip()
                candidate = re.sub(r"\s+,", ",", candidate)
                if is_valid_observation_sentence(candidate):
                    found_sent = candidate
                    break
            if found_sent:
                break
        if found_sent:
            observations.append(f"{label}: {found_sent}")

    return observations


def verify_tax_audit_report(full_text: str, filename: str = "") -> Dict[str, Any]:
    """
    Generic Form 3CB / Form 3CD Financial & Compliance Engine (zero hardcoded entity defaults).
    """
    txt = full_text or ""
    issues = []
    detailed_issues = []

    forms_found = []
    if re.search(r"form\s*(?:no\.?\s*)?3cb", txt, re.IGNORECASE):
        forms_found.append("3CB")
    if re.search(r"form\s*(?:no\.?\s*)?3cd", txt, re.IGNORECASE):
        forms_found.append("3CD")

    # 1. Identity Extraction (Assessee vs Auditor)
    assessee = None
    ass_patterns = [
        r"name\s*of\s*the\s*assessee\s*[:\-]?\s*([A-Z][A-Za-z0-9\ \t\.\&]{2,50}?)(?=\n|\r|$|2\.|address|pan|status)",
        r"in\s*the\s*case\s*of\s*(?:M/S\.?\s*)?([A-Z][A-Za-z0-9\ \t\.\&]{3,50}?)(?=\n|\r|,|pan|prop)",
        r"([A-Z][A-Z0-9\ \t]{2,35}(?:ENTERPRISES|TRADERS|INDUSTRIES|INFRA|EXPORTS|ASSOCIATES|PVT\.?\s*LTD|LIMITED))",
    ]
    for pat in ass_patterns:
        for m in re.finditer(pat, txt, re.IGNORECASE):
            cand = m.group(1).strip()
            if not any(w in cand.lower() for w in ["chartered", "accountant", "audit", "person referred"]):
                assessee = cand.title()
                break
        if assessee:
            break

    proprietor = None
    m_prop = re.search(r"(?:proprietor|prop\.?)\s*[:\-\(\)]*\s*([A-Za-z\ \t\.]{3,35}?)(?=\)|\n|\r|$|,|pan)", txt, re.IGNORECASE)
    if m_prop:
        proprietor = m_prop.group(1).strip().title()

    auditor_firm = None
    m_aud = re.search(r"(?:For\s+)?([A-Za-z][A-Za-z\ \t\.]{2,40}?\s*&\s*(?:Associates|Co\.?|Company))", txt, re.IGNORECASE)
    if m_aud:
        auditor_firm = m_aud.group(1).strip().title()

    signatory = None
    m_sig = re.search(r"(?:CA\.?\s+|Partner|Proprietor)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", txt)
    if m_sig:
        signatory = m_sig.group(1).strip()

    pan_match = PAN_RE.search(txt.upper())
    pan_no = pan_match.group(1) if pan_match else None

    ay_match = re.search(r"assessment\s*year\s*[:\-]?\s*(\d{4}\s*-\s*\d{2,4})", txt, re.IGNORECASE)
    assessment_year = ay_match.group(1).replace(" ", "") if ay_match else None

    py_match = re.search(r"previous\s*year\s*[:\-]?\s*(\d{4}\s*-\s*\d{2,4})", txt, re.IGNORECASE)
    previous_year = py_match.group(1).replace(" ", "") if py_match else None

    # 2. Clause 40 Extraction: Total Turnover, Gross Profit, Net Profit
    turnover = None
    gross_profit = None
    net_profit = None

    m_turn = re.search(
        r"(?:total\s*turnover|gross\s*turnover|turnover\s*of\s*the\s*assessee)[\s\S]{0,120}?(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d{6,9}(?:\.\d{1,2})?)",
        txt,
        re.IGNORECASE,
    )
    if m_turn:
        turnover = parse_indian_number(m_turn.group(1))

    m_gp = re.search(
        r"gross\s*profit\s*(?:/\s*turnover)?[\s\S]{0,100}?(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d{5,8}(?:\.\d{1,2})?)",
        txt,
        re.IGNORECASE,
    )
    if m_gp:
        gross_profit = parse_indian_number(m_gp.group(1))

    m_np = re.search(
        r"net\s*profit\s*(?:/\s*turnover)?[\s\S]{0,100}?(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d{5,8}(?:\.\d{1,2})?)",
        txt,
        re.IGNORECASE,
    )
    if m_np:
        net_profit = parse_indian_number(m_np.group(1))

    if turnover is None or gross_profit is None or net_profit is None:
        c40_pos = txt.lower().find("40.")
        if c40_pos == -1:
            c40_pos = txt.lower().find("turnover")
        if c40_pos != -1:
            c40_slice = txt[c40_pos : c40_pos + 1800]
            large_nums = [
                parse_indian_number(x)
                for x in re.findall(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\b|\b\d{5,9}(?:\.\d{1,2})?\b", c40_slice)
            ]
            large_nums = [n for n in large_nums if n and n > 50000 and n not in (1961, 2023, 2024, 2025, 2026)]
            if large_nums:
                sorted_desc = sorted(set(large_nums), reverse=True)
                if turnover is None and len(sorted_desc) >= 1:
                    turnover = sorted_desc[0]
                if gross_profit is None and len(sorted_desc) >= 2:
                    gross_profit = sorted_desc[1]
                if net_profit is None and len(sorted_desc) >= 3:
                    net_profit = sorted_desc[2]

    # 3. Generic Clause 18 / Annexure A Depreciation WDV Check
    dep_opening_or_total = None
    dep_allowable = None
    dep_closing_wdv = None
    dep_math_status = "NOT_RUN"

    dep_pos = txt.lower().find("depreciation allowable")
    if dep_pos != -1:
        dep_slice = txt[dep_pos : dep_pos + 2000]
        dep_nums = [parse_indian_number(x) for x in re.findall(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\b", dep_slice)]
        dep_nums = [n for n in dep_nums if n and n > 500]
        if len(dep_nums) >= 3:
            for i in range(len(dep_nums) - 2):
                a, b, c = dep_nums[i], dep_nums[i + 1], dep_nums[i + 2]
                if a > b > 0 and c > 0 and abs((a - b) - c) <= 2.0:
                    dep_opening_or_total, dep_allowable, dep_closing_wdv = a, b, c
                    dep_math_status = "PASSED"
                    break

    # 4. Complete Auditor Observations
    observation_items = extract_complete_auditor_observations(txt)
    auditor_qualifications = (
        " | ".join(observation_items)
        if observation_items
        else "No separate Form 3CB Para 3(a) qualifications extracted"
    )

    # 5. Filename vs Document Intake Hygiene Checks
    filename_period_status = "PASSED"
    filename_doctype_status = "PASSED"

    if filename:
        fn_low = filename.lower()
        fn_fy_match = re.search(r"FY[_\-\s]*(\d{4})[_\-\s]*(\d{2,4})", filename, re.IGNORECASE)
        if fn_fy_match and previous_year:
            fn_start_yr = fn_fy_match.group(1)
            doc_py_start = previous_year[:4]
            if doc_py_start and fn_start_yr != doc_py_start:
                filename_period_status = "FAILED"
                msg_period = (
                    f"Filename Period Discrepancy: Uploaded file is named '{filename}' (indicating FY {fn_start_yr}-{fn_fy_match.group(2)[-2:]}), "
                    f"whereas the document internally states Previous Year (FY) {previous_year} and Assessment Year (AY) {assessment_year or 'N/A'}."
                )
                issues.append(msg_period)
                detailed_issues.append({
                    "page": 1,
                    "bbox": None,
                    "text": msg_period,
                    "finding_code": "FILENAME_PERIOD_DISCREPANCY",
                    "category": "INTAKE_METADATA",
                })

        label_map = [
            (["balance_sheet", "balancesheet", "bal_sheet"], "Balance Sheet"),
            (["bank_statement", "bankstatement", "acct_stmt"], "Bank Statement"),
            (["salary_slip", "payslip", "pay_slip"], "Salary Slip"),
            (["gst_cert", "gstin"], "GST Certificate"),
            (["udyam"], "Udyam Certificate"),
        ]
        for kw_list, claimed_label in label_map:
            if any(k in fn_low for k in kw_list):
                filename_doctype_status = "FAILED"
                msg_type = (
                    f"Filename Document Type Discrepancy: Uploaded file is labeled '{filename}' (suggesting a {claimed_label}), "
                    f"whereas the document content is a Tax Audit Report (Form {forms_found[0] if forms_found else '3CB'}/{forms_found[-1] if forms_found else '3CD'})."
                )
                issues.append(msg_type)
                detailed_issues.append({
                    "page": 1,
                    "bbox": None,
                    "text": msg_type,
                    "finding_code": "FILENAME_DOCUMENT_TYPE_DISCREPANCY",
                    "category": "INTAKE_METADATA",
                })
                break

    turnover_extracted_ok = (turnover is not None and gross_profit is not None and net_profit is not None)
    if not turnover_extracted_ok:
        msg = (
            f"Clause 40 Extraction Incomplete: Turnover ({turnover}), Gross Profit ({gross_profit}), "
            f"or Net Profit ({net_profit}) could not be fully verified against source."
        )
        issues.append(msg)
        detailed_issues.append({"page": 1, "bbox": None, "text": msg, "category": "ARITHMETIC_LOGIC"})

    audit_coverage = {
        "assessee_identity": "PASSED" if (assessee and pan_no) else "FAILED",
        "form_3cb_3cd_recognition": "PASSED" if forms_found else "FAILED",
        "turnover_and_profit_extraction": "PASSED" if turnover_extracted_ok else "FAILED",
        "depreciation_arithmetic": dep_math_status,
        "bank_ledger_reconciliation": "NOT_APPLICABLE",
        "itr_taxable_income_extraction": "NOT_APPLICABLE",
    }

    intake_metadata_checks = {
        "filename_period_consistency": filename_period_status,
        "filename_document_type_consistency": filename_doctype_status,
    }

    has_math_failures = any(v in ("FAILED", "NOT_RUN") for v in audit_coverage.values() if v != "NOT_APPLICABLE")
    has_intake_failures = any(v == "FAILED" for v in intake_metadata_checks.values())
    has_any_issues = has_math_failures or has_intake_failures or bool(issues)

    return {
        "document_type": "TAX_AUDIT_REPORT",
        "status": "REVIEW_REQUIRED" if has_any_issues else "NO_MISMATCH_FOUND",
        "audit_coverage": audit_coverage,
        "intake_metadata_checks": intake_metadata_checks,
        "forms_detected": " and ".join(forms_found) if forms_found else "3CB and 3CD",
        "assessee_business_name": assessee or "Unknown Assessee",
        "proprietor_name": proprietor,
        "auditor_firm_name": auditor_firm,
        "auditor_signatory": signatory,
        "auditor_qualifications": auditor_qualifications,
        "auditor_observations_list": observation_items,
        "pan_number": pan_no,
        "assessment_year": assessment_year,
        "previous_year": previous_year,
        "total_turnover": turnover,
        "total_turnover_display": format_inr_compact(turnover),
        "gross_profit": gross_profit,
        "gross_profit_display": format_inr_compact(gross_profit),
        "net_profit": net_profit,
        "net_profit_display": format_inr_compact(net_profit),
        "depreciation_opening_block": dep_opening_or_total,
        "depreciation_allowable": dep_allowable,
        "depreciation_closing_wdv": dep_closing_wdv,
        "issues": issues,
        "detailed_issues": detailed_issues,
    }