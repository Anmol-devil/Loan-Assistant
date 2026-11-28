import os
import re
from typing import Any, Dict, List, Optional

PAN_RE = re.compile(r"\b([A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z])\b")
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")
UDYAM_RE = re.compile(r"\bUDYAM-[A-Z]{2}-\d{2}-\d{7}\b", re.IGNORECASE)
AADHAAR_12_RE = re.compile(r"(?<![0-9/])\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b(?![0-9/])")

from core.bank_metadata import BANK_NAME_RE, UNIVERSAL_IFSC_BANK_MAP as BANK_IFSC_MAP

def _detect_issuing_bank_hint(top_text: str, filename: str = "") -> Optional[str]:
    up = (top_text or "").upper()
    m_ifsc = re.search(r"IFSC(?:\s*CODE)?[\s:\|\-]*([A-Z]{4})[0O][A-Z0-9]{6}\b", up)
    if m_ifsc and m_ifsc.group(1) in BANK_IFSC_MAP:
        return BANK_IFSC_MAP[m_ifsc.group(1)]

    m_bank = BANK_NAME_RE.search(top_text or "")
    if m_bank:
        return re.sub(r"\s+", " ", m_bank.group(1)).strip().title()

    fn_low = (filename or "").lower()
    for code, bank_name in BANK_IFSC_MAP.items():
        first_word = bank_name.lower().split()[0]
        if first_word in fn_low and len(first_word) >= 4:
            return bank_name
    if re.search(r"\bsbi\b", fn_low):
        return "State Bank of India"
    if re.search(r"\bpnb\b", fn_low):
        return "Punjab National Bank"
    if re.search(r"\bbob\b", fn_low):
        return "Bank of Baroda"
    return None


def _detect_layout_variant(
    doc_code: str,
    file_ext: str,
    page_count: int,
    page_aspect_ratio: Optional[float],
    text_len: int,
    low_text: str,
) -> str:
    """
    Distinguishes full A4 e-Aadhaar / e-PAN documents from cropped PVC/mobile card scans.
    page_aspect_ratio = height / width (A4 portrait ~ 1.41, landscape ID card ~ 0.63).
    """
    if doc_code == "AADHAAR_CARD":
        if file_ext == ".pdf" and (
            (page_aspect_ratio and page_aspect_ratio >= 1.15)
            or any(k in low_text for k in ("enrolment no", "enrollment no", "download date", "information", "virtual id", "vid :"))
        ):
            return "FULL_E_DOCUMENT"
        if page_aspect_ratio and page_aspect_ratio <= 0.48:
            return "COMBINED_FRONT_BACK_CARD"
        return "ID_CARD_SCAN"

    if doc_code == "PAN_CARD":
        if file_ext == ".pdf" and (
            (page_aspect_ratio and page_aspect_ratio >= 1.15)
            or any(k in low_text for k in ("cut here", "e-pan", "electronic permanent account", "protean", "nsdl", "utiitsl"))
            or text_len > 650
        ):
            return "FULL_E_DOCUMENT"
        return "ID_CARD_SCAN"

    if doc_code == "BANK_STATEMENT":
        return "MULTI_PAGE_LEDGER" if page_count > 1 else "SINGLE_PAGE_LEDGER"

    return "STANDARD_DOCUMENT"


def detect_document_type(
    full_text: str,
    filename: str = "",
    file_ext: str = ".pdf",
    page_count: int = 1,
    page_aspect_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Weighted multi-signal Indian financial & KYC document detector.
    Prevents bank narrations (e.g., 'GST @18%') from misclassifying bank statements,
    and cleanly separates Full e-PAN / e-Aadhaar PDFs from card photos/scans.
    """
    txt = full_text or ""
    top_text = txt[:3000]
    low = txt.lower()
    top_low = top_text.lower()
    up = txt.upper()
    fn_low = (filename or "").lower()

    scores: Dict[str, float] = {
        "BANK_STATEMENT": 0.0,
        "PAN_CARD": 0.0,
        "AADHAAR_CARD": 0.0,
        "GST_CERTIFICATE": 0.0,
        "UDYAM_CERTIFICATE": 0.0,
        "TAX_AUDIT_REPORT": 0.0,
        "ITR_DOCUMENT": 0.0,
        "BALANCE_SHEET": 0.0,
        "SALARY_SLIP": 0.0,
    }
    sub_type: Optional[str] = None
    reasons: List[str] = []

    # 1. BANK STATEMENT SIGNALS (15+ Indian banks)
    bank_hint = _detect_issuing_bank_hint(top_text, filename)
    if bank_hint:
        scores["BANK_STATEMENT"] += 0.25
        reasons.append(f"bank_hint:{bank_hint}")
    if any(k in top_low for k in ("statement of account", "account statement", "detailed statement", "statement of axis", "statement period")):
        scores["BANK_STATEMENT"] += 0.45
    table_cols = sum(1 for k in ("opening balance", "closing balance", "withdrawal", "withdrawl", "deposit", "debit", "credit", "particulars", "narration", "running balance", "chq", "value date", "tran date") if k in low)
    if table_cols >= 3:
        scores["BANK_STATEMENT"] += 0.40
    if sum(1 for k in ("upi/", "neft", "rtgs", "imps", "ach-dr", "atm-cash") if k in low) >= 2:
        scores["BANK_STATEMENT"] += 0.25
    if any(k in fn_low for k in ("bank_statement", "bank statement", "acct_stmt", "transactionhistory", "optransaction")):
        scores["BANK_STATEMENT"] += 0.20

    # 2. AADHAAR (BOTH FULL E-AADHAAR LETTER & PVC/MOBILE CARD SCAN)
    if any(k in low for k in ("unique identification authority", "uidai", "mera aadhaar", "मेरा आधार", "aadhaar is proof of identity", "proof of identity, not of citizenship")):
        scores["AADHAAR_CARD"] += 0.75
    if any(k in low for k in ("aadhaar", "aadhar", "आधार", "enrolment no", "enrollment no", "vid :")):
        scores["AADHAAR_CARD"] += 0.30
    if AADHAAR_12_RE.search(txt) and any(k in low for k in ("government of india", "govt of india", "dob", "male", "female", "address")):
        scores["AADHAAR_CARD"] += 0.35
    if any(k in fn_low for k in ("aadhaar", "aadhar", "uidai", "eaadhaar")):
        scores["AADHAAR_CARD"] += 0.20

    # 3. PAN (BOTH FULL E-PAN PDF & OLD/NEW QR CARD SCAN)
    if "income tax department" in low or "permanent account number" in low or "आयकर विभाग" in txt:
        scores["PAN_CARD"] += 0.65
    if PAN_RE.search(up):
        scores["PAN_CARD"] += 0.20
    if any(k in fn_low for k in ("pan_card", "pancard", "pan card", "e-pan", "epan")):
        scores["PAN_CARD"] += 0.20
    # Penalize PAN_CARD if this is actually an ITR, Form 3CD, GST, or Bank Statement carrying a PAN
    if any(k in low for k in ("itr-v", "gross total income", "form no. 3cd", "form gstr", "statement of account", "opening balance")):
        scores["PAN_CARD"] -= 0.55

    # 4. TAX AUDIT REPORT (FORM 3CB / 3CD)
    if any(k in low for k in ("form no. 3cb", "form no. 3cd", "form 3cb", "form 3cd", "section 44ab", "rule 6g")):
        scores["TAX_AUDIT_REPORT"] += 0.90
        sub_type = "FORM_3CB_3CD"

    # 5. GST PORTAL DOCUMENTS (REG-06, GSTR-1, GSTR-3B, GSTR-2B)
    if re.search(r"\bform\s+gstr[\-\s]*3b\b|\bgstr[\-\s]*3b\b", top_low):
        scores["GST_CERTIFICATE"] += 0.92
        sub_type = "FORM GSTR-3B"
    elif re.search(r"\bform\s+gstr[\-\s]*2b\b|\bgstr[\-\s]*2b\b|auto[\-\s]*drafted\s+itc\s+statement", top_low):
        scores["GST_CERTIFICATE"] += 0.92
        sub_type = "FORM GSTR-2B"
    elif re.search(r"\bform\s+gstr[\-\s]*1\b|\bgstr[\-\s]*1\b|details\s+of\s+outward\s+supplies", top_low):
        scores["GST_CERTIFICATE"] += 0.92
        sub_type = "FORM GSTR-1"
    elif "gst reg-06" in top_low or ("goods and services tax" in top_low and GSTIN_RE.search(up)):
        scores["GST_CERTIFICATE"] += 0.90
        sub_type = "FORM GST REG-06"

    # 6. UDYAM MSME CERTIFICATE
    if "udyam registration" in low or "ministry of micro, small and medium" in low or UDYAM_RE.search(txt):
        scores["UDYAM_CERTIFICATE"] += 0.90

    # 7. ITR DOCUMENT
    if any(k in low for k in ("indian income tax return", "itr-v", "acknowledgement number", "gross total income")):
        scores["ITR_DOCUMENT"] += 0.85

    # 8. BALANCE SHEET
    if sum(1 for k in ("balance sheet as at", "equity and liabilities", "shareholders' funds", "non-current assets", "sundry debtors", "total assets") if k in low) >= 2:
        scores["BALANCE_SHEET"] += 0.85

    # 9. SALARY SLIP
    sal_hits = sum(1 for k in ("earnings", "deductions", "net pay", "gross earnings", "basic salary", "payslip", "salary slip", "provident fund", "hra") if k in low)
    if sal_hits >= 2:
        scores["SALARY_SLIP"] += 0.45 + 0.12 * sal_hits

    best_code, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score < 0.45:
        best_code = "UNKNOWN_DOCUMENT"
        best_score = 0.0

    category_labels = {
        "BANK_STATEMENT": "Bank Statement",
        "PAN_CARD": "PAN Card",
        "AADHAAR_CARD": "Aadhaar Card",
        "GST_CERTIFICATE": sub_type or "GST Document",
        "UDYAM_CERTIFICATE": "Udyam MSME Certificate",
        "TAX_AUDIT_REPORT": "Tax Audit Report (Form 3CB/3CD)",
        "ITR_DOCUMENT": "Income Tax Return (ITR)",
        "BALANCE_SHEET": "Balance Sheet",
        "SALARY_SLIP": "Salary Slip",
        "UNKNOWN_DOCUMENT": "Unknown Document",
    }

    variant = _detect_layout_variant(best_code, file_ext.lower(), page_count, page_aspect_ratio, len(txt), low)

    return {
        "doc_code": best_code,
        "doc_category": category_labels[best_code],
        "sub_type": sub_type,
        "layout_variant": variant,
        "issuing_bank_hint": bank_hint if best_code == "BANK_STATEMENT" else None,
        "confidence": round(min(0.99, best_score), 2),
        "scores": {k: round(v, 2) for k, v in scores.items() if v > 0},
        "signals": reasons,
    }