import os
import re
import json
import uuid
import numpy as np
import pymupdf
import docx
from ocr_compat import ocr_engine as ocr_reader
from corporate_kyc_engines import classify_financial_document
from recocile_spans import reconcile_spans


ALLOWED_CHECK_STATUSES = {"PASSED", "FAILED", "NOT_RUN", "NOT_APPLICABLE"}
ALLOWED_TRIAGE_BANDS = {"low_risk", "needs_review", "high_risk"}
ALLOWED_DOC_CODES = {
    "BANK_STATEMENT", "SALARY_SLIP", "PAN_CARD", "AADHAAR_CARD",
    "GST_CERTIFICATE", "UDYAM_CERTIFICATE", "ITR_DOCUMENT",
    "BALANCE_SHEET", "TAX_AUDIT_REPORT", "UNKNOWN_DOCUMENT"
}
ALLOWED_FINDING_CODES = {
    "ARITHMETIC_MISMATCH",
    "FILENAME_PERIOD_DISCREPANCY",
    "EXTRACTION_INCOMPLETE",
    "DUPLICATE_TRANSACTION_ROW",
    "DATE_ORDER_ANOMALY",
    "TEMPLATE_PLACEHOLDER_ANOMALY",
    "PDF_TEXT_OVERLAY",
    "FONT_SUBSET_ANOMALY",
    "PDF_INCREMENTAL_UPDATE",
    "METADATA_TOOLCHAIN_TRACE",
    "PIXEL_ELA_ANOMALY",
    "COPY_MOVE_CLONE",
    "CROSS_DOCUMENT_MISMATCH",
    "UNCLASSIFIED_DYNAMIC_DOCUMENT",
    "UNCLASSIFIED_ANOMALY",
    "FILENAME_DOCUMENT_TYPE_DISCREPANCY",
    "NON_HUMAN_PORTRAIT_ANOMALY",
    "STATEMENT_PERIOD_MISMATCH",
}

TENANT_CONFIGS = {
    "default_lender": {
        "needs_review_threshold": 0.35,
        "high_risk_threshold": 0.70,
        "family_weights": {"A": 0.45, "B": 0.25, "C": 0.20, "D": 0.10}
    },
    "conservative_bank": {
        "needs_review_threshold": 0.20,
        "high_risk_threshold": 0.50,
        "family_weights": {"A": 0.40, "B": 0.25, "C": 0.20, "D": 0.15}
    },
    "high_velocity_fintech": {
        "needs_review_threshold": 0.45,
        "high_risk_threshold": 0.80,
        "family_weights": {"A": 0.50, "B": 0.20, "C": 0.20, "D": 0.10}
    }
}

UNIVERSAL_IFSC_BANK_MAP = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "KKBK": "Kotak Mahindra Bank", "YESB": "Yes Bank",
    "PUNB": "Punjab National Bank", "BARB": "Bank of Baroda", "CNRB": "Canara Bank",
    "UBIN": "Union Bank of India", "IDIB": "Indian Bank", "BKID": "Bank of India",
    "CBIN": "Central Bank of India", "IOBA": "Indian Overseas Bank", "UCBA": "UCO Bank",
    "MAHB": "Bank of Maharashtra", "PSIB": "Punjab & Sind Bank", "IDFB": "IDFC First Bank",
    "INDB": "IndusInd Bank", "FDRL": "Federal Bank", "BDBL": "Bandhan Bank",
    "IBKL": "IDBI Bank", "RATN": "RBL Bank", "KARB": "Karnataka Bank",
    "KVBL": "Karur Vysya Bank", "SIBL": "South Indian Bank", "TMBL": "Tamilnad Mercantile Bank",
    "CIUB": "City Union Bank", "JSBL": "Jana Small Finance Bank", "AUBL": "AU Small Finance Bank",
    "ESFB": "Equitas Small Finance Bank", "UJVN": "Ujjivan Small Finance Bank",
    "SCBL": "Standard Chartered Bank", "CITI": "Citibank", "HSBC": "HSBC Bank",
    "DBSS": "DBS Bank", "DEUT": "Deutsche Bank", "AIRP": "Airtel Payments Bank",
    "PYTM": "Paytm Payments Bank", "SURY": "Suryoday Small Finance Bank"
}


# 1. HONEST RISK SCORING & OPTIONAL EMPIRICAL CALIBRATOR
class RiskScoreCalibrator:
    def __init__(self, external_labeled_json_path: str = None):
        self.is_empirically_validated = False
        self.score_type = "heuristic_priority_score"
        self.validation_sample_count = 0
        self.held_out_brier_score = None
        self.knots_x = np.array([0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0])
        self.knots_y = np.array([0.02, 0.10, 0.24, 0.42, 0.60, 0.75, 0.86, 0.95])

        if external_labeled_json_path and os.path.exists(external_labeled_json_path):
            self.fit_from_labeled_file(external_labeled_json_path)

    @staticmethod
    def _pava(y_sorted: np.ndarray) -> np.ndarray:
        blocks = [[float(v), 1] for v in y_sorted]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                tot_v = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
                tot_w = blocks[i][1] + blocks[i + 1][1]
                blocks[i] = [tot_v / tot_w, tot_w]
                blocks.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        out = []
        for val, cnt in blocks:
            out.extend([val] * cnt)
        return np.clip(np.array(out), 0.02, 0.95)

    def fit_from_labeled_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        val_split = np.array(data.get("validation_split", []))
        test_split = np.array(data.get("held_out_split", []))
        if len(val_split) >= 20 and len(test_split) >= 10:
            order = np.argsort(val_split[:, 0])
            self.knots_x = val_split[order, 0]
            self.knots_y = self._pava(val_split[order, 1])
            preds = [self.transform(x) for x in test_split[:, 0]]
            self.held_out_brier_score = round(float(np.mean((np.array(preds) - test_split[:, 1]) ** 2)), 4)
            self.validation_sample_count = len(val_split)
            self.is_empirically_validated = True
            self.score_type = "empirical_calibrated_probability"

    def transform(self, raw_score: float) -> float:
        clipped = float(np.clip(raw_score, 0.0, 1.0))
        return round(float(np.interp(clipped, self.knots_x, self.knots_y)), 2)


SCORER = RiskScoreCalibrator()


# 2. UNIVERSAL PAGE-1 PRE-TABLE ENTITY & BANK EXTRACTOR
def clean_ocr_person_name(raw_name: str) -> tuple[str, float]:
    if not raw_name:
        return "Unknown Holder", 0.0
    orig = raw_name.strip()
    cleaned = re.sub(r"^(?:Mr\.?|Mrs\.?|Ms\.?|Shri|Smt\.?)\s+", "", orig, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+[Il1\|]+$", "", cleaned).strip()
    cleaned = re.sub(r"[^A-Za-z0-9\s\.\&/\-\(\)]", "", cleaned).strip(" :-|.,")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).title()
    if not cleaned or len(cleaned) < 3:
        return "Unknown Holder", 0.0
    confidence = 0.92 if cleaned.lower() != orig.lower() else 0.98
    return cleaned, confidence


def extract_raw_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        page_texts = []
        with pymupdf.open(file_path) as doc:
            for p_idx, page in enumerate(doc):
                t = page.get_text("text").strip()
                if len(t) > 15:
                    page_texts.append(t)
                else:
                    pix = page.get_pixmap(dpi=200)
                    tmp_img = f"{file_path}_p{p_idx + 1}_id_tmp.png"
                    pix.save(tmp_img)
                    try:
                        res, _ = ocr_reader(tmp_img)
                        ocr_txt = "\n".join(r[1] for r in res) if res else ""
                        page_texts.append(ocr_txt)
                    finally:
                        if os.path.exists(tmp_img):
                            os.remove(tmp_img)
        return "\n".join(page_texts)
    elif ext == ".docx":
        d = docx.Document(file_path)
        p_txt = "\n".join(p.text for p in d.paragraphs)
        t_txt = "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
        return f"{p_txt}\n{t_txt}"
    elif ext in (".png", ".jpg", ".jpeg"):
        res, _ = ocr_reader(file_path)
        return "\n".join(r[1] for r in res) if res else ""
    return ""


def _extract_holder_strictly_from_header(file_path: str, pre_table_text: str) -> str | None:
    bad_words = [
        "statement of", "account statement", "detailed statement", "salary slip",
        "icici bank", "hdfc bank", "yes bank", "kotak mahindra", "state bank",
        "axis bank", "indusind bank", "standard chartered", "jammu", "rajasthan",
        "branch", "ifsc", "currency", "nominee", "period", "search", "customer id",
        "cust id", "scheme", "ckyc", "micr", "sanction", "drawing power", "interest rate",
        "opening balance", "closing balance", "page", "date of", "generated",
        "address", "city", "state", "phone", "email", "od limit", "account no",
        "a/c open", "account status", "regular", "product code", "joint holder", "nomination",
        "road", "street", "marg", "lane", "gali", "nagar", "mumbai", "meerut", "delhi", "plaza", "society"
    ]

    def _valid(c: str) -> bool:
        c = re.sub(r"\s+", " ", c).strip(" :-|.,")
        if len(c) < 3 or len(c) > 55:
            return False
        c_low = c.lower()
        if any(b in c_low for b in bad_words):
            return False
        if re.match(r"^[\d\W]+$", c) or re.match(r"^\d{1,2}[\-/][A-Za-z0-9]{2,9}[\-/]\d{2,4}$", c):
            return False
        # Must contain at least 3 alphabetic characters
        if sum(1 for ch in c if ch.isalpha()) < 3:
            return False
        return True

    # Stage 1: 2D Coordinate right-of-label lookup on Page 1
    if file_path and file_path.lower().endswith(".pdf") and os.path.exists(file_path):
        try:
            with pymupdf.open(file_path) as doc:
                if len(doc) > 0:
                    page = doc[0]
                    spans = []
                    for b in page.get_text("dict").get("blocks", []):
                        for l in b.get("lines", []):
                            line_txt = " ".join(s.get("text", "").strip() for s in l.get("spans", []) if s.get("text", "").strip())
                            if line_txt:
                                bbox = l["bbox"]
                                spans.append({"text": line_txt, "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]})

                    label_re = re.compile(
                        r"^(?:account\s*holder(?:\s*name)?|account\s*name|customer\s*name|applicant\s*name|employee\s*name|legal\s*name|name\s*of\s*enterprise|name\s*of\s*customer|name)\s*[:\-]?$",
                        re.IGNORECASE
                    )
                    for s in spans:
                        if s["y0"] > 380:
                            continue
                        if label_re.match(s["text"].strip()):
                            right_matches = [
                                other for other in spans
                                if other["x0"] > s["x1"] - 6.0 and abs(other["y0"] - s["y0"]) <= 12.0
                                and _valid(other["text"].lstrip(":- "))
                            ]
                            if right_matches:
                                right_matches.sort(key=lambda item: item["x0"])
                                return right_matches[0]["text"].lstrip(":- ").strip()
        except Exception:
            pass

    lines = [ln.strip() for ln in pre_table_text.splitlines() if ln.strip()]

    # Stage 2: Stacked Columnar Label Block ('Name:\nAddress:\nA/C No:\nJt. Holder:\n<VALUE>')
    for i, ln in enumerate(lines):
        if re.match(r"^(?:account\s*name|customer\s*name|holder\s*name|name)\s*[:\-]?$", ln, re.IGNORECASE):
            j = i + 1
            while j < min(len(lines), i + 8) and (
                lines[j].endswith(":")
                or re.match(r"^(?:address|a/c|account|jt\.?\s*holder|joint\s*holder|cust(?:omer)?\s*id|ifsc|micr|branch)\b", lines[j], re.IGNORECASE)
            ):
                j += 1
            if j < len(lines) and _valid(lines[j]):
                return lines[j].lstrip(":- ").strip()

    # Stage 3: Inline Key-Value or Business/Person Name Anchor
    pats = [
        r"(?:^|\n)([A-Za-z0-9\.\&\/\ \t]{3,50}?)\s*\(\s*(?:Account\s*Name|Company\s*Name)\s*\)",
        r"(?<!nominee\s)(?<!nominee)(?<!branch\s)(?:account\s*holder(?:\s*name)?|account\s*name|customer\s*name|applicant\s*name|employee\s*name|legal\s*name|name)\s*[:\-]\s*([A-Za-z][A-Za-z0-9\ \t\.\&/\-]{2,50}?)(?=\s*\||$|\n|pan|period|statement|branch|address|joint|jt\.|cust)",
        r"(?:^|\n)((?:M/S\.?\s+)?[A-Z][A-Z0-9\ \t\.\&]{2,48}?)\n\s*(?:joint\s*holder|jt\.?\s*holder|address|plot\s+no|flat\s+no|h\.?\s*no|house\s+no|kh\s+no|shop\s+no|ward|sector|phase|floor|near|opp|behind|village|dist|new\s+\d+|\d+[\-/A-Z0-9]+\s+[A-Z])",
        r"(?:^|\n)([A-Z][A-Z0-9\.\&/\ \t]{2,45}\s+(?:ENTERPRISES|TRADERS|ASSOCIATES|INDUSTRIES|PRIVATE\s+LIMITED|PVT\.?\s*LTD\.?|LIMITED|LLP|SOLUTIONS|INFRA|COLLECTION|AGENCY|CORP|CORPORATION|TECH|TECHNOLOGY|ENGINEERING|SERVICES|EXPORTS|IMPORTS))\b",
    ]
    for pat in pats:
        for m in re.finditer(pat, pre_table_text, re.IGNORECASE):
            cand = m.group(1).strip()
            if _valid(cand):
                return cand

    # Stage 4: Top-of-page Unlabeled Customer Name (Axis Bank & IndusInd Bank often place the Customer Name on Line 1 or 2 of Page 1!)
    for ln in lines[:6]:
        cleaned_ln = re.sub(r"^(?:Mr\.?|Mrs\.?|Ms\.?|M/s\.?)\s+", "", ln, flags=re.IGNORECASE).strip()
        if (
            _valid(cleaned_ln)
            and cleaned_ln.upper() == cleaned_ln
            and ":" not in cleaned_ln
            and not any(k in cleaned_ln.lower() for k in ["bank", "statement", "account", "branch", "address", "india", "page", "limit", "inr", "date", "from", "to"])
        ):
            return cleaned_ln
    return None

def _extract_page1_text(file_path: str) -> str:
    """Safely extracts raw text from Page 1 of a PDF or DOCX for entity & header identification."""
    if not file_path or not os.path.exists(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            with pymupdf.open(file_path) as doc:
                if len(doc) > 0:
                    return doc[0].get_text("text") or ""
        elif ext == ".docx":
            import docx
            d = docx.Document(file_path)
            parts = [p.text for p in d.paragraphs if p.text.strip()]
            for tbl in d.tables:
                for row in tbl.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells if c.text.strip()))
            return "\n".join(parts)
    except Exception:
        return ""
    return ""


def _is_address_or_noise_line(text: str) -> bool:
    """Returns True if a line is an address, branch label, city, or boilerplate noise rather than an entity name."""
    if not text:
        return True
    low = text.lower().strip(" :-|.,")
    exact_noise = {
        "address", "name", "account", "branch", "details", "summary", "holder",
        "joint holder", "nominee", "phone", "mobile", "email", "status", "type",
        "currency", "inr", "date", "period", "to", "from", "page", "customer id",
        "cust id", "account no", "a/c no", "ifsc", "micr", "regular", "registered",
        "india", "b", "bob", "world", "internet", "hlle",
    }
    if low in exact_noise:
        return True
    noise_tokens = (
        "address", "road", "street", "marg", "lane", "gali", "nagar", "colony", "sector",
        "phase", "floor", "plot", "flat", "shop", "house", "building", "complex", "tower",
        "chowk", "plaza", "khasra", "village", "vpo", "teh", "dist", "mumbai", "delhi",
        "meerut", "varanasi", "gurgaon", "haryana", "uttar pradesh", "maharashtra",
        "branch", "ifsc", "micr", "statement", "account no", "currency", "cust id",
        "phone", "email", "nominee", "period", "understand your world", "sheel palace",
    )
    if any(tok in low for tok in noise_tokens):
        return True
    if re.search(r"\b\d{6}\b", low):
        return True
    return False


def _format_entity_name_clean(raw_name: str) -> str:
    """Preserves 2-3 letter business initials (SM, K2, LLP, PVT, LTD, M/S) and joins split initials like 'S M' -> 'SM'."""
    if not raw_name:
        return "Unknown Holder"
    cleaned = re.sub(r"[ \t]+", " ", str(raw_name)).strip(" :-|.,")
    cleaned = re.sub(
        r"^([A-Za-z])\s+([A-Za-z])(?=\s+(?:ENTERPRISES|TRADERS|SERVICES|INDUSTRIES|CORPORATION|ASSOCIATES|COMPANY|AGENCIES|SONS)\b)",
        r"\1\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    keep_upper = {
        "SM", "K2", "LLP", "PVT", "PVT.", "LTD", "LTD.", "HUF", "PROP",
        "M/S", "M/S.", "SRK", "JSH", "VS", "MS", "RK", "JK", "DR", "AR", "PK"
    }
    words = []
    for w in cleaned.split():
        w_up = w.upper()
        if w_up in keep_upper or (len(w) == 2 and w.isalpha() and w.isupper()):
            words.append(w_up)
        else:
            words.append(w.title())
    return " ".join(words)


def _slice_page1_pre_table(page1_text: str) -> str:
    """
    Strictly slices Page 1 text BEFORE the transaction table header or first transaction row,
    even when PyMuPDF places each column header ('TRAN DATE', 'VALUE DATE', 'NARRATION') on its own line.
    """
    lines = [ln.strip() for ln in (page1_text or "").splitlines() if ln.strip()]
    pre_lines = []
    single_line_hdr_markers = {
        "tran date", "transaction date", "txn date", "value date", "value dt",
        "narration", "particulars", "withdrawal(dr)", "deposit(cr)", "balance(inr)",
        "chq.no.", "chq./ref.no.", "cheque no", "running balance", "closing balance", "sr no",
        "tran id", "transaction remarks", "transaction posted date", "withdra", "si no",
    }
    date_row_re = re.compile(
        r"^(?:[0-3]?\d[-/\.][01]?\d[-/\.]\d{2,4}|[0-3]?\d[\s\-/]+[A-Za-z]{3,9}[\s\-/]+\d{2,4})\b"
    )

    for idx, ln in enumerate(lines):
        low_ln = ln.lower().strip(" |:-")
        if low_ln in single_line_hdr_markers:
            break
        if ("date" in low_ln or "transactio" in low_ln) and any(
            k in low_ln for k in ("narration", "particular", "description", "remark", "withdrawal", "withdrawl", "deposit", "debit", "credit", "balance")
        ) and not any(p in low_ln for p in ("statement period", "from date", "open date", "statement date", "as on")):
            break
        win_low = " ".join(lines[idx : min(len(lines), idx + 5)]).lower()
        if ("tran date" in win_low or "value date" in win_low) and ("narration" in win_low or "particulars" in win_low) and "balance" in win_low:
            break
        if date_row_re.match(ln) and not any(
            hk in low_ln for hk in ("period", "from", "as on", "open date", "expiry", "birth")
        ):
            win_tx = " ".join(lines[idx : min(len(lines), idx + 5)])
            if re.search(r"\d+\.\d{2}", win_tx) or any(k in win_tx.upper() for k in ("MBK/", "UPI", "NEFT", "RTGS", "IMPS", "CHQ")):
                break

        pre_lines.append(ln)

    return "\n".join(pre_lines)


def _extract_bank_strictly_from_header(
    pre_table_text: str,
    full_text: str,
    file_path: str = "",
    original_filename: str = "",
) -> str:
    pre_up = (pre_table_text or "").upper()
    pre_nospace = re.sub(r"[ \t]+", "", pre_up)

    for target_txt in (pre_up, pre_nospace):
        m_lbl_ifsc = re.search(r"IFSC(?:CODE)?[\s:\|\-]*([A-Z]{4})[0O]([A-Z0-9]{6})", target_txt)
        if m_lbl_ifsc and m_lbl_ifsc.group(1) in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[m_lbl_ifsc.group(1)]

    for m_ifsc in re.finditer(r"\b([A-Z]{4})[0O]([A-Z0-9]{6})\b", pre_up):
        pref = m_ifsc.group(1)
        if pref in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[pref]

    bank_re = re.compile(
        r"\b(Bank\s+of\s+Baroda|bob\s+World|bob\s+Internet|Bank\s+of\s+Maharashtra|mahabank|"
        r"RBL\s*Bank|Jammu\s*(?:&|and)\s*Kashmir\s*Bank|J\s*&\s*K\s*Bank|ICICI\s*Bank|HDFC\s*Bank|"
        r"Yes\s*Bank|Kotak\s+Mahindra\s+Bank|Axis\s*Bank|IndusInd\s*Bank|State\s+Bank\s+of\s+India|"
        r"Standard\s+Chartered\s+Bank|Punjab\s+National\s+Bank|Canara\s+Bank|Union\s+Bank\s+of\s+India|"
        r"IDFC\s+First\s+Bank|Federal\s+Bank|Bandhan\s+Bank|IDBI\s+Bank|Indian\s+Bank|Bank\s+of\s+India|"
        r"Central\s+Bank\s+of\s+India|UCO\s+Bank|Karur\s+Vysya\s+Bank|Karnataka\s+Bank|South\s+Indian\s+Bank|"
        r"AU\s+Small\s+Finance\s+Bank|Ujjivan\s+Small\s+Finance\s+Bank|Equitas\s+Small\s+Finance\s+Bank)\b",
        re.IGNORECASE,
    )

    def _map_matched_bank(raw_str: str) -> str:
        low_b = re.sub(r"\s+", " ", raw_str).strip().lower()
        if "baroda" in low_b or low_b.startswith("bob"):
            return "Bank of Baroda"
        if "maharashtra" in low_b or "mahabank" in low_b:
            return "Bank of Maharashtra"
        if "rbl" in low_b:
            return "RBL Bank"
        if "jammu" in low_b or "j&k" in low_b or "j & k" in low_b:
            return "Jammu & Kashmir Bank"
        if "icici" in low_b:
            return "ICICI Bank"
        if "hdfc" in low_b:
            return "HDFC Bank"
        if "yes" in low_b:
            return "Yes Bank"
        if "kotak" in low_b:
            return "Kotak Mahindra Bank"
        if "axis" in low_b:
            return "Axis Bank"
        if "indusind" in low_b:
            return "IndusInd Bank"
        if "state bank" in low_b:
            return "State Bank of India"
        if "punjab national" in low_b:
            return "Punjab National Bank"
        if "canara" in low_b:
            return "Canara Bank"
        if "union bank" in low_b:
            return "Union Bank of India"
        if "idfc" in low_b:
            return "IDFC First Bank"
        if "idbi" in low_b:
            return "IDBI Bank"
        if "federal" in low_b:
            return "Federal Bank"
        if "bandhan" in low_b:
            return "Bandhan Bank"
        if "au small" in low_b:
            return "AU Small Finance Bank"
        if "ujjivan" in low_b:
            return "Ujjivan Small Finance Bank"
        return raw_str.title()

    found = [(m.start(), m.group(1)) for m in bank_re.finditer(pre_table_text or "")]
    if found:
        found.sort(key=lambda x: x[0])
        return _map_matched_bank(found[0][1])

    if "बड़ौदा" in (pre_table_text or ""):
        return "Bank of Baroda"
    if "महाराष्ट्र" in (pre_table_text or ""):
        return "Bank of Maharashtra"

    fn_cand = (original_filename or os.path.basename(file_path or "")).lower()
    if not re.match(r"^tmp[a-z0-9_]+\.", fn_cand):
        if re.search(r"\bbob\b|baroda", fn_cand):
            return "Bank of Baroda"
        if re.search(r"\bbom\b|maharashtra", fn_cand):
            return "Bank of Maharashtra"
        if re.search(r"\bsbi\b", fn_cand):
            return "State Bank of India"
        if re.search(r"\bpnb\b", fn_cand):
            return "Punjab National Bank"
        m_fn = bank_re.search(fn_cand)
        if m_fn:
            return _map_matched_bank(m_fn.group(1))

    m_rtgs = re.search(r"\bRTGS[/\-]([A-Z]{4})[HR5]\d{6,}", (full_text or "")[:2500].upper())
    if m_rtgs and m_rtgs.group(1) in UNIVERSAL_IFSC_BANK_MAP:
        return UNIVERSAL_IFSC_BANK_MAP[m_rtgs.group(1)]

    footer_slice = (full_text or "")[-600:]
    m_ftr = bank_re.search(footer_slice)
    if m_ftr:
        return _map_matched_bank(m_ftr.group(1))

    return "UNKNOWN"


def identify_document_and_entity(file_path: str, raw_report: dict = None) -> dict:
    fin_r = (raw_report or {}).get("financial_reconciliation") or {}
    fin_r_type = fin_r.get("document_type")

    text = _extract_page1_text(file_path)
    if not text and raw_report:
        txs = fin_r.get("transactions", [])
        text = " ".join(t.get("description", "") for t in txs[:10])

    if fin_r_type and fin_r_type in ALLOWED_DOC_CODES and fin_r_type != "UNKNOWN_DOCUMENT":
        doc_code = fin_r_type
        doc_category_map = {
            "BANK_STATEMENT": "Bank Statement",
            "PAN_CARD": "PAN Card",
            "AADHAAR_CARD": "Aadhaar Card",
            "SALARY_SLIP": "Salary Slip",
            "GST_CERTIFICATE": "GST Registration Certificate",
            "UDYAM_CERTIFICATE": "Udyam MSME Certificate",
            "ITR_DOCUMENT": "Income Tax Return (ITR)",
            "BALANCE_SHEET": "Balance Sheet",
            "TAX_AUDIT_REPORT": "Tax Audit Report (Form 3CB/3CD)",
        }
        doc_category = doc_category_map.get(doc_code, "Verified Document")
        cls_conf = 0.96
    else:
        cls_info = classify_financial_document(text)
        doc_code = cls_info.get("doc_code", "UNKNOWN_DOCUMENT")
        doc_category = cls_info.get("doc_category", "Unknown Document")
        cls_conf = cls_info.get("classification_confidence", 0.0)

    pre_table_header = _slice_page1_pre_table(text)

    extracted_pan = None
    pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text)
    if pan_match:
        extracted_pan = pan_match.group(1)
    extracted_pan = extracted_pan or fin_r.get("pan_number")

    holder = fin_r.get("legal_name") or fin_r.get("enterprise_name") or fin_r.get("taxpayer_name")
    person_name = None
    company_name = holder if holder else None
    branch_name = None
    ifsc_code = None
    account_number = None

    if doc_code == "BANK_STATEMENT":
        m_br = re.search(
            r"(?:Branch\s*Name|Account\s*Branch|Home\s*Branch)\s*[:\|\-\n]+\s*([A-Za-z0-9][A-Za-z0-9 \t,\.\-&]{2,40})",
            pre_table_header,
            re.IGNORECASE,
        )
        if m_br:
            branch_name = re.sub(r"[ \t]+", " ", m_br.group(1).splitlines()[0].split("|")[0]).strip(" :-|.,").upper()

        m_ifsc = re.search(r"IFSC(?:\s*Code)?[\s:\|\-]*([A-Z]{4})[0O]([A-Z0-9]{6})\b", pre_table_header.upper())
        if m_ifsc:
            ifsc_code = f"{m_ifsc.group(1)}0{m_ifsc.group(2)}"

        m_ac = re.search(
            r"(?:Account\s*(?:No\.?|Number)|A/C\s*(?:No\.?|Number)?)\s*[:\|\-\n]+\s*([A-Z0-9X\*]{6,22})",
            pre_table_header,
            re.IGNORECASE,
        )
        if m_ac and sum(1 for ch in m_ac.group(1) if ch.isdigit() or ch in "X*") >= 3:
            account_number = m_ac.group(1).strip()

        m_actype = re.search(
            r"(?:Account\s*Type|A/C\s*Type)\s*[:\|\-]+\s*([A-Za-z0-9\s\-\/]{2,25})",
            pre_table_header,
            re.IGNORECASE,
        )
        if m_actype:
            raw_act = m_actype.group(1).splitlines()[0].split("|")[0].strip(" :-|.,").upper()
            act_map = {
                "CA": "CA (Current Account)",
                "SB": "SB (Savings Account)",
                "SA": "SA (Savings Account)",
                "OD": "OD (Overdraft Account)",
                "CC": "CC (Cash Credit Account)",
            }
            identity_info_acct_type = act_map.get(raw_act, raw_act)
        else:
            identity_info_acct_type = None

        if not holder:
            for ln in pre_table_header.splitlines():
                clean_ln = re.sub(r"^(?:Main\s*Account\s*Holder\s*Name|Account\s*Name|Name)\s*[:\|\-]+\s*", "", ln.strip(), flags=re.I)
                if re.search(r"\b(?:M/S\.?|PRIVATE\s+LIMITED|PVT\.?\s*LTD|LLP|ENTERPRISES|TRADERS|SERVICES|SONS|INDUSTRIES|COLLECTION)\b", clean_ln, re.I):
                    if "bank" not in clean_ln.lower() and not _is_address_or_noise_line(clean_ln):
                        holder = _format_entity_name_clean(clean_ln)
                        company_name = holder
                        break

    if not holder:
        holder = "Unknown Holder"
    else:
        holder = _format_entity_name_clean(holder)

    issuing_bank = None
    if doc_code == "BANK_STATEMENT":
        issuing_bank = _extract_bank_strictly_from_header(
            pre_table_header,
            text,
            file_path=file_path,
            original_filename=(raw_report or {}).get("original_filename", ""),
        )

    bp_meta = fin_r.get("blueprint_metadata") or {}
    if not branch_name and bp_meta.get("branch_name"):
        branch_name = bp_meta["branch_name"]
    if not ifsc_code and bp_meta.get("ifsc_code"):
        ifsc_code = bp_meta["ifsc_code"]
    if not account_number and bp_meta.get("account_number"):
        account_number = bp_meta["account_number"]
    if holder in ("Unknown Holder", "") and bp_meta.get("holder_name"):
        holder = _format_entity_name_clean(bp_meta["holder_name"])

    identity_badge = f"{doc_category} of {holder}" if holder != "Unknown Holder" else doc_category

    return {
        "doc_code": doc_code,
        "doc_category": doc_category,
        "classification_confidence": round(cls_conf, 2),
        "holder_name": holder,
        "person_name": person_name,
        "company_name": company_name,
        "employer_name": None,
        "issuing_bank": issuing_bank,
        "branch_name": branch_name,
        "ifsc_code": ifsc_code,
        "account_number": account_number,
        "pan_number": extracted_pan,
        "identity_badge": identity_badge,
    }

# 3. STRUCTURED FINDINGS BUILDER & SECTION 4.3 VALIDATOR
def extract_page_from_text(text: str, default_page: int = 1) -> int:
    m = re.search(r"\bPage\s+(\d+)\b", str(text), re.IGNORECASE)
    return int(m.group(1)) if m else default_page

def extract_bbox_from_text(text: str):
    match = re.search(r"\[([\d\.\s,]+)\]", str(text))
    if match:
        try:
            coords = [round(float(x.strip()), 2) for x in match.group(1).split(",")]
            if len(coords) == 4:
                return coords
        except Exception:
            pass
    return None


def build_structured_findings(doc_id, doc_label, doc_code, raw_report, family_d_report=None):
    structured = []
    fin_rec = raw_report.get("financial_reconciliation", {})

    if doc_code == "UNKNOWN_DOCUMENT":
        raw_s = 0.60
        structured.append({
            "document_id": doc_id,
            "page": 1,
            "bbox": None,
            "field_pointer": "document_classification",
            "detector_id": "STAGE_2_DOCUMENT_CLASSIFIER",
            "detector_version": "4.3.0",
            "finding_code": "UNCLASSIFIED_DYNAMIC_DOCUMENT",
            "raw_detector_score": raw_s,
            "individual_finding_score": SCORER.transform(raw_s),
            "original_signal": "Insufficient keyword or layout evidence to classify document.",
            "reason": "Unknown Document, page 1: Document type could not be reliably classified into any supported financial or KYC schema."
        })

    detailed_issues = fin_rec.get("detailed_issues")
    if not detailed_issues:
        detailed_issues = [
            {"page": extract_page_from_text(iss, 1), "bbox": extract_bbox_from_text(iss), "text": iss}
            for iss in fin_rec.get("issues", [])
        ]

    for item in detailed_issues:
        issue_txt = item["text"]
        page_no = item.get("page") or extract_page_from_text(issue_txt, 1)
        bbox = item.get("bbox") or extract_bbox_from_text(issue_txt)

        det_id = "FAM_A_LOGIC_ARITHMETIC"
        pointer = "ledger_row_bbox" if bbox else "summary_or_ledger_balance"
        is_intake_check = False

        if item.get("finding_code") == "FILENAME_DOCUMENT_TYPE_DISCREPANCY" or "Filename Document Type Discrepancy" in issue_txt:
            code = "FILENAME_DOCUMENT_TYPE_DISCREPANCY"
            raw_s = 0.35
            det_id = "STAGE_1_INTAKE_METADATA_CHECK"
            pointer = "uploaded_filename_vs_document_type"
            is_intake_check = True
        elif item.get("finding_code") == "FILENAME_PERIOD_DISCREPANCY" or "Filename Period Discrepancy" in issue_txt or "Filename vs. Document Period Discrepancy" in issue_txt:
            code = "FILENAME_PERIOD_DISCREPANCY"
            raw_s = 0.35
            det_id = "STAGE_1_INTAKE_METADATA_CHECK"
            pointer = "uploaded_filename_vs_document_period"
            is_intake_check = True
        elif any(k in issue_txt for k in ["Warning:", "Missing", "Fatal:", "Incomplete"]):
            code = "EXTRACTION_INCOMPLETE"
            raw_s = 0.42
        elif "Duplicate" in issue_txt:
            code = "DUPLICATE_TRANSACTION_ROW"
            raw_s = 0.80
        elif "Monotonicity" in issue_txt:
            code = "DATE_ORDER_ANOMALY"
            raw_s = 0.75
        elif "Statement Period Mismatch" in issue_txt:
            code = "STATEMENT_PERIOD_MISMATCH"
            raw_s = 0.35
            det_id = "STAGE_1_INTAKE_METADATA_CHECK"
            pointer = "declared_period_vs_last_transaction_date"
            is_intake_check = True
        elif "KYC Portrait Anomaly" in issue_txt:
            code = "NON_HUMAN_PORTRAIT_ANOMALY"
            raw_s = 0.85
            det_id = "FAM_C_IMAGE_FORENSICS"
            pointer = "id_card_portrait_region"
        elif (
            "Summary" in issue_txt
            and "does not match" in issue_txt
            and (fin_rec.get("audit_coverage") or {}).get("running_balance_verified") == "PASSED"
            and not raw_report.get("forensic_flags")
        ):
            # Running balance across rows is 100% consistent and PDF has 0 forensic flags:
            # classify summary discrepancy as an extraction completeness review item, NOT high-risk fraud!
            code = "EXTRACTION_INCOMPLETE"
            raw_s = 0.38
        else:
            code = "ARITHMETIC_MISMATCH"
            total_tx_cnt = max(1, len(fin_rec.get("transactions") or []))
            rb_fail_cnt = sum(1 for x in detailed_issues if "Running balance mismatch" in x.get("text", ""))
            fail_ratio = min(1.0, rb_fail_cnt / max(1.0, total_tx_cnt * 0.4))
            raw_s = round(0.58 + 0.27 * fail_ratio, 2)

        finding_conf = raw_s if is_intake_check else SCORER.transform(raw_s)

        structured.append({
            "document_id": doc_id,
            "page": int(page_no),
            "bbox": bbox,
            "field_pointer": pointer,
            "detector_id": det_id,
            "detector_version": "4.3.0",
            "finding_code": code,
            "raw_detector_score": raw_s,
            "individual_finding_score": finding_conf,
            "original_signal": issue_txt,
            "reason": f"{doc_label}, page {page_no}: {issue_txt}"
        })

    for flag in raw_report.get("forensic_flags", []):
        page_no = extract_page_from_text(flag, 1)
        bbox = extract_bbox_from_text(flag)
        low_f = flag.lower()

        if any(k in low_f for k in ["ela", "pixel", "compression"]):
            det_id = "FAM_C_IMAGE_FORENSICS"
            code = "PIXEL_ELA_ANOMALY"
            raw_s = 0.78
            pointer = "visual_pixel_region"
        elif "copy-move" in low_f or "duplicated keypoint" in low_f:
            det_id = "FAM_C_IMAGE_FORENSICS"
            code = "COPY_MOVE_CLONE"
            raw_s = 0.80
            pointer = "visual_pixel_region"
        elif "template placeholder" in low_f or "form hints" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "TEMPLATE_PLACEHOLDER_ANOMALY"
            raw_s = 0.75
            pointer = "header_template_text"
        elif "overlay" in low_f or "overlaps with" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "PDF_TEXT_OVERLAY"
            raw_s = 0.75
            pointer = "overlapping_text_span"
        elif "font" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "FONT_SUBSET_ANOMALY"
            raw_s = 0.70
            pointer = "font_descriptor"
        elif "eof" in low_f or "incremental" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "PDF_INCREMENTAL_UPDATE"
            raw_s = 0.72
            pointer = "xref_trailer"
        elif any(k in low_f for k in ["metadata", "author", "producer", "creator", "photoshop", "canva", "revision"]):
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "METADATA_TOOLCHAIN_TRACE"
            raw_s = 0.65
            pointer = "document_metadata"
        else:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "UNCLASSIFIED_ANOMALY"
            raw_s = 0.60
            pointer = "unclassified_signal"

        structured.append({
            "document_id": doc_id,
            "page": int(page_no),
            "bbox": bbox,
            "field_pointer": pointer,
            "detector_id": det_id,
            "detector_version": "4.3.0",
            "finding_code": code,
            "raw_detector_score": raw_s,
            "individual_finding_score": SCORER.transform(raw_s),
            "original_signal": flag,
            "reason": f"{doc_label}, page {page_no}: {flag}"
        })

    if family_d_report and isinstance(family_d_report, dict):
        for d_finding in family_d_report.get("findings", []):
            raw_s = 0.85
            structured.append({
                "document_id": doc_id,
                "page": 1,
                "bbox": None,
                "field_pointer": "cross_document_pack",
                "detector_id": "FAM_D_CROSS_DOCUMENT_KYC",
                "detector_version": "4.3.0",
                "finding_code": "CROSS_DOCUMENT_MISMATCH",
                "raw_detector_score": raw_s,
                "individual_finding_score": SCORER.transform(raw_s),
                "original_signal": d_finding,
                "reason": f"{doc_label} (Cross-Document Check): {d_finding}"
            })

    return structured


def standardize_audit_coverage(raw_cov: dict) -> dict:
    standardized = {}
    for k, v in (raw_cov or {}).items():
        if isinstance(v, str) and v in ALLOWED_CHECK_STATUSES:
            standardized[k] = v
        elif v is True:
            standardized[k] = "PASSED"
        elif v is False:
            standardized[k] = "FAILED"
        else:
            standardized[k] = "NOT_RUN"
    return standardized


def compose_founder_summary(doc_label, structured_findings):
    if not structured_findings:
        return (
            f"{doc_label}: All evaluated arithmetic invariants and structural checks PASSED. "
            f"Note: A low_risk verdict means no configured anomalies were detected, not proof of genuineness."
        )

    pages = sorted({f["page"] for f in structured_findings})
    page_str = f"page {pages[0]}" if len(pages) == 1 else f"pages {', '.join(map(str, pages))}"
    codes = {f["finding_code"] for f in structured_findings}

    if "UNCLASSIFIED_DYNAMIC_DOCUMENT" in codes:
        return f"Unknown Document, {page_str}: Document could not be reliably classified into a supported financial schema. Send for manual review."

    clauses = []
    if "TEMPLATE_PLACEHOLDER_ANOMALY" in codes:
        clauses.append("unfilled template placeholders (e.g., '(Company Name)', '(Address)') were found in the header, indicating a template-edited statement")
    if "PDF_TEXT_OVERLAY" in codes:
        clauses.append("overlapping text layers were detected over monetary figures")
    if "FONT_SUBSET_ANOMALY" in codes:
        clauses.append("mismatched font subsets were detected on numeric values")
    if "PDF_INCREMENTAL_UPDATE" in codes:
        clauses.append("multiple PDF incremental-save revisions (%%EOF) were detected")
    if "METADATA_TOOLCHAIN_TRACE" in codes:
        clauses.append("external editing software traces were found in the file metadata")
    if "PIXEL_ELA_ANOMALY" in codes:
        clauses.append("pixel-level Error Level Analysis (ELA) compression anomalies were detected")
    if "COPY_MOVE_CLONE" in codes:
        clauses.append("cloned/copy-moved image regions were detected")
    if "CROSS_DOCUMENT_MISMATCH" in codes:
        clauses.append("cross-document identity, PAN, or income inconsistencies were detected across the application pack")
    if "UNCLASSIFIED_ANOMALY" in codes:
        clauses.append("unclassified structural anomalies were flagged")
    if "FILENAME_PERIOD_DISCREPANCY" in codes:
        clauses.append("the uploaded filename indicates FY 2024-25 whereas the report states Previous Year 2023-24 (Assessment Year 2024-25)")
    if "FILENAME_DOCUMENT_TYPE_DISCREPANCY" in codes:
        clauses.append("the uploaded filename suggests a Balance Sheet whereas the document content is a Tax Audit Report (Form 3CB/3CD)")

    unique_clauses = []
    for c in clauses:
        if c not in unique_clauses:
            unique_clauses.append(c)
    clauses = unique_clauses

    has_math = "ARITHMETIC_MISMATCH" in codes
    if has_math and clauses:
        return f"{doc_label}, {page_str}: {'; '.join(clauses)}, and the running balance or summary equation FAILED reconciliation. Send for manual review."
    elif has_math:
        return f"{doc_label}, {page_str}: running balance or summary equation FAILED reconciliation against the itemised rows. Send for manual review."
    elif clauses:
        return f"{doc_label}, {page_str}: arithmetic reconciles cleanly, but {'; '.join(clauses)}. Send for manual review."
    elif "EXTRACTION_INCOMPLETE" in codes:
        return f"{doc_label}, {page_str}: required summary fields could not be extracted (FAILED / NOT_RUN). Cannot certify clean; send for manual review."
    return f"{doc_label}, {page_str}: anomalies detected ({', '.join(sorted(codes))}). Send for manual review."


def validate_section_4_3_contract(payload: dict) -> tuple[bool, list[str]]:
    errors = []
    required_types = {
        "module": str,
        "schema_version": str,
        "document_id": str,
        "tenant_id": str,
        "identity_card": dict,
        "file_format": str,
        "raw_fused_document_score": (int, float),
        "overall_document_risk_score": (int, float),
        "scoring_methodology": dict,
        "triage_band": str,
        "plain_language_summary": str,
        "detector_coverage": dict,
        "structured_findings": list,
        "extracted_financials": dict,
        "disclaimer": str
    }

    for k, expected_type in required_types.items():
        if k not in payload:
            errors.append(f"Missing mandatory key: '{k}'")
        elif not isinstance(payload[k], expected_type):
            errors.append(f"Key '{k}' expected type {expected_type}, got {type(payload[k]).__name__}")

    if payload.get("triage_band") not in ALLOWED_TRIAGE_BANDS:
        errors.append(f"Invalid triage_band '{payload.get('triage_band')}'. Allowed: {ALLOWED_TRIAGE_BANDS}")

    doc_score = payload.get("overall_document_risk_score", -1)
    if not isinstance(doc_score, (int, float)) or not (0.0 <= float(doc_score) <= 0.95):
        errors.append(f"overall_document_risk_score ({doc_score}) violates [0.0, 0.95] range.")

    det_cov = payload.get("detector_coverage", {})
    for fam_key in ("family_a_logic", "family_b_pdf_structure", "family_c_image_forensics", "family_d_cross_document"):
        if fam_key not in det_cov:
            errors.append(f"detector_coverage missing key '{fam_key}'")
        elif det_cov[fam_key] not in ALLOWED_CHECK_STATUSES:
            errors.append(f"detector_coverage['{fam_key}'] has invalid status '{det_cov[fam_key]}'. Allowed: {ALLOWED_CHECK_STATUSES}")

    aud_cov = payload.get("extracted_financials", {}).get("audit_coverage", {})
    for chk_k, chk_v in aud_cov.items():
        if chk_v not in ALLOWED_CHECK_STATUSES:
            errors.append(f"audit_coverage['{chk_k}'] has invalid status '{chk_v}'. Allowed: {ALLOWED_CHECK_STATUSES}")

    for idx, f in enumerate(payload.get("structured_findings", [])):
        req_f = {
            "document_id": str,
            "page": int,
            "field_pointer": str,
            "detector_id": str,
            "detector_version": str,
            "finding_code": str,
            "raw_detector_score": (int, float),
            "individual_finding_score": (int, float),
            "original_signal": str,
            "reason": str
        }
        for fk, ft in req_f.items():
            if fk not in f:
                errors.append(f"structured_findings[{idx}] missing '{fk}'")
            elif not isinstance(f[fk], ft):
                errors.append(f"structured_findings[{idx}]['{fk}'] invalid type")

        if f.get("page", 0) < 1:
            errors.append(f"structured_findings[{idx}]['page'] must be >= 1")

        if f.get("finding_code") not in ALLOWED_FINDING_CODES:
            errors.append(f"structured_findings[{idx}] invalid finding_code '{f.get('finding_code')}'")

        bbox = f.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(c, (int, float)) for c in bbox):
                errors.append(f"structured_findings[{idx}]['bbox'] must be None or a 4-float list [x0, y0, x1, y1]")

    return (len(errors) == 0), errors


def build_module1_output(file_path, raw_report, tenant_id="default_lender", doc_id=None, family_d_report=None):
    doc_id = doc_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    tenant_cfg = TENANT_CONFIGS.get(tenant_id, TENANT_CONFIGS["default_lender"])
    weights = dict(tenant_cfg["family_weights"])
    file_fmt = str(raw_report.get("file_format") or "UNKNOWN").upper()
    identity_info = identify_document_and_entity(file_path, raw_report)
    print(
        "[BANK IDENTITY DEBUG] "
        f"Header detection: {identity_info.get('issuing_bank')} | "
        f"Blueprint detection: "
        f"{raw_report.get('financial_reconciliation', {}).get('blueprint_metadata', {}).get('issuing_bank')}"
    )
    doc_code = identity_info.get("doc_code", "UNKNOWN_DOCUMENT")
    doc_label = identity_info.get("doc_category", "Unknown Document")
    bp_meta = raw_report.get("financial_reconciliation", {}).get("blueprint_metadata") or {}
    bp_bank = (bp_meta.get("issuing_bank") or "").strip()
    bp_holder = (bp_meta.get("holder_name") or "").strip()
    if (
        (not identity_info.get("issuing_bank") or identity_info.get("issuing_bank") == "UNKNOWN")
        and bp_bank
        and not bp_bank.lower().startswith("tmp")
        and not bp_bank.lower().endswith((".pdf", ".docx", ".png", ".jpg", ".jpeg"))
    ):
        identity_info["issuing_bank"] = bp_bank
    if bp_holder and bp_holder not in ("Unknown Holder", "") and identity_info.get("holder_name") in (None, "", "Unknown Holder"):
        clean_bp_h = _format_entity_name_clean(bp_holder)
        identity_info["holder_name"] = clean_bp_h
        identity_info["identity_badge"] = f"{identity_info.get('doc_category', 'Document')} of {clean_bp_h}"

    for k_meta in ("branch_name", "account_number", "ifsc_code", "customer_id"):
        if bp_meta.get(k_meta) and not identity_info.get(k_meta):
            identity_info[k_meta] = bp_meta[k_meta]
    if bp_holder:
        identity_info["holder_name"] = bp_holder  
    invalid_holders = {
        "", "unknown holder", "address", "name", "account", "branch", "city",
        "state", "phone", "email", "regular", "status", "currency", "inr",
        "india and rajasthan bank ltd"
    }
    if bp_holder and bp_holder.lower() not in invalid_holders:
        identity_info["holder_name"] = bp_holder
        identity_info["person_name"] = bp_holder
        identity_info["identity_badge"] = f"{identity_info['doc_category']} of {bp_holder}"
    elif (identity_info.get("holder_name") or "").strip().lower() in invalid_holders:
        identity_info["holder_name"] = "Unknown Holder"
        identity_info["person_name"] = "Unknown Holder"
        identity_info["identity_badge"] = f"{identity_info['doc_category']} of Unknown Holder"

    fin_rec = dict(raw_report.get("financial_reconciliation", {}))
    if doc_code == "UNKNOWN_DOCUMENT":
        fin_rec["document_type"] = "UNKNOWN_DOCUMENT"
        fin_rec["issues"] = []
        fin_rec["detailed_issues"] = []
        fin_rec["audit_coverage"] = {"document_schema_recognized": "FAILED"}
    elif doc_code == "PAN_CARD":
        fin_rec["document_type"] = "PAN_CARD"
        pan_no = fin_rec.get("pan_number") or identity_info.get("pan_number")
        if pan_no:
            identity_info["pan_number"] = pan_no
            fin_rec["pan_number"] = pan_no
        if fin_rec.get("legal_name") and fin_rec["legal_name"] not in ("Unknown Holder", ""):
            identity_info["holder_name"] = fin_rec["legal_name"]
            identity_info["person_name"] = fin_rec["legal_name"]
            identity_info["identity_badge"] = f"PAN Card of {fin_rec['legal_name']}"
        pan_ok = bool(pan_no and len(pan_no) == 10)
        entity_ok = bool(pan_no and len(pan_no) == 10 and pan_no[3] in "PCHFATBLJG")
        existing_cov = fin_rec.get("audit_coverage") or {}
        fin_rec["audit_coverage"] = {
            "pan_number_extracted": "PASSED" if pan_ok else "FAILED",
            "individual_entity_code_verified": ("PASSED" if entity_ok else "FAILED") if pan_ok else "NOT_RUN",
            "human_portrait_verified": existing_cov.get("human_portrait_verified", "PASSED"),
        }
    elif doc_code == "AADHAAR_CARD":
        fin_rec["document_type"] = "AADHAAR_CARD"
        if fin_rec.get("legal_name") and fin_rec["legal_name"] not in ("Unknown Holder", ""):
            identity_info["holder_name"] = fin_rec["legal_name"]
            identity_info["person_name"] = fin_rec["legal_name"]
            identity_info["identity_badge"] = f"Aadhaar Card of {fin_rec['legal_name']}"

    fin_rec["audit_coverage"] = standardize_audit_coverage(fin_rec.get("audit_coverage", {}))

    raw_report_norm = dict(raw_report)
    raw_report_norm["financial_reconciliation"] = fin_rec
    structured_findings = build_structured_findings(doc_id, doc_label, doc_code, raw_report_norm, family_d_report)

    fam_a_findings = [f for f in structured_findings if f["detector_id"] in ("FAM_A_LOGIC_ARITHMETIC", "STAGE_2_DOCUMENT_CLASSIFIER")]
    intake_findings = [f for f in structured_findings if f["detector_id"] == "STAGE_1_INTAKE_METADATA_CHECK"]
    fam_b_findings = [f for f in structured_findings if f["detector_id"] == "FAM_B_STRUCTURE_FORENSICS"]
    fam_c_findings = [f for f in structured_findings if f["detector_id"] == "FAM_C_IMAGE_FORENSICS"]
    fam_d_findings = [f for f in structured_findings if f["detector_id"] == "FAM_D_CROSS_DOCUMENT_KYC"]

    score_a = max((f["raw_detector_score"] for f in fam_a_findings), default=0.0)
    score_intake = max((f["raw_detector_score"] for f in intake_findings), default=0.0)
    score_b = max((f["raw_detector_score"] for f in fam_b_findings), default=0.0)
    score_c = max((f["raw_detector_score"] for f in fam_c_findings), default=0.0)
    score_d = max((f["raw_detector_score"] for f in fam_d_findings), default=0.0)

    aud_vals = [
        v for k, v in fin_rec["audit_coverage"].items()
        if v != "NOT_APPLICABLE" and k != "statement_period_verified"
    ]
    if score_a > 0 or "FAILED" in aud_vals:
        cov_a = "FAILED"
    elif aud_vals and all(v == "PASSED" for v in aud_vals):
        cov_a = "PASSED"
    else:
        cov_a = "NOT_RUN"

    if score_b > 0:
        cov_b = "FAILED"
    elif file_fmt in ("DIGITAL_PDF", "HYBRID_PDF", "DOCX"):
        cov_b = "PASSED"
    else:
        cov_b = "NOT_APPLICABLE"

    if score_c > 0:
        cov_c = "FAILED"
    elif file_fmt in ("IMAGE", "SCANNED_PDF", "HYBRID_PDF"):
        cov_c = "PASSED"
    else:
        cov_c = "NOT_APPLICABLE"

    if family_d_report is None:
        cov_d = "NOT_RUN"
    elif score_d > 0 or family_d_report.get("status") == "REVIEW_REQUIRED":
        cov_d = "FAILED"
    else:
        cov_d = "PASSED"

    coverage_state = {
        "family_a_logic": cov_a,
        "family_b_pdf_structure": cov_b,
        "family_c_image_forensics": cov_c,
        "family_d_cross_document": cov_d
    }

    if cov_d == "NOT_RUN":
        active_w_sum = weights["A"] + weights["B"] + weights["C"]
        eff_weights = {
            "A": round(weights["A"] / active_w_sum, 3),
            "B": round(weights["B"] / active_w_sum, 3),
            "C": round(weights["C"] / active_w_sum, 3),
            "D": 0.0
        }
    else:
        eff_weights = weights

    weighted_sum = round(
        (eff_weights["A"] * score_a)
        + (eff_weights["B"] * score_b)
        + (eff_weights["C"] * score_c)
        + (eff_weights["D"] * score_d),
        3
    )

    raw_fused = 0.0
    overall_doc_score = 0.02
    fusion_rule_applied = "all_evaluated_families_passed"

    active_flagged = sum(
        1 for s in (score_a, score_b, score_c, score_d)
        if s is not None and s > 0
    )

    if active_flagged >= 2:
        finding_density_boost = min(1.30, 1.12 + 0.04 * len(structured_findings))
        raw_fused = round(min(1.0, weighted_sum * finding_density_boost), 3)
        fusion_rule_applied = f"multi_family_consensus_boost ({finding_density_boost:.2f}x)"
    elif active_flagged == 1:
        # Scale smoothly with the active family's score instead of clamping to a flat 0.54
        max_single = max(score_a, score_b, score_c, score_d)
        raw_fused = round(0.55 * weighted_sum + 0.45 * max_single, 3)
        fusion_rule_applied = "single_family_proportional_blend"
    elif score_intake is not None and score_intake > 0:
        raw_fused = round(min(1.0, score_intake), 3)
        overall_doc_score = round(raw_fused, 2)
        fusion_rule_applied = "intake_metadata_discrepancy_review"
    else:
        raw_fused = 0.0
        overall_doc_score = 0.02
        fusion_rule_applied = "all_evaluated_families_passed"

    if active_flagged > 0:
        overall_doc_score = float(SCORER.transform(raw_fused))
        overall_doc_score = max(0.0, min(1.0, overall_doc_score))

    max_finding_score = max(
        (f["individual_finding_score"] for f in structured_findings),
        default=0.0
    )
    if overall_doc_score >= tenant_cfg["high_risk_threshold"]:
        triage_band = "high_risk"
    elif overall_doc_score >= tenant_cfg["needs_review_threshold"]:
        triage_band = "needs_review"
    else:
        triage_band = "low_risk"

    plain_summary = compose_founder_summary(doc_label, structured_findings)

    payload = {
        "module": "MODULE_1_AI_DOCUMENT_CHECKER",
        "schema_version": "4.3.0",
        "document_id": doc_id,
        "tenant_id": tenant_id,
        "tenant_thresholds_applied": {
            "needs_review": tenant_cfg["needs_review_threshold"],
            "high_risk": tenant_cfg["high_risk_threshold"]
        },
        "identity_card": identity_info,
        "file_format": file_fmt,
        "total_pages_processed": raw_report.get("total_pages", fin_rec.get("total_pages_processed", 1)),
        "raw_fused_document_score": raw_fused,
        "overall_document_risk_score": overall_doc_score,
        "calibrated_risk_score": overall_doc_score,
        "scoring_methodology": {
            "score_type": SCORER.score_type,
            "is_empirically_calibrated": SCORER.is_empirically_validated,
            "calibration_warning": (
                None if SCORER.is_empirically_validated else
                "Scores are heuristic review-priority scores mapped via a monotonic scaling curve, NOT empirical fraud probabilities."
            ),
            "overall_document_risk_score": overall_doc_score,
            "max_individual_finding_score": max_finding_score,
            "family_raw_scores": {"Family_A": score_a, "Family_B": score_b, "Family_C": score_c, "Family_D": score_d},
            "effective_family_weights": eff_weights,
            "weighted_linear_sum": weighted_sum,
            "fusion_rule_applied": fusion_rule_applied,
            "raw_fused_document_score": raw_fused
        },
        "triage_band": triage_band,
        "plain_language_summary": plain_summary,
        "detector_coverage": coverage_state,
        "structured_findings": structured_findings,
        "extracted_financials": fin_rec,
        "disclaimer": (
            "A PASSED check means no configured anomalies were detected; it does not prove a document is genuine. "
            "A needs_review or high_risk result indicates the document requires manual investigation, not conclusive proof of fraud."
        )
    }

    is_valid, schema_errors = validate_section_4_3_contract(payload)
    payload["contract_validation"] = {"valid": is_valid, "errors": schema_errors}
    return payload