import os
import re
from typing import Optional, Tuple

import docx
import pymupdf

from ocr_compat import run_ocr_normalized

# Single canonical IFSC-prefix -> Issuing Bank map (41 Indian & foreign banks)
UNIVERSAL_IFSC_BANK_MAP = {
    "SBIN": "State Bank of India",
    "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank",
    "KKBK": "Kotak Mahindra Bank",
    "YESB": "Yes Bank",
    "PUNB": "Punjab National Bank",
    "BARB": "Bank of Baroda",
    "CNRB": "Canara Bank",
    "UBIN": "Union Bank of India",
    "IDIB": "Indian Bank",
    "BKID": "Bank of India",
    "CBIN": "Central Bank of India",
    "IOBA": "Indian Overseas Bank",
    "UCBA": "UCO Bank",
    "MAHB": "Bank of Maharashtra",
    "PSIB": "Punjab & Sind Bank",
    "IDFB": "IDFC First Bank",
    "INDB": "IndusInd Bank",
    "FDRL": "Federal Bank",
    "BDBL": "Bandhan Bank",
    "IBKL": "IDBI Bank",
    "RATN": "RBL Bank",
    "JAKA": "Jammu & Kashmir Bank",
    "KARB": "Karnataka Bank",
    "KVBL": "Karur Vysya Bank",
    "SIBL": "South Indian Bank",
    "TMBL": "Tamilnad Mercantile Bank",
    "CIUB": "City Union Bank",
    "JSBL": "Jana Small Finance Bank",
    "AUBL": "AU Small Finance Bank",
    "ESFB": "Equitas Small Finance Bank",
    "UJVN": "Ujjivan Small Finance Bank",
    "SURY": "Suryoday Small Finance Bank",
    "SCBL": "Standard Chartered Bank",
    "CITI": "Citibank",
    "HSBC": "HSBC Bank",
    "DBSS": "DBS Bank",
    "DEUT": "Deutsche Bank",
    "AIRP": "Airtel Payments Bank",
    "PYTM": "Paytm Payments Bank",
}

# Alias for backward compatibility with bank_parser/bank_schema.py
UNIVERSAL_IFSC_MAP = UNIVERSAL_IFSC_BANK_MAP

BANK_NAME_RE = re.compile(
    r"\b("
    r"Standard\s+Chartered(?:\s+Bank)?"
    r"|Bank\s+of\s+Baroda|bob\s+World|bob\s+Internet"
    r"|Bank\s+of\s+Maharashtra|mahabank|RBL\s*Bank"
    r"|Jammu\s*(?:&|and)\s*Kashmir\s*Bank|J\s*&\s*K\s*Bank"
    r"|ICICI\s*Bank|HDFC\s*Bank|Yes\s*Bank|Kotak\s+Mahindra\s+Bank"
    r"|Axis\s*Bank|IndusInd\s*Bank|State\s+Bank\s+of\s+India"
    r"|Punjab\s+National\s+Bank|Canara\s+Bank|Union\s+Bank\s+of\s+India"
    r"|IDFC\s+First\s+Bank|Federal\s+Bank|Bandhan\s+Bank|IDBI\s+Bank"
    r"|Indian\s+Bank|Bank\s+of\s+India|Central\s+Bank\s+of\s+India|UCO\s+Bank"
    r"|Karur\s+Vysya\s+Bank|Karnataka\s+Bank|South\s+Indian\s+Bank"
    r"|AU\s+Small\s+Finance\s+Bank|Ujjivan\s+Small\s+Finance\s+Bank|Equitas\s+Small\s+Finance\s+Bank"
    r")\b",
    re.IGNORECASE,
)

BAD_BRANCH_WORDS = {
    "details", "detail", "info", "information", "summary", "name",
    "code", "address", "ifsc", "micr", "phone", "email", "contact",
    "manager", "office", "number", "no", "id", "n/a", "na", "none",
}

_KEEP_UPPER_ENTITY_TOKENS = {
    "SM", "K2", "LLP", "PVT", "PVT.", "LTD", "LTD.", "HUF", "PROP",
    "M/S", "M/S.", "SRK", "JSH", "VS", "MS", "RK", "JK", "DR", "AR", "PK",
}


def normalize_issuing_bank_name(
    raw_val: Optional[str],
    fallback_bank: Optional[str] = None,
) -> Optional[str]:
    if not raw_val:
        return fallback_bank
    s = str(raw_val).strip()
    if len(s) < 3:
        return fallback_bank

    if re.match(r"^tmp[a-z0-9_]+\.[a-z0-9]+$", s.lower()) or s.lower().endswith(
        (".pdf", ".docx", ".png", ".jpg", ".jpeg")
    ):
        s_clean = re.sub(r"\.(?:pdf|docx|png|jpg|jpeg)$", "", s, flags=re.I)
    else:
        s_clean = s

    m_ifsc = re.search(r"\b([A-Za-z]{4})[0O][A-Za-z0-9]{6}\b", s_clean)
    if m_ifsc and m_ifsc.group(1).upper() in UNIVERSAL_IFSC_BANK_MAP:
        return UNIVERSAL_IFSC_BANK_MAP[m_ifsc.group(1).upper()]
    if len(s_clean) >= 4 and s_clean[:4].upper() in UNIVERSAL_IFSC_BANK_MAP and any(
        ch.isdigit() for ch in s_clean
    ):
        return UNIVERSAL_IFSC_BANK_MAP[s_clean[:4].upper()]

    low = re.sub(r"\s+", " ", s_clean).strip().lower()
    if "rbl" in low or "ratnakar" in low:
        return "RBL Bank"
    if "hdfc" in low:
        return "HDFC Bank"
    if "icici" in low:
        return "ICICI Bank"
    if "yes" in low:
        return "Yes Bank"
    if "axis" in low:
        return "Axis Bank"
    if "kotak" in low:
        return "Kotak Mahindra Bank"
    if "indusind" in low:
        return "IndusInd Bank"
    if "jammu" in low or "j&k" in low or "j & k" in low:
        return "Jammu & Kashmir Bank"
    if "state bank" in low or re.search(r"\bsbi\b", low):
        return "State Bank of India"
    if "punjab national" in low or re.search(r"\bpnb\b", low):
        return "Punjab National Bank"
    if "baroda" in low or low.startswith("bob") or re.search(r"\bbob\b", low):
        return "Bank of Baroda"
    if "maharashtra" in low or "mahabank" in low:
        return "Bank of Maharashtra"
    if "canara" in low:
        return "Canara Bank"
    if "union bank" in low:
        return "Union Bank of India"
    if "idfc" in low:
        return "IDFC First Bank"
    if "idbi" in low:
        return "IDBI Bank"
    if "federal" in low:
        return "Federal Bank"
    if "bandhan" in low:
        return "Bandhan Bank"
    if "au small" in low:
        return "AU Small Finance Bank"
    if "ujjivan" in low:
        return "Ujjivan Small Finance Bank"
    if "equitas" in low:
        return "Equitas Small Finance Bank"
    if "indian bank" in low:
        return "Indian Bank"
    if "bank of india" in low:
        return "Bank of India"
    if "central bank" in low:
        return "Central Bank of India"
    if "uco" in low:
        return "UCO Bank"
    if "karur" in low:
        return "Karur Vysya Bank"
    if "karnataka" in low:
        return "Karnataka Bank"
    if "south indian" in low:
        return "South Indian Bank"
    if "standard chartered" in low:
        return "Standard Chartered Bank"
    if (
        "bank" in low
        and not low.startswith("tmp")
        and "detected" not in low
        and not any(w in low for w in ("statement", "fy_", "fy-", "acct_stmt", "passbook", "transaction"))
        and "_" not in s_clean
    ):
        return s_clean.title()
    return fallback_bank


_normalize_issuing_bank_name = normalize_issuing_bank_name


def extract_branch_name(pre_table_text: str) -> Optional[str]:
    if not pre_table_text:
        return None
    m_br = re.search(
        r"\b(?:Branch\s*Name|Home\s*Branch|Account\s*Branch|Base\s*Branch)\b[\s:\|\-]+([A-Za-z][A-Za-z0-9 \t,\.\-&/]{2,50})",
        pre_table_text,
        re.IGNORECASE,
    )
    if not m_br:
        m_br = re.search(
            r"\bBranch\b\s*[:\|\-]+[\s:\|\-]*([A-Za-z][A-Za-z0-9 \t,\.\-&/]{2,50})",
            pre_table_text,
            re.IGNORECASE,
        )
    if not m_br:
        return None
    cand_br = re.sub(r"[ \t]+", " ", m_br.group(1).splitlines()[0].split("|")[0]).strip(" :-|.,")
    if (
        cand_br
        and cand_br.lower() not in BAD_BRANCH_WORDS
        and not any(
            k in cand_br.lower()
            for k in ("account type", "account no", "account number", "currency", "statement date", "ifsc", "micr")
        )
    ):
        return cand_br.upper()
    return None


def clean_ocr_person_name(raw_name: str) -> Tuple[str, float]:
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
    words = []
    for w in cleaned.split():
        w_up = w.upper()
        if w_up in _KEEP_UPPER_ENTITY_TOKENS or (len(w) == 2 and w.isalpha() and w.isupper()):
            words.append(w_up)
        else:
            words.append(w.title())
    return " ".join(words)


_format_entity_title = _format_entity_name_clean


def _is_address_or_noise_line(text: str) -> bool:
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


def _is_address_or_branch_line(text: str) -> bool:
    if _is_address_or_noise_line(text):
        return True
    low = text.lower().strip(" :-|.,")
    extra_exact = {
        "address 1", "address 2", "address 3", "account name", "account number",
        "jt. holder", "jt holder", "eerut", "city", "zip code", "rblbank", "rbl bank",
    }
    if low in extra_exact:
        return True
    extra_tokens = (
        "bank", "eerut", "ghaziabad", "muzaffarnagar", "srinagar", "jammu",
        "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad",
        "kashmir", "rajasthan", "gujarat", "punjab", "bihar", "west bengal",
        "karnataka", "tamil nadu", "account", "cust", "zip code", "nomination",
        "date", "page", "pin", "ckyc", "neft", "rtgs", "imps", "upi/", "by clg",
        "to clg", "transfer", "chq", "cheque", "joint holder", "opening balance",
        "closing balance", "drawing power", "sanction", "limit", "interest",
        "over draft", "fatehullapur", "fatehpur",
    )
    if any(b in low for b in extra_tokens):
        return True
    if re.search(r"\-\s*0?\d{2,4}$", low):
        return True
    return False


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
                        res = run_ocr_normalized(tmp_img)
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
        res = run_ocr_normalized(file_path)
        return "\n".join(r[1] for r in res) if res else ""
    return ""


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
            d = docx.Document(file_path)
            parts = [p.text for p in d.paragraphs if p.text.strip()]
            for tbl in d.tables:
                for row in tbl.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells if c.text.strip()))
            return "\n".join(parts)
    except Exception:
        return ""
    return ""


def _slice_page1_pre_table(page1_text: str) -> str:
    """Strictly slices Page 1 text BEFORE the transaction table header or first transaction row."""
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
        if low_ln in single_line_hdr_markers or low_ln.startswith(
            ("cheque no/", "withdra", "tran id", "si no", "sl no")
        ):
            break
        win_low = " ".join(lines[idx : min(len(lines), idx + 8)]).lower()
        if (
            "value" in win_low
            and "date" in win_low
            and "balance" in win_low
            and ("withdra" in win_low or "deposit" in win_low)
        ):
            break
        if (
            ("date" in low_ln or "transactio" in low_ln)
            and any(
                k in low_ln
                for k in (
                    "narration", "particular", "description", "remark",
                    "withdrawal", "withdrawl", "deposit", "debit", "credit", "balance",
                )
            )
            and not any(
                p in low_ln
                for p in ("statement period", "from date", "open date", "statement date", "as on")
            )
        ):
            break
        win5_low = " ".join(lines[idx : min(len(lines), idx + 5)]).lower()
        if (
            ("tran date" in win5_low or "value date" in win5_low)
            and ("narration" in win5_low or "particulars" in win5_low)
            and "balance" in win5_low
        ):
            break
        if date_row_re.match(ln) and not any(
            hk in low_ln for hk in ("period", "from", "as on", "open date", "expiry", "birth")
        ):
            win_tx = " ".join(lines[idx : min(len(lines), idx + 5)])
            if re.search(r"\d+\.\d{2}", win_tx) or any(
                k in win_tx.upper() for k in ("MBK/", "UPI", "NEFT", "RTGS", "IMPS", "CHQ")
            ):
                break

        pre_lines.append(ln)

    return "\n".join(pre_lines)


def _extract_holder_strictly_from_header(file_path: str, pre_table_text: str) -> Optional[str]:
    bad_words = [
        "statement of", "account statement", "detailed statement", "salary slip",
        "icici bank", "hdfc bank", "yes bank", "kotak mahindra", "state bank",
        "axis bank", "indusind bank", "standard chartered", "jammu", "rajasthan",
        "branch", "ifsc", "currency", "nominee", "period", "search", "customer id",
        "cust id", "scheme", "ckyc", "micr", "sanction", "drawing power", "interest rate",
        "opening balance", "closing balance", "page", "date of", "generated",
        "address", "city", "state", "phone", "email", "od limit", "account no",
        "a/c open", "account status", "regular", "product code", "joint holder", "nomination",
        "road", "street", "marg", "lane", "gali", "nagar", "mumbai", "meerut", "delhi", "plaza", "society",
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
        return sum(1 for ch in c if ch.isalpha()) >= 3

    if file_path and file_path.lower().endswith(".pdf") and os.path.exists(file_path):
        try:
            with pymupdf.open(file_path) as doc:
                if len(doc) > 0:
                    page = doc[0]
                    spans = []
                    for b in page.get_text("dict").get("blocks", []):
                        for l in b.get("lines", []):
                            line_txt = " ".join(
                                s.get("text", "").strip() for s in l.get("spans", []) if s.get("text", "").strip()
                            )
                            if line_txt:
                                bbox = l["bbox"]
                                spans.append(
                                    {"text": line_txt, "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}
                                )

                    label_re = re.compile(
                        r"^(?:account\s*holder(?:\s*name)?|account\s*name|customer\s*name|applicant\s*name|"
                        r"employee\s*name|legal\s*name|name\s*of\s*enterprise|name\s*of\s*customer|name)\s*[:\-]?$",
                        re.IGNORECASE,
                    )
                    for s in spans:
                        if s["y0"] > 380:
                            continue
                        if label_re.match(s["text"].strip()):
                            right_matches = [
                                other
                                for other in spans
                                if other["x0"] > s["x1"] - 6.0
                                and abs(other["y0"] - s["y0"]) <= 12.0
                                and _valid(other["text"].lstrip(":- "))
                            ]
                            if right_matches:
                                right_matches.sort(key=lambda item: item["x0"])
                                return right_matches[0]["text"].lstrip(":- ").strip()
        except Exception:
            pass

    lines = [ln.strip() for ln in pre_table_text.splitlines() if ln.strip()]

    for i, ln in enumerate(lines):
        if re.match(r"^(?:account\s*name|customer\s*name|holder\s*name|name)\s*[:\-]?$", ln, re.IGNORECASE):
            j = i + 1
            while j < min(len(lines), i + 8) and (
                lines[j].endswith(":")
                or re.match(
                    r"^(?:address|a/c|account|jt\.?\s*holder|joint\s*holder|cust(?:omer)?\s*id|ifsc|micr|branch)\b",
                    lines[j],
                    re.IGNORECASE,
                )
            ):
                j += 1
            if j < len(lines) and _valid(lines[j]):
                return lines[j].lstrip(":- ").strip()

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

    for ln in lines[:6]:
        cleaned_ln = re.sub(r"^(?:Mr\.?|Mrs\.?|Ms\.?|M/s\.?)\s+", "", ln, flags=re.IGNORECASE).strip()
        if (
            _valid(cleaned_ln)
            and cleaned_ln.upper() == cleaned_ln
            and ":" not in cleaned_ln
            and not any(
                k in cleaned_ln.lower()
                for k in ["bank", "statement", "account", "branch", "address", "india", "page", "limit", "inr", "date", "from", "to"]
            )
        ):
            return cleaned_ln
    return None


def _extract_bank_strictly_from_header(
    pre_table_text: str,
    full_text: str,
    file_path: str = "",
    original_filename: str = "",
) -> str:
    top_chunk = (full_text or "")[:1200]
    ifsc_m = re.search(r"\bIFSC(?:\s+Code)?\s*[:\-]?\s*([A-Z]{4})[0O][A-Z0-9]{6}\b", top_chunk, re.I)
    if ifsc_m and ifsc_m.group(1).upper() in UNIVERSAL_IFSC_BANK_MAP:
        return UNIVERSAL_IFSC_BANK_MAP[ifsc_m.group(1).upper()]

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

    found = [(m.start(), m.group(1)) for m in BANK_NAME_RE.finditer(pre_table_text or "")]
    if found:
        found.sort(key=lambda x: x[0])
        mapped = normalize_issuing_bank_name(found[0][1])
        if mapped:
            return mapped

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
        m_fn = BANK_NAME_RE.search(fn_cand)
        if m_fn:
            mapped = normalize_issuing_bank_name(m_fn.group(1))
            if mapped:
                return mapped

    m_rtgs = re.search(r"\bRTGS[/\-]([A-Z]{4})[HR5]\d{6,}", (full_text or "")[:2500].upper())
    if m_rtgs and m_rtgs.group(1) in UNIVERSAL_IFSC_BANK_MAP:
        return UNIVERSAL_IFSC_BANK_MAP[m_rtgs.group(1)]

    footer_slice = (full_text or "")[-600:]
    m_ftr = BANK_NAME_RE.search(footer_slice)
    if m_ftr:
        mapped = normalize_issuing_bank_name(m_ftr.group(1))
        if mapped:
            return mapped

    return "UNKNOWN"