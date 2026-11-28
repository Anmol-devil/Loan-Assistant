import re
from typing import Any
from typing import Optional, Tuple

# Pre-compiled canonical Indian currency regexes (fast, module-level cache)
# Matches: 12,34,567.89 | 1,234.50 | 450.00 | .67 | -.67 | 12,345.00Dr | (500.00)
MONEY_STRICT_RE = re.compile(
    r"(?:(?<!\d)[+\-−]?\s*(?:₹|Rs\.?|INR)?\s*)?"
    r"(?:\(\s*)?"
    r"(?:\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{1,2}|\.\d{1,2})"
    r"(?:\s*\))?"
    r"(?:\s*(?:Dr|Cr|DR|CR)\.?)?"
)

# Permissive token check for single-cell numeric values (supports IndusInd small integers like '1', '5', '100')
MONEY_CELL_RE = re.compile(
    r"^\s*(?:₹|Rs\.?|INR)?\s*([+\-−]?)\s*(\(?)\s*"
    r"(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?|\.\d{1,2})"
    r"\s*(\)?)\s*(Dr|Cr|DR|CR)?\.?\s*$"
)

# Common Indian bank statement dates: DD/MM/YYYY, DD-MM-YY, DD-Mon-YYYY, DD Mon YYYY
DATE_TOKEN_RE = re.compile(
    r"\b(?:"
    r"(?:0?[1-9]|[12]\d|3[01])[-/\.](?:0?[1-9]|1[0-2])[-/\.](?:\d{4}|\d{2})"
    r"|"
    r"(?:0?[1-9]|[12]\d|3[01])[\-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[\-\s,]+(?:\d{4}|\d{2})"
    r"|"
    r"(?:\d{4})[-/\.](?:0?[1-9]|1[0-2])[-/\.](?:0?[1-9]|[12]\d|3[01])"
    r")\b",
    re.IGNORECASE,
)


def parse_amount_with_sign(
    raw: Any,
    allow_small_integer: bool = True,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Parses Indian bank currency strings into (signed_float, dr_cr_tag).
    Handles:
      - Lakhs/crores commas: '12,34,567.89' -> (1234567.89, None)
      - Axis sub-₹1 decimals: '.67' -> (0.67, None)
      - J&K no-space suffixes: '45,210.50Dr' -> (-45210.50, 'DR')
      - Parenthesized negatives: '(1,250.00)' -> (-1250.0, 'DR')
      - IndusInd small integers: '1' -> (1.0, None) when allow_small_integer=True
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s or s in ("-", "--", "NA", "N/A", "NIL"):
        return None, None

    m = MONEY_CELL_RE.match(s)
    if not m:
        return None, None

    sign_char, open_p, num_str, close_p, drcr = m.groups()
    if (open_p and not close_p) or (close_p and not open_p):
        return None, None

    # Guard against bare integers unless allowed (prevents picking up cheque/ref numbers)
    if "." not in num_str and "," not in num_str:
        if not allow_small_integer or len(num_str) > 6:
            return None, None

    clean_num = num_str.replace(",", "")
    if clean_num.startswith("."):
        clean_num = "0" + clean_num

    try:
        val = float(clean_num)
    except ValueError:
        return None, None

    tag = drcr.upper() if drcr else None
    is_neg = (sign_char in ("-", "−")) or bool(open_p) or (tag == "DR")
    if is_neg and val > 0:
        val = -val

    return round(val, 2), tag


def parse_indian_float(raw: Any, allow_small_integer: bool = True) -> Optional[float]:
    """Convenience wrapper returning only the parsed float (or None)."""
    val, _ = parse_amount_with_sign(raw, allow_small_integer=allow_small_integer)
    return val


def is_date_token(text: str) -> bool:
    """Fast check for whether a string contains a valid Indian statement date."""
    return bool(text and DATE_TOKEN_RE.search(text))