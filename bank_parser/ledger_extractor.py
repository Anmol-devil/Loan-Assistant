import re
from typing import Dict, Any, List, Optional, Tuple
from .structural_scanner import DATE_TOKEN_RE

INLINE_DEC_RE = re.compile(
    r"(?<![A-Za-z0-9\./\-])([+-]?(?:(?:\d{1,3}(?:,\d{2,3})+|\d{1,9})[\.,:\-]\d{2}|\.\d{2})(?:\s*\(?[CD]r\.?\)?)?)(?![A-Za-z0-9\./\-])",
    re.I,
)

TIME_TOKEN_RE = re.compile(r"^\d{2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?$", re.I)

PAGE_LOCAL_STOP_MARKERS = (
    "page total",
    "total :",
    "total:",
    "grand total",
    "cumulative total",
    "transaction total",
    "statement summary",
    "summary for account",
    "b. summary",
    "total transaction count",
    "total debit count",
    "total credit count",
    "total debit amount",
    "total credit amount",
    "charge breakup",
    "end of statement",
    "++++ end",
    "opening balance:",
    "total withdrawals:",
    "total deposits:",
    "mandatory disclaimer",
    "transaction codes in your account",
)

NON_TX_ROW_MARKERS = (
    "page total",
    "total :",
    "total:",
    "grand total",
    "cumulative total",
    "transaction total",
    "statement summary",
    "summary for account",
    "total transaction count",
    "total debit count",
    "total credit count",
    "total debit amount",
    "total credit amount",
    "closing balance",
    "carried forward",
    "c/f",
    "charge breakup",
    "end of statement",
    "++++ end",
    "uncleared amount",
    "mandatory disclaimer",
)


def _clean_ocr_number(raw_str: str) -> str:
    s = (raw_str or "").strip()
    if DATE_TOKEN_RE.match(s) or TIME_TOKEN_RE.match(s):
        return s
    s = re.sub(r"^([+-]?[\d,]+)[:\-](\d{2})$", r"\1.\2", s)
    s = re.sub(r"^([+-]?[\d,]+),(\d{2})$", r"\1.\2", s)
    m_dots = re.match(r"^([+-]?\d{1,3}(?:\.\d{3})+)\.(\d{2})$", s)
    if m_dots:
        s = m_dots.group(1).replace(".", ",") + "." + m_dots.group(2)
    return s

def _parse_monetary_token(text: str) -> Tuple[Optional[float], Optional[str]]:
    if not text:
        return None, None
    s = _clean_ocr_number(text)
    if (DATE_TOKEN_RE.match(s) and len(s) <= 12) or TIME_TOKEN_RE.match(s):
        return None, None
    s_clean = re.sub(r"(?:INR|Rs\.?|₹|\$|\s)", "", s, flags=re.I)
    m = re.match(
        r"^([+-]?(?:(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?|\.\d{2}))(?:\(?([CD]r)\.?\)?)?([+-])?$",
        s_clean,
        re.I,
    )
    if not m:
        return None, None
    num_part = m.group(1).replace(",", "")
    if num_part.startswith("."):
        num_part = "0" + num_part
    elif num_part.startswith("-."):
        num_part = "-0." + num_part[2:]
    elif num_part.startswith("+."):
        num_part = "0." + num_part[2:]
    ind = m.group(2).upper() if m.group(2) else ("DR" if m.group(3) == "-" else None)
    try:
        return round(float(num_part), 2), ind
    except ValueError:
        return None, None

def _is_left_date_span(sp: Dict[str, Any]) -> bool:
    if not isinstance(sp, dict):
        return False
    txt = (sp.get("text") or "").strip()
    # Ignore date-range strings inside narration like '1-05-2024to29-06-2024'
    if re.search(r"\d{2,4}\s*to\s*\d{1,2}", txt, re.I):
        return False
    return (sp.get("cx_norm", 1.0) <= 0.38 or (sp.get("bbox_norm") or [1.0])[0] <= 0.28) and bool(
        DATE_TOKEN_RE.match(txt)
    )


def classify_page_role(page_obj: Dict[str, Any]) -> str:
    if not isinstance(page_obj, dict):
        return "INFO_DISCLAIMER"
    h_idx = page_obj.get("header_line_idx")
    date_rows = int(page_obj.get("date_row_count") or 0)
    clusters = page_obj.get("numeric_x_clusters") or []
    has_sum = bool(page_obj.get("has_summary_keywords"))

    if h_idx is not None and (date_rows >= 1 or len(clusters) >= 1):
        return "TRANSACTION"
    if date_rows >= 1 or len(clusters) >= 2:
        return "CONTINUATION"
    if has_sum:
        return "SUMMARY"
    return "INFO_DISCLAIMER"


def _adapt_columns_to_page(
    page_obj: Dict[str, Any],
    base_deb_cx: float,
    base_cred_cx: float,
    base_bal_cx: float,
    base_bal_x1: float,
) -> Tuple[float, float, float, float]:
    deb_cx, cred_cx, bal_cx, bal_x1 = base_deb_cx, base_cred_cx, base_bal_cx, base_bal_x1
    lines = page_obj.get("lines") or []
    h_idx = page_obj.get("header_line_idx")

    if h_idx is not None and h_idx < len(lines):
        hdr_spans = list(lines[h_idx])
        for k_off in range(1, 5):
            if h_idx + k_off < len(lines):
                nxt = lines[h_idx + k_off]
                if any(
                    DATE_TOKEN_RE.match(s.get("text", "")) or re.search(r"(?:\d+\.\d{2}|(?<![A-Za-z0-9])\.\d{2})", s.get("text", ""))
                    for s in nxt
                ):
                    break
                hdr_spans.extend(nxt)
        for s in hdr_spans:
            t_low = (s.get("text") or "").lower().strip()
            if any(k in t_low for k in ("withdrawal", "withdrawl", "debit", "dr amt", "dr amount")):
                deb_cx = float(s["cx_norm"])
            elif any(k in t_low for k in ("deposit", "credit", "cr amt", "cr amount")):
                cred_cx = float(s["cx_norm"])
            elif "balance" in t_low and "opening" not in t_low:
                bal_cx = float(s["cx_norm"])
                bal_x1 = float(s["bbox_norm"][2])

    p_clusters = page_obj.get("numeric_x_clusters") or []
    if p_clusters:
        rightmost = p_clusters[-1]
        if abs(rightmost["x1_normalized"] - bal_x1) <= 0.08:
            bal_x1 = float(rightmost["x1_normalized"])
            bal_cx = float(rightmost["cx_normalized"])
        if len(p_clusters) >= 3:
            c_deb = p_clusters[-3]["cx_normalized"]
            c_cred = p_clusters[-2]["cx_normalized"]
            if c_deb < c_cred < bal_cx:
                deb_cx = float(c_deb)
                cred_cx = float(c_cred)

    return deb_cx, cred_cx, bal_cx, bal_x1


def _extract_inline_summary_from_line(joined_low: str, stated_sum_dict: Dict[str, Any]) -> None:
    m_ob = re.search(r"opening\s+bal(?:ance)?\s*[:\-]\s*([+-]?[\d,]+\.\d{2}(?:\s*\(?[cd]r\)?)?)", joined_low, re.I)
    if m_ob and stated_sum_dict.get("opening_balance") is None:
        v, ind = _parse_monetary_token(m_ob.group(1))
        if v is not None:
            stated_sum_dict["opening_balance"] = -abs(v) if (ind == "DR" or m_ob.group(1).strip().startswith("-")) else v
            stated_sum_dict["opening_balance_source"] = "EXTRACTED_FROM_STATEMENT"

    m_td = re.search(r"total\s+(?:withdrawals?|withdrawl|debits?)\s*[:\-]\s*([\d,]+\.\d{2})", joined_low, re.I)
    if m_td and stated_sum_dict.get("total_debits") is None:
        v, _ = _parse_monetary_token(m_td.group(1))
        if v is not None and v >= 0:
            stated_sum_dict["total_debits"] = abs(v)

    m_tc = re.search(r"total\s+(?:deposits?|credits?)\s*[:\-]\s*([\d,]+\.\d{2})", joined_low, re.I)
    if m_tc and stated_sum_dict.get("total_credits") is None:
        v, _ = _parse_monetary_token(m_tc.group(1))
        if v is not None and v >= 0:
            stated_sum_dict["total_credits"] = abs(v)

    m_cb = re.search(
        r"closing\s+bal(?:ance)?(?!\s+figure|\s+includes)\s*[:\-]\s*([+-]?[\d,]+\.\d{2}(?:\s*\(?[cd]r\)?)?)",
        joined_low,
        re.I,
    )
    if m_cb and stated_sum_dict.get("closing_balance") is None:
        v, ind = _parse_monetary_token(m_cb.group(1))
        if v is not None:
            stated_sum_dict["closing_balance"] = -abs(v) if (ind == "DR" or m_cb.group(1).strip().startswith("-")) else v


def extract_transactions_with_blueprint(scan_data: Dict[str, Any], blueprint: Dict[str, Any]) -> Dict[str, Any]:
    scan_data = scan_data or {}
    blueprint = blueprint or {}
    tb = blueprint.get("table_structure_blueprint")
    if not isinstance(tb, dict):
        tb = {}
    cols = tb.get("columns")
    if not isinstance(cols, dict):
        cols = {}

    table_start_y = float(tb.get("table_start_y_points") or 0.0)

    bal_col = cols.get("balance") if isinstance(cols.get("balance"), dict) else {}
    deb_col = cols.get("debit") if isinstance(cols.get("debit"), dict) else {}
    cred_col = cols.get("credit") if isinstance(cols.get("credit"), dict) else {}

    pages_list = scan_data.get("pages") or []
    p1_clusters = (pages_list[0].get("numeric_x_clusters") or []) if pages_list and isinstance(pages_list[0], dict) else []
    default_bal_x1 = p1_clusters[-1]["x1_normalized"] if p1_clusters else 0.92
    default_bal_cx = p1_clusters[-1]["cx_normalized"] if p1_clusters else 0.85

    bal_x_range = bal_col.get("x_range_normalized")
    if isinstance(bal_x_range, list) and len(bal_x_range) >= 2 and bal_x_range[1] is not None:
        base_bal_x1 = float(bal_x_range[1])
    else:
        base_bal_x1 = float(default_bal_x1)

    base_bal_cx = float(bal_col.get("cx_normalized") if bal_col.get("cx_normalized") is not None else default_bal_cx)
    base_deb_cx = float(deb_col["cx_normalized"]) if deb_col.get("cx_normalized") is not None else (
        p1_clusters[-3]["cx_normalized"] if len(p1_clusters) >= 3 else 0.62
    )
    base_cred_cx = float(cred_col["cx_normalized"]) if cred_col.get("cx_normalized") is not None else (
        p1_clusters[-2]["cx_normalized"] if len(p1_clusters) >= 3 else 0.74
    )

    stated_sum_dict = blueprint.get("stated_summary")
    if not isinstance(stated_sum_dict, dict):
        stated_sum_dict = {}
        blueprint["stated_summary"] = stated_sum_dict

    transactions: List[Dict[str, Any]] = []
    per_page_stats: List[Dict[str, Any]] = []
    aligned_numeric_hits = 0
    total_numeric_spans = 0
    total_candidate_balances = 0

    # Keep all monetary numbers found on each page (including orphan sub-lines) for post-reversal delta healing
    page_all_numbers: Dict[int, List[Tuple[float, float]]] = {}

    for page_obj in pages_list:
        if not isinstance(page_obj, dict):
            continue
        p_num = page_obj.get("page") or 1
        page_role = classify_page_role(page_obj)
        deb_cx_norm, cred_cx_norm, bal_cx_norm, bal_x1_norm = _adapt_columns_to_page(
            page_obj, base_deb_cx, base_cred_cx, base_bal_cx, base_bal_x1
        )

        # Record all monetary numbers on this page for boundary healing
        p_nums: List[Tuple[float, float]] = []
        for sp in page_obj.get("spans") or []:
            txt_s = (sp.get("text") or "").strip()
            v_s, _ = _parse_monetary_token(txt_s)
            if v_s is not None and ("." in _clean_ocr_number(txt_s) or "," in txt_s):
                p_nums.append((float(sp.get("cy_points", 0.0)), abs(v_s)))
            else:
                for m_in in INLINE_DEC_RE.finditer(txt_s):
                    v_in, _ = _parse_monetary_token(m_in.group(1))
                    if v_in is not None:
                        p_nums.append((float(sp.get("cy_points", 0.0)), abs(v_in)))
        page_all_numbers[p_num] = p_nums

        lines = page_obj.get("lines") or []
        h_idx = page_obj.get("header_line_idx")

        first_dated_idx = None
        for idx_l, ln in enumerate(lines):
            if any(_is_left_date_span(sp) for sp in ln):
                ln_low = " ".join((s.get("text") or "").lower() for s in ln)
                if any(
                    hdr_k in ln_low
                    for hdr_k in ("from:", "to:", "from date", "to date", "open date", "a/c open", "generated on", "period:")
                ):
                    continue
                first_dated_idx = idx_l
                break
        if h_idx is not None:
            start_line = h_idx + 1
        else:
            start_line = 0

        body_lines: List[List[Dict[str, Any]]] = []
        in_summary_section = False

        for ln in lines[start_line:]:
            if not ln:
                continue
            if (
                p_num == 1
                and table_start_y > 0
                and h_idx is None
                and max(s["bbox_points"][1] for s in ln) < table_start_y - 2.0
            ):
                continue
            joined_low = " ".join((s.get("text") or "").lower() for s in ln).strip()

            if any(
                k in joined_low
                for k in ("opening balance", "total withdrawal", "total deposit", "total debit", "total credit", "closing bal")
            ):
                _extract_inline_summary_from_line(joined_low, stated_sum_dict)

            if any(sm in joined_low for sm in PAGE_LOCAL_STOP_MARKERS):
                if body_lines and not any(_is_left_date_span(s) for s in body_lines[-1]):
                    if "page total" in joined_low:
                        body_lines.pop()
                in_summary_section = True
                continue

            if in_summary_section:
                if not any(_is_left_date_span(s) for s in ln):
                    continue
                in_summary_section = False

            if any(
                ign in joined_low
                for ign in [
                    "page no.:", "we understand your world", "closing balance includes funds",
                    "contents of this statement", "registered office address:", "state account branch gstn",
                    "statement of account", "computer generated", "system generated", "printed by",
                    "od limit", "account branch", "cust id", "a/c open date", "yes touch phonebanking",
                    "cin l65190", "email us at", "sms \"help\""
                ]
            ):
                continue

            if not any(re.search(r"(?:\d+\.\d{2}|(?<![A-Za-z0-9])\.\d{2})", s.get("text", "")) for s in ln):
                if (
                    any(
                        h_w in joined_low
                        for h_w in (
                            "transactio", "n date", "value date", "transaction remarks", "withdrawl",
                            "amt (inr)", "balance (in", "cheque no/reference"
                        )
                    )
                    or joined_low in ("r)", "n", "id", "inr)")
                ) and not any(_is_left_date_span(s) for s in ln):
                    continue

            body_lines.append(ln)

        def is_strictly_balance_col_span(sp: Dict[str, Any]) -> bool:
            cx_n = float(sp.get("cx_norm", 0.0))
            x1_n = float((sp.get("bbox_norm") or [0, 0, 0, 0])[2])
            dist_bal = min(abs(cx_n - bal_cx_norm), abs(x1_n - bal_x1_norm))
            dist_deb = abs(cx_n - deb_cx_norm) if deb_cx_norm else 9.0
            dist_cred = abs(cx_n - cred_cx_norm) if cred_cx_norm else 9.0

            if dist_bal > min(dist_deb, dist_cred) + 0.015 and x1_n < bal_x1_norm - 0.04:
                return False
            return dist_bal <= 0.10 or x1_n >= max(0.84, bal_x1_norm - 0.05)

        def get_line_balance_spans(ln_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            joined_l = " ".join((s.get("text") or "").lower() for s in ln_spans)
            if any(sm in joined_l for sm in NON_TX_ROW_MARKERS):
                return []
            b_spans = []
            for sp in ln_spans:
                txt_sp = sp.get("text") or ""
                val, _ = _parse_monetary_token(txt_sp)
                if val is not None and ("." in _clean_ocr_number(txt_sp) or "," in txt_sp):
                    if is_strictly_balance_col_span(sp):
                        b_spans.append(sp)
                elif INLINE_DEC_RE.search(txt_sp) and sp["bbox_norm"][2] >= max(0.84, bal_x1_norm - 0.05):
                    b_spans.append(sp)
            return b_spans

        unpacked_body_lines: List[List[Dict[str, Any]]] = []
        for ln in body_lines:
            b_sps = get_line_balance_spans(ln)
            if len(b_sps) >= 2:
                b_sps.sort(key=lambda s: s["cy_points"])
                b_cys = [bs["cy_points"] for bs in b_sps]
                sub_lns: List[List[Dict[str, Any]]] = [[] for _ in b_cys]
                for sp in ln:
                    best_k = min(range(len(b_cys)), key=lambda k: abs(sp["cy_points"] - b_cys[k]))
                    sub_lns[best_k].append(sp)
                for sl in sub_lns:
                    if sl:
                        sl.sort(key=lambda s: s["bbox_points"][0])
                        unpacked_body_lines.append(sl)
            else:
                unpacked_body_lines.append(ln)
        body_lines = unpacked_body_lines

        page_candidate_balances = sum(len(get_line_balance_spans(ln)) for ln in body_lines)
        total_candidate_balances += page_candidate_balances

        bal_anchor_idxs = [
            i for i, ln in enumerate(body_lines)
            if len(get_line_balance_spans(ln)) > 0
            or any(k in " ".join((s.get("text") or "").lower() for s in ln) for k in ("opening balance", "brought forward", "b/f"))
        ]

        if len(bal_anchor_idxs) >= 1:
            anchor_idxs = bal_anchor_idxs
        else:
            def fallback_dated_anchor(ln_spans: List[Dict[str, Any]]) -> bool:
                joined_l = " ".join((s.get("text") or "").lower() for s in ln_spans)
                if any(sm in joined_l for sm in NON_TX_ROW_MARKERS):
                    return False
                has_left_date = any(_is_left_date_span(sp) for sp in ln_spans)
                has_dec = any(
                    (_parse_monetary_token(sp.get("text") or "")[0] is not None and "." in _clean_ocr_number(sp.get("text") or ""))
                    or bool(INLINE_DEC_RE.search(sp.get("text") or ""))
                    for sp in ln_spans
                )
                return has_left_date and has_dec

            anchor_idxs = [i for i, ln in enumerate(body_lines) if fallback_dated_anchor(ln)]

        if not anchor_idxs:
            per_page_stats.append({
                "page": p_num,
                "page_role": page_role,
                "candidate_balances": page_candidate_balances,
                "transactions_extracted": 0,
            })
            continue

        anchor_cys = [
            sum(s["cy_points"] for s in body_lines[a_i]) / len(body_lines[a_i])
            for a_i in anchor_idxs
        ]

        # Determine if Date + Debit/Credit sits on a sub-line BELOW the Balance anchor (PNB layout)
        # or ABOVE the Balance anchor.
        anchors_with_dates = sum(1 for a_i in anchor_idxs if any(_is_left_date_span(s) for s in body_lines[a_i]))
        dates_on_separate_subline = (anchors_with_dates < len(anchor_idxs) * 0.5)
        dated_line_cys = [
            sum(s["cy_points"] for s in ln) / len(ln)
            for ln in body_lines
            if any(_is_left_date_span(s) for s in ln)
        ]
        # In PNB, the first Balance line on the page is ABOVE the first Date+Amount line (anchor_cys[0] < dated_line_cys[0])
        subline_sits_below_anchor = (
            dates_on_separate_subline
            and bool(dated_line_cys)
            and anchor_cys[0] < dated_line_cys[0] - 1.5
        )

        grouped_blocks: List[List[Dict[str, Any]]] = [[] for _ in anchor_idxs]
        for l_i, ln in enumerate(body_lines):
            if l_i in anchor_idxs:
                pos = anchor_idxs.index(l_i)
                grouped_blocks[pos].extend(ln)
            else:
                ln_cy = sum(s["cy_points"] for s in ln) / len(ln)
                if dates_on_separate_subline:
                    if subline_sits_below_anchor:
                        above_cands = [k for k, a_cy in enumerate(anchor_cys) if a_cy <= ln_cy + 3.0]
                        best_pos = above_cands[-1] if above_cands else 0
                    else:
                        below_cands = [k for k, a_cy in enumerate(anchor_cys) if a_cy >= ln_cy - 3.0]
                        best_pos = below_cands[0] if below_cands else (len(anchor_cys) - 1)
                else:
                    best_pos = min(range(len(anchor_cys)), key=lambda k: abs(ln_cy - anchor_cys[k]))

                if abs(ln_cy - anchor_cys[best_pos]) <= 48.0:
                    grouped_blocks[best_pos].extend(ln)

        page_tx_count = 0

        for block_spans in grouped_blocks:
            block_spans.sort(key=lambda s: (round(s["cy_points"], 1), s["cx_points"]))
            row_joined = " ".join((s.get("text") or "").lower() for s in block_spans)

            if any(sm in row_joined for sm in NON_TX_ROW_MARKERS) and not any(
                u in row_joined for u in ["upi", "neft", "rtgs", "imps", "chq"]
            ):
                continue

            if any(k in row_joined for k in ["opening balance", "brought forward", "b/f"]) and not any(
                u in row_joined for u in ["upi", "neft", "rtgs", "imps", "chq"]
            ):
                for sp in block_spans:
                    txt_sp = sp.get("text") or ""
                    v_ob, i_ob = _parse_monetary_token(txt_sp)
                    if v_ob is not None and ("." in _clean_ocr_number(txt_sp) or "," in txt_sp):
                        signed_ob = -abs(v_ob) if (i_ob == "DR" or txt_sp.strip().startswith("-")) else v_ob
                        stated_sum_dict["opening_balance"] = signed_ob
                        stated_sum_dict["opening_balance_source"] = "EXTRACTED_FROM_STATEMENT"
                continue

            date_str = None
            desc_parts: List[str] = []
            row_monetary_tokens: List[Tuple[float, float, float, Optional[str], str]] = []
            dr_cr_ind = None
            bal_dr_cr_ind = None

            for sp in block_spans:
                txt = (sp.get("text") or "").strip()
                if not txt or TIME_TOKEN_RE.match(txt):
                    continue

                if _is_left_date_span(sp):
                    m_dt = DATE_TOKEN_RE.match(txt)
                    if m_dt:
                        if not date_str:
                            date_str = m_dt.group(0)
                        txt = txt[m_dt.end():].strip()
                        if not txt or TIME_TOKEN_RE.match(txt):
                            continue

                if re.match(r"^\(?([CD]r)\.?\)?$", txt, re.I):
                    flg = "DR" if "D" in txt.upper() else "CR"
                    if sp["cx_norm"] >= 0.70 or abs(sp["cx_norm"] - bal_cx_norm) <= 0.14:
                        bal_dr_cr_ind = flg
                    else:
                        dr_cr_ind = flg
                    continue

                val, ind = _parse_monetary_token(txt)
                is_int_amt_in_col = (
                    val is not None
                    and re.fullmatch(r"\d{1,3}", txt) is not None
                    and min(abs(sp["cx_norm"] - deb_cx_norm), abs(sp["cx_norm"] - cred_cx_norm)) <= 0.055
                )
                if val is not None and ("." in _clean_ocr_number(txt) or "," in txt or is_int_amt_in_col):
                    total_numeric_spans += 1
                    row_monetary_tokens.append((sp["cx_norm"], sp["bbox_norm"][2], val, ind, txt))
                elif " " in txt and INLINE_DEC_RE.search(txt):
                    x0_n, _, x1_n, _ = sp["bbox_norm"]
                    w_n = max(0.02, x1_n - x0_n)
                    t_len = max(1, len(txt))
                    last_end = 0
                    for m_in in INLINE_DEC_RE.finditer(txt):
                        prefix_txt = txt[last_end:m_in.start(1)].strip()
                        if prefix_txt and not TIME_TOKEN_RE.match(prefix_txt):
                            desc_parts.append(prefix_txt)
                        sub_t = m_in.group(1)
                        sub_v, sub_i = _parse_monetary_token(sub_t)
                        if sub_v is not None:
                            total_numeric_spans += 1
                            sub_cx = round(x0_n + ((m_in.start(1) + m_in.end(1)) / (2.0 * t_len)) * w_n, 4)
                            sub_x1 = round(x0_n + (m_in.end(1) / t_len) * w_n, 4)
                            row_monetary_tokens.append((sub_cx, sub_x1, sub_v, sub_i, sub_t))
                        last_end = m_in.end(1)
                    suffix_txt = txt[last_end:].strip()
                    if suffix_txt and not TIME_TOKEN_RE.match(suffix_txt):
                        desc_parts.append(suffix_txt)
                else:
                    desc_parts.append(txt)

            if not row_monetary_tokens:
                continue

            desc_clean = re.sub(r"^(?:R\)|n\)|Id)\s+", "", " ".join(desc_parts).strip())
            if any(sm in desc_clean.lower() for sm in NON_TX_ROW_MARKERS):
                continue

            if not date_str:
                date_str = transactions[-1]["date"] if transactions else f"Page-{p_num}"

            row_monetary_tokens.sort(key=lambda item: item[0])
            b_cx, b_x1, b_v, b_ind, b_txt = row_monetary_tokens[-1]

            if len(row_monetary_tokens) == 1 and abs(b_cx - bal_cx_norm) > 0.11 and b_x1 < bal_x1_norm - 0.06:
                if transactions and transactions[-1]["credit"] == 0.0 and transactions[-1]["debit"] == 0.0:
                    prev_tx = transactions[-1]
                    if abs(b_cx - deb_cx_norm) <= abs(b_cx - cred_cx_norm):
                        prev_tx["debit"] = abs(b_v)
                    else:
                        prev_tx["credit"] = abs(b_v)
                    if date_str and prev_tx["date"].startswith("Page-"):
                        prev_tx["date"] = date_str
                    if desc_clean:
                        prev_tx["description"] = f"{prev_tx['description']} {desc_clean}".strip()
                continue

            aligned_numeric_hits += 1
            eff_b_ind = b_ind or bal_dr_cr_ind
            bal_val = -abs(b_v) if (eff_b_ind == "DR" or b_txt.startswith("-")) else b_v
            non_bal_tokens = [
                t for t in row_monetary_tokens[:-1]
                if abs(t[0] - bal_cx_norm) > 0.06 and t[1] < bal_x1_norm - 0.04
            ]

            debit_val = 0.0
            credit_val = 0.0
            non_bal_tokens = row_monetary_tokens[:-1]
            non_zero_cands = [t for t in non_bal_tokens if abs(t[2]) > 0.001]
            active_cands = non_zero_cands if non_zero_cands else non_bal_tokens

            prev_b = (
                transactions[-1]["balance"]
                if transactions
                else stated_sum_dict.get("opening_balance")
            )

            if active_cands:
                aligned_numeric_hits += 1
                chosen_cand = active_cands[-1]
                if prev_b is not None and len(active_cands) >= 2:
                    expected_delta = round(bal_val - prev_b, 2)
                    for cand in active_cands:
                        if abs(abs(cand[2]) - abs(expected_delta)) <= 1.0:
                            chosen_cand = cand
                            break

                a_cx, a_x1, a_v, a_ind, a_txt = chosen_cand
                is_neg_token = a_txt.strip().startswith("-") or (a_v < 0)

                if is_neg_token and (prev_b is not None and abs((prev_b + abs(a_v)) - bal_val) <= 1.0):
                    debit_val = -abs(a_v)
                    credit_val = 0.0
                elif "reversal ft-dr" in desc_clean.lower() and (prev_b is not None and abs((prev_b + abs(a_v)) - bal_val) <= 1.0):
                    debit_val = -abs(a_v)
                    credit_val = 0.0
                elif a_ind == "DR" or dr_cr_ind == "DR":
                    debit_val = abs(a_v)
                elif a_ind == "CR" or dr_cr_ind == "CR":
                    credit_val = abs(a_v)
                elif deb_cx_norm is not None and cred_cx_norm is not None and abs(deb_cx_norm - cred_cx_norm) > 0.025:
                    d_dist = abs(a_cx - deb_cx_norm)
                    c_dist = abs(a_cx - cred_cx_norm)
                    if d_dist <= c_dist:
                        debit_val = -abs(a_v) if is_neg_token else abs(a_v)
                    else:
                        credit_val = -abs(a_v) if is_neg_token else abs(a_v)
                elif prev_b is not None and (
                    abs((prev_b - abs(a_v)) - bal_val) <= 1.0
                    or (prev_b < 0 and bal_val > 0 and abs((prev_b - abs(a_v)) - (-bal_val)) <= 1.0)
                ):
                    debit_val = abs(a_v)
                elif prev_b is not None and (
                    abs((prev_b + abs(a_v)) - bal_val) <= 1.0
                    or (prev_b < 0 and bal_val > 0 and abs((prev_b + abs(a_v)) - (-bal_val)) <= 1.0)
                ):
                    credit_val = abs(a_v)
                else:
                    if prev_b is not None and bal_val < prev_b:
                        debit_val = abs(a_v)
                    else:
                        credit_val = abs(a_v)

            rx0 = min(s["bbox_points"][0] for s in block_spans)
            ry0 = min(s["bbox_points"][1] for s in block_spans)
            rx1 = max(s["bbox_points"][2] for s in block_spans)
            ry1 = max(s["bbox_points"][3] for s in block_spans)

            transactions.append({
                "page": p_num,
                "bbox": [round(rx0, 2), round(ry0, 2), round(rx1, 2), round(ry1, 2)],
                "date": date_str,
                "description": desc_clean,
                "credit": round(credit_val, 2),
                "debit": round(debit_val, 2),
                "balance": round(bal_val, 2),
            })
            page_tx_count += 1

        per_page_stats.append({
            "page": p_num,
            "page_role": page_role,
            "candidate_balances": page_candidate_balances,
            "transactions_extracted": page_tx_count,
        })

    # Detect Reverse Chronological Order (e.g. PNB OpTransactionHistory)
    if len(transactions) >= 3:
        fwd_hits = sum(
            1 for i in range(len(transactions) - 1)
            if abs((transactions[i]["balance"] + transactions[i + 1]["credit"] - transactions[i + 1]["debit"]) - transactions[i + 1]["balance"]) <= 1.0
        )
        rev_hits = sum(
            1 for i in range(len(transactions) - 1)
            if abs((transactions[i + 1]["balance"] + transactions[i]["credit"] - transactions[i]["debit"]) - transactions[i]["balance"]) <= 1.0
        )
        if rev_hits > fwd_hits and rev_hits >= (len(transactions) - 1) * 0.45:
            transactions.reverse()
            tb["chronological_order"] = "REVERSE_CHRONOLOGICAL"

    # Post-Reversal Cross-Page & Boundary Delta Healing:
    for idx in range(1, len(transactions)):
        prev_b = transactions[idx - 1]["balance"]
        tx = transactions[idx]
        delta = round(tx["balance"] - prev_b, 2)
        abs_delta = abs(delta)
        curr_net = round(tx["credit"] - tx["debit"], 2)

        if abs_delta > 0 and (
            abs(curr_net - delta) > 1.0
            or (tx["credit"] == 0.0 and tx["debit"] == 0.0 and abs_delta >= 0.01)
        ):
            p_curr = tx["page"]
            row_cy = (tx["bbox"][1] + tx["bbox"][3]) / 2.0
            found_on_page = False

            # Check current page (within 65pt of row_cy or at page top/bottom) and adjacent pages (top/bottom boundary)
            for p_search in (p_curr, p_curr - 1, p_curr + 1):
                for sp_cy, sp_val in page_all_numbers.get(p_search, []):
                    if abs(sp_val - abs_delta) <= 0.5:
                        if p_search == p_curr and abs(sp_cy - row_cy) <= 75.0:
                            found_on_page = True
                            break
                        elif p_search != p_curr:
                            found_on_page = True
                            break
                if found_on_page:
                    break

            if found_on_page:
                if delta > 0:
                    tx["credit"], tx["debit"] = round(abs_delta, 2), 0.0
                else:
                    tx["debit"], tx["credit"] = round(abs_delta, 2), 0.0
    is_od_summary = (
        (stated_sum_dict.get("opening_balance") is not None and stated_sum_dict["opening_balance"] < 0)
        or (stated_sum_dict.get("closing_balance") is not None and stated_sum_dict["closing_balance"] < 0)
    )
    if not is_od_summary:
        for i in range(1, len(transactions) - 1):
            prev_b = transactions[i - 1]["balance"]
            curr_t = transactions[i]
            next_t = transactions[i + 1]
            if curr_t["debit"] < 0 or curr_t["credit"] < 0 or next_t["debit"] < 0 or next_t["credit"] < 0:
                continue
            amt1 = max(abs(curr_t["debit"]), abs(curr_t["credit"]))
            amt2 = max(abs(next_t["debit"]), abs(next_t["credit"]))
            if amt1 > 0 and amt2 > 0:
                curr_ok = abs((prev_b + curr_t["credit"] - curr_t["debit"]) - curr_t["balance"]) <= 1.0
                if not curr_ok:
                    for s1 in (-1, 1):
                        for s2 in (-1, 1):
                            if abs((prev_b + s1 * amt1 + s2 * amt2) - next_t["balance"]) <= 1.5:
                                if s1 < 0:
                                    curr_t["debit"], curr_t["credit"] = amt1, 0.0
                                else:
                                    curr_t["credit"], curr_t["debit"] = amt1, 0.0
                                if s2 < 0:
                                    next_t["debit"], next_t["credit"] = amt2, 0.0
                                else:
                                    next_t["credit"], next_t["debit"] = amt2, 0.0
                                break

    if len(transactions) >= 2 and stated_sum_dict.get("opening_balance") is None:
        r1 = transactions[0]
        r1_amt = max(abs(r1["debit"]), abs(r1["credit"]))
        if r1_amt > 0:
            desc_r1 = (r1.get("description") or "").upper()
            if any(k in desc_r1 for k in ("ACH DR", "RTGS/RATN", "NEFT DR", "RTGS DR", "CHQ PAID", "WITHDRAW", "DR-", "DEBIT")):
                r1["debit"], r1["credit"] = r1_amt, 0.0
            elif any(k in desc_r1 for k in ("NEFT CR", "RTGS CR", "CHQ DEP", "CASH DEP", "IMPS", "FROM ")):
                r1["credit"], r1["debit"] = r1_amt, 0.0

    if transactions and stated_sum_dict.get("opening_balance") is None:
        first_tx = transactions[0]
        stated_sum_dict["opening_balance"] = round(first_tx["balance"] - first_tx["credit"] + first_tx["debit"], 2)
        stated_sum_dict["opening_balance_source"] = "INFERRED_FROM_FIRST_TRANSACTION"
    elif stated_sum_dict.get("opening_balance") is not None and not stated_sum_dict.get("opening_balance_source"):
        stated_sum_dict["opening_balance_source"] = "EXTRACTED_FROM_STATEMENT"

    alignment_ratio = min(1.0, (aligned_numeric_hits / max(1, total_numeric_spans))) if total_numeric_spans > 0 else 0.0
    candidate_coverage_ratio = (
        min(1.0, len(transactions) / float(max(1, total_candidate_balances)))
        if total_candidate_balances > 0
        else 1.0
    )

    return {
        "transactions": transactions,
        "per_page_summary": per_page_stats,
        "numeric_alignment_ratio": round(alignment_ratio, 3),
        "valid_anchor_rows_found": len(transactions),
        "total_candidate_balances": total_candidate_balances,
        "candidate_coverage_ratio": round(candidate_coverage_ratio, 3),
        "chronological_order": tb.get("chronological_order", "CHRONOLOGICAL"),
    }