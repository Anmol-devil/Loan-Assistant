import os
import sys
import json
import re

def reconcile_spans(pages_spans, y_tolerance=6.0, full_text=""):
    issues = []
    detailed_issues = []
    transactions = []
    
    from multiformated_reconciliation import (
        extract_summary_from_full_text,
        extract_summary_cards,
        Stitch_Vertical_Wrapped_Numbers,
        parse_amount,
        group_into_rows,
        DATE_RE,
        parse_amount_with_sign,
        normalize_ocr_amount_str,
        run_math_checks,
    )
    summary = {
        "opening_balance": [],
        "summary_credits": [],
        "summary_debits": [],
        "closing_balance": []
    }

    if full_text:
        txt_summary = extract_summary_from_full_text(full_text)
        for k, vals in txt_summary.items():
            items = [vals] if isinstance(vals, str) else vals
            target = summary.setdefault(k, [])
            for v in items:
                if v not in target:
                    target.append(v)

    last_known_header_map = {}
    per_page_stats = []
    last_valid_sl_no = 0
    for page_idx, page_spans in enumerate(pages_spans, start=1):
        initial_rows = group_into_rows(page_spans, y_tolerance=y_tolerance)
        header_map = {}
        header_y = None
        header_min_y = None

        for r_idx, row in enumerate(initial_rows):
            window_spans = list(row)
            curr_cy = sum(s["cy"] for s in row) / len(row)
            for k_off in range(1, 5):
                if r_idx + k_off < len(initial_rows):
                    next_row = initial_rows[r_idx + k_off]
                    next_has_date = any(DATE_RE.match(s["text"].strip()) for s in next_row)
                    next_has_decimal = any(re.search(r"\d+\.\d{2}", s["text"]) for s in next_row)
                    next_cy = sum(s["cy"] for s in next_row) / len(next_row)

                    if not next_has_date and not next_has_decimal and abs(next_cy - curr_cy) <= 42.0:
                        window_spans.extend(next_row)
                    else:
                        break

            joined = " ".join(s["text"].lower() for s in window_spans)
            if (
                ("date" in joined or "transactio" in joined)
                and "balance" in joined
                and "opening balance" not in joined
                and "dr count" not in joined
                and "recover date" not in joined
                and any(k in joined for k in [
                    "credit", "debit", "deposit", "withdrawal", "withdrawl", "withdra",
                    "amount", "amt", "dr/cr", "dr", "cr", "narration", "description",
                    "remarks", "particulars"
                ])
            ):
                header_y = max(s["cy"] for s in window_spans)
                header_min_y = min(s["bbox"][1] for s in window_spans)
                for s in window_spans:
                    txt = s["text"].lower().strip()
                    if txt in ("sl", "si", "sl no", "si no", "s.no", "sr no", "sr.no", "s.no.", "#"):
                        header_map["sl_no"] = s["cx"]
                    elif "tran" in txt and ("id" in txt or len(txt) <= 7) and "date" not in txt and "part" not in txt:
                        header_map["tran_id"] = s["cx"]
                    elif "posted" in txt:
                        header_map["posted_date"] = s["cx"]
                    elif "value" in txt and ("date" in txt or "dt" in txt):
                        header_map["value_date"] = s["cx"]
                    elif ("date" in txt or txt == "transactio") and "generation" not in txt and "value" not in txt:
                        header_map.setdefault("date", s["cx"])
                    elif any(k in txt for k in ["desc", "particular", "narration", "detail", "remark"]):
                        header_map["description"] = s["cx"]
                    elif any(k in txt for k in ["branch", "code", "init.", "init br", "init. br", "init.br", "br."]) or txt in ("init", "br"):
                        header_map["branch_code"] = s["cx"]
                    elif any(k in txt for k in ["chq", "ref", "cheque", "inst no", "instrument"]):
                        header_map["ref"] = s["cx"]
                    elif txt in ("dr/cr", "cr/dr", "dr / cr", "cr / dr", "dr|cr", "cr|dr", "type"):
                        header_map["dr_cr_col"] = s["cx"]
                    elif ("withdrawal" in txt and "deposit" in txt) or ("dr" in txt and "cr" in txt) or "withdrawal(dr)" in txt or "deposit(cr)" in txt:
                        header_map["amount_cr_dr"] = s["cx"]
                    elif "amount" in txt and "balance" not in txt and "total" not in txt and "withdrawal" not in txt and "deposit" not in txt:
                        header_map["single_amount"] = s["cx"]
                    elif any(k in txt for k in ["credit", "deposit", "(cr)"]) and "count" not in txt:
                        header_map["credit"] = s["cx"]
                    elif any(k in txt for k in ["debit", "withdrawal", "withdrawl", "withdra", "wal (dr)", "(dr)"]) and "count" not in txt:
                        header_map["debit"] = s["cx"]
                    elif "balance" in txt and not any(x in txt for x in ["available", "effective", "opening"]):
                        header_map["balance"] = s["cx"]
                break

        if page_idx == 1 and header_min_y is not None and not (summary["opening_balance"] and summary["closing_balance"]):
            page_summary = extract_summary_cards(page_spans, header_min_y)
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

        monetary_col_keys = [k for k in ("debit", "credit", "balance", "amount_cr_dr", "single_amount") if k in header_map]
        num_cols = [header_map[k] for k in monetary_col_keys]
        if num_cols:
            page_spans = Stitch_Vertical_Wrapped_Numbers(page_spans, num_cols)

        bal_col_x = header_map.get("balance", 500)
        bal_cys = sorted([
            s["cy"] for s in page_spans
            if (not header_y or s["cy"] > header_y + 3.0)
            and (abs(s["cx"] - bal_col_x) <= 75 or abs(s["bbox"][2] - bal_col_x) <= 75)
            and parse_amount_with_sign(s["text"].strip())[0] is not None
        ])
        eff_y_tol = y_tolerance
        if len(bal_cys) >= 15:
            gaps = [bal_cys[i + 1] - bal_cys[i] for i in range(len(bal_cys) - 1) if bal_cys[i + 1] - bal_cys[i] > 1.5]
            if gaps:
                min_gaps = sorted(gaps)[:max(1, len(gaps) // 4)]
                avg_min_gap = sum(min_gaps) / len(min_gaps)
                eff_y_tol = min(y_tolerance, max(2.2, avg_min_gap * 0.35))

        rows = group_into_rows(page_spans, y_tolerance=eff_y_tol)

        body_rows = []
        for row in rows:
            row_cy = sum(s["cy"] for s in row) / len(row)
            if header_y and row_cy <= header_y + 3.0:
                continue
            row_joined_lower = " ".join(s["text"].lower() for s in row).strip()

            if re.search(r"\bpage\s*[:\-]?\s*\d+\s*(?:of|/)\s*\d+\b", row_joined_lower):
                continue

            if row_joined_lower in ("yes bank", "icici bank", "axis bank", "indusind bank", "canara bank", "hdfc bank", "r)", "n", "id"):
                continue

            row_has_any_date_or_amt = any(
                DATE_RE.match(s["text"].strip()) or re.search(r"\d+\.\d{2}", normalize_ocr_amount_str(s["text"]))
                for s in row
            )
            if not row_has_any_date_or_amt and any(skip_k in row_joined_lower for skip_k in [
                "statement of account", "statement for a/c", "statement of axis", "system generated", "phone banking",
                "service tax registration", "discrepancies in this statement", "computer generated",
                "registered office", "corporate identity number", "limits and effective", "unless the constituent",
                "branch code", "account branch", "joint holder", "customer id", "ifsc code", "micr code", "nominee"
            ]):
                continue

            if ("date" in row_joined_lower or "transactio" in row_joined_lower) and "balance" in row_joined_lower and any(w in row_joined_lower for w in ["particulars", "narration", "description", "remarks", "debit", "withdrawal", "withdrawl"]):
                continue

            if "closing balance" in row_joined_lower and not any(u in row_joined_lower for u in ["upi/", "imps/", "neft/", "rtgs/", "mob/"]):
                for s in row:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None and abs(val) > 0.001:
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        summary["closing_balance"] = [signed_v]
                break

            if any(stop_k in row_joined_lower for stop_k in [
                "page total", "period totals", "transaction total", "cumulative total", "dr count",
                "total transaction count", "total debit count", "total credit count",
                "total debit amount", "total credit amount", "summary for account", "b. summary",
                "reconciliation note", "statement summary", "end of statement", "end of the statement",
                "--end of statement", "++++ end", "charge breakup", "recover date", "legends", "disclaimer", "closing balance as on"
            ]):
                break
            body_rows.append(row)

        date_x_max = max(header_map.get("date", 85), header_map.get("value_date", 145)) + 50

        def is_valid_balance_span(s):
            txt = normalize_ocr_amount_str(s["text"].strip())
            if not (abs(s["cx"] - bal_col_x) <= 65 or abs(s["bbox"][2] - bal_col_x) <= 65):
                return False, None
            # Ensure this span is closer to the 'balance' column than to 'debit', 'credit', or 'single_amount'
            if monetary_col_keys:
                def _c_dist(col_k):
                    hx = header_map[col_k]
                    return min(abs(s["cx"] - hx), abs(s["bbox"][2] - (hx + 18.0)), abs(s["bbox"][0] - (hx - 18.0)))
                closest_k = min(monetary_col_keys, key=_c_dist)
                if closest_k != "balance":
                    return False, None

            if re.match(r"^\d{2}$", txt) and abs(s["cx"] - bal_col_x) <= 45:
                return True, round(float("0." + txt), 2)
            if re.match(r"^\d{6,9}$", txt) and abs(s["cx"] - bal_col_x) <= 45:
                return True, round(float(txt[:-2] + "." + txt[-2:]), 2)
            val, _ = parse_amount_with_sign(txt)
            if val is not None and ("." in txt or "," in txt):
                return True, val
            return False, None

        def get_row_balance_spans(r):
            b_spans = []
            for s in r:
                ok, _ = is_valid_balance_span(s)
                if ok:
                    b_spans.append(s)
            return b_spans

        unpacked_body_rows = []
        for r in body_rows:
            b_spans = get_row_balance_spans(r)
            if len(b_spans) >= 2:
                b_spans.sort(key=lambda s: s["cy"])
                b_cys = [bs["cy"] for bs in b_spans]
                sub_rows = [[] for _ in b_cys]
                for s in r:
                    best_k = min(range(len(b_cys)), key=lambda k: abs(s["cy"] - b_cys[k]))
                    sub_rows[best_k].append(s)
                for sr in sub_rows:
                    if sr:
                        sr.sort(key=lambda s: s["cx"])
                        unpacked_body_rows.append(sr)
            else:
                unpacked_body_rows.append(r)
        body_rows = unpacked_body_rows

        def row_has_balance_span(r):
            return len(get_row_balance_spans(r)) > 0

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
                    for s in row:
                        closest_pos = min(range(len(anchor_cys)), key=lambda idx: abs(s["cy"] - anchor_cys[idx]))
                        tx_blocks[closest_pos].append(s)

        page_uses_decimals = (
            sum(
                1
                for b_r in body_rows
                for s_x in b_r
                if (abs(s_x["cx"] - bal_col_x) <= 75 or abs(s_x["bbox"][2] - bal_col_x) <= 75) and "." in normalize_ocr_amount_str(s_x["text"])
            )
            >= max(1, len(anchor_indices) // 2)
        )
        page_tx_count = 0

        for block in tx_blocks:
            block.sort(key=lambda s: (round(s["cy"], 1), s["cx"]))
            row_joined_lower = " ".join(s["text"].lower() for s in block)

            if any(tot_k in row_joined_lower for tot_k in [
                "transaction total", "total :", "total:", "grand total", "summary", "charge breakup",
                "total debit count", "total credit count", "total debit amount", "total credit amount",
                "total transaction count"
            ]):
                continue

            if bool(re.search(r"\b(?:brought\s+forward|opening\s+balance|b/f)\b", row_joined_lower)) and not any(u in row_joined_lower for u in ["upi/", "imps/", "neft/", "rtgs/", "mob/"]):
                bf_candidates = []
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None and ("." in normalize_ocr_amount_str(s["text"]) or "," in s["text"]):
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        bf_candidates.append((s["cx"], signed_v))
                if bf_candidates:
                    bf_candidates.sort(key=lambda x: x[0])
                    non_zero_bf = [v for _, v in bf_candidates if abs(v) > 0.001]
                    chosen_bf = non_zero_bf[-1] if non_zero_bf else bf_candidates[-1][1]
                    if page_idx == 1:
                        if not summary["opening_balance"] or summary["opening_balance"][0] == 0.0:
                            summary["opening_balance"] = [chosen_bf]
                        elif chosen_bf not in summary["opening_balance"]:
                            summary["opening_balance"].insert(0, chosen_bf)
                if "opening balance" in row_joined_lower or "brought forward" in row_joined_lower or "b/f" in row_joined_lower:
                    continue

            if bool(re.search(r"\b(?:closing\s+balance|carried\s+forward|c/f)\b", row_joined_lower)) and not any(u in row_joined_lower for u in ["upi/", "imps/", "neft/", "rtgs/", "mob/"]):
                cf_candidates = []
                for s in block:
                    val, ind = parse_amount_with_sign(s["text"])
                    if val is not None and "." in normalize_ocr_amount_str(s["text"]):
                        signed_v = -abs(val) if (ind == "DR" or s["text"].strip().startswith("-")) else val
                        cf_candidates.append((s["cx"], signed_v))
                if cf_candidates:
                    cf_candidates.sort(key=lambda x: x[0])
                    non_zero_cf = [v for _, v in cf_candidates if abs(v) > 0.001]
                    chosen_cf = non_zero_cf[-1] if non_zero_cf else cf_candidates[-1][1]
                    summary["closing_balance"] = [chosen_cf]
                continue

            date_str = None
            sl_no_val = None
            used_ids = set()

            for idx, s in enumerate(block):
                t_clean = s["text"].strip().rstrip(".")
                if "sl_no" in header_map and abs(s["cx"] - header_map["sl_no"]) <= 18 and t_clean.isdigit():
                    cand_sl = int(t_clean)
                    if sl_no_val is None and (
                        (last_valid_sl_no == 0 and 1 <= cand_sl <= 3)
                        or (last_valid_sl_no > 0 and last_valid_sl_no < cand_sl <= last_valid_sl_no + 5)
                    ):
                        sl_no_val = cand_sl
                        last_valid_sl_no = cand_sl
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
                "balance": None,
                "_bal_cy": (ry0 + ry1) / 2.0
            }

            cr_dr_amounts = []
            single_col_amounts = []
            candidate_monetary_spans = []
            row_dr_cr_flags = []

            for idx, s in enumerate(block):
                if idx in used_ids:
                    continue
                raw_txt = s["text"].strip()
                txt = normalize_ocr_amount_str(raw_txt)
                if txt in ("—", "-", "--", "₹", "|", ".00", "00", "R)", "n"):
                    continue

                if "branch_code" in header_map and abs(s["cx"] - header_map["branch_code"]) <= 38 and "." not in txt and "," not in txt:
                    continue

                if re.match(r"^\(?(?:DR|CR|CF|CH|CI)\.?\)?$", txt, re.IGNORECASE) and ("dr_cr_col" in header_map or s["cx"] > date_x_max):
                    clean_flg = "DR" if "DR" in txt.upper() else "CR"
                    row_dr_cr_flags.append((s["cx"], clean_flg))
                    continue

                ok_bal_special, special_bal_val = is_valid_balance_span(s)
                if ok_bal_special and "." not in txt:
                    val, ind = special_bal_val, None
                    txt = f"{special_bal_val:.2f}"
                else:
                    val, ind = parse_amount_with_sign(txt)

                if val is not None and "." not in txt and "single_amount" in header_map and abs(s["cx"] - header_map["single_amount"]) <= 45:
                    if len(txt) >= 6:
                        val = round(float(txt[:-2] + "." + txt[-2:]), 2)
                        txt = f"{val:.2f}"
                    elif txt == "100":
                        txt = "100.00"

                if val is not None and ("[" in raw_txt or "]" in raw_txt):
                    val = None

                if ind == "MALFORMED_DOT":
                    msg_mal = (
                        f"Page {page_idx} ({date_str}): Malformed Monetary Format Anomaly — "
                        f"Amount '{raw_txt}' uses a period '.' instead of a thousands comma ','."
                    )
                    issues.append(msg_mal)
                    detailed_issues.append({"page": page_idx, "bbox": s["bbox"], "text": msg_mal})

                if val is not None and ("." in txt or "," in txt) and monetary_col_keys:
                    def col_dist(col_k):
                        hx = header_map[col_k]
                        return min(abs(s["cx"] - hx), abs(s["bbox"][2] - (hx + 18.0)), abs(s["bbox"][0] - (hx - 18.0)))

                    closest_mon_col = min(monetary_col_keys, key=col_dist)
                    dist_mon = col_dist(closest_mon_col)
                else:
                    closest_mon_col = None
                    dist_mon = 999.0

                closest_col = min(header_map.keys(), key=lambda c: abs(s["cx"] - header_map[c]))
                is_monetary_col = closest_col in ("credit", "debit", "balance", "amount_cr_dr", "single_amount")
                dist_to_col = abs(s["cx"] - header_map[closest_col])

                if val is not None and ind in ("CR", "DR") and not txt.startswith("-"):
                    cr_dr_amounts.append((s["cx"], val, ind, txt))
                elif val is not None and closest_mon_col is not None and dist_mon <= 95.0:
                    candidate_monetary_spans.append((val, s["cx"], s["bbox"][2], closest_mon_col, txt))
                    if closest_mon_col == "balance":
                        is_neg_bal = txt.startswith("-") or ind == "DR"
                        tx["balance"] = -abs(val) if is_neg_bal else abs(val)
                        tx["_bal_cy"] = s["cy"]
                    elif closest_mon_col in ("single_amount", "amount_cr_dr"):
                        single_col_amounts.append(abs(val))
                    elif closest_mon_col == "credit":
                        if val > 0:
                            tx["credit"] = abs(val)
                    elif closest_mon_col == "debit":
                        tx["debit"] = val
                elif val is not None and is_monetary_col and dist_to_col <= 65.0 and not page_uses_decimals:
                    if closest_col == "balance":
                        tx["balance"] = -abs(val) if (txt.startswith("-") or ind == "DR") else abs(val)
                        tx["_bal_cy"] = s["cy"]
                    elif closest_col in ("single_amount", "amount_cr_dr"):
                        single_col_amounts.append(abs(val))
                    elif closest_col == "credit":
                        if val > 0 or tx["credit"] == 0.0:
                            tx["credit"] = abs(val)
                    elif closest_col == "debit":
                        tx["debit"] = (val)
                else:
                    if closest_col not in ("value_date", "posted_date", "sl_no", "branch_code", "dr_cr_col"):
                        tx["description"].append(raw_txt)

            row_dr_cr_flag = None
            bal_dr_cr_flag = None
            if row_dr_cr_flags:
                row_dr_cr_flags.sort(key=lambda x: x[0])
                for flg_x, flg_v in row_dr_cr_flags:
                    if flg_x > bal_col_x - 15.0:
                        bal_dr_cr_flag = flg_v
                    else:
                        row_dr_cr_flag = flg_v
                if row_dr_cr_flag is None:
                    row_dr_cr_flag = row_dr_cr_flags[0][1]

            if bal_dr_cr_flag == "DR" and tx["balance"] is not None:
                tx["balance"] = -abs(tx["balance"])

            if single_col_amounts:
                amt_v = single_col_amounts[-1]
                prev_b = summary["opening_balance"][0] if (not transactions and summary["opening_balance"]) else (transactions[-1]["balance"] if transactions else None)
                if prev_b is not None and tx["balance"] is not None:
                    if abs((prev_b - amt_v) - tx["balance"]) <= 1.0:
                        tx["debit"] = amt_v
                    elif abs((prev_b + amt_v) - tx["balance"]) <= 1.0:
                        tx["credit"] = amt_v
                    elif abs((prev_b - (amt_v / 100.0)) - tx["balance"]) <= 0.05:
                        tx["debit"] = round(amt_v / 100.0, 2)
                    elif abs((prev_b + (amt_v / 100.0)) - tx["balance"]) <= 0.05:
                        tx["credit"] = round(amt_v / 100.0, 2)
                    elif row_dr_cr_flag == "DR":
                        tx["debit"] = amt_v
                    elif row_dr_cr_flag == "CR":
                        tx["credit"] = amt_v
                    elif tx["balance"] < prev_b:
                        tx["debit"] = amt_v
                    else:
                        tx["credit"] = amt_v
                else:
                    if row_dr_cr_flag == "DR":
                        tx["debit"] = amt_v
                    else:
                        tx["credit"] = amt_v

            if len(cr_dr_amounts) >= 2:
                cr_dr_amounts.sort(key=lambda item: item[0])
                amt_cx, amt_val, amt_ind, _ = cr_dr_amounts[-2]
                bal_cx, bal_val, bal_ind, _ = cr_dr_amounts[-1]
                if amt_ind == "CR":
                    tx["credit"] = abs(amt_val)
                else:
                    tx["debit"] = abs(amt_val)
                tx["balance"] = -abs(bal_val) if bal_ind == "DR" else abs(bal_val)
            elif len(cr_dr_amounts) == 1:
                amt_cx, amt_val, amt_ind, _ = cr_dr_amounts[0]
                if tx["balance"] is not None and abs(amt_cx - header_map.get("balance", 500)) <= 35 and tx["credit"] == 0 and tx["debit"] == 0:
                    tx["balance"] = -abs(amt_val) if amt_ind == "DR" else abs(amt_val)
                elif tx["balance"] is None and abs(amt_cx - header_map.get("balance", 500)) <= 45:
                    tx["balance"] = -abs(amt_val) if amt_ind == "DR" else abs(amt_val)
                else:
                    if amt_ind == "CR":
                        tx["credit"] = abs(amt_val)
                    else:
                        tx["debit"] = abs(amt_val)

            prev_b = summary["opening_balance"][0] if (not transactions and summary["opening_balance"]) else (transactions[-1]["balance"] if transactions else None)
            if prev_b is not None and tx["balance"] is not None and len(candidate_monetary_spans) >= 2:
                non_bal_cands = [c[0] for c in candidate_monetary_spans if c[3] != "balance" and abs(c[0]) > 0]
                expected_delta = round(tx["balance"] - prev_b, 2)
                if abs((tx["credit"] - tx["debit"]) - expected_delta) > 1.0:
                    for cand_v in non_bal_cands:
                        if abs(abs(cand_v) - abs(expected_delta)) <= 1.0:
                            if expected_delta > 0:
                                tx["credit"], tx["debit"] = abs(cand_v), 0.0
                            else:
                                tx["debit"], tx["credit"] = abs(cand_v), 0.0
                            break

            tx["description"] = " ".join(tx["description"]).strip()
            if tx["balance"] is not None:
                if tx["credit"] == 0.0 and tx["debit"] == 0.0 and not tx["description"] and transactions and abs(tx["balance"] - transactions[-1]["balance"]) < 0.01:
                    continue
                transactions.append(tx)
                page_tx_count += 1

        per_page_stats.append({"page": page_idx, "transactions_extracted": page_tx_count})

    fwd_hits = 0
    rev_hits = 0
    for i in range(len(transactions) - 1):
        t_curr, t_next = transactions[i], transactions[i + 1]
        if abs((t_curr["balance"] + t_next["credit"] - t_next["debit"]) - t_next["balance"]) <= 1.0:
            fwd_hits += 1
        if abs((t_next["balance"] + t_curr["credit"] - t_curr["debit"]) - t_curr["balance"]) <= 1.0:
            rev_hits += 1

    is_reverse_chrono = (len(transactions) >= 3 and rev_hits > fwd_hits and rev_hits >= len(transactions) * 0.5)
    if is_reverse_chrono:
        transactions.reverse()

    if transactions:
        first_t = transactions[0]
        if bool(re.search(r"\b(?:b/f|brought\s+forward|opening\s+balance)\b", first_t["description"].lower())) and first_t["credit"] == 0 and first_t["debit"] == 0:
            if not summary["opening_balance"] or summary["opening_balance"][0] == 0.0:
                summary["opening_balance"] = [first_t["balance"]]
            elif first_t["balance"] not in summary["opening_balance"]:
                summary["opening_balance"].insert(0, first_t["balance"])
            transactions = transactions[1:]

    if transactions and (transactions[0]["credit"] > 0 or transactions[0]["debit"] > 0):
        inferred_first_open = round(transactions[0]["balance"] - transactions[0]["credit"] + transactions[0]["debit"], 2)
        if not summary["opening_balance"]:
            summary["opening_balance"] = [inferred_first_open]
        elif abs(summary["opening_balance"][0] - inferred_first_open) > 1.5:
            tot_c_raw = round(sum(t["credit"] for t in transactions), 2)
            tot_d_raw = round(sum(t["debit"] for t in transactions), 2)
            last_b_raw = round(transactions[-1]["balance"], 2)
            if abs((inferred_first_open + tot_c_raw - tot_d_raw) - last_b_raw) <= 2.0:
                summary["opening_balance"][0] = inferred_first_open

    for idx, tx in enumerate(transactions):
        prev_b = summary["opening_balance"][0] if (idx == 0 and summary["opening_balance"]) else (transactions[idx - 1]["balance"] if idx > 0 else None)
        if prev_b is None:
            tx.pop("_bal_cy", None)
            continue

        if prev_b < 0 and tx["balance"] > 0:
            neg_bal = -abs(tx["balance"])
            if (tx["credit"] > 0 or tx["debit"] > 0) and abs((prev_b + tx["credit"] - tx["debit"]) - neg_bal) <= 1.5:
                tx["balance"] = neg_bal
            else:
                for look_i in range(idx + 1, min(len(transactions), idx + 5)):
                    fut_tx = transactions[look_i]
                    if fut_tx["credit"] > 0 or fut_tx["debit"] > 0:
                        prev_test = -abs(transactions[look_i - 1]["balance"]) if look_i - 1 >= idx else neg_bal
                        if abs((prev_test + fut_tx["credit"] - fut_tx["debit"]) - fut_tx["balance"]) <= 1.5 or abs((prev_test + fut_tx["credit"] - fut_tx["debit"]) - (-fut_tx["balance"])) <= 1.5:
                            tx["balance"] = neg_bal
                            break

        delta = round(tx["balance"] - prev_b, 2)
        abs_delta = abs(delta)
        current_net = round(tx["credit"] - tx["debit"], 2)

        if abs_delta > 0 and abs(current_net - delta) > 1.0:
            if abs(tx["credit"] - abs_delta) <= 1.0 or abs(tx["debit"] - abs_delta) <= 1.0:
                if delta > 0:
                    tx["credit"], tx["debit"] = abs_delta, 0.0
                else:
                    tx["debit"], tx["credit"] = abs_delta, 0.0
            else:
                p_idx = tx["page"] - 1
                bal_cy = tx.get("_bal_cy", (tx["bbox"][1] + tx["bbox"][3]) / 2.0)
                matched_on_page = False

                if 0 <= p_idx < len(pages_spans):
                    for s in pages_spans[p_idx]:
                        if abs(s["cy"] - bal_cy) <= 22.0:
                            norm_s = normalize_ocr_amount_str(s["text"])
                            for m_num in re.finditer(r"[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?", norm_s):
                                v, _ = parse_amount_with_sign(m_num.group(0))
                                if v is not None and (abs(abs(v) - abs_delta) <= 1.0 or abs((abs(v) / 100.0) - abs_delta) <= 0.05):
                                    matched_on_page = True
                                    break
                        if matched_on_page:
                            break

                if not matched_on_page:
                    int_part = int(round(abs_delta))
                    indian_fmt = f"{int_part:,}"
                    if len(str(int_part)) > 3:
                        last3 = str(int_part)[-3:]
                        rest = str(int_part)[:-3]
                        indian_fmt = ",".join([rest[max(0, i - 2):i] for i in range(len(rest), 0, -2)][::-1]) + "," + last3
                    if (
                        indian_fmt in tx["description"]
                        or str(int_part) in tx["description"]
                        or f"{abs_delta:.2f}" in tx["description"]
                        or ("oll:" in tx["description"].lower() and abs(abs_delta - int_part) < 0.01)
                    ):
                        matched_on_page = True

                if matched_on_page:
                    if delta > 0:
                        tx["credit"], tx["debit"] = abs_delta, 0.0
                    else:
                        tx["debit"], tx["credit"] = abs_delta, 0.0

        tx.pop("_bal_cy", None)

    res = run_math_checks(summary, transactions, issues, detailed_issues, is_reverse_chrono=is_reverse_chrono, full_text=full_text)
    res["total_pages_processed"] = len(pages_spans)
    res["per_page_summary"] = per_page_stats
    return res
