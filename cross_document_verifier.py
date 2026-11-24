import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

PAN_REGEX = re.compile(r"^[A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]$")

PAN_ENTITY_MAP = {
    "P": ("individual", "proprietorship", "proprietor"),
    "C": ("company", "private limited", "public limited", "pvt", "ltd"),
    "F": ("firm", "partnership", "llp", "limited liability partnership"),
    "H": ("huf", "hindu undivided family"),
    "T": ("trust",),
    "A": ("aop", "association of persons"),
    "B": ("boi", "body of individuals"),
}

STOP_TOKENS = {
    "m", "s", "ms", "mr", "mrs", "shri", "smt", "kumari", "dr",
    "pvt", "private", "ltd", "limited", "llp", "co", "company",
    "and", "the", "of", "prop", "proprietor", "enterprises",
}


def validate_pan(pan_str: str, allow_all_entities: bool = True) -> Tuple[bool, str]:
    """Validates 10-character Indian PAN format and entity type (4th character)."""
    if not pan_str:
        return False, "PAN not provided."
    cleaned = pan_str.strip().upper()
    if not PAN_REGEX.match(cleaned):
        return False, f"Invalid PAN structure '{cleaned}' (expected 3 letters, entity code, 1 letter, 4 digits, 1 letter)."
    if not allow_all_entities and cleaned[3] != "P":
        return False, f"PAN entity type '{cleaned[3]}' is not 'P' (Individual Person)."
    return True, f"Valid PAN (Entity Code: {cleaned[3]})."


def _clean_name_tokens(name: str) -> List[str]:
    s = re.sub(r"\bM\s*/\s*S\.?\b", " ", str(name or ""), flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z\s]", " ", s).lower()
    return [t for t in s.split() if t and t not in STOP_TOKENS]


def name_similarity(name1: str, name2: str) -> float:
    """
    Indian-entity-aware name similarity: strips M/S, Pvt Ltd, honorifics,
    handles word-order swaps (token sort), and supports subset matches.
    """
    t1 = _clean_name_tokens(name1)
    t2 = _clean_name_tokens(name2)
    if not t1 or not t2:
        return 0.0

    s1_raw, s2_raw = " ".join(t1), " ".join(t2)
    s1_sort, s2_sort = " ".join(sorted(t1)), " ".join(sorted(t2))

    raw_ratio = SequenceMatcher(None, s1_raw, s2_raw).ratio()
    sort_ratio = SequenceMatcher(None, s1_sort, s2_sort).ratio()

    set1, set2 = set(t1), set(t2)
    overlap = len(set1 & set2) / max(1, min(len(set1), len(set2)))
    # Only boost subset overlap when at least 2 meaningful tokens match (or single exact token)
    subset_score = overlap if (len(set1 & set2) >= 2 or (len(set1) == 1 and set1 == set2)) else 0.0

    return round(max(raw_ratio, sort_ratio, subset_score * 0.92), 3)


def _norm_dob(dob: Optional[str]) -> Optional[str]:
    if not dob:
        return None
    m = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", str(dob))
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
    return str(dob).strip()


def _make_finding(
    code: str,
    pillar: str,
    severity: str,
    message: str,
    docs: List[str],
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "finding_code": code,
        "family": "FAMILY_D_CROSS_DOCUMENT",
        "pillar": pillar,
        "severity": severity,  # CRITICAL | HIGH | MEDIUM | LOW
        "documents_compared": docs,
        "message": message,
        "evidence": evidence or {},
    }


def _extract_canonical_doc(doc: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalizes either a Module 1 contract JSON or raw verify_any_file() output."""
    id_card = doc.get("identity_card") or {}
    fin = doc.get("extracted_financials") or doc.get("financial_reconciliation") or {}

    doc_code = (
        id_card.get("doc_code")
        or fin.get("document_type")
        or doc.get("document_type")
        or "UNKNOWN_DOCUMENT"
    )
    label = (
        fin.get("form_display_label")
        or id_card.get("doc_category")
        or doc_code
    )
    filename = doc.get("filename") or id_card.get("filename") or f"{label} #{idx + 1}"

    # Build a pool of valid entity/person names on this document (handles Proprietorship trade vs legal names)
    raw_names = [
        id_card.get("holder_name"),
        fin.get("legal_name"),
        fin.get("trade_name"),
        fin.get("enterprise_name"),
        fin.get("assessee_business_name"),
        fin.get("proprietor_name"),
        fin.get("authorized_signatory"),
        fin.get("employee_name"),
        (fin.get("blueprint_metadata") or {}).get("holder_name"),
    ]
    bad_names = {"unknown", "unknown holder", "unknown assessee", "verified holder", "not found", "n/a"}
    names_pool = []
    for n in raw_names:
        if n and str(n).strip().lower() not in bad_names:
            clean_n = str(n).strip()
            if clean_n not in names_pool:
                names_pool.append(clean_n)

    pan = (
        id_card.get("pan_number")
        or fin.get("pan_number")
        or fin.get("embedded_pan")
        or (fin.get("gstin")[2:12] if fin.get("gstin") and len(str(fin.get("gstin"))) == 15 else None)
    )
    pan = str(pan).strip().upper() if pan else None

    return {
        "doc_id": f"DOC_{idx + 1}",
        "label": label,
        "filename": filename,
        "doc_code": doc_code,
        "gst_form_type": fin.get("gst_form_type"),
        "pan": pan,
        "gstin": fin.get("gstin"),
        "constitution": fin.get("constitution_of_business"),
        "names_pool": names_pool,
        "primary_name": names_pool[0] if names_pool else None,
        "father_name": fin.get("father_name"),
        "dob": _norm_dob(fin.get("date_of_birth")),
        "account_number": id_card.get("account_number") or (fin.get("blueprint_metadata") or {}).get("account_number"),
        "financial_year": fin.get("financial_year") or fin.get("previous_year"),
        "assessment_year": fin.get("assessment_year"),
        "tax_period": fin.get("tax_period"),
        "fin": fin,
    }


def verify_loan_dossier(document_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Unified Family D Cross-Document Verification Engine for Indian Retail & MSME Loans.
    Accepts 2+ Module 1 outputs in any combination.
    """
    docs = [_extract_canonical_doc(d, i) for i, d in enumerate(document_outputs or [])]
    findings: List[Dict[str, Any]] = []
    checks: Dict[str, str] = {
        "pan_cross_consistency": "NOT_APPLICABLE",
        "entity_name_cross_consistency": "NOT_APPLICABLE",
        "kyc_dob_and_father_consistency": "NOT_APPLICABLE",
        "pan_constitution_alignment": "NOT_APPLICABLE",
        "gst_1_vs_3b_sales_reconciliation": "NOT_APPLICABLE",
        "gst_3b_vs_2b_itc_reconciliation": "NOT_APPLICABLE",
        "bank_credits_vs_declared_turnover": "NOT_APPLICABLE",
        "bank_vs_salary_slip_reconciliation": "NOT_APPLICABLE",
        "tax_audit_vs_itr_alignment": "NOT_APPLICABLE",
    }

    if len(docs) < 2:
        return {
            "family": "FAMILY_D_CROSS_DOCUMENT",
            "status": "NOT_RUN",
            "documents_evaluated": len(docs),
            "checks_matrix": checks,
            "risk_score_contribution": 0,
            "structured_findings": [],
            "findings": [],
        }

    # ------------------------------------------------------------------
    # PILLAR 1: IDENTITY & KYC GRAPH (PAN, Name Pool, DOB, Father, Constitution)
    # ------------------------------------------------------------------
    pan_docs = [(d["label"], d["pan"]) for d in docs if d["pan"]]
    if len(pan_docs) >= 2:
        unique_pans = {p for _, p in pan_docs}
        if len(unique_pans) > 1:
            checks["pan_cross_consistency"] = "FAILED"
            findings.append(_make_finding(
                "CROSS_DOC_PAN_MISMATCH",
                "IDENTITY_KYC",
                "CRITICAL",
                f"Conflicting PAN numbers detected across dossier: {dict(pan_docs)}.",
                [lbl for lbl, _ in pan_docs],
                {"pans_found": dict(pan_docs)},
            ))
        else:
            checks["pan_cross_consistency"] = "PASSED"

    # Cross-check Entity / Holder Names using Name Pools (prevents Proprietorship false positives)
    named_docs = [d for d in docs if d["names_pool"]]
    if len(named_docs) >= 2:
        name_mismatch_found = False
        for i in range(len(named_docs)):
            for j in range(i + 1, len(named_docs)):
                d1, d2 = named_docs[i], named_docs[j]
                # Best match across all valid aliases on both documents
                best_sim = max(
                    name_similarity(n1, n2)
                    for n1 in d1["names_pool"]
                    for n2 in d2["names_pool"]
                )
                # If both docs share the exact same valid PAN and one is an Individual PAN (Proprietorship),
                # treat trade-name vs individual-name gently unless both are personal KYC cards.
                same_verified_pan = bool(d1["pan"] and d1["pan"] == d2["pan"])
                both_personal_kyc = {d1["doc_code"], d2["doc_code"]}.issubset({"PAN_CARD", "AADHAAR_CARD", "SALARY_SLIP"})

                if best_sim < 0.72:
                    if same_verified_pan and not both_personal_kyc and d1["pan"][3] == "P":
                        continue  # Proprietorship trade name linked by verified identical PAN
                    name_mismatch_found = True
                    findings.append(_make_finding(
                        "CROSS_DOC_NAME_MISMATCH",
                        "IDENTITY_KYC",
                        "CRITICAL" if both_personal_kyc else "HIGH",
                        f"Holder/Entity name mismatch between '{d1['label']}' ({d1['primary_name']}) and "
                        f"'{d2['label']}' ({d2['primary_name']}) (max similarity: {best_sim}).",
                        [d1["label"], d2["label"]],
                        {"doc1_names": d1["names_pool"], "doc2_names": d2["names_pool"], "similarity": best_sim},
                    ))
        checks["entity_name_cross_consistency"] = "FAILED" if name_mismatch_found else "PASSED"

    # KYC DOB & Father's Name (PAN vs Aadhaar)
    dob_docs = [(d["label"], d["dob"]) for d in docs if d["dob"]]
    father_docs = [(d["label"], d["father_name"]) for d in docs if d["father_name"]]
    if len(dob_docs) >= 2 or len(father_docs) >= 2:
        kyc_fail = False
        if len(dob_docs) >= 2 and len({v for _, v in dob_docs}) > 1:
            kyc_fail = True
            findings.append(_make_finding(
                "CROSS_DOC_DOB_MISMATCH",
                "IDENTITY_KYC",
                "CRITICAL",
                f"Date of Birth mismatch across KYC documents: {dict(dob_docs)}.",
                [lbl for lbl, _ in dob_docs],
                {"dobs": dict(dob_docs)},
            ))
        if len(father_docs) >= 2:
            f0_lbl, f0_val = father_docs[0]
            for f_lbl, f_val in father_docs[1:]:
                f_sim = name_similarity(f0_val, f_val)
                if f_sim < 0.78:
                    kyc_fail = True
                    findings.append(_make_finding(
                        "CROSS_DOC_FATHER_NAME_MISMATCH",
                        "IDENTITY_KYC",
                        "HIGH",
                        f"Father's Name mismatch between '{f0_lbl}' ('{f0_val}') and '{f_lbl}' ('{f_val}') (similarity: {f_sim}).",
                        [f0_lbl, f_lbl],
                        {"father_names": {f0_lbl: f0_val, f_lbl: f_val}, "similarity": f_sim},
                    ))
        checks["kyc_dob_and_father_consistency"] = "FAILED" if kyc_fail else "PASSED"

    # PAN 4th-char vs GST Constitution of Business
    const_docs = [d for d in docs if d["constitution"] and d["pan"]]
    if const_docs:
        const_fail = False
        for d in const_docs:
            p4 = d["pan"][3]
            expected_kws = PAN_ENTITY_MAP.get(p4, ())
            c_low = d["constitution"].lower()
            if expected_kws and not any(k in c_low for k in expected_kws):
                const_fail = True
                findings.append(_make_finding(
                    "PAN_CONSTITUTION_MISMATCH",
                    "IDENTITY_KYC",
                    "HIGH",
                    f"PAN entity code '{p4}' ({d['pan']}) conflicts with declared Constitution of Business '{d['constitution']}' on {d['label']}.",
                    [d["label"]],
                    {"pan": d["pan"], "constitution": d["constitution"]},
                ))
        checks["pan_constitution_alignment"] = "FAILED" if const_fail else "PASSED"

    # ------------------------------------------------------------------
    # PILLAR 2: GST TRIO RECONCILIATION (GSTR-1 vs GSTR-3B vs GSTR-2B)
    # ------------------------------------------------------------------
    gstr1_by_key = {}
    gstr3b_by_key = {}
    gstr2b_by_key = {}
    for d in docs:
        if d["doc_code"] == "GST_CERTIFICATE":
            key = (d["gstin"] or "ANY", (d["financial_year"] or "").strip(), (d["tax_period"] or "").strip().lower())
            if d["gst_form_type"] == "FORM GSTR-1":
                gstr1_by_key[key] = d
            elif d["gst_form_type"] == "FORM GSTR-3B":
                gstr3b_by_key[key] = d
            elif d["gst_form_type"] == "FORM GSTR-2B":
                gstr2b_by_key[key] = d

    # 2A. GSTR-1 vs GSTR-3B Outward Sales
    common_1_3b = set(gstr1_by_key.keys()) & set(gstr3b_by_key.keys())
    if common_1_3b:
        gst_sales_fail = False
        for key in common_1_3b:
            d1, d3 = gstr1_by_key[key], gstr3b_by_key[key]
            v1 = (d1["fin"].get("gst_financial_summary") or {}).get("total_taxable_value")
            v3 = ((d3["fin"].get("gst_financial_summary") or {}).get("outward_taxable_supplies") or {}).get("taxable_value")
            if v1 is not None and v3 is not None:
                diff = abs(v1 - v3)
                pct = (diff / max(1.0, max(v1, v3))) * 100.0
                if diff > 1000.0 and pct > 5.0:
                    gst_sales_fail = True
                    findings.append(_make_finding(
                        "GSTR1_VS_GSTR3B_SALES_MISMATCH",
                        "GST_RECONCILIATION",
                        "CRITICAL" if pct > 15.0 else "HIGH",
                        f"GSTR-1 taxable sales (₹{v1:,.2f}) vs GSTR-3B declared outward sales (₹{v3:,.2f}) "
                        f"differ by ₹{diff:,.2f} ({pct:.1f}%) for period {key[1]} {key[2].title()}.",
                        [d1["label"], d3["label"]],
                        {"gstr1_sales": v1, "gstr3b_sales": v3, "variance_pct": round(pct, 2)},
                    ))
        checks["gst_1_vs_3b_sales_reconciliation"] = "FAILED" if gst_sales_fail else "PASSED"

    # 2B. GSTR-3B Claimed ITC vs GSTR-2B Eligible ITC
    common_3b_2b = set(gstr3b_by_key.keys()) & set(gstr2b_by_key.keys())
    if common_3b_2b:
        itc_fail = False
        for key in common_3b_2b:
            d3, d2 = gstr3b_by_key[key], gstr2b_by_key[key]
            itc_3b_dict = (d3["fin"].get("gst_financial_summary") or {}).get("net_itc_available") or {}
            itc_2b_dict = (d2["fin"].get("gst_financial_summary") or {}).get("eligible_itc_part_a") or {}
            claimed = sum(float(itc_3b_dict.get(k, 0) or 0) for k in ("igst", "cgst", "sgst", "cess"))
            eligible = float(itc_2b_dict.get("total_eligible_itc") or 0)
            if eligible > 0 and claimed > eligible * 1.05 and (claimed - eligible) > 1000.0:
                itc_fail = True
                excess = claimed - eligible
                findings.append(_make_finding(
                    "EXCESS_ITC_CLAIMED_3B_VS_2B",
                    "GST_RECONCILIATION",
                    "CRITICAL",
                    f"GSTR-3B Net ITC claimed (₹{claimed:,.2f}) exceeds GSTR-2B auto-drafted eligible ITC "
                    f"(₹{eligible:,.2f}) by ₹{excess:,.2f} for {key[1]} {key[2].title()}.",
                    [d3["label"], d2["label"]],
                    {"claimed_3b_itc": claimed, "eligible_2b_itc": eligible, "excess_itc": round(excess, 2)},
                ))
        checks["gst_3b_vs_2b_itc_reconciliation"] = "FAILED" if itc_fail else "PASSED"

    # ------------------------------------------------------------------
    # PILLAR 3: BANK CREDITS VS DECLARED TURNOVER (Form 3CD / GSTR)
    # ------------------------------------------------------------------
    bank_docs = [d for d in docs if d["doc_code"] == "BANK_STATEMENT"]
    audit_docs = [d for d in docs if d["doc_code"] == "TAX_AUDIT_REPORT" and d["fin"].get("total_turnover")]
    if bank_docs and audit_docs:
        total_bank_credits = sum(
            float((b["fin"].get("summary") or b["fin"]).get("total_credits") or 0)
            for b in bank_docs
        )
        declared_turnover = float(audit_docs[0]["fin"]["total_turnover"])
        if total_bank_credits > 0 and declared_turnover > 0:
            ratio = total_bank_credits / declared_turnover
            if ratio < 0.35 or ratio > 2.0:
                checks["bank_credits_vs_declared_turnover"] = "FAILED"
                findings.append(_make_finding(
                    "BANK_CREDITS_VS_TURNOVER_ANOMALY",
                    "BANKING_VS_REVENUE",
                    "HIGH",
                    f"Total bank credits (₹{total_bank_credits:,.2f}) represent {ratio * 100:.1f}% of Form 3CD "
                    f"declared turnover (₹{declared_turnover:,.2f}) — outside normal 35%–200% underwriting band.",
                    [bank_docs[0]["label"], audit_docs[0]["label"]],
                    {"total_bank_credits": total_bank_credits, "declared_turnover": declared_turnover, "ratio": round(ratio, 3)},
                ))
            else:
                checks["bank_credits_vs_declared_turnover"] = "PASSED"

    # ------------------------------------------------------------------
    # PILLAR 4: TAX AUDIT (FORM 3CD) VS ITR ALIGNMENT
    # ------------------------------------------------------------------
    itr_docs = [d for d in docs if d["doc_code"] == "ITR_DOCUMENT"]
    if audit_docs and itr_docs:
        a_doc, i_doc = audit_docs[0], itr_docs[0]
        a_np = a_doc["fin"].get("net_profit")
        i_gti = i_doc["fin"].get("gross_total_income")
        a_ay, i_ay = a_doc["assessment_year"], i_doc["assessment_year"]
        if a_ay and i_ay and a_ay != i_ay:
            checks["tax_audit_vs_itr_alignment"] = "FAILED"
            findings.append(_make_finding(
                "ASSESSMENT_YEAR_MISMATCH",
                "TAX_AUDIT_ALIGNMENT",
                "MEDIUM",
                f"Period mismatch: Tax Audit Report is for AY {a_ay}, while ITR is for AY {i_ay}.",
                [a_doc["label"], i_doc["label"]],
                {"audit_ay": a_ay, "itr_ay": i_ay},
            ))
        elif a_np and i_gti and i_gti > a_np * 1.5 and (i_gti - a_np) > 100000:
            checks["tax_audit_vs_itr_alignment"] = "FAILED"
            findings.append(_make_finding(
                "ITR_VS_AUDIT_PROFIT_VARIANCE",
                "TAX_AUDIT_ALIGNMENT",
                "MEDIUM",
                f"ITR Gross Total Income (₹{i_gti:,.2f}) is significantly higher than Form 3CD Net Profit (₹{a_np:,.2f}); verify other income sources.",
                [a_doc["label"], i_doc["label"]],
                {"audit_net_profit": a_np, "itr_gross_total_income": i_gti},
            ))
        elif a_np or i_gti:
            checks["tax_audit_vs_itr_alignment"] = "PASSED"

    # ------------------------------------------------------------------
    # PILLAR 5: BANK STATEMENT VS SALARY SLIP
    # ------------------------------------------------------------------
    salary_docs = [d for d in docs if d["doc_code"] == "SALARY_SLIP"]
    if bank_docs and salary_docs:
        all_txs = []
        for b in bank_docs:
            all_txs.extend(b["fin"].get("transactions") or [])
        sal_fail = False
        for s_doc in salary_docs:
            net_pay = s_doc["fin"].get("net_pay")
            employer = str(s_doc["fin"].get("employer_name") or "").lower()
            emp_tokens = [t for t in re.findall(r"[a-z]+", employer) if len(t) > 3 and t not in STOP_TOKENS]
            if net_pay:
                matched_credit = False
                for tx in all_txs:
                    cr = float(tx.get("credit") or 0)
                    desc = str(tx.get("description") or "").lower()
                    is_sal_desc = ("salary" in desc or "sal " in desc or any(t in desc for t in emp_tokens))
                    if cr > 0 and abs(cr - float(net_pay)) <= 50.0 and (is_sal_desc or not emp_tokens):
                        matched_credit = True
                        break
                if not matched_credit:
                    sal_fail = True
                    findings.append(_make_finding(
                        "SALARY_CREDIT_NOT_FOUND_IN_BANK",
                        "RETAIL_INCOME",
                        "CRITICAL",
                        f"Salary Slip Net Pay (₹{float(net_pay):,.2f}) from '{s_doc['fin'].get('employer_name') or 'Employer'}' "
                        f"does not match any salary credit in the uploaded bank statement(s).",
                        [bank_docs[0]["label"], s_doc["label"]],
                        {"net_pay": net_pay, "employer": s_doc["fin"].get("employer_name")},
                    ))
        checks["bank_vs_salary_slip_reconciliation"] = "FAILED" if sal_fail else "PASSED"

    # Compute Family D Risk Score Contribution (0 to 50 pts)
    sev_weights = {"CRITICAL": 35, "HIGH": 22, "MEDIUM": 12, "LOW": 5}
    fam_d_score = min(50, sum(sev_weights.get(f["severity"], 10) for f in findings))

    return {
        "family": "FAMILY_D_CROSS_DOCUMENT",
        "status": "REVIEW_REQUIRED" if findings else "NO_MISMATCH_FOUND",
        "documents_evaluated": len(docs),
        "checks_matrix": checks,
        "risk_score_contribution": fam_d_score,
        "structured_findings": findings,
        "findings": [f["message"] for f in findings],
    }


# Backward-compatible wrappers for existing callers
def verify_corporate_msme_pack(document_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return verify_loan_dossier(document_outputs)


def verify_application_pack(statement_data, salary_slip_data, application_data):
    docs = [
        {"identity_card": {"doc_code": "BANK_STATEMENT", "doc_category": "Bank Statement", "holder_name": statement_data.get("account_holder")}, "extracted_financials": statement_data},
        {"identity_card": {"doc_code": "SALARY_SLIP", "doc_category": "Salary Slip", "holder_name": salary_slip_data.get("employee_name")}, "extracted_financials": salary_slip_data},
        {"identity_card": {"doc_code": "APPLICATION_FORM", "doc_category": "Application Form", "holder_name": application_data.get("applicant_name"), "pan_number": application_data.get("pan")}, "extracted_financials": application_data},
    ]
    return verify_loan_dossier(docs)