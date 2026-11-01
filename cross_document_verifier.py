import re
from difflib import SequenceMatcher

PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


def validate_pan(pan_str):
    """Validates 10-digit Indian PAN format and individual entity type."""
    if not pan_str:
        return False, "PAN not provided."
    cleaned = pan_str.strip().upper()
    if not PAN_REGEX.match(cleaned):
        return False, "Invalid PAN structure (expected 5 letters, 4 digits, 1 letter)."
    if cleaned[3] != "P":
        return False, f"PAN entity type '{cleaned[3]}' is not 'P' (Individual Person)."
    return True, "Valid Individual PAN."


def name_similarity(name1, name2):
    """Computes string similarity score between applicant names."""
    n1 = re.sub(r"[^a-zA-Z\s]", "", str(name1)).lower().strip()
    n2 = re.sub(r"[^a-zA-Z\s]", "", str(name2)).lower().strip()
    return round(SequenceMatcher(None, n1, n2).ratio(), 3)


def verify_application_pack(statement_data, salary_slip_data, application_data):
    """
    Family D: Cross-checks Bank Statement against Salary Slip and Application Form
    for Individual Retail Loans.
    """
    findings = []
    audit_checks = {
        "pan_format_valid": False,
        "name_cross_match": False,
        "employer_narration_match": False,
        "net_pay_credit_match": False
    }

    # 1. PAN Check
    applicant_pan = application_data.get("pan", "")
    pan_valid, pan_msg = validate_pan(applicant_pan)
    audit_checks["pan_format_valid"] = pan_valid
    if not pan_valid:
        findings.append(f"PAN Validation Error: {pan_msg}")

    # 2. Name Consistency (Statement vs Salary Slip vs Application)
    stmt_name = statement_data.get("account_holder", "")
    app_name = application_data.get("applicant_name", "")
    slip_name = salary_slip_data.get("employee_name", "")

    sim_app_stmt = name_similarity(app_name, stmt_name)
    sim_slip_stmt = name_similarity(slip_name, stmt_name)

    if sim_app_stmt < 0.80 or sim_slip_stmt < 0.80:
        findings.append(
            f"Identity Mismatch: Application Name ('{app_name}'), Statement Name ('{stmt_name}'), "
            f"and Payslip Name ('{slip_name}') do not match with high confidence."
        )
    else:
        audit_checks["name_cross_match"] = True

    # 3. Employer Narration Matching
    employer = salary_slip_data.get("employer_name", "").lower()
    employer_tokens = [
        t for t in re.findall(r"\w+", employer)
        if len(t) > 3 and t not in ("private", "limited", "pvt", "ltd")
    ]

    transactions = statement_data.get("transactions", [])
    found_employer_narration = False

    for tx in transactions:
        if tx.get("credit", 0) > 0:
            desc_lower = tx.get("description", "").lower()
            if any(token in desc_lower for token in employer_tokens):
                found_employer_narration = True
                break

    if not found_employer_narration and employer_tokens:
        findings.append(
            f"Employer Narration Warning: Declared employer '{salary_slip_data.get('employer_name')}' "
            f"not found in statement credit narrations."
        )
    else:
        audit_checks["employer_narration_match"] = True

    # 4. Net Salary Amount Cross-Match
    net_pay = salary_slip_data.get("net_pay")
    if net_pay:
        salary_credits = [
            tx["credit"] for tx in transactions
            if "salary" in tx.get("description", "").lower() or (tx["credit"] > 0 and found_employer_narration)
        ]
        if not any(abs(c - net_pay) <= 10.0 for c in salary_credits):
            findings.append(
                f"Income Cross-Check Mismatch: Salary slip Net Pay (₹{net_pay:,.2f}) does not match "
                f"any salary credit in the bank statement."
            )
        else:
            audit_checks["net_pay_credit_match"] = True

    status = "REVIEW_REQUIRED" if findings else "NO_MISMATCH_FOUND"

    return {
        "family": "FAMILY_D_CROSS_CHECKS",
        "status": status,
        "audit_checks": audit_checks,
        "findings": findings
    }


def verify_corporate_msme_pack(document_outputs: list) -> dict:
    """
    Family D Corporate MSME Pack Verifier:
    Cross-checks any combination of Bank Statement, GST Certificate,
    Udyam Certificate, ITR, Balance Sheet, and PAN Card.
    """
    findings = []
    pans_found = {}
    names_found = {}

    for doc in document_outputs:
        id_card = doc.get("identity_card", {})
        fin = doc.get("extracted_financials", {})
        cat = id_card.get("doc_category", "Document")
        holder = id_card.get("holder_name") or fin.get("legal_name") or fin.get("enterprise_name")
        pan = id_card.get("pan_number") or fin.get("embedded_pan") or fin.get("pan_number")

        if holder and holder != "Unknown Holder":
            names_found[cat] = holder
        if pan:
            pans_found[cat] = pan

    # 1. Cross-Check PANs across all documents
    unique_pans = set(pans_found.values())
    if len(unique_pans) > 1:
        findings.append(
            f"Corporate PAN Mismatch Across Dossier: Conflicting PAN numbers detected across documents: {pans_found}."
        )

    # 2. Cross-Check Entity / Holder Names across all documents
    name_list = list(names_found.items())
    for i in range(len(name_list)):
        for j in range(i + 1, len(name_list)):
            cat1, n1 = name_list[i]
            cat2, n2 = name_list[j]
            sim = name_similarity(n1, n2)
            if sim < 0.70 and n1.lower() not in n2.lower() and n2.lower() not in n1.lower():
                findings.append(
                    f"Entity Name Mismatch: '{cat1}' belongs to '{n1}', while '{cat2}' belongs to '{n2}' (similarity: {sim})."
                )

    return {
        "family": "FAMILY_D_CORPORATE_MSME_CROSS_CHECK",
        "status": "REVIEW_REQUIRED" if findings else "NO_MISMATCH_FOUND",
        "pans_verified": pans_found,
        "entities_verified": names_found,
        "findings": findings
    }