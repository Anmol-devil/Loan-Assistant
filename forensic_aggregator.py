import cv2
import numpy as np


def detect_copy_move_forgery(image_path, min_matches=18):
    """
    Family C (Classical): Uses ORB keypoint matching to detect cloned/copy-moved
    regions (e.g., duplicated transaction rows or stamps) inside an image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is None or len(keypoints) < 20:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors)

    cloned_pairs = []
    for m in matches:
        # Ignore self-matches; look for identical feature descriptors at different coordinates
        pt1 = np.array(keypoints[m.queryIdx].pt)
        pt2 = np.array(keypoints[m.trainIdx].pt)
        spatial_dist = np.linalg.norm(pt1 - pt2)

        if m.distance < 12 and spatial_dist > 45.0:
            cloned_pairs.append((pt1, pt2))

    if len(cloned_pairs) >= min_matches:
        return [
            f"Copy-Move Forgery Alert: Detected {len(cloned_pairs)} duplicated keypoint clusters "
            f"(possible cloned row, signature, or stamp)."
        ]
    return []


def compute_weighted_risk_score(report):
    """
    Implements Section 4.2 Weighted Evidence Aggregator.
    Distinguishes 'CHECKED_AND_CLEAN', 'FLAGGED', 'INCOMPLETE', and 'NOT_APPLICABLE'.
    """
    file_fmt = report.get("file_format", "UNKNOWN")
    flags = report.get("forensic_flags", [])
    fin = report.get("financial_reconciliation", {})
    math_issues = fin.get("issues", [])
    coverage = fin.get("audit_coverage", {})

    # Separate physical flags into Family B (Structure/Metadata/Font) vs Family C (Image ELA/Pixel)
    family_b_flags = [f for f in flags if any(k in f.lower() for k in ["metadata", "font", "overlay", "docx", "revision", "author"])]
    family_c_flags = [f for f in flags if any(k in f.lower() for k in ["ela", "pixel", "patch", "copy-move", "compression"])]

    # 1. Family A: Logic & Arithmetic Checks (Weight: 45 points)
    if math_issues:
        fam_a_state = "FLAGGED"
        fam_a_score = 45
    elif not all(coverage.values()):
        fam_a_state = "INCOMPLETE_EXTRACTION"
        fam_a_score = 25
    else:
        fam_a_state = "CHECKED_AND_CLEAN"
        fam_a_score = 0

    # 2. Family B: Document Structure & Font Forensics (Weight: 30 points)
    if file_fmt in ("DIGITAL_PDF", "DOCX"):
        if family_b_flags:
            fam_b_state = "FLAGGED"
            fam_b_score = min(30, 15 * len(family_b_flags))
        else:
            fam_b_state = "CHECKED_AND_CLEAN"
            fam_b_score = 0
    else:
        # Image files Still check EXIF/PNG headers
        if family_b_flags:
            fam_b_state = "FLAGGED (HEADER)"
            fam_b_score = 20
        else:
            fam_b_state = "NOT_APPLICABLE (FLAT_IMAGE)"
            fam_b_score = 0

    # 3. Family C: Image & Pixel Forensics (Weight: 25 points)
    if file_fmt in ("IMAGE", "SCANNED_PDF"):
        if family_c_flags:
            fam_c_state = "FLAGGED"
            fam_c_score = 25
        else:
            fam_c_state = "CHECKED_AND_CLEAN"
            fam_c_score = 0
    else:
        fam_c_state = "NOT_APPLICABLE (BORN_DIGITAL)"
        fam_c_score = 0

    total_risk = min(100, fam_a_score + fam_b_score + fam_c_score)

    if total_risk >= 40:
        risk_tier = "HIGH RISK (REJECT / MANUAL AUDIT)"
    elif total_risk >= 15:
        risk_tier = "MEDIUM RISK (REVIEW REQUIRED)"
    else:
        risk_tier = "LOW RISK (VERIFIED CLEAN)"

    return {
        "risk_score": total_risk,
        "risk_tier": risk_tier,
        "family_matrix": [
            {"Family": "Family A: Arithmetic & Logic", "State": fam_a_state, "Risk Contribution": f"+{fam_a_score} pts"},
            {"Family": "Family B: File Structure & Fonts", "State": fam_b_state, "Risk Contribution": f"+{fam_b_score} pts"},
            {"Family": "Family C: Pixel ELA & Copy-Move", "State": fam_c_state, "Risk Contribution": f"+{fam_c_score} pts"},
        ]
    }