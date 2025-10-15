import json
from rapidfuzz import fuzz
import re

def clean_text(text):
    """Normalize text for matching: remove non-alphanumeric and uppercase"""
    if not text:
        return ""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()


def load_oem_db(json_path):
    """Load OEM database from a JSON file"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_text(ocr_text, oem_db, threshold=80):
    """Match OCR text against OEM DB using fuzzy matching"""
    ocr_clean = clean_text(ocr_text)
    best_match = None
    best_score = 0

    for entry in oem_db:
        target_text = entry.get("name", "")
        aliases = entry.get("aliases", [])
        candidates = [target_text] + aliases

        for candidate in candidates:
            candidate_clean = clean_text(candidate)
            score = fuzz.token_sort_ratio(ocr_clean, candidate_clean)
            if score > best_score:
                best_score = score
                best_match = entry

    if best_match and best_score >= threshold:
        return {
            "match_found": True,
            "name": best_match.get("name", ""),
            "manufacturer": best_match.get("manufacturer", ""),
            "description": best_match.get("description", ""),
            "score": best_score
        }
    else:
        return {
            "match_found": False,
            "name": None,
            "manufacturer": None,
            "description": None,
            "score": best_score
        }
