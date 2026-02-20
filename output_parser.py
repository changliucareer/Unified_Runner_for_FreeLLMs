import os
import json
import re
import csv
from typing import Dict, Any

GBV_CLASSIFIER_LABEL_MAP = {
    # Hate-speech-CNERG/bert-base-uncased-hatexplain
    # 0 = hate speech, 1 = normal, 2 = offensive
    "hate_explain": {
        0: {"contains_gbv": True,  "category": "hate_speech"},
        1: {"contains_gbv": False, "category": "none"}, # Corrected: 1 is Normal
        2: {"contains_gbv": True,  "category": "offensive_language"} # Corrected: 2 is Offensive
    },

    # cardiffnlp/twitter-roberta-base-hate
    # 0 = non-hate, 1 = hate
    "cardiff_hate": {
        0: {"contains_gbv": False, "category": "none"},
        1: {"contains_gbv": True,  "category": "hate_speech"}
    },

    # tum-nlp/bertweet-sexism (Trained on EDOS)
    # 0 = not sexist, 1 = sexist
    "sexism_edos_tum": {
        0: {"contains_gbv": False, "category": "none"},
        1: {"contains_gbv": True,  "category": "sexism"}
    },

    # NLP-LTU/bertweet-large-sexism-detector (Trained on EDOS)
    # 0 = not sexist, 1 = sexist
    "sexism_edos_ltu": {
        0: {"contains_gbv": False, "category": "none"},
        1: {"contains_gbv": True,  "category": "sexism"}
    }
}


###-  Strip markdown fences, Extract the FULL outermost JSON block -###
def extract_json_block(text: str) -> str:
    """
    Extract full JSON block from LLM output.
    Handles markdown fences and multi-line JSON.
    """

    if not text:
        return ""

    # Remove markdown fences if present
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end+1]

    return ""

def repair_json(text: str) -> str:
    """
    Attempt minimal repair for common LLM JSON errors.
    """
    if not text:
        return ""

    # Replace single quotes
    text = text.replace("'", '"')

    # Fix lowercase booleans accidentally quoted
    text = text.replace('"true"', 'true')
    text = text.replace('"false"', 'false')

    # If JSON missing closing brace
    if text.count("{") > text.count("}"):
        text += "}"

    return text

def safe_json_load(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = repair_json(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to close common truncation
        if text.count("{") > text.count("}"):
            text += "}"
        if text.count("[") > text.count("]"):
            text += "]"

        try:
            return json.loads(text)
        except:
            return {}

def fallback_boolean_detection(text: str) -> bool:
    """
    If JSON completely broken, detect boolean manually.
    """
    if re.search(r'"contains_appearance"\s*:\s*true', text, re.IGNORECASE):
        return True
    if re.search(r'"contains_appearance"\s*:\s*false', text, re.IGNORECASE):
        return False
    return ""


###-- Updated to handle valence and more robust parsing --###
def infer_sub_category_from_reason(reason: str) -> str:
    """
    Fallback heuristic: infer sub-category from LLM reason text.
    """
    if not reason:
        return ""

    reason_lower = reason.lower()

    if "face" in reason_lower or "facial" in reason_lower:
        return "facial_features"
    if "body" in reason_lower:
        return "body_features"
    if "clothing" in reason_lower or "dress" in reason_lower:
        return "clothing_or_dress"
    if "groom" in reason_lower or "clean" in reason_lower:
        return "cleanliness_or_grooming"
    if "evaluat" in reason_lower or "look" in reason_lower:
        return "evaluative_framing"

    return ""


def parse_appearance_output_file(input_jsonl: str,
                                  original_comments: Dict[str, str],
                                  output_csv: str):

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(input_jsonl, "r") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.writer(outfile)
        writer.writerow([
            "comment_id",
            "comment",
            "contains_appearance",
            "sub_category",
            "appearance_valence",
            "segments",
            "reason"
        ])

        for line in infile:
            record = json.loads(line)

            cid = record["cid"]
            raw_output = record["raw_output"]

            json_block = extract_json_block(raw_output)
            parsed = safe_json_load(json_block)

            if not parsed:
                contains = fallback_boolean_detection(raw_output)
                sub_category = ""
                valence = ""
                segments = []
                reason = ""
            else:
                contains = parsed.get("contains_appearance", "")
                sub_category = parsed.get("appearance_sub_category", "")
                valence = parsed.get("appearance_valence", "")
                segments = parsed.get("segments", [])
                reason = parsed.get("reason", "")

                # If model didn't provide sub_category but reason exists
                if not sub_category and reason:
                    sub_category = infer_sub_category_from_reason(reason)

            writer.writerow([
                cid,
                original_comments.get(cid, ""),
                contains,
                sub_category,
                valence,
                "; ".join(segments) if isinstance(segments, list) else segments,
                reason
            ])

    print(f"✅ Appearance parsed file saved to {output_csv}")

###-- New function for GBV parsing with more fields --###
def parse_gbv_classifier_output_file(input_jsonl: str,
                                     original_comments: dict,
                                     output_csv: str,
                                     model_name: str):
    """
    Parse outputs from classification-based GBV models.
    Maps raw label integers into interpretable categories.
    """

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    label_map = GBV_CLASSIFIER_LABEL_MAP.get(model_name, {})

    with open(input_jsonl, "r") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.writer(outfile)

        writer.writerow([
            "comment_id",
            "comment",
            "contains_gbv",
            "gbv_category",
            "raw_label",
            "confidence"
        ])

        for line in infile:
            record = json.loads(line)

            cid = record.get("cid")
            raw_label = record.get("label")
            confidence = record.get("confidence")

            mapping = label_map.get(raw_label, {
                "contains_gbv": "",
                "category": "unknown"
            })

            contains_gbv = mapping["contains_gbv"]
            category = mapping["category"]

            writer.writerow([
                cid,
                original_comments.get(cid, ""),
                contains_gbv,
                category,
                raw_label,
                confidence
            ])

    print(f"✅ Parsed classifier output saved to {output_csv}")


### -- Improved fallback function for GBV boolean detection --###
def fallback_gbv_boolean_detection(text: str):
    """
    Fallback detection if JSON parsing fails.
    Attempts structured detection first, then light heuristic.
    """

    if not text:
        return ""

    if re.search(r'"contains_gbv"\s*:\s*true', text, re.IGNORECASE):
        return True

    if re.search(r'"contains_gbv"\s*:\s*false', text, re.IGNORECASE):
        return False

    # Light heuristic fallback (only if JSON totally broken)
    heuristic_keywords = [
        "misogyn", "slut", "bitch", "whore",
        "rape", "kill", "sexual", "sexist"
    ]

    for word in heuristic_keywords:
        if re.search(word, text, re.IGNORECASE):
            return True

    return ""


def parse_gbv_output_file(input_jsonl: str,
                          original_comments: Dict[str, str],
                          output_csv: str):
    """
    Parse outputs from LLM-based GBV detection.
    Robust to:
    - key drift (secondary vs subcategories)
    - target vs target_group
    - truncated JSON
    - non-list segments
    """

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(input_jsonl, "r") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.writer(outfile)

        writer.writerow([
            "comment_id",
            "comment",
            "contains_gbv",
            "gbv_primary_category",
            "gbv_secondary_categories",
            "target",
            "segments",
            "reason"
        ])

        for line in infile:
            record = json.loads(line)

            cid = record.get("cid")
            raw_output = record.get("raw_output", "")

            json_block = extract_json_block(raw_output)
            parsed = safe_json_load(json_block)

            # ------------------------
            # CASE 1: JSON FAILED
            # ------------------------
            if not parsed:
                contains = fallback_gbv_boolean_detection(raw_output)
                primary = ""
                secondary = []
                target = ""
                segments = []
                reason = ""

            # ------------------------
            # CASE 2: JSON PARSED
            # ------------------------
            else:
                contains = parsed.get("contains_gbv", "")

                # Robust key handling
                primary = (
                    parsed.get("gbv_primary_category")
                    or parsed.get("primary_category")
                    or ""
                )

                secondary = (
                    parsed.get("gbv_secondary_categories")
                    or parsed.get("gbv_subcategories")
                    or parsed.get("secondary_categories")
                    or []
                )

                target = (
                    parsed.get("target")
                    or parsed.get("target_group")
                    or ""
                )

                segments = parsed.get("segments", [])
                reason = parsed.get("reason", "")

                # Normalize list fields safely
                if isinstance(secondary, str):
                    secondary = [secondary]

                if not isinstance(secondary, list):
                    secondary = []

                if isinstance(segments, str):
                    segments = [segments]

                if not isinstance(segments, list):
                    segments = []

                # If model says contains_gbv = False
                # force structural consistency
                if contains is False:
                    primary = ""
                    secondary = []
                    target = ""
                    segments = []

            writer.writerow([
                cid,
                original_comments.get(cid, ""),
                contains,
                primary,
                "; ".join(secondary),
                target,
                "; ".join(segments),
                reason
            ])

    print(f"✅ GBV parsed file saved to {output_csv}")