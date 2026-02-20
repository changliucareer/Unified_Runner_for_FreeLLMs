    # -------------------------
    # Prompt Builder
    # -------------------------
###--  updated prompt to enforce subcategory assignment rules --###
def build_prompt_appearance(comment: str) -> str:
    return f"""
    You are an information extraction system.

    Task:
    Detect appearance-related content in the text and determine its valence.

    Rules:
    - If appearance is detected (contains_appearance = true),
    you MUST assign exactly ONE appearance_sub_category.
    - If appearance is NOT detected (contains_appearance = false),
    appearance_sub_category MUST be null,
    appearance_valence MUST be null,
    and segments MUST be an empty list.

    Definition:
    Appearance-related content refers to descriptions of a person's:
    - Facial features
    - Body features
    - Clothing or dress
    - Cleanliness or grooming
    - Evaluative framing of physical looks

    Valence categories:
    - negative (insulting, mocking, degrading)
    - positive (complimenting but appearance-focused, may reinforce objectification)
    - neutral (descriptive without evaluative tone)

    Output format (STRICT JSON ONLY):
    {{
    "contains_appearance": true or false,
    "appearance_sub_category":
        "facial_features" or
        "body_features" or
        "clothing_or_dress" or
        "cleanliness_or_grooming" or
        "evaluative_framing" or null,
    "appearance_valence": "negative" or "positive" or "neutral" or null,
    "segments": ["exact text spans"],
    "reason": "short explanation"
    }}

    Text:
    \"\"\"{comment}\"\"\"
    """
