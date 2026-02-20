    # -------------------------
    # Prompt Builder
    # -------------------------


# def build_prompt_appearance(comment: str) -> str:
#     return f"""
#         You are an information extraction system.

#         Task:
#         Detect appearance-related content in the given text.

#         Definition:
#         Appearance-related content refers to descriptions of a person's:
#         - Facial features
#         - Body features
#         - Clothing or dress
#         - Cleanliness or grooming

#         Instructions:
#         1. Answer ONLY in valid JSON.
#         2. Do NOT include explanations outside JSON.
#         3. If no appearance-related content exists, return empty list for segments.

#         Output format:
#         {{
#         "contains_appearance": true or false,
#         "segments": ["segment1", "segment2"],
#         "reason": "short explanation"
#         }}

#         Text:
#         \"\"\"{comment}\"\"\"
#         """

### ------------------ add valence detection to appearance prompt ------------------ ###
# def build_prompt_appearance(comment: str) -> str:
#     return f"""
#     You are an information extraction system.

#     Task:
#     Detect appearance-related content in the text and determine its valence. If appearance is detected, you MUST assign exactly one appearance_sub_category.

#     Definition:
#     Appearance-related content refers to descriptions of a person's:
#     - Facial features
#     - Body features
#     - Clothing or dress
#     - Cleanliness or grooming
#     - Evaluative framing of physical looks

#     Valence categories:
#     - negative (insulting, mocking, degrading)
#     - positive (complimenting but appearance-focused, may reinforce objectification)
#     - neutral (descriptive without evaluative tone)

#     Instructions:
#     1. Use ONLY valid JSON.
#     2. Do NOT include explanations outside JSON.
#     3. If no appearance-related content exists, return empty list for segments and valence as null.

#     Output format:
#     {{
#     "contains_appearance": true or false,
#     "appearance_sub_category": 
#         "facial_features" or
#         "body_features" or
#         "clothing_or_dress" or
#         "cleanliness_or_grooming" or
#         "evaluative_framing" or null,
#     "appearance_valence": "negative" or "positive" or "neutral" or null,
#     "segments": ["segment1", "segment2"],
#     "reason": "short explanation"
#     }}

#     Text:
#     \"\"\"{comment}\"\"\"
#     """

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

### ------------------ add valence detection to GBV prompt ------------------ ###
# def build_prompt_gbv(comment: str) -> str:
#     return f"""
#     You are an information extraction system.

#     Task:
#     Detect whether the text contains Gender-Based Violence (GBV) according to the DHDC Harms Map taxonomy.

#     Primary GBV Categories:
#     - Sexual Harassment
#     - Misogyny
#     - Dehumanisation
#     - Threat of Violence
#     - Gendered Stereotyping
#     - Gendered Discreditation

#     Instructions:
#     1. Only use categories from the list above.
#     2. Appearance references alone do NOT constitute GBV unless linked to one of the above categories.
#     3. Return ONLY valid JSON.

#     Output format:
#     {{
#     "contains_gbv": true or false,
#     "gbv_primary_category": "one category or null",
#     "gbv_subcategories": ["optional finer labels"],
#     "target_group": "who is targeted or null",
#     "segments": ["text spans"],
#     "reason": "short explanation"
#     }}

#     Text:
#     \"\"\"{comment}\"\"\"
#   """

# 
### --- UPDATED build_prompt_gbv() (Taxonomy-Aligned) --- ###
def build_prompt_gbv(comment: str) -> str:
    return f"""
    You are an expert content analysis system.

    Your task is to detect Gender-Based Violence (GBV) against women MPs
    using the DHDC Harms Map taxonomy provided below.

    Only classify as GBV if the content fits one or more of the defined categories.

    You must return STRICTLY valid JSON.
    Do not include explanations outside JSON.

    --------------------------------------------------------------------
    DHDC GBV TAXONOMY
    --------------------------------------------------------------------

    PRIMARY CATEGORIES:

    1. Pejoratives
    - Misogynistic pejoratives (gendered slurs, sexualised insults,
        insults referencing female body parts, animal comparisons)
    - Racist or religion-based pejoratives targeting women

    2. Disrespectful Actions
    - Disinformation (including gendered disinformation, AI-generated harm)
    - Silencing (tone policing, patronising language, trivialising argument,
        mansplaining, derailing, racial gaslighting)
    - Anti-feminist discourse (denying discrimination, victim-blaming,
        claiming women have too much power, endorsing sexist practices)

    3. Benevolent Othering
    - Benevolent sexism
    - Reductive appearance-based compliments
    - Framing women as inspirational objects
    - Reinforcing stereotypical femininity

    4. Derogatory Content
    - Failure as woman (bad mother/wife, too aggressive, unfeminine)
    - Unfit for role as MP
        * Intellectual inferiority (stupid, incompetent, token hire,
            emotionally unstable)
        * Moral inferiority (liar, attention-seeking, disloyal,
            sexually promiscuous, exchanging sex for power)
    - Dehumanisation
        * Objectification
        * Sexualising content
        * Body shaming
        * Hygiene shaming

    5. Threats
    - Threats of physical violence
    - Threats of sexual violence
    - Threats to loved ones
    - Threats to privacy (doxing)
    - Supporting violence

    6. Violent Offences
    - Stalking
    - Sexual offences (sexual harassment, image-based abuse,
        deepfake sexual content, cyberflashing)

    --------------------------------------------------------------------

    IMPORTANT RULES:

    - If no GBV is present, set contains_gbv to false.
    - If contains_gbv is false:
    -- All other fields must be null or empty list.
    - If contains_gbv is true:
    -- You MUST select exactly ONE primary category.
    -- You MAY include additional relevant primary categories in
        gbv_secondary_categories if clearly present.
    -- Use exact category names as written above.
    -- Extract exact quoted text spans when possible.

    --------------------------------------------------------------------

    OUTPUT FORMAT (JSON ONLY):

    {{
    "contains_gbv": true or false,
    "gbv_primary_category": "Pejoratives" or
                            "Disrespectful Actions" or
                            "Benevolent Othering" or
                            "Derogatory Content" or
                            "Threats" or
                            "Violent Offences" or null,
    "gbv_secondary_categories": ["optional additional primary categories"],
    "target": "individual woman / women as a group / other / null",
    "segments": ["exact quoted spans from text"],
    "reason": "brief explanation referencing taxonomy language"
    }}

    --------------------------------------------------------------------

    Text:
    \"\"\"{comment}\"\"\"
    """