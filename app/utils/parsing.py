"""Parsing utilities for extracting structured components from model outputs."""

import re
import json
from typing import Dict, List, Optional, Any, Set


# ---------------------------------------------------------------------------
# Lookup tables used by normalize_answer
# ---------------------------------------------------------------------------

# Boolean / null equivalences → canonical digit string
_BOOL_MAP: Dict[str, str] = {
    "false": "0", "no": "0", "none": "0",
    "n/a": "0", "na": "0", "null": "0",
    "not available": "0", "not applicable": "0",
    "true": "1", "yes": "1",
}

# English number words → digit string (0-19 + tens)
_NUM_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}

# Ordinal words → cardinal digit string
_ORDINAL_WORDS: Dict[str, str] = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12",
}

# Month abbreviations → full name
_MONTH_MAP: Dict[str, str] = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "sept": "september",
    "oct": "october", "nov": "november", "dec": "december",
}

# Ordinal suffix pattern: "1st", "2nd", "3rd", "4th", ...
_ORDINAL_SUFFIX_RE = re.compile(r"^(-?\d+)(?:st|nd|rd|th)$")

# Compound number word pattern: "twenty one", "forty five", etc.
_COMPOUND_NUM_RE = re.compile(
    r"^(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[\s-]"
    r"(one|two|three|four|five|six|seven|eight|nine)$"
)
_COMPOUND_TENS: Dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_COMPOUND_ONES: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

# Trailing count-noun pattern after a number: "5 bars", "3 items", etc.
_TRAILING_COUNT_NOUN_RE = re.compile(
    r"^(-?\d+(?:\.\d+)?)"
    r"\s+"
    r"(?:bars?|lines?|items?|points?|segments?|slices?|"
    r"data\s+points?|categories|entries|values?|columns?|rows?)$"
)

# Negative word pattern: "negative 5" → "-5"
_NEGATIVE_WORD_RE = re.compile(r"^negative\s+(\d+(?:\.\d+)?)$")

# Chart-specific lead-in phrases to strip (extended)
_LEADIN_RE = re.compile(
    r"^(?:"
    r"the\s+answer\s+is"
    r"|answer\s+is"
    r"|it\s+is"
    r"|it's"
    r"|approximately"
    r"|about"
    r"|the\s+\w+\s+(?:color|colour|label|value|name|bar|line|segment|category|slice)\s+is"
    r"|the\s+(?:color|colour|label|value|name|bar|line|segment|category)\s+is"
    r"|the\s+(?:highest|lowest|largest|smallest|maximum|minimum|greatest|least)\s+(?:value\s+)?is"
    r"|the\s+(?:correct\s+)?answer\s+(?:would\s+be|would\s+be\s+approximately)"
    r")\s+",
    re.IGNORECASE,
)


def parse_response(text: str) -> Dict[str, Any]:
    """
    Parse model output into structured components.

    Expected format:
        <think>
        <type>chart_type</type>
        <table>{"columns": [...], "rows": [...]}</table>
        reasoning steps...
        </think>
        <answer>final answer</answer>

    Args:
        text: Raw model output

    Returns:
        Dictionary with keys: type, table, reasoning, answer, raw
    """
    result = {
        "type": "",
        "table": {},
        "table_raw": "",
        "table_parse_success_strict": False,
        "reasoning": "",
        "answer": "",
        "raw": text,
        "parse_success": True,
    }

    # Extract chart type
    type_match = re.search(r"<type>(.*?)</type>", text, re.DOTALL)
    if type_match:
        result["type"] = type_match.group(1).strip().lower()

    # Extract table JSON
    table_match = re.search(r"<table>(.*?)</table>", text, re.DOTALL)
    if table_match:
        table_str = table_match.group(1).strip()
        result["table_raw"] = table_str
        # Strict JSON parse for format reward consistency
        try:
            json.loads(table_str)
            result["table_parse_success_strict"] = True
        except json.JSONDecodeError:
            result["table_parse_success_strict"] = False
        # Lenient parse for downstream table similarity
        result["table"] = parse_json_table(table_str)

    # Extract answer
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()

    # Extract reasoning (content in <think> after </table> and before </think>)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        # Remove type and table tags to get reasoning
        reasoning = re.sub(r"<type>.*?</type>", "", think_content, flags=re.DOTALL)
        reasoning = re.sub(r"<table>.*?</table>", "", reasoning, flags=re.DOTALL)
        result["reasoning"] = reasoning.strip()

    # Check parse success
    if not result["type"] or not result["answer"]:
        result["parse_success"] = False

    return result


def _normalize_table_obj(obj: Any) -> Dict[str, Any]:
    """Return a canonical table dict or {} for malformed parsed JSON."""
    if not isinstance(obj, dict):
        return {}

    columns = obj.get("columns", [])
    rows = obj.get("rows", [])
    if not isinstance(columns, list) or not isinstance(rows, list):
        return {}

    return {"columns": columns, "rows": rows}


def parse_json_table(table_str: str) -> Dict[str, Any]:
    """
    Parse JSON table string into dictionary.

    Handles common formatting issues in model outputs.

    Args:
        table_str: JSON string of table

    Returns:
        Parsed dictionary or empty dict on failure
    """
    if not table_str:
        return {}

    # Clean common issues
    table_str = table_str.strip()

    # If wrapped in ```json ... ``` take the inner block
    if "```json" in table_str:
        try:
            table_str = table_str.split("```json", 1)[-1].split("```", 1)[0].strip()
        except Exception:
            pass

    # Try direct parse
    try:
        return _normalize_table_obj(json.loads(table_str))
    except json.JSONDecodeError:
        pass

    # Try fixing common issues
    try:
        # Replace single quotes with double quotes
        fixed = table_str.replace("'", '"')
        return _normalize_table_obj(json.loads(fixed))
    except json.JSONDecodeError:
        pass

    # Try extracting just the JSON object
    try:
        match = re.search(r"\{.*\}", table_str, re.DOTALL)
        if match:
            return _normalize_table_obj(json.loads(match.group()))
    except json.JSONDecodeError:
        pass

    return {}


def extract_numbers(text: str) -> Set[float]:
    """
    Extract all numeric values from text.

    Handles integers, floats, percentages, and negative numbers.

    Args:
        text: Input text

    Returns:
        Set of numeric values found
    """
    if not text:
        return set()

    numbers = set()

    # Pattern for numbers (handles decimals, negatives, percentages)
    pattern = r"-?\d+\.?\d*%?"

    for match in re.finditer(pattern, text):
        num_str = match.group()

        # Handle percentage
        is_percentage = num_str.endswith("%")
        if is_percentage:
            num_str = num_str[:-1]

        try:
            value = float(num_str)
            if is_percentage:
                value = value / 100
            numbers.add(value)
        except ValueError:
            continue

    return numbers


def extract_table_values(table: Dict[str, Any]) -> Set[float]:
    """
    Extract all numeric values from a table dictionary.

    Args:
        table: Table dictionary with 'columns' and 'rows' keys

    Returns:
        Set of numeric values in the table
    """
    values = set()

    if not table or not isinstance(table, dict):
        return values

    def extract_from_value(v):
        """Recursively extract numbers from a value."""
        if isinstance(v, (int, float)):
            values.add(float(v))
        elif isinstance(v, str):
            values.update(extract_numbers(v))
        elif isinstance(v, list):
            for item in v:
                extract_from_value(item)
        elif isinstance(v, dict):
            for val in v.values():
                extract_from_value(val)

    # Extract from rows
    rows = table.get("rows", [])
    for row in rows:
        extract_from_value(row)

    # Extract from columns (in case they contain numeric data)
    columns = table.get("columns", [])
    for col in columns:
        extract_from_value(col)

    return values


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.

    Handles:
    - Lead-in phrase stripping (generic + chart-specific)
    - Boolean / null equivalences (false/no → "0", true/yes → "1")
    - Negative word form ("negative 5" → "-5")
    - Ordinal suffixes ("2nd" → "2") and ordinal words ("second" → "2")
    - Number words ("three" → "3", "twenty one" → "21")
    - Month abbreviations ("jan" → "january")
    - Trailing count nouns ("5 bars" → "5")
    - British/American spelling, currency symbols, commas, punctuation

    Args:
        answer: Raw answer string

    Returns:
        Normalized lowercase string
    """
    if not answer:
        return ""

    # Strip and lowercase
    normalized = answer.strip().lower()

    # Strip chart-specific and generic lead-in phrases (iterative — handles nested)
    prev = None
    while prev != normalized:
        prev = normalized
        normalized = _LEADIN_RE.sub("", normalized).strip()

    # Normalize british/american spelling variants
    normalized = normalized.replace("grey", "gray")
    normalized = normalized.replace("colour", "color")

    # Remove currency words/symbols and commas
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\b(usd|dollar|dollars|eur|euro|euros|gbp|pound|pounds)\b", "", normalized)
    normalized = re.sub(r"[$£€]", "", normalized).strip()

    # Remove trailing punctuation
    normalized = re.sub(r"[.,;:!?]+$", "", normalized)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    if not normalized:
        return ""

    # --- Semantic equivalences (applied to the full string only for safety) ---

    # Negative word form: "negative 5" → "-5"
    neg_m = _NEGATIVE_WORD_RE.match(normalized)
    if neg_m:
        normalized = "-" + neg_m.group(1)

    # Ordinal suffix: "1st" → "1", "2nd" → "2", etc.
    ord_m = _ORDINAL_SUFFIX_RE.match(normalized)
    if ord_m:
        normalized = ord_m.group(1)

    # Boolean / null equivalences
    if normalized in _BOOL_MAP:
        normalized = _BOOL_MAP[normalized]

    # Ordinal words: "second" → "2"
    if normalized in _ORDINAL_WORDS:
        normalized = _ORDINAL_WORDS[normalized]

    # Compound number words: "twenty one" → "21", "forty-five" → "45"
    comp_m = _COMPOUND_NUM_RE.match(normalized)
    if comp_m:
        tens = _COMPOUND_TENS[comp_m.group(1)]
        ones = _COMPOUND_ONES[comp_m.group(2)]
        normalized = str(tens + ones)

    # Single number words: "three" → "3"
    if normalized in _NUM_WORDS:
        normalized = _NUM_WORDS[normalized]

    # Month abbreviations: "jan" → "january"
    if normalized in _MONTH_MAP:
        normalized = _MONTH_MAP[normalized]

    # Trailing count nouns after a number: "5 bars" → "5"
    count_m = _TRAILING_COUNT_NOUN_RE.match(normalized)
    if count_m:
        normalized = count_m.group(1)

    return normalized


def _parse_numeric_with_units(text: str) -> Optional[float]:
    """
    Parse a numeric value with optional magnitude suffixes/words.

    Magnitude treatment:
      - "hundred" is expanded: "2 hundred" → 200.0
      - k/m/b/t/thousand/million/billion/trillion are treated cosmetically
        (stripped) so that "1.5 million" and "1.5" both parse to 1.5 and
        compare equal within the training reward tolerance.
      - "%" returns the raw number (50% → 50.0, consistent with label format).

    Args:
        text: String potentially containing a number with units

    Returns:
        Float value or None if no numeric content found
    """
    if not text:
        return None

    cleaned = text.strip().lower()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")

    # Handle percent word
    cleaned = cleaned.replace("percent", "%")

    # Handle "hundred" before the main regex so "2 hundred" → 200
    hundred_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s+hundred\b", cleaned
    )
    if hundred_match:
        # Check nothing else follows (avoid "2 hundred thousand" → just 200)
        tail = cleaned[hundred_match.end():].strip()
        if not tail or tail.startswith("%"):
            return float(hundred_match.group(1)) * 100

    pattern = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(%|k|m|b|t|thousand|million|billion|trillion)?"
    )

    match = pattern.search(cleaned)
    if not match:
        return None

    number = float(match.group(1))
    suffix = match.group(2)

    if suffix == "%":
        return number

    # Treat large magnitude suffixes as cosmetic
    # (labels and predictions usually use the same format)
    if suffix in {"k", "m", "b", "t", "thousand", "million", "billion", "trillion"}:
        return number

    return number


def try_parse_numeric(value: str) -> Optional[float]:
    """
    Try to parse a string as a numeric value.

    Args:
        value: String to parse

    Returns:
        Float value or None if not numeric
    """
    if not value:
        return None

    # First try parsing with unit suffixes/words
    parsed = _parse_numeric_with_units(value)
    if parsed is not None:
        return parsed

    # Clean the string
    cleaned = value.strip()

    # Remove common prefixes/suffixes
    cleaned = re.sub(r"^[$%]", "", cleaned)
    cleaned = re.sub(r"[$%]$", "", cleaned)
    cleaned = cleaned.replace(",", "")  # Remove thousand separators

    try:
        return float(cleaned)
    except ValueError:
        return None


def split_reasoning_steps(reasoning: str) -> List[str]:
    """
    Split reasoning into individual steps.

    Handles various step markers: numbers, bullets, newlines.

    Args:
        reasoning: Reasoning text

    Returns:
        List of reasoning steps
    """
    if not reasoning:
        return []

    # Try splitting by numbered steps (1., 2., etc.)
    numbered = re.split(r"\n?\d+[.)]\s*", reasoning)
    if len(numbered) > 1:
        return [s.strip() for s in numbered if s.strip()]

    # Try splitting by bullet points
    bulleted = re.split(r"\n?[-*]\s*", reasoning)
    if len(bulleted) > 1:
        return [s.strip() for s in bulleted if s.strip()]

    # Try splitting by "Step" markers
    stepped = re.split(r"\n?Step\s*\d*[:.]\s*", reasoning, flags=re.IGNORECASE)
    if len(stepped) > 1:
        return [s.strip() for s in stepped if s.strip()]

    # Try splitting by <step-1>: style tags
    tagged = re.split(r"\n?<step-\d+>:\s*", reasoning, flags=re.IGNORECASE)
    if len(tagged) > 1:
        return [s.strip() for s in tagged if s.strip()]

    # Fall back to sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", reasoning)
    return [s.strip() for s in sentences if s.strip()]


def check_format_compliance(text: str) -> Dict[str, bool]:
    """
    Check if output follows expected format.

    Args:
        text: Model output

    Returns:
        Dictionary with compliance flags
    """
    checks = {
        "has_think_tags": bool(re.search(r"<think>.*</think>", text, re.DOTALL)),
        "has_answer_tags": bool(re.search(r"<answer>.*</answer>", text, re.DOTALL)),
        "has_type_tags": bool(re.search(r"<type>.*</type>", text, re.DOTALL)),
        "has_table_tags": bool(re.search(r"<table>.*</table>", text, re.DOTALL)),
        "proper_order": False,
        "single_tags": True,
    }

    # Check order: <think> before <answer>
    think_pos = text.find("<think>")
    answer_pos = text.find("<answer>")
    if think_pos >= 0 and answer_pos >= 0:
        checks["proper_order"] = think_pos < answer_pos

    # Check for duplicate tags
    for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
        if text.count(tag) > 1:
            checks["single_tags"] = False
            break

    checks["fully_compliant"] = all(checks.values())

    return checks
