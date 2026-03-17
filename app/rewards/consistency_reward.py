"""
Consistency reward: verifies the model's reasoning references the values it
extracted into <table>, rather than hallucinating different numbers.

A model that actually read the chart will cite the same numbers in its
<table> and its reasoning steps.  A model that guessed the answer and
back-filled a plausible-sounding trace will have mismatches.

Score in [0, 1]:
  1.0  all table values appear in reasoning (or table has no numeric values)
  0.0  no table block found, or table has numbers but none appear in reasoning

Design decisions:
  - Tolerance matching: 2% relative OR 0.01 absolute, so 16.666 matches 16.67
  - Neutral score (1.0) when table has no numbers: categorical charts like
    "highest category" have text-only tables; we cannot penalise those rollouts
  - Fallback reasoning extraction: if </think> is missing (truncated output),
    we use everything after </table> instead of returning 0 unfairly
  - Column headers included: year columns (2020, 2021 ...) are valid values
    the model may reference in reasoning
"""

import json
import re
from typing import List, Set

# Matches: optional minus, digits with optional comma thousands, optional decimal
# Examples: 16.67, -3, 1,234.5, 0.8, 2020
# NOTE: \d+ (not \d{1,3}) so "2020" matches as one token, not "202" + "0"
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

_TABLE_RE = re.compile(r"<table>(.*?)</table>", re.DOTALL | re.IGNORECASE)

# Primary: text between </table> and </think>
# Fallback: everything after </table> (handles truncated outputs)
_REASON_STRICT_RE = re.compile(r"</table>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_REASON_LOOSE_RE  = re.compile(r"</table>(.*)",          re.DOTALL | re.IGNORECASE)

# Tolerance thresholds for "close enough"
_REL_TOL = 0.02   # 2% relative tolerance
_ABS_TOL = 0.01   # absolute tolerance for small numbers

# "negative 5.2" or "minus 5.2" → -5.2
_NEG_WORD_RE = re.compile(r"\b(?:negative|minus)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_num(s: str) -> float:
    """Parse a number string, stripping comma separators."""
    return float(s.replace(",", ""))


def _nums_close(a: float, b: float) -> bool:
    """
    True if a and b are numerically close.

    Uses the looser of 2% relative tolerance and 0.01 absolute tolerance
    so that rounding differences (e.g. 16.666 vs 16.67) count as matching.
    """
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return (abs(a - b) / denom < _REL_TOL) or (abs(a - b) < _ABS_TOL)


def _nums_from_table(table_str: str) -> Set[float]:
    """
    Extract all numeric values from the model's JSON table output.

    Parses both:
      - column headers that are numeric (e.g. year columns: 2020, 2021)
      - row cell values

    Falls back to regex scan of the raw string if JSON parse fails.
    """
    try:
        obj = json.loads(table_str.strip())
    except Exception:
        # JSON parse failed (e.g. partial output) — regex scan the raw block
        nums: Set[float] = set()
        for m in _NUM_RE.findall(table_str):
            try:
                nums.add(_parse_num(m))
            except ValueError:
                pass
        return nums

    nums = set()

    # Numeric column headers (e.g. year axes)
    for col in obj.get("columns", []):
        try:
            nums.add(_parse_num(str(col)))
        except (ValueError, TypeError):
            pass

    # Row cell values
    for row in obj.get("rows", []):
        for val in row:
            try:
                nums.add(_parse_num(str(val)))
            except (ValueError, TypeError):
                pass

    return nums


def _nums_from_text(text: str) -> List[float]:
    """
    Extract all numbers from free-form reasoning text.

    Handles:
      - Standard numerics: 16.67, -3, 1,234.5
      - Word-form negatives: "negative 5.2" → -5.2, "minus 3" → -3
    """
    nums = []
    # Word-form negatives first ("negative 5.2" → -5.2)
    for m in _NEG_WORD_RE.findall(text):
        try:
            nums.append(-_parse_num(m))
        except ValueError:
            pass
    # Standard regex (also picks up the bare digit in "negative 5.2" as +5.2,
    # which is intentional — both +5.2 and -5.2 end up in the list so that
    # _nums_close can match either sign convention used in the reasoning)
    for m in _NUM_RE.findall(text):
        try:
            nums.append(_parse_num(m))
        except ValueError:
            pass
    return nums


def _extract_reasoning(completion: str) -> str:
    """
    Extract the reasoning block that comes after the table.

    Strategy:
      1. Try </table>...</think>  (ideal — bounded region)
      2. Fall back to everything after </table>  (handles truncated output)
      3. Return empty string if no </table> found
    """
    m = _REASON_STRICT_RE.search(completion)
    if m:
        return m.group(1)
    m = _REASON_LOOSE_RE.search(completion)
    if m:
        return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_consistency_reward(completion: str) -> float:
    """
    Score how consistently the model's reasoning references its own table.

    Args:
        completion: Full model output string (everything after the prompt).

    Returns:
        Float in [0, 1].
    """
    table_m = _TABLE_RE.search(completion)
    if not table_m:
        return 0.0

    table_nums = _nums_from_table(table_m.group(1))

    # Table has no numeric values (e.g. pure categorical chart).
    # We cannot verify grounding via numbers — return neutral score.
    if not table_nums:
        return 1.0

    reasoning = _extract_reasoning(completion)
    if not reasoning:
        return 0.0

    reason_nums = _nums_from_text(reasoning)
    if not reason_nums:
        return 0.0

    # For each table value, check if at least one reasoning number is close to it
    matched = sum(
        1 for t in table_nums
        if any(_nums_close(t, r) for r in reason_nums)
    )
    return matched / len(table_nums)


def consistency_reward_batch(completions: List[str]) -> List[float]:
    """Compute consistency reward for a batch of completions."""
    return [compute_consistency_reward(c) for c in completions]
