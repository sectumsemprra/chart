"""
Consistency reward: verifies the model's reasoning references the values it
extracted into <table>, rather than hallucinating different numbers.

A model that actually read the chart will have matching numbers in its table
and its reasoning steps.  A model that guessed the answer and back-filled a
plausible-sounding reasoning trace will have mismatches.

Score in [0, 1]:
  1.0  all table values appear in the reasoning steps
  0.0  no table extracted, or none of its values appear in reasoning
"""

import json
import re
from typing import List

# Matches integers and decimals, with optional leading minus and comma thousands
_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

_TABLE_RE = re.compile(r"<table>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_REASON_RE = re.compile(r"</table>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _nums_from_table(table_str: str) -> set:
    """Extract all numeric leaf values from the model's JSON table output."""
    try:
        obj = json.loads(table_str.strip())
    except Exception:
        # Fallback: regex scan on raw string
        return {round(float(m.replace(",", "")), 4)
                for m in _NUM_RE.findall(table_str)}

    nums = set()
    for row in obj.get("rows", []):
        for val in row:
            try:
                nums.add(round(float(str(val).replace(",", "")), 4))
            except (ValueError, TypeError):
                pass
    return nums


def _nums_from_text(text: str) -> set:
    """Extract all numbers appearing in free-form reasoning text."""
    return {round(float(m.replace(",", "")), 4) for m in _NUM_RE.findall(text)}


def compute_consistency_reward(completion: str) -> float:
    """
    Score how consistently the model's reasoning references its own table.

    Args:
        completion: Full model output string.

    Returns:
        Float in [0, 1].
    """
    table_m = _TABLE_RE.search(completion)
    reason_m = _REASON_RE.search(completion)

    if not table_m or not reason_m:
        return 0.0

    table_nums = _nums_from_table(table_m.group(1))
    if not table_nums:
        return 0.0

    reason_nums = _nums_from_text(reason_m.group(1))

    overlap = sum(1 for n in table_nums if n in reason_nums)
    return overlap / len(table_nums)


def consistency_reward_batch(completions: List[str]) -> List[float]:
    """Compute consistency reward for a list of completions."""
    return [compute_consistency_reward(c) for c in completions]
