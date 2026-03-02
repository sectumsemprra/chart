"""
HCPC-v2: Hierarchical Correct-Path Consistency (improved).
 
Three fixes over the original HCPC:
 
1. SOFT FILTERING (dead-signal fix)
   Original: hard gate — if num_correct < 2, reward = 0.
   v2: every rollout gets a soft weight:
       w_i = answer_score_i * clamp(table_sim_i / threshold, 0, 1)
   C_type, C_table, D_reason are weighted averages over ALL rollouts,
   so hard examples still receive a training signal.
 
2. STRATEGY-TYPE DIVERSITY (richer diversity signal)
   Original: D_reason = 1 - avg_pairwise_semantic_similarity.
   v2: D_reason = 0.5 * D_semantic + 0.5 * D_strategy
   where D_strategy measures coverage of distinct reasoning strategies
   (direct-read, comparison, arithmetic, estimation, trend).
 
3. DECOUPLED REWARD
   Original: R_HCPC = correct_rate * weighted_sum  (sparse when few correct)
   v2: R_HCPC = mean_i(w_i) * weighted_sum          (dense, always non-zero)
"""
 
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import Counter
 
from utils.parsing import parse_response, normalize_answer, try_parse_numeric
from utils.similarity import compute_pairwise_similarity, compute_table_similarity
 
 
# ---------------------------------------------------------------------------
# Strategy keyword classifier
# ---------------------------------------------------------------------------
 
_STRATEGY_KEYWORDS: Dict[str, List[str]] = {
    "direct_read": [
        "shows", "displays", "indicates", "according to", "from the chart",
        "the value is", "is shown", "directly", "reads",
    ],
    "comparison": [
        "compare", "greater", "less", "more than", "fewer", "highest", "lowest",
        "larger", "smaller", "maximum", "minimum", "most", "least", "between",
        "difference", "versus", "vs",
    ],
    "arithmetic": [
        "sum", "total", "add", "plus", "subtract", "minus", "multiply",
        "divide", "calculate", "compute", "average", "mean", "percent",
        " + ", " - ", " × ", " / ", " = ",
    ],
    "estimation": [
        "approximately", "about", "roughly", "around", "estimate", "nearly",
        "close to", "~",
    ],
    "trend": [
        "increase", "decrease", "grow", "decline", "trend", "over time",
        "rise", "fall", "pattern", "consistent", "steady",
    ],
}
 
 
def _classify_strategies(text: str) -> List[str]:
    """Return all strategy types present in a reasoning text."""
    if not text:
        return ["direct_read"]
    tl = text.lower()
    found = [s for s, kws in _STRATEGY_KEYWORDS.items() if any(k in tl for k in kws)]
    return found if found else ["direct_read"]
 
 
def _strategy_diversity(reasonings: List[str], weights: List[float]) -> float:
    """
    Weighted strategy-type diversity in [0, 1].
 
    High when rollouts use different strategy types.
    """
    if not reasonings:
        return 0.0
    total_w = sum(weights)
    if total_w < 1e-8:
        return 0.0
 
    strategy_w: Dict[str, float] = Counter()
    for reasoning, w in zip(reasonings, weights):
        strategies = _classify_strategies(reasoning)
        share = w / max(len(strategies), 1)
        for s in strategies:
            strategy_w[s] += share
 
    total = sum(strategy_w.values())
    if total < 1e-8:
        return 0.0
 
    n_buckets = len(_STRATEGY_KEYWORDS)
    entropy = -sum((c / total) * math.log(c / total + 1e-12) for c in strategy_w.values())
    max_entropy = math.log(n_buckets + 1e-12)
    return min(entropy / max_entropy, 1.0)
 
 
# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
 
@dataclass
class HCPCv2Result:
    """Result of HCPC-v2 computation."""
    reward: float
    per_rollout_rewards: List[float]
    c_type: float
    c_table: float
    d_reason: float
    d_semantic: float
    d_strategy: float
    soft_weights: List[float]
    details: Dict[str, Any]
 
 
# ---------------------------------------------------------------------------
# Computer
# ---------------------------------------------------------------------------
 
class HCPCv2Computer:
    """
    HCPC-v2: soft-filtered, strategy-aware version of HCPC.
 
    All rollouts contribute via continuous soft weights; no hard binary gate.
    """
 
    def __init__(
        self,
        w_type: float = 1.0,
        w_table: float = 2.0,
        w_reason: float = 1.5,
        table_sim_threshold: float = 0.6,
        answer_tolerance: float = 0.05,
        semantic_weight: float = 0.5,
    ):
        self.w_type = w_type
        self.w_table = w_table
        self.w_reason = w_reason
        self.table_sim_threshold = table_sim_threshold
        self.answer_tolerance = answer_tolerance
        self.semantic_weight = semantic_weight
        self.strategy_weight = 1.0 - semantic_weight
 
    def compute(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> HCPCv2Result:
        if not rollouts:
            return self._zero(0)
 
        n = len(rollouts)
        parsed = [parse_response(r) for r in rollouts]
 
        # Step 1: soft weights
        soft_weights = self._soft_weights(parsed, ground_truth)
        total_w = sum(soft_weights)
 
        if total_w < 1e-6:
            return self._zero(n)
 
        # Step 2–4: metrics
        c_type = self._type_consistency(parsed, soft_weights)
        c_table = self._table_consistency(parsed, soft_weights)
        d_semantic, d_strategy, d_reason = self._diversity(parsed, soft_weights)
 
        quality = (self.w_type * c_type
                   + self.w_table * c_table
                   + self.w_reason * d_reason)
 
        # Per-rollout: each rollout's bonus scales with its own soft weight
        per_rollout = [w * quality for w in soft_weights]
 
        # Group reward: mean soft weight × quality
        reward = (total_w / n) * quality
 
        return HCPCv2Result(
            reward=reward,
            per_rollout_rewards=per_rollout,
            c_type=c_type,
            c_table=c_table,
            d_reason=d_reason,
            d_semantic=d_semantic,
            d_strategy=d_strategy,
            soft_weights=soft_weights,
            details={"total_weight": total_w, "quality": quality, "n": n},
        )
 
    # ------------------------------------------------------------------
    # Soft weights
    # ------------------------------------------------------------------
 
    def _soft_weights(self, parsed: List[Dict], gt: Dict) -> List[float]:
        """w_i = answer_score_i * clamp(table_sim_i / threshold, 0, 1)"""
        gt_table = gt.get("table", {})
        gt_answer = gt.get("label", "")
        weights = []
        for p in parsed:
            ans_score = 1.0 if self._answers_match(p.get("answer", ""), gt_answer) else 0.1
            if gt_table:
                sim = compute_table_similarity(p.get("table", {}), gt_table)
                tbl_factor = min(sim / self.table_sim_threshold, 1.0)
            else:
                tbl_factor = 1.0
            weights.append(ans_score * tbl_factor)
        return weights
 
    def _answers_match(self, pred: str, label: str) -> bool:
        pn = try_parse_numeric(pred)
        ln = try_parse_numeric(label)
        if pn is not None and ln is not None:
            denom = abs(ln) if ln != 0 else 1.0
            return abs(pn - ln) / denom <= self.answer_tolerance
        pn_ = normalize_answer(pred)
        ln_ = normalize_answer(label)
        if not pn_ or not ln_:
            return False
        return pn_ == ln_ or pn_ in ln_ or ln_ in pn_
 
    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
 
    def _type_consistency(self, parsed: List[Dict], weights: List[float]) -> float:
        total_w = sum(weights)
        if total_w < 1e-8:
            return 0.0
        type_w: Dict[str, float] = Counter()
        for p, w in zip(parsed, weights):
            t = p.get("type", "").lower().strip()
            if t:
                type_w[t] += w
        if not type_w:
            return 1.0
        return type_w.most_common(1)[0][1] / total_w
 
    def _table_consistency(self, parsed: List[Dict], weights: List[float]) -> float:
        n = len(parsed)
        if n < 2:
            return 1.0
        tables = [p.get("table", {}) for p in parsed]
        sim_sum = weight_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                wp = weights[i] * weights[j]
                if wp < 1e-10:
                    continue
                sim_sum += wp * compute_table_similarity(tables[i], tables[j])
                weight_sum += wp
        return sim_sum / weight_sum if weight_sum > 1e-8 else 1.0
 
    def _diversity(
        self, parsed: List[Dict], weights: List[float]
    ) -> Tuple[float, float, float]:
        reasonings = [p.get("reasoning", "") for p in parsed]
 
        # Semantic: 1 - avg pairwise similarity (cap at 8 pairs for speed)
        pairs = [(reasonings[i], reasonings[j])
                 for i in range(len(reasonings))
                 for j in range(i + 1, len(reasonings))
                 if weights[i] * weights[j] > 1e-10
                 and reasonings[i] and reasonings[j]]
        if len(pairs) >= 2:
            avg_sim, _ = compute_pairwise_similarity([p[0] for p in pairs[:8]])
            d_semantic = max(0.0, 1.0 - avg_sim)
        else:
            d_semantic = 0.0
 
        d_strategy = _strategy_diversity(reasonings, weights)
        d_combined = self.semantic_weight * d_semantic + self.strategy_weight * d_strategy
        return d_semantic, d_strategy, d_combined
 
    def _zero(self, n: int) -> HCPCv2Result:
        return HCPCv2Result(
            reward=0.0,
            per_rollout_rewards=[0.0] * n,
            c_type=0.0, c_table=0.0,
            d_reason=0.0, d_semantic=0.0, d_strategy=0.0,
            soft_weights=[0.0] * n,
            details={"reason": "empty_or_zero_weight"},
        )
 
 
# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
 
def compute_hcpc_v2_reward(
    rollouts: List[str],
    ground_truth: Dict[str, Any],
    w_type: float = 1.0,
    w_table: float = 2.0,
    w_reason: float = 1.5,
    table_sim_threshold: float = 0.6,
    semantic_weight: float = 0.5,
) -> float:
    computer = HCPCv2Computer(
        w_type=w_type,
        w_table=w_table,
        w_reason=w_reason,
        table_sim_threshold=table_sim_threshold,
        semantic_weight=semantic_weight,
    )
    return computer.compute(rollouts, ground_truth).reward