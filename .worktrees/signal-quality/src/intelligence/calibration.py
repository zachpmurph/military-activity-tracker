"""
intelligence/calibration.py
────────────────────────────
Repeatable calibration harness for the intelligence classifier.

Detects and corrects two failure modes:

  1. MISCLASSIFICATION  – margin between winner and runner-up is too small;
                          high confidence assigned to low-data regions.
  2. SCORE COLLAPSE     – many regions produce identical (class, score) outputs,
                          usually because one feature dominates while another
                          feature's weight slot is consistently inactive
                          (e.g. change_score_norm=0 when no activity-change
                          data is available, leaving spike_flag to account
                          for ~38-40% of every ANOMALY score).

Signal-environment gating
─────────────────────────
The system operates in two regimes:

  Signal-rich   spikes / coordination / new-entries / change-data present.
                Score clustering here → weight bias → calibration runs.

  Signal-poor   none of the above.  The classifier correctly assigns similar
                ROUTINE scores to every region.  Score clustering here is the
                EXPECTED output and must NOT trigger weight adjustment.

The gate uses two metrics computed from the four event-detection features:
  signal_sufficiency  – batch-mean signal level [0, 1]
  input_diversity     – population std of per-region signal scores

When both are below their thresholds (is_legitimate_clustering is True) the
loop records the iteration state and exits without modifying any weights.

Operational loop
────────────────
  1. run classify_regions() on a fixed feature list
  2. compute CalibrationReport (7 metrics, including signal environment)
  3. gate: exit if is_legitimate_clustering
  4. detect CalibrationFailure list (5 rules, 3 gated)
  5. propose_adjustments() → bounded weight deltas
  6. apply deltas, go to 1
  7. stop when converged or max_iterations reached

No ML.  No schema changes.  Every adjustment is ±max_weight_delta and the
sum of each profile's weights is kept at exactly 1.0.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import intelligence.classifier as _cls_module
from intelligence.classifier import (
    Classification,
    RegionFeatures,
    RegionIntelligence,
    _score_region,
    classify_regions,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS: Dict[str, float] = {
    # Margin (gap between 1st and 2nd class score)
    "min_margin":              10.0,   # individual region; below → LOW_MARGIN
    "min_mean_margin":         15.0,   # fleet average; below → system issue

    # Score-cluster detection
    # A bucket is (class, round(score / 5) * 5).  If one bucket holds more
    # than this fraction of all regions, clustering is flagged.
    "max_cluster_fraction":    0.20,

    # Within-class score spread.  Std < threshold → scores are collapsing.
    "min_score_std":            3.0,

    # Class distribution
    "max_class_fraction":      0.55,   # no single class > 55 % of outputs

    # Confidence ↔ margin alignment (Pearson r).  Low r means confidence
    # scores are driven by volume/coverage rather than classification evidence.
    "min_conf_margin_corr":    0.60,

    # Feature dominance: average fraction of class score from one feature.
    # If a single feature accounts for this share, it is over-contributing.
    "max_feature_dominance":   0.40,

    # Small-data / high-confidence abuse
    "small_data_cap":           5,     # aircraft_count threshold for "small data"
    "high_conf_cap":           50,     # confidence above which small-data is suspect
    "max_small_data_high_conf": 0.15,  # max fraction of regions in this state

    # Calibration loop
    "max_weight_delta":        0.05,   # max absolute change per feature per iteration

    # Signal-environment gating
    # These two thresholds together distinguish "legitimate clustering caused by
    # uniform low-signal inputs" from "pathological clustering caused by biased
    # weights."  Calibration is suppressed in the former case.
    "min_signal_sufficiency":  0.10,   # batch mean signal [0,1]; below → low-signal env
    "min_input_diversity":     0.08,   # std of per-region signal scores; below → uniform inputs

    # FEATURE_DOMINANCE_IMBALANCE detection
    # A feature is "imbalanced" when it contributes > ratio of the class score
    # across > coverage fraction of regions in that class, in a signal-rich env.
    # Distinct from FEATURE_COLLAPSE_ANOMALY: that rule targets inactive features;
    # this rule targets over-active features.
    "feature_dominance_ratio":    0.60,   # per-region contribution fraction above which one feature dominates
    "feature_dominance_coverage": 0.60,   # min fraction of class regions that must be dominated

    # FEATURE_COLLAPSE_ANOMALY detection
    # A feature is considered "collapsed" when it has non-trivial weight in the
    # ANOMALY profile but near-zero activation across all ANOMALY-classified regions.
    # Root cause: the feature is always zero/near-zero so another feature (spike_flag)
    # dominates.  Adjusting that feature's weight is the correct fix.
    "feature_collapse_min_weight":  0.10,   # minimum profile weight to be considered non-trivial
    "feature_collapse_max_std":     0.05,   # activation std below this → collapsed
    "feature_collapse_max_mean":    0.50,   # activation mean below this → near-zero activation
}


# ─────────────────────────────────────────────────────────────────────────────
# 1b.  SIGNAL SUFFICIENCY
#
# A "low-signal environment" is one where the four event-detection features are
# all near zero for essentially every region in the batch.  In that state the
# classifier correctly assigns similar ROUTINE scores to every region — score
# clustering is the EXPECTED output, not a calibration failure.
#
# Distinguishing the two clustering regimes:
#
#   Legitimate:   uniform inputs → uniform outputs.
#                 signal_sufficiency LOW  AND  input_diversity LOW
#                 → calibration must be SUPPRESSED (weights are correct)
#
#   Pathological: diverse inputs → identical outputs.
#                 signal_sufficiency OK   OR  input_diversity HIGH
#                 → calibration should run (weights are biased)
#
# The four chosen features are the only ones that can fire from real events
# (spike_flag and coordination_flag are boolean; new_entry_ratio and
# change_score_norm are continuous).  Persistent/structural features such as
# persistence_score, military_ratio, or flow_balance can be non-zero even when
# nothing is happening, so they are intentionally excluded.
# ─────────────────────────────────────────────────────────────────────────────

def _per_region_signal(r: "RegionIntelligence", norm_denom: float) -> float:
    """
    Single-region signal score in [0, 1].

    Weights
    ───────
    spike_flag         0.35  strongest binary event marker
    coordination_flag  0.25  binary; requires simultaneous multi-aircraft activity
    new_entry_ratio    0.25  continuous; fraction of aircraft unseen in baseline
    change_score_norm  0.15  continuous; normalised activity-change magnitude

    Weights sum to 1.0; a region with all four signals fully active scores 1.0.
    """
    f = r.features
    return (
        0.35 * float(f.spike_flag)
        + 0.25 * float(f.coordination_flag)
        + 0.25 * f.new_entry_ratio
        + 0.15 * min(f.change_score / norm_denom, 1.0)
    )


def compute_signal_sufficiency(results: List["RegionIntelligence"]) -> float:
    """
    Mean signal level across all classified regions in the batch.

    Returns 0.0 for an empty batch; 1.0 if every region has every signal at max.

    A value below THRESHOLDS["min_signal_sufficiency"] (default 0.10) means the
    environment is signal-poor: no spikes, no coordinated activity, no new entries,
    no activity-change data.  Score clustering in this state is EXPECTED and must
    not trigger weight adjustment.
    """
    if not results:
        return 0.0
    norm_denom = max(
        max((r.features.change_score for r in results), default=0.0),
        1.0,
    )
    return sum(_per_region_signal(r, norm_denom) for r in results) / len(results)


def compute_input_diversity(results: List["RegionIntelligence"]) -> float:
    """
    Population standard deviation of per-region signal scores.

    Interpretation
    ──────────────
    LOW  (< THRESHOLDS["min_input_diversity"], default 0.08)
         All regions carry similar signal levels.  Clustering in the output is
         LEGITIMATE: uniform inputs → uniform outputs by design.

    HIGH (>= threshold)
         Inputs are heterogeneous but outputs converge anyway.
         This is PATHOLOGICAL clustering and warrants calibration.
    """
    if len(results) < 2:
        return 0.0
    norm_denom = max(
        max((r.features.change_score for r in results), default=0.0),
        1.0,
    )
    scores = [_per_region_signal(r, norm_denom) for r in results]
    mean_s = sum(scores) / len(scores)
    return math.sqrt(sum((s - mean_s) ** 2 for s in scores) / len(scores))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationReport:
    """Six metrics computed from a list of RegionIntelligence outputs."""

    n_regions: int

    # M1 – Margin distribution
    mean_margin:          float   # mean score gap between winner and runner-up
    low_margin_fraction:  float   # fraction with margin < THRESHOLDS["min_margin"]

    # M2 – Score clustering
    max_cluster_fraction: float   # largest (class, score-bucket) group / n_regions
    score_std_by_class:   Dict[str, float]  # within-class score std per class

    # M3 – Class distribution
    class_fractions:      Dict[str, float]  # fraction of all regions per class
    dominant_class:       str
    dominant_class_fraction: float

    # M4 – Confidence ↔ margin alignment
    conf_margin_corr:     float   # Pearson r(confidence, margin)

    # M5 – Small-data / high-confidence
    small_data_high_conf_fraction: float

    # M6 – Per-class feature dominance
    # feature_dominance[class_name][feature_name] = mean fraction of class score
    feature_dominance:    Dict[str, Dict[str, float]]

    # M7 – Signal environment (used for gating; not a calibration error on its own)
    signal_sufficiency:   float   # mean signal level [0, 1] across the batch
    input_diversity:      float   # population std of per-region signal scores

    # Derived: list of metric names that are outside their threshold
    violations: List[str] = field(default_factory=list)

    # Debug: batch-aggregated mean per-class feature contribution breakdown.
    # {class_name: {feature_name: mean_normalized_contribution}}
    # Populated by evaluate(); None when constructed directly (e.g. in tests).
    feature_contributions_summary: Optional[Dict] = None

    def __post_init__(self) -> None:
        self.violations = _compute_violations(self)

    # ── environment properties ────────────────────────────────────────────────

    @property
    def is_low_signal(self) -> bool:
        """
        True when the batch mean signal is below the sufficiency threshold.

        In this state the classifier correctly produces similar ROUTINE scores
        for most regions — that is expected behaviour, not a weight error.
        """
        return self.signal_sufficiency < THRESHOLDS["min_signal_sufficiency"]

    @property
    def is_legitimate_clustering(self) -> bool:
        """
        True when score clustering is caused by uniformly low-signal inputs
        rather than by biased weights.

        Both conditions must hold simultaneously:
          1. Low signal sufficiency  — the environment carries no events.
          2. Low input diversity     — all regions look alike to the classifier.

        When True the calibration loop skips weight adjustment because there is
        nothing to correct: the weights are producing accurate outputs.
        """
        return (
            self.is_low_signal
            and self.input_diversity < THRESHOLDS["min_input_diversity"]
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def is_converged(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        env = "LOW-SIGNAL" if self.is_low_signal else "signal-ok"
        lines = [
            f"n={self.n_regions}  mean_margin={self.mean_margin:.1f}  "
            f"low_margin={self.low_margin_fraction:.0%}  "
            f"max_cluster={self.max_cluster_fraction:.0%}  "
            f"dominant={self.dominant_class}({self.dominant_class_fraction:.0%})  "
            f"corr={self.conf_margin_corr:.2f}  "
            f"signal={self.signal_sufficiency:.3f}[{env}]  "
            f"diversity={self.input_diversity:.3f}",
        ]
        if self.violations:
            lines.append("VIOLATIONS: " + ", ".join(self.violations))
        else:
            lines.append("CONVERGED")
        return "\n".join(lines)


def _compute_violations(r: CalibrationReport) -> List[str]:
    v: List[str] = []
    t = THRESHOLDS

    # ── Always-active checks ──────────────────────────────────────────────────
    if r.mean_margin < t["min_mean_margin"]:
        v.append("mean_margin")

    for cls, std in r.score_std_by_class.items():
        if std < t["min_score_std"] and std >= 0:
            v.append(f"score_std[{cls}]")

    if r.conf_margin_corr < t["min_conf_margin_corr"]:
        v.append("conf_margin_corr")

    if r.small_data_high_conf_fraction > t["max_small_data_high_conf"]:
        v.append("small_data_high_conf")

    for cls, dom in r.feature_dominance.items():
        if dom:
            top_feat = max(dom, key=dom.__getitem__)
            if dom[top_feat] > t["max_feature_dominance"]:
                v.append(f"feature_dominance[{cls}:{top_feat}]")

    # FEATURE_DOMINANCE_IMBALANCE — gated: a single feature carrying >60% of
    # class score is only concerning in signal-rich environments where the inputs
    # are diverse.  In low-signal batches every score is dominated by structural
    # features (e.g. persistence_score) by design.
    if not r.is_low_signal:
        for cls, dom in r.feature_dominance.items():
            if dom:
                top_feat = max(dom, key=dom.__getitem__)
                top_frac = dom[top_feat]
                if top_frac > t["feature_dominance_ratio"]:
                    v.append(f"feature_dominance:{cls}:{top_feat}:{top_frac:.2f}")

    # ── Signal-gated checks ───────────────────────────────────────────────────
    # These three violations are suppressed in low-signal environments because
    # the observed patterns (tight margins, ROUTINE dominance, score clustering)
    # are the CORRECT classifier outputs when all event features are near zero.
    # Firing them would falsely indicate weight miscalibration and cause the
    # loop to chase noise.

    # SCORE_CLUSTER — only pathological when inputs are diverse.
    # Uniform low-signal inputs → identical ROUTINE scores → legitimate, skip.
    if r.max_cluster_fraction > t["max_cluster_fraction"] and not r.is_legitimate_clustering:
        v.append("score_cluster")

    # CLASS_DOMINANCE — ROUTINE dominance in a low-signal environment is correct.
    # Non-ROUTINE dominance (e.g. ANOMALY taking 80% of outputs) is always flagged.
    if r.dominant_class_fraction > t["max_class_fraction"]:
        if r.dominant_class != "ROUTINE" or not r.is_low_signal:
            v.append("class_skew")

    # LOW_MARGIN — thin margins are expected when all features are near zero;
    # every profile produces a similar score so the classifier cannot spread them.
    if r.low_margin_fraction > 0.10 and not r.is_low_signal:
        v.append("low_margin_fraction")

    return v


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient; returns 0.0 for degenerate inputs."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / denom if denom > 1e-9 else 0.0


def _feature_vector_map(f: RegionFeatures, context: Dict[str, float]) -> Dict[str, float]:
    """
    Reconstruct the same feature-name → value mapping used by _score_region.
    Kept in sync with classifier._score_region manually; must be updated if
    new features are added.
    """
    max_inflow = max(context.get("max_inflow", 1), 1)
    max_change = max(context.get("max_change", 40.0), 1.0)
    max_flow   = 10

    return {
        "military_ratio":      f.military_ratio,
        "staging_flag":        float(f.staging_flag),
        "persistence_score":   f.persistence_score,
        "coordination_flag":   float(f.coordination_flag),
        "projection_flag":     float(f.projection_flag),
        "new_entry_ratio":     f.new_entry_ratio,
        "spike_flag":          float(f.spike_flag),
        "aircraft_density":    f.aircraft_density,
        "novelty":             f.novelty,
        "pos_flow_balance":    max(f.flow_balance, 0.0),
        "neg_flow_balance":    max(-f.flow_balance, 0.0),
        "inflow_norm":         min(f.inflow_count / max_inflow, 1.0),
        "change_score_norm":   min(f.change_score / max_change, 1.0),
        "any_flow":            min((f.outflow_count + f.inflow_count) / (max_flow * 2), 1.0),
        "low_persistence":     1.0 - f.persistence_score,
        "unmilitary":          1.0 - f.military_ratio,
        "no_spike":            float(not f.spike_flag),
        "low_new_entry_ratio": 1.0 - f.new_entry_ratio,
    }


def _compute_feature_dominance(
    results: List[RegionIntelligence],
    profiles: Dict,
    context: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    For each class, compute the average fraction of the class score contributed
    by each feature.  Returns {class_name: {feature_name: mean_fraction}}.
    """
    buckets: Dict[str, List[Dict[str, float]]] = {c.value: [] for c in Classification}

    for r in results:
        cls_name = r.classification.value
        fv = _feature_vector_map(r.features, context)
        weights = profiles.get(r.classification, {})
        score = r.score
        if score <= 0:
            continue
        fractions = {
            feat: (weights[feat] * fv.get(feat, 0.0) * 100) / score
            for feat in weights
        }
        buckets[cls_name].append(fractions)

    dominance: Dict[str, Dict[str, float]] = {}
    for cls_name, recs in buckets.items():
        if not recs:
            dominance[cls_name] = {}
            continue
        n = len(recs)
        avg: Dict[str, float] = {}
        for feat in recs[0]:
            avg[feat] = sum(rec.get(feat, 0.0) for rec in recs) / n
        dominance[cls_name] = avg

    return dominance


def compute_feature_contributions(
    result:   RegionIntelligence,
    profiles: Dict,
    context:  Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-class contribution breakdown for a single region::

        {"ROUTINE": {"persistence_score": 0.62, ...}, "ANOMALY": {...}, ...}

    Contributions are normalized to sum to 1.0 per class (weight × feature_value,
    then divided by class total).  Pass ``context`` for batch-consistent
    normalization; defaults to a single-region context when omitted.
    """
    if context is None:
        context = {
            "max_inflow": max(result.features.inflow_count, 1),
            "max_change": max(result.features.change_score, 40.0),
        }
    fv  = _feature_vector_map(result.features, context)
    out: Dict[str, Dict[str, float]] = {}
    for cls_enum, weights in profiles.items():
        raw   = {feat: weights[feat] * fv.get(feat, 0.0) for feat in weights}
        total = sum(raw.values())
        if total > 0:
            out[cls_enum.value] = {f: round(v / total, 4) for f, v in raw.items()}
        else:
            out[cls_enum.value] = {f: 0.0 for f in raw}
    return out


def evaluate(
    results: List[RegionIntelligence],
    profiles: Optional[Dict] = None,
) -> CalibrationReport:
    """
    Compute all six calibration metrics from a list of RegionIntelligence objects.
    `profiles` defaults to the current module-level _PROFILES.
    """
    if profiles is None:
        profiles = _cls_module._PROFILES

    if not results:
        return CalibrationReport(
            n_regions=0, mean_margin=0.0, low_margin_fraction=0.0,
            max_cluster_fraction=0.0, score_std_by_class={},
            class_fractions={}, dominant_class="", dominant_class_fraction=0.0,
            conf_margin_corr=0.0, small_data_high_conf_fraction=0.0,
            feature_dominance={},
            signal_sufficiency=0.0,
            input_diversity=0.0,
        )

    n = len(results)
    t = THRESHOLDS

    # --- M1: Margin ---
    margins: List[float] = []
    for r in results:
        sorted_scores = sorted(r.all_scores.values(), reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        margins.append(margin)

    mean_margin = sum(margins) / n
    low_margin_fraction = sum(1 for m in margins if m < t["min_margin"]) / n

    # --- M2: Score clustering ---
    # Bucket = (class_value, round(score / 5) * 5)
    bucket_counts: Dict[Tuple, int] = {}
    for r in results:
        bucket = (r.classification.value, round(r.score / 5) * 5)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    max_cluster = max(bucket_counts.values()) if bucket_counts else 0
    max_cluster_fraction = max_cluster / n

    # Within-class score std
    class_scores: Dict[str, List[float]] = {c.value: [] for c in Classification}
    for r in results:
        class_scores[r.classification.value].append(r.score)
    score_std_by_class: Dict[str, float] = {}
    for cls, scores in class_scores.items():
        if len(scores) < 2:
            score_std_by_class[cls] = -1.0   # sentinel: not enough data
            continue
        mean_s = sum(scores) / len(scores)
        score_std_by_class[cls] = math.sqrt(sum((s - mean_s) ** 2 for s in scores) / len(scores))

    # --- M3: Class distribution ---
    class_counts: Dict[str, int] = {c.value: 0 for c in Classification}
    for r in results:
        class_counts[r.classification.value] += 1
    class_fractions = {cls: cnt / n for cls, cnt in class_counts.items()}
    dominant_class = max(class_counts, key=class_counts.__getitem__)
    dominant_class_fraction = class_counts[dominant_class] / n

    # --- M4: Confidence ↔ margin alignment ---
    confidences = [float(r.confidence) for r in results]
    conf_margin_corr = _pearson(confidences, margins)

    # --- M5: Small-data / high-confidence ---
    small_data_high_conf = sum(
        1 for r in results
        if r.features.aircraft_count <= t["small_data_cap"]
        and r.confidence > t["high_conf_cap"]
    )
    small_data_high_conf_fraction = small_data_high_conf / n

    # --- M6: Feature dominance ---
    context = {
        "max_inflow": max((r.features.inflow_count for r in results), default=1),
        "max_change": max((r.features.change_score for r in results), default=40.0),
    }
    feature_dominance = _compute_feature_dominance(results, profiles, context)

    # --- M7: Signal environment ---
    signal_sufficiency = compute_signal_sufficiency(results)
    input_diversity    = compute_input_diversity(results)

    # --- Debug: feature_contributions_summary ---
    # Batch-aggregated mean normalized contribution per class, averaged over
    # results assigned to that class.  Uses the same batch context as M6.
    _contrib_by_cls: Dict[str, List[Dict[str, float]]] = {c.value: [] for c in Classification}
    for r in results:
        per_cls = compute_feature_contributions(r, profiles, context)
        _contrib_by_cls[r.classification.value].append(
            per_cls.get(r.classification.value, {})
        )
    feature_contributions_summary: Dict[str, Dict[str, float]] = {}
    for cls_name, recs in _contrib_by_cls.items():
        if not recs:
            feature_contributions_summary[cls_name] = {}
            continue
        feats = recs[0].keys()
        feature_contributions_summary[cls_name] = {
            feat: round(sum(rec.get(feat, 0.0) for rec in recs) / len(recs), 4)
            for feat in feats
        }

    return CalibrationReport(
        n_regions                     = n,
        mean_margin                   = round(mean_margin, 2),
        low_margin_fraction           = round(low_margin_fraction, 4),
        max_cluster_fraction          = round(max_cluster_fraction, 4),
        score_std_by_class            = {k: round(v, 2) for k, v in score_std_by_class.items()},
        class_fractions               = {k: round(v, 4) for k, v in class_fractions.items()},
        dominant_class                = dominant_class,
        dominant_class_fraction       = round(dominant_class_fraction, 4),
        conf_margin_corr              = round(conf_margin_corr, 4),
        small_data_high_conf_fraction = round(small_data_high_conf_fraction, 4),
        feature_dominance             = feature_dominance,
        signal_sufficiency            = round(signal_sufficiency, 4),
        input_diversity               = round(input_diversity, 4),
        feature_contributions_summary = feature_contributions_summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FAILURE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationFailure:
    region:        RegionIntelligence
    rule:          str    # e.g. "LOW_MARGIN", "SCORE_CLUSTER"
    severity:      float  # 0.0 (mild) – 1.0 (critical)
    detail:        str    # human-readable explanation
    suggested_fix: str    # which profile + feature to adjust

    def __lt__(self, other: "CalibrationFailure") -> bool:
        return self.severity > other.severity   # sort descending by severity


def detect_failures(
    results:  List[RegionIntelligence],
    profiles: Optional[Dict] = None,
) -> List[CalibrationFailure]:
    """
    Apply six rules to flag individual regions for inspection.
    Returns a list sorted by severity (highest first).

    Rules 1 (LOW_MARGIN), 2 (SCORE_CLUSTER), and 4 (CLASS_DOMINANCE for ROUTINE)
    are suppressed in low-signal environments where the observed clustering is a
    correct classifier output, not a weight error.  Rules 3, 5, and 6 are always
    active because data-volume, ANOMALY-overfit, and feature-collapse concerns
    apply regardless of signal level.

    Parameters
    ──────────
    results   Classified regions from the current iteration.
    profiles  Weight profiles used in classification.  Required for
              FEATURE_COLLAPSE_ANOMALY (rule 6); if None that rule is skipped.
    """
    if not results:
        return []

    t      = THRESHOLDS
    n      = len(results)
    failures: List[CalibrationFailure] = []

    # ── Environment assessment (used to gate rules 1, 2, 4) ──────────────────
    _sufficiency   = compute_signal_sufficiency(results)
    _diversity     = compute_input_diversity(results)
    _low_signal    = _sufficiency < t["min_signal_sufficiency"]
    _legit_cluster = _low_signal and _diversity < t["min_input_diversity"]

    # Pre-compute margins per region
    margin_map: Dict[int, float] = {}
    for i, r in enumerate(results):
        sv = sorted(r.all_scores.values(), reverse=True)
        margin_map[i] = sv[0] - (sv[1] if len(sv) > 1 else 0.0)

    # --- Rule 1: LOW_MARGIN ---
    # Suppressed in low-signal environments: when all features are near zero
    # every profile produces similar scores, so thin margins are expected and
    # NOT a sign that the weights need widening.
    if not _low_signal:
        for i, r in enumerate(results):
            mg = margin_map[i]
            if mg < t["min_margin"]:
                sev = 1.0 - mg / t["min_margin"]   # closer to 0 → severity → 1
                failures.append(CalibrationFailure(
                    region        = r,
                    rule          = "LOW_MARGIN",
                    severity      = round(sev, 3),
                    detail        = f"margin={mg:.1f} < threshold={t['min_margin']}; "
                                    f"runner-up={sorted(r.all_scores, key=r.all_scores.__getitem__, reverse=True)[1]}",
                    suggested_fix = f"widen {r.classification.value} profile away from runner-up",
                ))

    # --- Rule 2: SCORE_CLUSTER ---
    # Suppressed when clustering is legitimate (uniform low-signal inputs).
    # Only fire when inputs are diverse but outputs converge — that is weight bias.
    if not _legit_cluster:
        bucket_members: Dict[Tuple, List[int]] = {}
        for i, r in enumerate(results):
            bucket = (r.classification.value, round(r.score / 5) * 5)
            bucket_members.setdefault(bucket, []).append(i)

        cluster_threshold = max(3, int(n * t["max_cluster_fraction"]))
        for (cls, score_bucket), members in bucket_members.items():
            if len(members) >= cluster_threshold:
                sev = min(1.0, len(members) / n / t["max_cluster_fraction"])
                for i in members:
                    r = results[i]
                    failures.append(CalibrationFailure(
                        region        = r,
                        rule          = "SCORE_CLUSTER",
                        severity      = round(sev, 3),
                        detail        = f"{len(members)} regions share {cls}@{score_bucket}±2.5 "
                                        f"({len(members)/n:.0%} of fleet)",
                        suggested_fix = f"reduce dominant feature weight in {cls} profile",
                    ))

    # --- Rule 3: HIGH_CONF_LOW_DATA ---
    # Always active: thin-data / high-confidence is a concern in any environment.
    for r in results:
        if (r.features.aircraft_count <= t["small_data_cap"]
                and r.confidence > t["high_conf_cap"]):
            sev = min(1.0, (r.confidence - t["high_conf_cap"]) / 50.0)
            failures.append(CalibrationFailure(
                region        = r,
                rule          = "HIGH_CONF_LOW_DATA",
                severity      = round(sev, 3),
                detail        = f"conf={r.confidence} but aircraft_count={r.features.aircraft_count}",
                suggested_fix = "increase confidence margin denominator or require data volume",
            ))

    # --- Rule 4: CLASS_DOMINANCE ---
    # ROUTINE dominating in a low-signal environment is expected behaviour: the
    # classifier correctly identifies that everything is routine.  Only flag if
    # a non-ROUTINE class dominates, or if ROUTINE dominates in a signal-rich env.
    class_counts: Dict[str, int] = {}
    for r in results:
        class_counts[r.classification.value] = class_counts.get(r.classification.value, 0) + 1

    dominant_cls_name = max(class_counts, key=class_counts.__getitem__)
    _routine_ok = _low_signal and dominant_cls_name == "ROUTINE"

    if not _routine_ok:
        for cls, cnt in class_counts.items():
            frac = cnt / n
            if frac > t["max_class_fraction"]:
                sev = min(1.0, (frac - t["max_class_fraction"]) / (1.0 - t["max_class_fraction"]))
                for r in results:
                    if r.classification.value == cls:
                        failures.append(CalibrationFailure(
                            region        = r,
                            rule          = "CLASS_DOMINANCE",
                            severity      = round(sev, 3),
                            detail        = f"{cls} assigned to {frac:.0%} of regions",
                            suggested_fix = f"reduce {cls} profile weights or tighten required signals",
                        ))
                break   # flag only the worst offending class

    # --- Rule 5: ANOMALY_OVERFIT ---
    # Specific to the described failure mode: ANOMALY capturing low-volume
    # spike regions that may be noise rather than genuine anomalies.
    small_anomalies = [
        r for r in results
        if r.classification == Classification.ANOMALY
        and r.features.aircraft_count <= t["small_data_cap"]
    ]
    small_total = sum(
        1 for r in results if r.features.aircraft_count <= t["small_data_cap"]
    )
    if small_total > 0 and len(small_anomalies) / small_total > 0.40:
        rate = len(small_anomalies) / small_total
        sev  = min(1.0, rate / 0.60)
        for r in small_anomalies:
            failures.append(CalibrationFailure(
                region        = r,
                rule          = "ANOMALY_OVERFIT",
                severity      = round(sev, 3),
                detail        = f"ANOMALY captures {rate:.0%} of small-volume regions; "
                                f"spike_flag alone may be driving classification",
                suggested_fix = "reduce spike_flag or novelty weight in ANOMALY; "
                                "raise change_score_norm weight to require corroborating data",
            ))

    # --- Rule 6: FEATURE_COLLAPSE_ANOMALY ---
    # Detects ANOMALY-profile features that carry non-trivial weight but show
    # near-zero activation variance across all ANOMALY-classified regions.
    # This indicates the feature is structurally inert (e.g., change_score_norm
    # is always 0.0), forcing other features (spike_flag) to dominate.
    # Self-gating in low-signal batches: no ANOMALY results → rule naturally skips.
    if profiles is not None:
        anomaly_results = [r for r in results if r.classification == Classification.ANOMALY]
        if len(anomaly_results) >= 2:
            # Build context for feature vector reconstruction
            _ctx = {
                "max_inflow": max((r.features.inflow_count for r in anomaly_results), default=1),
                "max_change": max(
                    max((r.features.change_score for r in anomaly_results), default=0.0),
                    40.0,
                ),
            }
            anomaly_profile = profiles.get(Classification.ANOMALY, {})
            # Collect per-region feature vectors for ANOMALY regions
            feature_activations: Dict[str, List[float]] = {}
            for r in anomaly_results:
                fv = _feature_vector_map(r.features, _ctx)
                for fname, fval in fv.items():
                    feature_activations.setdefault(fname, []).append(fval)

            for fname, activations in feature_activations.items():
                weight = anomaly_profile.get(fname, 0.0)
                if weight < t["feature_collapse_min_weight"]:
                    continue
                mean_act = sum(activations) / len(activations)
                std_act  = math.sqrt(
                    sum((a - mean_act) ** 2 for a in activations) / len(activations)
                )
                if std_act < t["feature_collapse_max_std"] and mean_act < t["feature_collapse_max_mean"]:
                    sev = min(1.0, weight / t["max_feature_dominance"])
                    for r in anomaly_results:
                        failures.append(CalibrationFailure(
                            region        = r,
                            rule          = "FEATURE_COLLAPSE_ANOMALY",
                            severity      = round(sev, 3),
                            detail        = (
                                f"feature '{fname}' has weight={weight:.3f} in ANOMALY profile "
                                f"but mean={mean_act:.3f}, std={std_act:.3f} across ANOMALY regions; "
                                f"feature is structurally inactive"
                            ),
                            suggested_fix = (
                                f"reduce '{fname}' weight in ANOMALY profile; "
                                f"redistribute to a corroborating feature with non-zero activation"
                            ),
                        ))

    # --- Rule 7: FEATURE_DOMINANCE_IMBALANCE ---
    # Fires when a single feature contributes > feature_dominance_ratio of the
    # class score for ≥ feature_dominance_coverage of regions in that class, in
    # a signal-rich environment.  Distinct from FEATURE_COLLAPSE_ANOMALY: that
    # rule targets features that are INACTIVE; this rule targets features that
    # are OVER-ACTIVE (e.g. persistence_score swamping ROUTINE classification).
    if not _low_signal and profiles is not None:
        _ctx7 = {
            "max_inflow": max((r.features.inflow_count for r in results), default=1),
            "max_change": max((r.features.change_score for r in results), default=40.0),
        }
        _by_class: Dict[str, List[RegionIntelligence]] = {}
        for r in results:
            _by_class.setdefault(r.classification.value, []).append(r)

        for cls_name, cls_results in _by_class.items():
            if len(cls_results) < 2:
                continue
            try:
                cls_enum = Classification(cls_name)
            except ValueError:
                continue
            cls_profile = profiles.get(cls_enum, {})
            if not cls_profile:
                continue

            # Per-region: which feature dominates and by how much?
            dominated   = 0
            feat_counts: Dict[str, int] = {}
            for r in cls_results:
                fv    = _feature_vector_map(r.features, _ctx7)
                raw   = {feat: cls_profile[feat] * fv.get(feat, 0.0) for feat in cls_profile}
                total = sum(raw.values())
                if total <= 0:
                    continue
                top_f = max(raw, key=raw.__getitem__)
                if raw[top_f] / total > t["feature_dominance_ratio"]:
                    dominated += 1
                    feat_counts[top_f] = feat_counts.get(top_f, 0) + 1

            coverage = dominated / len(cls_results)
            if coverage >= t["feature_dominance_coverage"] and feat_counts:
                dom_feat  = max(feat_counts, key=feat_counts.__getitem__)
                sev       = min(1.0, coverage / t["feature_dominance_coverage"])
                for r in cls_results:
                    failures.append(CalibrationFailure(
                        region        = r,
                        rule          = "FEATURE_DOMINANCE_IMBALANCE",
                        severity      = round(sev, 3),
                        detail        = (
                            f"'{dom_feat}' dominates {cls_name} score in {coverage:.0%} "
                            f"of {cls_name} regions "
                            f"(threshold {t['feature_dominance_ratio']:.0%})"
                        ),
                        suggested_fix = (
                            f"reduce '{dom_feat}' weight in {cls_name} profile by 10%; "
                            f"redistribute proportionally to other features"
                        ),
                    ))

    failures.sort()
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WEIGHT ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeightAdjustment:
    classification: str
    feature:        str
    old_weight:     float
    new_weight:     float
    delta:          float
    reason:         str


def propose_adjustments(
    results:  List[RegionIntelligence],
    profiles: Optional[Dict] = None,
    report:   Optional[CalibrationReport] = None,
) -> List[WeightAdjustment]:
    """
    Identify over-contributing features in failing profiles and return bounded
    weight adjustments that keep each profile summing to exactly 1.0.

    Strategy
    ────────
    For each violated class:
      1. Find the feature with the highest average score fraction (the
         over-contributor) among the regions in that class.
      2. Compute the excess above max_feature_dominance threshold.
      3. Reduce that feature by min(excess / 2, max_weight_delta).
      4. Redistribute the removed weight proportionally to the features
         whose current weight is lowest (the under-contributors), excluding
         the feature we just cut.
    """
    if profiles is None:
        profiles = _cls_module._PROFILES
    if report is None:
        report = evaluate(results, profiles)

    # Safety: no weight adjustments in low-signal or legitimate-clustering envs.
    # In these states the weights are producing correct outputs; adjusting them
    # would cause the classifier to over-fit to the absence of events.
    if report.is_low_signal or report.is_legitimate_clustering:
        return []

    t = THRESHOLDS
    adjustments: List[WeightAdjustment] = []

    for cls, dom in report.feature_dominance.items():
        if not dom:
            continue
        top_feat = max(dom, key=dom.__getitem__)
        top_frac = dom[top_feat]
        if top_frac <= t["max_feature_dominance"]:
            continue
        # If this feature is already covered by the stronger FEATURE_DOMINANCE_IMBALANCE
        # block below (mean fraction > feature_dominance_ratio), skip here to avoid
        # applying two independent reductions to the same feature.
        if top_frac > t["feature_dominance_ratio"]:
            continue

        # Find Classification enum for this class name
        try:
            cls_enum = Classification(cls)
        except ValueError:
            continue

        profile = dict(profiles.get(cls_enum, {}))
        if top_feat not in profile:
            continue

        # How much to reduce
        excess     = top_frac - t["max_feature_dominance"]
        raw_delta  = min(excess / 2.0, t["max_weight_delta"])
        delta      = round(raw_delta, 4)

        old_w = profile[top_feat]
        new_w = max(0.01, old_w - delta)   # never drop to 0
        actual_delta = old_w - new_w

        # Redistribute the removed weight to the lowest-weight features
        others = {f: w for f, w in profile.items() if f != top_feat}
        if not others:
            continue
        # Sort by current weight ascending (distribute to under-contributors first)
        sorted_others = sorted(others, key=others.__getitem__)
        n_recv = max(1, len(sorted_others) // 2)   # distribute to bottom half
        per_feat = actual_delta / n_recv

        new_profile = dict(profile)
        new_profile[top_feat] = new_w

        adjustments.append(WeightAdjustment(
            classification = cls,
            feature        = top_feat,
            old_weight     = round(old_w, 4),
            new_weight     = round(new_w, 4),
            delta          = round(-actual_delta, 4),
            reason         = (
                f"{top_feat} accounts for {top_frac:.1%} of {cls} scores "
                f"(threshold {t['max_feature_dominance']:.0%}); "
                f"reducing by {actual_delta:.4f}"
            ),
        ))

        for feat in sorted_others[:n_recv]:
            adj_new = new_profile[feat] + per_feat
            adjustments.append(WeightAdjustment(
                classification = cls,
                feature        = feat,
                old_weight     = round(others[feat], 4),
                new_weight     = round(adj_new, 4),
                delta          = round(per_feat, 4),
                reason         = f"redistribution from {top_feat} reduction",
            ))

    # ── FEATURE_DOMINANCE_IMBALANCE adjustments ───────────────────────────────
    # For violations with mean contribution > feature_dominance_ratio (0.60),
    # apply a proportional 10% reduction (×0.9) with weights clamped to [0.05, 0.80].
    # Removed weight is redistributed proportionally across all other features.
    _handled: set = set()   # (cls_name, feature) pairs already processed
    for viol in report.violations:
        if not viol.startswith("feature_dominance:"):
            continue
        parts = viol.split(":")
        if len(parts) != 4:
            continue
        _, cls_name, top_feat, _ = parts
        key = (cls_name, top_feat)
        if key in _handled:
            continue
        _handled.add(key)

        try:
            cls_enum = Classification(cls_name)
        except ValueError:
            continue

        profile = dict(profiles.get(cls_enum, {}))
        if top_feat not in profile:
            continue

        old_w   = profile[top_feat]
        new_w   = max(0.05, min(0.80, old_w * 0.9))
        removed = old_w - new_w
        if removed <= 0:
            continue

        others       = {f: w for f, w in profile.items() if f != top_feat and w > 0}
        total_others = sum(others.values())
        if not others or total_others <= 0:
            continue

        adjustments.append(WeightAdjustment(
            classification = cls_name,
            feature        = top_feat,
            old_weight     = round(old_w, 4),
            new_weight     = round(new_w, 4),
            delta          = round(new_w - old_w, 4),
            reason         = (
                f"FEATURE_DOMINANCE_IMBALANCE: '{top_feat}' dominates {cls_name} "
                f"(mean fraction > {t['feature_dominance_ratio']:.0%}); "
                f"reducing weight ×0.9"
            ),
        ))

        for feat, w in others.items():
            share   = w / total_others
            adj_new = max(0.05, min(0.80, w + removed * share))
            adjustments.append(WeightAdjustment(
                classification = cls_name,
                feature        = feat,
                old_weight     = round(w, 4),
                new_weight     = round(adj_new, 4),
                delta          = round(adj_new - w, 4),
                reason         = (
                    f"redistribution from FEATURE_DOMINANCE_IMBALANCE "
                    f"reduction of '{top_feat}' in {cls_name}"
                ),
            ))

    return adjustments


def apply_adjustments(
    adjustments: List[WeightAdjustment],
    profiles:    Optional[Dict] = None,
) -> Dict:
    """
    Apply a list of WeightAdjustments to a profiles dict (deep copy).
    Returns the new profiles dict with weights re-normalised to sum exactly 1.0.
    """
    if profiles is None:
        profiles = _cls_module._PROFILES

    new_profiles = copy.deepcopy(profiles)

    for adj in adjustments:
        try:
            cls_enum = Classification(adj.classification)
        except ValueError:
            continue
        if cls_enum in new_profiles and adj.feature in new_profiles[cls_enum]:
            new_profiles[cls_enum][adj.feature] = adj.new_weight

    # Re-normalise each profile to sum exactly 1.0
    for cls_enum, profile in new_profiles.items():
        total = sum(profile.values())
        if abs(total - 1.0) > 1e-6:
            new_profiles[cls_enum] = {f: round(w / total, 6) for f, w in profile.items()}

    return new_profiles


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONVERGENCE CRITERIA
# ─────────────────────────────────────────────────────────────────────────────

CONVERGENCE_CRITERIA = {
    "mean_margin":               "≥ 15.0  (average classification separation)",
    "conf_margin_corr":          "≥  0.60 (confidence tracks classification certainty)",
    "max_cluster_fraction":      "≤ 20 %  (largest same-class/score-bucket group)",
    "min_score_std[each class]": "≥  3.0  (within-class score spread)",
}

# Violation prefixes that gate loop convergence.  Secondary violations
# (class_skew, feature_dominance, low_margin_fraction, small_data_high_conf)
# are diagnostic but do NOT block the loop from terminating.
_CONVERGENCE_VIOLATION_PREFIXES: frozenset = frozenset({
    "mean_margin",
    "conf_margin_corr",
    "score_cluster",
})
_CONVERGENCE_VIOLATION_PREFIX_STR = "score_std["


def check_convergence(report: CalibrationReport) -> bool:
    """
    Return True when the four operationally critical metrics are all clear.

    Only the following violation types block convergence:
      • mean_margin          — classification margins too thin
      • conf_margin_corr     — confidence not tracking classification certainty
      • score_cluster        — pathological output clustering
      • score_std[<class>]   — within-class score spread collapsed (threshold 3.0)

    Secondary violations (class_skew, feature_dominance, low_margin_fraction,
    small_data_high_conf) are informational and do not prevent termination.
    """
    blocking = [
        v for v in report.violations
        if v in _CONVERGENCE_VIOLATION_PREFIXES
        or v.startswith(_CONVERGENCE_VIOLATION_PREFIX_STR)
    ]
    return len(blocking) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ITERATIVE CALIBRATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration:   int
    report:      CalibrationReport
    adjustments: List[WeightAdjustment]
    profiles:    Dict   # profile state AT THE START of this iteration
    converged:   bool


def run_calibration_loop(
    features_list:  List[RegionFeatures],
    profiles:       Optional[Dict] = None,
    max_iterations: int = 10,
    min_aircraft:   int = 2,
    verbose:        bool = True,
) -> Tuple[Dict, List[IterationRecord]]:
    """
    Run the full calibration loop.

    Parameters
    ──────────
    features_list   RegionFeatures to classify each iteration (fixed dataset)
    profiles        Starting weight profiles; defaults to module _PROFILES
    max_iterations  Safety cap; loop exits early if converged
    min_aircraft    Passed through to classify_regions
    verbose         Print iteration summaries to stdout

    Returns
    ───────
    (final_profiles, history)
      final_profiles  Adjusted weight dict; apply to _PROFILES to update
      history         List of IterationRecord, one per iteration
    """
    if profiles is None:
        profiles = copy.deepcopy(_cls_module._PROFILES)
    else:
        profiles = copy.deepcopy(profiles)

    history: List[IterationRecord] = []
    prev_violations: int = 999

    for iteration in range(1, max_iterations + 1):
        # ── Step 1: classify with current profiles ──────────────────────────
        original_profiles = _cls_module._PROFILES
        _cls_module._PROFILES = profiles
        try:
            results = classify_regions(features_list, min_aircraft=min_aircraft)
        finally:
            _cls_module._PROFILES = original_profiles

        # ── Step 2: compute metrics ──────────────────────────────────────────
        report = evaluate(results, profiles)

        # ── Step 3: detect failures (for record-keeping) ─────────────────────
        failures = detect_failures(results, profiles)

        # ── Step 4: check convergence ────────────────────────────────────────
        converged = check_convergence(report)

        if verbose:
            print(f"\n[CALIBRATION] Iteration {iteration}/{max_iterations}")
            print(f"  {report.summary()}")
            if failures:
                unique_rules = {f.rule for f in failures}
                print(f"  Failures: {len(failures)} regions flagged {sorted(unique_rules)}")

        record = IterationRecord(
            iteration   = iteration,
            report      = report,
            adjustments = [],
            profiles    = copy.deepcopy(profiles),
            converged   = converged,
        )

        # ── Gate: skip calibration when clustering is legitimate ─────────────
        # Legitimate clustering = low-signal environment + low input diversity.
        # In this state the weights are CORRECT; adjusting them would cause the
        # classifier to over-fit to the absence of events and misclassify regions
        # once signal returns.
        if report.is_legitimate_clustering:
            if verbose:
                print(
                    f"  [GATE] Low-signal environment detected "
                    f"(sufficiency={report.signal_sufficiency:.3f}, "
                    f"diversity={report.input_diversity:.3f}).  "
                    f"Score clustering is legitimate — skipping weight adjustment."
                )
            history.append(record)
            break

        if converged:
            if verbose:
                print(f"  Converged after {iteration} iteration(s).")
            history.append(record)
            break

        # ── Step 5: propose and apply weight adjustments ─────────────────────
        adjustments = propose_adjustments(results, profiles, report)

        if not adjustments:
            if verbose:
                print("  No adjustments proposed; stopping early (stuck).")
            history.append(record)
            break

        # Stagnation guard: if violations didn't improve, stop
        n_violations = len(report.violations)
        if n_violations >= prev_violations and iteration > 2:
            if verbose:
                print(f"  Violations not improving ({n_violations} >= {prev_violations}); stopping.")
            history.append(record)
            break
        prev_violations = n_violations

        if verbose:
            for adj in adjustments:
                sign = "+" if adj.delta > 0 else ""
                print(f"  ADJ {adj.classification:22} {adj.feature:<25} "
                      f"{adj.old_weight:.4f} -> {adj.new_weight:.4f}  ({sign}{adj.delta:.4f})")

        profiles = apply_adjustments(adjustments, profiles)
        record.adjustments = adjustments
        history.append(record)

    return profiles, history


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DIFF HELPER
# ─────────────────────────────────────────────────────────────────────────────

def diff_profiles(before: Dict, after: Dict) -> str:
    """Return a human-readable summary of weight changes between two profiles."""
    lines: List[str] = []
    for cls_enum in Classification:
        b = before.get(cls_enum, {})
        a = after.get(cls_enum, {})
        cls_lines: List[str] = []
        for feat in sorted(set(b) | set(a)):
            bw = b.get(feat, 0.0)
            aw = a.get(feat, 0.0)
            if abs(aw - bw) > 1e-5:
                sign = "+" if aw > bw else ""
                cls_lines.append(f"    {feat:<25} {bw:.4f} -> {aw:.4f}  ({sign}{aw - bw:.4f})")
        if cls_lines:
            lines.append(f"  {cls_enum.value}:")
            lines.extend(cls_lines)
    return "\n".join(lines) if lines else "  (no changes)"
