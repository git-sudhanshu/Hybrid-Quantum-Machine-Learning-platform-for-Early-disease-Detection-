"""
Q-Dx :: Shared evaluation metrics
=================================

ONE metric implementation, used by BOTH the classical and the quantum model.

That is the whole point of this module. If the SVM computed its own accuracy
and the VQC computed its own, a subtle difference -- a different positive
label, a different rounding, a different handling of the zero-division case --
would quietly corrupt the comparison the entire project rests on. Here there
is exactly one confusion matrix function and one specificity formula, so the
two models are measured by identical rulers.

WHY SENSITIVITY AND SPECIFICITY, NOT JUST ACCURACY
--------------------------------------------------
This is a medical screening problem, and the two error types cost wildly
different amounts:

    False negative -> a malignant tumour is called benign. The patient goes
                      home. This is the error that kills someone.
    False positive -> a benign mass is flagged malignant. The patient gets a
                      follow-up biopsy. Distressing and expensive, not fatal.

Accuracy averages these two together as if they were equivalent. They are
not. Sensitivity (how many cancers we catch) and specificity (how many
healthy patients we correctly clear) must be reported separately, always.

    sensitivity = TP / (TP + FN)      "of the sick, how many did we catch?"
    specificity = TN / (TN + FP)      "of the healthy, how many did we clear?"

scikit-learn has no specificity scorer, so it is computed from the confusion
matrix here.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

POSITIVE_LABEL = 1          # 1 = malignant = disease present. See data_loader.


# ==========================================================================
# Core metric block -- both models return exactly this shape
# ==========================================================================
def compute_metrics(y_true, y_pred, y_score=None, *,
                    model_name: str = "model",
                    training_time: float | None = None,
                    inference_time: float | None = None,
                    n_train_samples: int | None = None,
                    threshold: float | None = None) -> dict:
    """Every metric the project reports, from one place.

    y_score is the continuous decision score (probability for the SVM, the
    rescaled expectation value for the VQC). ROC-AUC is computed from it
    rather than from hard labels, because AUC measures ranking quality across
    ALL thresholds -- computing it from 0/1 predictions throws that away and
    silently reports a much worse number.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    roc_auc, fpr, tpr = None, None, None
    if y_score is not None:
        y_score = np.asarray(y_score).ravel()
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
            f, t, _ = roc_curve(y_true, y_score)
            fpr, tpr = f.tolist(), t.tolist()
        except ValueError:
            pass        # only one class present in y_true

    n_test = len(y_true)
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred,
                                           pos_label=POSITIVE_LABEL,
                                           zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred,
                                          pos_label=POSITIVE_LABEL,
                                          zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL,
                             zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "training_time": training_time,
        "inference_time": inference_time,
        "inference_time_per_sample": (inference_time / n_test
                                      if inference_time and n_test else None),
        "n_test_samples": n_test,
        "n_train_samples": n_train_samples,
        "threshold": threshold,
        "missed_cancers": int(fn),        # plain-language headline numbers
        "false_alarms": int(fp),
    }


# ==========================================================================
# Bootstrap confidence intervals
# ==========================================================================
def bootstrap_ci(y_true, y_pred, y_score=None, metric: str = "accuracy",
                 n_boot: int = 2000, alpha: float = 0.05,
                 random_state: int = 42) -> dict:
    """Percentile bootstrap confidence interval for any single metric.

    WHY THIS MATTERS FOR THIS PROJECT
    ---------------------------------
    Our test set is ~114 patients. A difference of two or three patients moves
    accuracy by nearly 2 percentage points. Reporting "97.2% vs 96.5%" as
    though that were a real difference is the most common mistake in
    hackathon ML, and a sharp judge will ask about it.

    Resampling the test set with replacement 2000 times shows how much the
    metric would wobble on a different sample of patients from the same
    population. If two models' intervals overlap heavily, they are not
    distinguishable and we should say so.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_score = None if y_score is None else np.asarray(y_score).ravel()
    n = len(y_true)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue                      # degenerate resample, skip
        if metric == "roc_auc":
            if y_score is None:
                raise ValueError("roc_auc needs y_score")
            vals.append(roc_auc_score(yt, y_score[idx]))
        else:
            m = compute_metrics(yt, yp)
            vals.append(m[metric])

    vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "point_estimate": float(np.mean(vals)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_width": float(hi - lo),
        "confidence": 1 - alpha,
        "n_bootstrap": len(vals),
    }


# ==========================================================================
# McNemar's test -- are two models actually different?
# ==========================================================================
def mcnemar_test(y_true, pred_a, pred_b,
                 name_a: str = "model A", name_b: str = "model B") -> dict:
    """Test whether two classifiers differ significantly on the SAME test set.

    WHY MCNEMAR RATHER THAN COMPARING ACCURACIES
    --------------------------------------------
    Both models are evaluated on the identical patients, so their errors are
    paired, not independent. A two-sample test would ignore that and be
    wrong. McNemar looks only at the patients where the two models DISAGREE:

            b = A wrong, B right
            c = A right, B wrong

    Under the null hypothesis "the models are equally good", each disagreement
    is a coin flip, so b ~ Binomial(b + c, 0.5). We use the exact binomial
    version rather than the chi-square approximation because our disagreement
    counts are small (single digits), where the approximation is unreliable.

    p > 0.05 means we CANNOT claim one model is better. For this project that
    is a perfectly good result to report -- and reporting it honestly is
    stronger than pretending a three-patient difference is a finding.
    """
    y_true = np.asarray(y_true).ravel()
    a_wrong = np.asarray(pred_a).ravel() != y_true
    b_wrong = np.asarray(pred_b).ravel() != y_true

    both_wrong = int((a_wrong & b_wrong).sum())
    both_right = int((~a_wrong & ~b_wrong).sum())
    only_a_wrong = int((a_wrong & ~b_wrong).sum())
    only_b_wrong = int((~a_wrong & b_wrong).sum())

    n_disagree = only_a_wrong + only_b_wrong
    if n_disagree == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(min(only_a_wrong, only_b_wrong),
                                        n_disagree, 0.5,
                                        alternative="two-sided").pvalue)

    return {
        "test": "McNemar (exact binomial)",
        "model_a": name_a,
        "model_b": name_b,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        f"only_{name_a}_wrong": only_a_wrong,
        f"only_{name_b}_wrong": only_b_wrong,
        "n_disagreements": n_disagree,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
        "interpretation": (
            f"{name_a} and {name_b} differ significantly (p={p_value:.4f})"
            if p_value < 0.05 else
            f"No significant difference between {name_a} and {name_b} "
            f"(p={p_value:.4f}); on this test set they are statistically "
            f"indistinguishable."
        ),
    }


# ==========================================================================
# Clinical operating point -- deliverable 4
# ==========================================================================
def operating_points(y_true, y_score, targets=(0.90, 0.95, 0.99)) -> list[dict]:
    """For each target sensitivity, the threshold that achieves it and what
    it costs in false alarms -- expressed in PATIENTS, not percentages.

    This is what turns a classifier into decision support. A clinician does
    not want "the accuracy-optimal threshold"; they want to say "I am not
    willing to miss more than 1 in 20 cancers" and then see the price.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n_pos = int((y_true == POSITIVE_LABEL).sum())
    n_neg = int(len(y_true) - n_pos)

    candidates = np.unique(y_score)
    rows = []
    for target in targets:
        best = None
        for t in candidates:
            pred = (y_score >= t).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            if sens >= target and (best is None or spec > best["specificity"]):
                best = {"target_sensitivity": target, "threshold": float(t),
                        "sensitivity": sens, "specificity": spec,
                        "cancers_caught": tp, "cancers_missed": fn,
                        "false_alarms": fp, "correctly_cleared": tn,
                        "achievable": True}
        if best is None:
            best = {"target_sensitivity": target, "threshold": None,
                    "sensitivity": None, "specificity": None,
                    "cancers_caught": None, "cancers_missed": None,
                    "false_alarms": None, "correctly_cleared": None,
                    "achievable": False}
        best["total_positives"] = n_pos
        best["total_negatives"] = n_neg
        rows.append(best)
    return rows


# ==========================================================================
# Comparison table
# ==========================================================================
COMPARE_FIELDS = ["accuracy", "sensitivity", "specificity", "precision",
                  "f1", "roc_auc", "training_time", "inference_time"]


def compare_models(*results: dict) -> dict:
    """Build the classical-vs-quantum table the dashboard renders.

    Takes any number of metric dicts from compute_metrics and pivots them.
    No interpretation, no ranking, no highlighting a winner -- whichever
    model is better on a given row is whatever the numbers say.
    """
    names = [r.get("model", f"model_{i}") for i, r in enumerate(results)]
    table = {}
    for field in COMPARE_FIELDS:
        table[field] = {n: r.get(field) for n, r in zip(names, results)}
    return {"models": names, "metrics": table}


def format_comparison(comparison: dict) -> str:
    """Plain-text rendering of the comparison table, for terminal and logs."""
    names = comparison["models"]
    width = max(18, max(len(n) for n in names) + 2)
    lines = ["Metric".ljust(16) + "".join(n.ljust(width) for n in names),
             "-" * (16 + width * len(names))]
    for field, row in comparison["metrics"].items():
        cells = ""
        for n in names:
            v = row[n]
            cells += ("n/a" if v is None else
                      (f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}")).ljust(width)
        lines.append(field.ljust(16) + cells)
    return "\n".join(lines)
