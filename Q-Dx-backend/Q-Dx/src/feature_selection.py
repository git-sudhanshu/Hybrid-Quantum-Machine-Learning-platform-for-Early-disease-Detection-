"""
Q-Dx :: Feature selection
=========================

Optional filtering step that runs BEFORE PCA.

WHY BOTH SELECTION AND PCA?
---------------------------
They solve different problems and are not redundant:

  Feature SELECTION keeps a subset of the ORIGINAL measurements and discards
  the rest. What survives is still interpretable -- "worst concave points" is
  a thing a pathologist recognises.

  PCA (dimensionality REDUCTION) keeps information from ALL features but
  rewrites it into new axes that are combinations of everything. Nothing is
  discarded, but nothing stays interpretable either.

Running selection first means PCA is computed on informative columns only, so
noisy or constant features cannot contribute variance that PCA then treats as
signal. PCA maximises variance, and variance is not the same thing as
usefulness -- a loud, irrelevant measurement can hijack a component.

DEFAULT IS OFF
--------------
On WDBC all 30 features carry signal, so the default pipeline keeps them all
and the headline results are unaffected. This module exists because an
UPLOADED dataset may well contain junk columns, and because "feature
selection" is a named component of deliverable 1.

LEAKAGE
-------
Selectors are fitted on training data only, exactly like the scaler and PCA.
Selecting features using the full dataset is one of the most common and most
damaging leaks in applied ML: the test set silently votes on which columns
the model is allowed to see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import (SelectKBest, VarianceThreshold,
                                       f_classif, mutual_info_classif)

RANDOM_STATE = 42


# ==========================================================================
# Ranking (diagnostic -- tells you what matters before you decide to cut)
# ==========================================================================
def rank_features(X, y, method: str = "anova", feature_names=None) -> list[dict]:
    """Score every feature against the target, best first.

    method="anova"        ANOVA F-statistic. Fast, but only detects LINEAR
                          separation between the two classes.
    method="mutual_info"  Mutual information. Slower, but catches non-linear
                          and non-monotonic relationships that F-tests miss --
                          worth checking on biomedical data, where thresholds
                          and U-shaped risk curves are common.

    Ranking is diagnostic only; nothing is removed. Run it, look at it, then
    decide.
    """
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X, float)
    y = np.asarray(y).ravel()
    if feature_names is None:
        feature_names = (list(X.columns) if isinstance(X, pd.DataFrame)
                         else [f"f{i}" for i in range(X_arr.shape[1])])

    if method == "anova":
        scores, pvalues = f_classif(X_arr, y)
    elif method == "mutual_info":
        scores = mutual_info_classif(X_arr, y, random_state=RANDOM_STATE)
        pvalues = [None] * len(scores)
    else:
        raise ValueError("method must be 'anova' or 'mutual_info'")

    rows = [{"feature": n, "score": float(s),
             "p_value": (None if p is None else float(p))}
            for n, s, p in zip(feature_names, scores, pvalues)]
    return sorted(rows, key=lambda r: r["score"], reverse=True)


# ==========================================================================
# Selection (transformers, fitted on train only)
# ==========================================================================
def build_variance_filter(threshold: float = 0.0):
    """Drop features with variance at or below `threshold`.

    threshold=0.0 removes only constant columns -- always safe, and it
    prevents a divide-by-zero in downstream scaling. Genuinely useful on
    uploaded data, where an all-zeros or single-value column is common.
    """
    return VarianceThreshold(threshold=threshold)


def build_kbest_selector(k: int, method: str = "anova"):
    """Keep the k highest-scoring features. Returns an unfitted transformer:
    fit it on X_train ONLY, then transform both train and test."""
    score_func = f_classif if method == "anova" else mutual_info_classif
    return SelectKBest(score_func=score_func, k=k)


def select_features(X_train, y_train, X_test, k: int, method: str = "anova",
                    feature_names=None):
    """Fit a k-best selector on training data and apply it to both splits.

    Returns (X_train_sel, X_test_sel, selected_names, selector).
    """
    X_tr = (X_train.to_numpy() if isinstance(X_train, pd.DataFrame)
            else np.asarray(X_train, float))
    X_te = (X_test.to_numpy() if isinstance(X_test, pd.DataFrame)
            else np.asarray(X_test, float))
    if feature_names is None:
        feature_names = (list(X_train.columns) if isinstance(X_train, pd.DataFrame)
                         else [f"f{i}" for i in range(X_tr.shape[1])])

    if k >= X_tr.shape[1]:
        return X_tr, X_te, list(feature_names), None      # nothing to cut

    selector = build_kbest_selector(k, method).fit(X_tr, np.asarray(y_train).ravel())
    mask = selector.get_support()
    kept = [n for n, m in zip(feature_names, mask) if m]
    return selector.transform(X_tr), selector.transform(X_te), kept, selector


# ==========================================================================
# Manual check:  python -m src.feature_selection
# ==========================================================================
if __name__ == "__main__":
    from .data_loader import DATASET_CONFIG, load_dataset, split_X_y

    df = load_dataset()
    X, y = split_X_y(df)

    print("=" * 68)
    print("FEATURE RANKING -- which measurements separate malignant from benign")
    print("=" * 68)

    anova = rank_features(X, y, "anova")
    mi = rank_features(X, y, "mutual_info")

    print(f"\n{'rank':>4}  {'ANOVA F-test':32s}  {'mutual information':32s}")
    for i in range(10):
        print(f"{i+1:>4}  {anova[i]['feature']:24s}{anova[i]['score']:8.1f}  "
              f"{mi[i]['feature']:24s}{mi[i]['score']:8.3f}")

    print(f"\nweakest 5 by ANOVA:")
    for r in anova[-5:]:
        print(f"    {r['feature']:28s} F={r['score']:7.2f}  p={r['p_value']:.4f}")

    n_weak = sum(1 for r in anova if r["p_value"] is not None and r["p_value"] > 0.05)
    print(f"\nfeatures failing significance at p>0.05: {n_weak} of {len(anova)}")
    print("-> on WDBC almost every measurement carries signal, which is why")
    print("   the default pipeline keeps all 30 and lets PCA do the reduction.")

    vt = build_variance_filter(0.0).fit(X)
    print(f"\nconstant (zero-variance) columns: "
          f"{int((~vt.get_support()).sum())}")
