"""
Q-Dx :: Dataset loading and inspection
======================================

Owns exactly one thing: getting a clean, well-described DataFrame into memory.
No cleaning, no scaling, no splitting, no PCA -- those live in preprocessing.py.

DATASET: Wisconsin Breast Cancer (Diagnostic), a.k.a. WDBC
    Source   : UCI ML Repository, bundled inside scikit-learn
               (sklearn.datasets.load_breast_cancer) -- no download required.
    Samples  : 569
    Features : 30 numeric (10 cell-nucleus measurements x mean / standard
               error / "worst" value, computed from digitised images of fine
               needle aspirates of breast masses)
    Target   : binary diagnosis
    Missing  : none
    Duplicates: none

WHY THIS DATASET (defend this to judges)
----------------------------------------
1. It ships with scikit-learn, so every teammate loads byte-identical data
   with no download, no account and no file to misplace. In a 36-hour build
   that removes an entire class of failure.
2. 30 features reducing to 4 principal components is an 87% cut -- a genuine
   justification for the hybrid architecture. A dataset with 8 features would
   make "we needed classical dimensionality reduction" a hard sell.
3. 21 feature pairs correlate above |r| = 0.9 (radius, perimeter and area are
   geometrically linked, and each appears three times over). PCA therefore has
   real redundancy to exploit rather than merely discarding signal.
4. No missing values and no duplicates, so cleaning effort is near zero.

KNOWN LIMITATION, stated up front
---------------------------------
WDBC is an easy benchmark: an RBF SVM reaches ROC-AUC ~0.99 even on only 4
principal components. The classical baseline will be strong, so the quantum
model may tie or lose. That is an acceptable and honest outcome -- a credible
baseline you lose to is worth more than a weak one you beat.

!!! THE LABEL TRAP !!!
----------------------
scikit-learn ships this dataset encoded as malignant=0, benign=1. Left alone,
"the positive class" would be BENIGN, and every sensitivity/specificity figure
in the project would silently describe how well the model detects HEALTHY
people. For a disease-detection platform that is backwards and dangerous.

This module therefore RE-ENCODES the target so that:

        1 = malignant = disease present = the positive class
        0 = benign    = no disease

Every metric downstream -- classical and quantum -- depends on this. Do not
change it without changing both models' evaluation code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# --------------------------------------------------------------------------
# Centralised dataset configuration.
# Keep dataset-specific facts HERE so that swapping datasets later means
# editing one block rather than hunting through the codebase.
# --------------------------------------------------------------------------
DATASET_CONFIG = {
    "name": "Wisconsin Breast Cancer (Diagnostic)",
    "short_name": "WDBC",
    "source": "UCI ML Repository via sklearn.datasets.load_breast_cancer",
    "target_column": "diagnosis",
    "positive_label": 1,
    "positive_class_name": "malignant",
    "negative_class_name": "benign",
    "task": "binary classification",
    "random_state": 42,
}

TARGET = DATASET_CONFIG["target_column"]


# ==========================================================================
# LOAD
# ==========================================================================
def load_dataset() -> pd.DataFrame:
    """Return the full dataset as a DataFrame: 30 feature columns + target.

    The target column is named by DATASET_CONFIG["target_column"] and is
    re-encoded so 1 = malignant (disease present). See the LABEL TRAP note
    in this module's docstring -- this flip is deliberate and load-bearing.
    """
    bunch = load_breast_cancer()

    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)

    # bunch.target: 0 = malignant, 1 = benign  ->  flip to 1 = malignant.
    df[TARGET] = 1 - bunch.target

    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Feature columns only -- everything except the target."""
    return [c for c in df.columns if c != TARGET]


def split_X_y(df: pd.DataFrame):
    """Convenience accessor: (X DataFrame, y Series). No splitting, no copying
    of preprocessing logic -- that belongs in preprocessing.py."""
    return df[get_feature_names(df)], df[TARGET]


# ==========================================================================
# INSPECT
# ==========================================================================
def inspect_dataset(df: pd.DataFrame) -> dict:
    """Full diagnostic report on the loaded data. Everything measured, nothing
    asserted -- this is what you show a judge who asks 'did you actually look
    at your data?'"""
    features = get_feature_names(df)
    y = df[TARGET]
    n_pos = int((y == DATASET_CONFIG["positive_label"]).sum())
    n_neg = int(len(y) - n_pos)

    missing_per_col = df.isna().sum()

    # Flag strongly correlated feature pairs -- the justification for PCA.
    corr = df[features].corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    high_corr_pairs = int((np.abs(corr[iu]) > 0.9).sum())

    return {
        "dataset_name": DATASET_CONFIG["name"],
        "source": DATASET_CONFIG["source"],
        "rows": int(len(df)),
        "n_features": len(features),
        "feature_names": features,
        "target_name": TARGET,
        "positive_class": DATASET_CONFIG["positive_class_name"],
        "negative_class": DATASET_CONFIG["negative_class_name"],
        "positive_cases": n_pos,
        "negative_cases": n_neg,
        "positive_rate": round(n_pos / len(df), 4),
        "class_balance": {
            DATASET_CONFIG["positive_class_name"]: n_pos,
            DATASET_CONFIG["negative_class_name"]: n_neg,
        },
        "missing_values_total": int(missing_per_col.sum()),
        "columns_with_missing": missing_per_col[missing_per_col > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": len(df.select_dtypes(include="number").columns),
        "categorical_columns": len(df.select_dtypes(exclude="number").columns),
        "highly_correlated_pairs": high_corr_pairs,
        "total_feature_pairs": len(iu[0]),
    }


def get_dataset_summary(df: pd.DataFrame, train_samples=None, test_samples=None,
                        reduced_features=None) -> dict:
    """Compact summary for the Streamlit dashboard.

    Split sizes and the reduced feature count are passed in rather than
    computed here, because this module deliberately does not know how the data
    will be split or reduced. Call it again after preprocessing to fill them.
    """
    info = inspect_dataset(df)
    return {
        "dataset_name": info["dataset_name"],
        "source": info["source"],
        "samples": info["rows"],
        "features": info["n_features"],
        "positive_cases": info["positive_cases"],
        "negative_cases": info["negative_cases"],
        "positive_class": info["positive_class"],
        "missing_values": info["missing_values_total"],
        "duplicate_rows": info["duplicate_rows"],
        "train_samples": train_samples,
        "test_samples": test_samples,
        "reduced_features": reduced_features,
    }


# ==========================================================================
# Manual check:  python -m src.data_loader
# ==========================================================================
if __name__ == "__main__":
    df = load_dataset()
    info = inspect_dataset(df)

    print("=" * 62)
    print(info["dataset_name"])
    print("=" * 62)
    print(f"source            : {info['source']}")
    print(f"rows x features   : {info['rows']} x {info['n_features']}")
    print(f"target column     : {info['target_name']}")
    print(f"positive class    : {info['positive_class']} (encoded as 1)")
    print(f"class balance     : {info['positive_cases']} {info['positive_class']} / "
          f"{info['negative_cases']} {info['negative_class']} "
          f"({info['positive_rate']*100:.1f}% positive)")
    print(f"missing values    : {info['missing_values_total']}")
    print(f"duplicate rows    : {info['duplicate_rows']}")
    print(f"numeric / categorical columns : "
          f"{info['numeric_columns']} / {info['categorical_columns']}")
    print(f"feature pairs |r| > 0.9       : "
          f"{info['highly_correlated_pairs']} of {info['total_feature_pairs']}"
          "   <- why PCA is justified")

    print("\nfirst 8 features  :")
    for f in info["feature_names"][:8]:
        print(f"    {f}")
    print(f"    ... and {info['n_features'] - 8} more")

    print("\nsanity check on the label flip:")
    print(f"    mean of target = {df[TARGET].mean():.4f} "
          f"(should be ~0.373 -- malignant is the MINORITY class)")
    print("\nhead of the frame:")
    print(df[[*info['feature_names'][:3], TARGET]].head())
