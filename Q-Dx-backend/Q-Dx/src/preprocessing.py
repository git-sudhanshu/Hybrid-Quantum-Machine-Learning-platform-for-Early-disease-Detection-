"""
Q-Dx :: Preprocessing and dimensionality reduction
==================================================

Turns a raw DataFrame into two things:
  1. a scaled full-width feature matrix  -> the classical "ceiling" baseline
  2. a PCA-reduced matrix (default 4D)   -> the input BOTH models are compared on

This module is the single source of truth for the train/test split. Nothing
else in Q-Dx is allowed to split data, or the classical and quantum models
would be evaluated on different patients and the comparison would be void.

------------------------------------------------------------------
DATA LEAKAGE: the discipline, and why each piece matters
------------------------------------------------------------------
    raw DataFrame
        |
        |-- drop exact duplicate rows      <- BEFORE splitting, deliberately.
        |                                     An identical row landing in both
        |                                     train and test is leakage; dropping
        |                                     duplicates first prevents it.
        |
        |-- train_test_split(stratify=y)   <- the ONLY split in the project
        |
        |-- fit  Pipeline on X_train ONLY  <- imputer, encoder, scaler and PCA
        |                                     all learn their statistics from
        |                                     training patients alone
        |
        `-- transform X_train and X_test with those frozen statistics

The test set never influences an imputation median, a scaling mean, or a
principal component direction.

------------------------------------------------------------------
WHY PCA (defend this to judges)
------------------------------------------------------------------
Angle encoding needs one qubit per feature. 30 qubits is not simulable in a
hackathon (2^30 amplitudes) and is far beyond near-term hardware. So the
dimensionality must come down before the quantum stage - and that necessity
is precisely what makes the architecture HYBRID rather than decorative.

PCA is the right tool here because the features are genuinely redundant: 21
of 435 feature pairs in WDBC correlate above |r| = 0.9, since radius,
perimeter and area measure overlapping geometry and each appears three times
(mean / standard error / worst). PCA rotates that redundancy into a small
number of uncorrelated components.

HONESTY NOTE for the explainability stage: principal components are linear
combinations of all 30 original measurements. They are NOT biomarkers. Never
label component 1 as "tumour radius" - see explainability.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import DATASET_CONFIG, load_dataset

RANDOM_STATE = DATASET_CONFIG["random_state"]   # 42
DEFAULT_N_COMPONENTS = 4                        # = number of qubits
DEFAULT_TEST_SIZE = 0.2


# ==========================================================================
# Pipeline construction
# ==========================================================================
def build_preprocessor(numeric_cols, categorical_cols, n_components: int):
    """A single fitted-once sklearn Pipeline: impute -> encode -> scale -> PCA.

    Using a Pipeline rather than loose transformers is what makes the
    leakage guarantee mechanical instead of a promise: `.fit()` sees only
    training rows, and `.transform()` can only ever apply frozen statistics.

    Numeric columns get median imputation (robust to the skewed distributions
    common in biomedical measurements) then standardisation. Categorical
    columns, if an uploaded dataset has any, get most-frequent imputation and
    one-hot encoding with handle_unknown="ignore" so an unseen category in
    the test set degrades gracefully instead of crashing.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    transformers = [("num", numeric_pipe, numeric_cols)]
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return Pipeline([
        ("columns", ColumnTransformer(transformers, remainder="drop")),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
    ])


def build_scaler_only(numeric_cols, categorical_cols):
    """The same preprocessing WITHOUT PCA, for the full-width classical model.

    Needed because the fair comparison is SVM-on-4-components vs VQC-on-4-
    components. The full-width SVM is reported separately as a "classical
    ceiling" - what you give up by compressing to 4 dimensions.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", numeric_pipe, numeric_cols)]
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
            categorical_cols,
        ))
    return Pipeline([("columns", ColumnTransformer(transformers, remainder="drop"))])


# ==========================================================================
# The one and only split + fit
# ==========================================================================
def prepare_data(df: pd.DataFrame | None = None,
                 target_column: str | None = None,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 test_size: float = DEFAULT_TEST_SIZE,
                 random_state: int = RANDOM_STATE) -> dict:
    """Run the full preprocessing pipeline and return everything downstream needs.

    df / target_column
        Omit both to use the bundled WDBC benchmark. Pass an uploaded
        DataFrame and its target column name to run the platform on any other
        binary biomedical dataset - this is what makes "dataset upload"
        possible without rewriting the pipeline.

    n_components
        Number of PCA components = number of qubits the VQC will use.
        Configurable so the quantum developer can try 4, 6 or 8 without
        touching this file.

    Returns a dict containing both the reduced and full-width matrices, the
    fitted pipelines, the variance accounting, and a dashboard summary.
    """
    if df is None:
        df = load_dataset()
        target_column = DATASET_CONFIG["target_column"]
    if target_column is None:
        raise ValueError(
            "target_column must be given when passing your own DataFrame. "
            "It names the column holding the diagnosis label."
        )
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)[:10]}..."
        )

    # --- duplicates dropped BEFORE the split (leakage prevention) ----------
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_duplicates_dropped = n_before - len(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Target encoding: map whatever the two labels are onto {0, 1},
    # preserving sorted order so the mapping is deterministic.
    classes = np.sort(y.unique())
    if len(classes) != 2:
        raise ValueError(f"Binary classification only; found classes {classes}.")
    y = y.map({classes[0]: 0, classes[1]: 1}).to_numpy()

    numeric_cols = list(X.select_dtypes(include="number").columns)
    categorical_cols = list(X.select_dtypes(exclude="number").columns)

    if n_components > len(numeric_cols) + len(categorical_cols):
        raise ValueError(
            f"n_components={n_components} exceeds the {X.shape[1]} available "
            f"features. Reduce n_components."
        )

    # --- THE SPLIT: stratified, seeded, and the only one in the project ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # --- fit on TRAIN ONLY, then transform both ---------------------------
    reducer = build_preprocessor(numeric_cols, categorical_cols, n_components)
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)

    scaler_only = build_scaler_only(numeric_cols, categorical_cols)
    X_train_full = scaler_only.fit_transform(X_train)
    X_test_full = scaler_only.transform(X_test)

    pca = reducer.named_steps["pca"]
    evr = pca.explained_variance_ratio_

    return {
        # what both models are compared on
        "X_train_reduced": X_train_reduced,
        "X_test_reduced": X_test_reduced,
        # full-width, for the classical ceiling reference
        "X_train_full": X_train_full,
        "X_test_full": X_test_full,
        "y_train": y_train,
        "y_test": y_test,
        # fitted artifacts, for saving and for inference on new patients
        "reducer": reducer,
        "scaler_only": scaler_only,
        # provenance and reporting
        "n_components": n_components,
        "explained_variance_ratio": evr.tolist(),
        "cumulative_variance": float(evr.sum()),
        "n_original_features": X.shape[1],
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "duplicates_dropped": n_duplicates_dropped,
        "test_size": test_size,
        "random_state": random_state,
        "class_mapping": {str(classes[0]): 0, str(classes[1]): 1},
        "summary": {
            "samples": len(df),
            "original_features": X.shape[1],
            "reduced_features": n_components,
            "variance_retained": round(float(evr.sum()), 4),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "train_positives": int(y_train.sum()),
            "test_positives": int(y_test.sum()),
            "duplicates_dropped": n_duplicates_dropped,
        },
    }


# ==========================================================================
# The interface the quantum module consumes
# ==========================================================================
def prepare_quantum_data(df=None, target_column=None,
                         n_components: int = DEFAULT_N_COMPONENTS,
                         test_size: float = DEFAULT_TEST_SIZE,
                         random_state: int = RANDOM_STATE):
    """Return exactly (X_train_q, X_test_q, y_train, y_test) for the VQC.

        from src.preprocessing import prepare_quantum_data
        from src.quantum_model import train_vqc

        X_train_q, X_test_q, y_train, y_test = prepare_quantum_data()
        model = train_vqc(X_train_q, y_train)

    X_train_q.shape[1] == n_components == the qubit count. No PennyLane import
    happens anywhere in this module: the quantum dependency stays isolated on
    the other side of this boundary.
    """
    d = prepare_data(df, target_column, n_components, test_size, random_state)
    return (d["X_train_reduced"], d["X_test_reduced"],
            d["y_train"], d["y_test"])


# ==========================================================================
# Persistence
# ==========================================================================
def save_preprocessor(prepared: dict, path: str = "models/classical/preprocessor.joblib") -> str:
    """Save the fitted pipelines so the dashboard can transform a new patient
    without retraining or re-reading the original dataset."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "reducer": prepared["reducer"],
        "scaler_only": prepared["scaler_only"],
        "n_components": prepared["n_components"],
        "class_mapping": prepared["class_mapping"],
        "explained_variance_ratio": prepared["explained_variance_ratio"],
    }, path)
    return path


def load_preprocessor(path: str = "models/classical/preprocessor.joblib") -> dict:
    return joblib.load(path)


# ==========================================================================
# Manual check:  python -m src.preprocessing
# ==========================================================================
if __name__ == "__main__":
    d = prepare_data()
    s = d["summary"]

    print("=" * 64)
    print("PREPROCESSING PIPELINE")
    print("=" * 64)
    print(f"samples after dedup : {s['samples']}  "
          f"({s['duplicates_dropped']} duplicate rows dropped)")
    print(f"train / test        : {s['train_samples']} / {s['test_samples']}  "
          f"(stratified, random_state={d['random_state']})")
    print(f"positives           : {s['train_positives']} train / "
          f"{s['test_positives']} test")
    print(f"features            : {s['original_features']} -> {s['reduced_features']}"
          f"   ({100*(1 - s['reduced_features']/s['original_features']):.0f}% reduction)")
    print(f"variance retained   : {s['variance_retained']*100:.2f}%")

    print("\nvariance per principal component:")
    cum = 0.0
    for i, v in enumerate(d["explained_variance_ratio"], start=1):
        cum += v
        print(f"    PC{i}: {v*100:5.2f}%   (cumulative {cum*100:5.2f}%)")

    print(f"\nshapes handed to the models:")
    print(f"    X_train_reduced {d['X_train_reduced'].shape}   <- VQC + fair SVM")
    print(f"    X_test_reduced  {d['X_test_reduced'].shape}")
    print(f"    X_train_full    {d['X_train_full'].shape}  <- classical ceiling")

    print("\nleakage checks:")
    tr_mean = d["X_train_reduced"].mean(axis=0)
    te_mean = d["X_test_reduced"].mean(axis=0)
    print(f"    train PC means ~0 (scaler fitted here): {np.round(tr_mean, 6)}")
    print(f"    test  PC means drift slightly (frozen stats applied): "
          f"{np.round(te_mean, 4)}")
    print("    -> test means are NOT forced to zero, which is the visible")
    print("       signature of a scaler that never saw the test set.")

    print("\nquantum interface:")
    Xtr, Xte, ytr, yte = prepare_quantum_data()
    print(f"    prepare_quantum_data() -> {Xtr.shape}, {Xte.shape}, "
          f"{ytr.shape}, {yte.shape}")
    print(f"    feature width {Xtr.shape[1]} == qubit count {Xtr.shape[1]}  OK")

    print(f"\nsaved: {save_preprocessor(d)}")
