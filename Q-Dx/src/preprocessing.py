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

from .data_loader import DEFAULT_DATASET, get_config, load_dataset

RANDOM_STATE = 42
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
def prepare_data(dataset: str = DEFAULT_DATASET,
                 df: pd.DataFrame | None = None,
                 target_column: str | None = None,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 test_size: float = DEFAULT_TEST_SIZE,
                 random_state: int = RANDOM_STATE,
                 train_cap: int | None = -1) -> dict:
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
    cfg = get_config(dataset)
    if train_cap == -1:                      # -1 means "use the dataset default"
        train_cap = cfg.get("quantum_train_cap")

    if df is None:
        df = load_dataset(dataset)
        target_column = cfg["target_column"]
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

    # --- optional training cap, for quantum tractability -------------------
    # Simulating a circuit costs CPU per sample, so a 55,000-patient VQC is not
    # trainable in a hackathon. When a cap applies we take a stratified
    # subsample and BOTH models train on it, so the comparison stays fair.
    # The uncapped training set is kept as well, for the full-data classical
    # reference. The cap is reported, never hidden.
    X_train_uncapped, y_train_uncapped = X_train, y_train
    capped = False
    if train_cap is not None and len(y_train) > train_cap:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=train_cap, stratify=y_train,
            random_state=random_state)
        capped = True

    # --- fit on TRAIN ONLY, then transform both ---------------------------
    reducer = build_preprocessor(numeric_cols, categorical_cols, n_components)
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)

    scaler_only = build_scaler_only(numeric_cols, categorical_cols)
    X_train_full = scaler_only.fit_transform(X_train)
    X_test_full = scaler_only.transform(X_test)

    # Uncapped view, fitted separately on the full training set so the
    # full-data classical model is not handicapped by the subsample's statistics.
    if capped:
        reducer_unc = build_preprocessor(numeric_cols, categorical_cols, n_components)
        X_train_reduced_all = reducer_unc.fit_transform(X_train_uncapped)
        X_test_reduced_all = reducer_unc.transform(X_test)
    else:
        X_train_reduced_all, X_test_reduced_all = X_train_reduced, X_test_reduced

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
        # uncapped training view -- classical model trained on everything
        "X_train_reduced_all": X_train_reduced_all,
        "X_test_reduced_all": X_test_reduced_all,
        "y_train_all": y_train_uncapped,
        # fitted artifacts, for saving and for inference on new patients
        "reducer": reducer,
        "scaler_only": scaler_only,
        # provenance and reporting
        "dataset": dataset,
        "disease": cfg["disease"],
        "dataset_name": cfg["name"],
        "positive_class": cfg["positive_class_name"],
        "train_capped": capped,
        "train_cap": train_cap,
        "n_train_uncapped": int(len(y_train_uncapped)),
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
            "train_capped": capped,
            "train_samples_uncapped": int(len(y_train_uncapped)),
            "disease": cfg["disease"],
            "dataset_name": cfg["name"],
        },
    }


# ==========================================================================
# The interface the quantum module consumes
# ==========================================================================
def prepare_quantum_data(dataset: str = DEFAULT_DATASET, df=None,
                         target_column=None,
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
    d = prepare_data(dataset, df, target_column, n_components, test_size,
                     random_state)
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
        "dataset": prepared["dataset"],
        "disease": prepared["disease"],
        "explained_variance_ratio": prepared["explained_variance_ratio"],
    }, path)
    return path


def load_preprocessor(path: str = "models/classical/preprocessor.joblib") -> dict:
    return joblib.load(path)


# ==========================================================================
# Manual check:  python -m src.preprocessing
# ==========================================================================
if __name__ == "__main__":
    from .data_loader import DATASETS

    for key in DATASETS:
        try:
            d = prepare_data(key)
        except FileNotFoundError as e:
            print(f"\n!! {key}: {e}\n"); continue
        s_ = d["summary"]
        print("=" * 68)
        print(f"{d['disease'].upper()}  --  {d['dataset_name']}")
        print("=" * 68)
        print(f"  samples   : {s_['samples']:,}  "
              f"({s_['duplicates_dropped']} duplicates dropped)")
        print(f"  split     : {s_['train_samples']:,} train / "
              f"{s_['test_samples']:,} test (stratified, seed={d['random_state']})")
        if d["train_capped"]:
            print(f"  CAPPED    : training subsampled {d['n_train_uncapped']:,} -> "
                  f"{s_['train_samples']:,} for quantum tractability")
            print(f"              both models train on this subsample (fair);")
            print(f"              the full {d['n_train_uncapped']:,} is kept for the")
            print(f"              full-data classical reference.")
        print(f"  reduction : {s_['original_features']} -> {s_['reduced_features']} "
              f"components, {s_['variance_retained']*100:.2f}% variance")
        print(f"  shapes    : train {d['X_train_reduced'].shape}, "
              f"test {d['X_test_reduced'].shape}")
        tr = d["X_train_reduced"].mean(axis=0); te = d["X_test_reduced"].mean(axis=0)
        print(f"  leakage   : train means {tr.round(4)}  "
              f"test means {te.round(4)} (not zeroed)")
        print(f"  saved     : {save_preprocessor(d, f'models/classical/preprocessor_{key}.joblib')}")
        print()
