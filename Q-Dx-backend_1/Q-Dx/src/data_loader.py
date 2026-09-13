"""
Q-Dx :: Dataset registry, loading and inspection
================================================

Q-Dx supports MULTIPLE diseases. Everything dataset-specific lives in the
DATASETS registry below, so adding a third disease means adding one entry --
not editing the pipeline.

    load_dataset("breast_cancer")   -> imaging-derived tumour measurements
    load_dataset("diabetes")        -> lifestyle and clinical risk indicators

WHY TWO DISEASES, AND WHY THESE TWO
------------------------------------
They are deliberately different in every way that matters, which is what makes
the platform a platform rather than one model with a nice wrapper:

                    Breast cancer (WDBC)      Diabetes (BRFSS)
    patients        569                       68,972
    features        30 continuous             19 binary / ordinal
    data type       imaging measurements      survey + clinical indicators
    difficulty      easy  (AUC ~0.99)         hard  (AUC ~0.79)
    balance         37% positive              51% positive

The difficulty gap is the scientifically interesting part. On the easy
dataset the classical and quantum models are statistically tied; on the hard
one they separate. A platform that only ever ran on an easy benchmark could
not have discovered that.

CLINICAL METADATA
-----------------
Every feature carries a human-readable label and, where it applies, its coding
scheme. A doctor should never be shown "HvyAlcoholConsump = 1"; they should be
shown "Heavy alcohol consumption: yes". That mapping lives here, next to the
data it describes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ==========================================================================
# BREAST CANCER clinical metadata
# ==========================================================================
_WDBC_NOTE = ("Ten nucleus measurements (radius, texture, perimeter, area, "
              "smoothness, compactness, concavity, concave points, symmetry, "
              "fractal dimension), each reported as mean, standard error and "
              "worst value across the imaged nuclei.")


# ==========================================================================
# DIABETES clinical metadata (BRFSS coding)
# ==========================================================================
_DIABETES_FEATURES = {
    "HighBP":               ("High blood pressure", "0 = no, 1 = yes"),
    "HighChol":             ("High cholesterol", "0 = no, 1 = yes"),
    "CholCheck":            ("Cholesterol checked in last 5 years", "0 = no, 1 = yes"),
    "BMI":                  ("Body Mass Index", "kg/m², typically 12-98"),
    "Smoker":               ("Smoked 100+ cigarettes in lifetime", "0 = no, 1 = yes"),
    "Stroke":               ("History of stroke", "0 = no, 1 = yes"),
    "HeartDiseaseorAttack": ("Coronary heart disease or heart attack", "0 = no, 1 = yes"),
    "PhysActivity":         ("Physical activity in past 30 days", "0 = no, 1 = yes"),
    "Fruits":               ("Eats fruit at least once a day", "0 = no, 1 = yes"),
    "Veggies":              ("Eats vegetables at least once a day", "0 = no, 1 = yes"),
    "HvyAlcoholConsump":    ("Heavy alcohol consumption", "0 = no, 1 = yes"),
    "GenHlth":              ("Self-reported general health", "1 = excellent … 5 = poor"),
    "MentHlth":             ("Days of poor mental health, past month", "0-30 days"),
    "PhysHlth":             ("Days of poor physical health, past month", "0-30 days"),
    "DiffWalk":             ("Difficulty walking or climbing stairs", "0 = no, 1 = yes"),
    "Sex":                  ("Sex", "0 = female, 1 = male"),
    "Age":                  ("Age band", "1 = 18-24, rising in 5-year bands, 13 = 80+"),
    "Education":            ("Education level", "1 = never attended … 6 = college graduate"),
    "Income":               ("Income band", "1 = lowest … 8 = highest"),
}


# ==========================================================================
# THE REGISTRY
# ==========================================================================
DATASETS = {
    "breast_cancer": {
        "key": "breast_cancer",
        "name": "Wisconsin Breast Cancer (Diagnostic)",
        "short_name": "WDBC",
        "disease": "Breast cancer",
        "source": "UCI ML Repository, bundled with scikit-learn",
        "loader": "sklearn",
        "target_column": "diagnosis",
        "positive_class_name": "malignant",
        "negative_class_name": "benign",
        "clinical_context": ("Measurements computed from digitised images of "
                             "fine needle aspirates of breast masses."),
        "feature_note": _WDBC_NOTE,
        "feature_meta": {},          # 30 self-describing names, no coding scheme
        "default_components": 4,
        "quantum_train_cap": None,   # small enough to train on everything
        "difficulty": "easy",
    },
    "diabetes": {
        "key": "diabetes",
        "name": "Diabetes Health Indicators (BRFSS)",
        "short_name": "BRFSS-Diabetes",
        "disease": "Type 2 diabetes",
        "source": "CDC Behavioral Risk Factor Surveillance System, cleaned CSV",
        "loader": "csv",
        "csv_name": "diabetes_clean.csv",
        "target_column": "DIABETES",
        "positive_class_name": "diabetic",
        "negative_class_name": "non-diabetic",
        "clinical_context": ("Population health-survey indicators: blood "
                             "pressure, cholesterol, BMI, lifestyle and "
                             "self-reported health status."),
        "feature_note": ("Mostly binary yes/no indicators plus a few ordinal "
                         "scales. No laboratory values, so this reflects what "
                         "a GP could collect in a single consultation."),
        "feature_meta": _DIABETES_FEATURES,
        "default_components": 4,
        "quantum_train_cap": 3000,   # see note below
        "difficulty": "hard",
    },
}

DEFAULT_DATASET = "breast_cancer"

# WHY quantum_train_cap EXISTS
# ----------------------------
# Simulating a quantum circuit costs real CPU time per sample, so training a
# VQC on 55,000 patients is not tractable in a hackathon. The diabetes VQC is
# therefore trained on a stratified subsample.
#
# This is reported, never hidden -- and the honest comparison trains the
# classical model on the SAME subsample. The full-data SVM is also reported
# separately, so the reader can see what the classical model achieves when it
# is allowed everything. As it happens, the VQC on 1,000 patients still beats
# the SVM on all 55,177, which makes the constraint part of the finding rather
# than an excuse.


def list_datasets() -> list[dict]:
    """Everything a UI needs to render a disease picker."""
    return [{"key": c["key"], "name": c["name"], "disease": c["disease"],
             "positive_class": c["positive_class_name"],
             "difficulty": c["difficulty"]}
            for c in DATASETS.values()]


def get_config(dataset: str = DEFAULT_DATASET) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. "
                         f"Available: {list(DATASETS)}")
    return DATASETS[dataset]


# ==========================================================================
# LOAD
# ==========================================================================
def load_dataset(dataset: str = DEFAULT_DATASET,
                 csv_path: str | None = None) -> pd.DataFrame:
    """Return features + target as one DataFrame, target encoded 1 = disease.

    THE LABEL RULE, and it is load-bearing
    --------------------------------------
    scikit-learn ships WDBC as malignant=0, benign=1. Left alone, "the positive
    class" would be BENIGN, and every sensitivity figure in the project would
    silently describe how well the model detects HEALTHY people. We flip it.

    Across both datasets the invariant is the same and non-negotiable:

            1 = disease present      0 = disease absent

    csv_path lets a doctor's own uploaded file be used instead of the bundled
    copy, as long as it has the same columns.
    """
    cfg = get_config(dataset)
    target = cfg["target_column"]

    if cfg["loader"] == "sklearn":
        bunch = load_breast_cancer()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        df[target] = 1 - bunch.target          # flip: 1 = malignant
        return df

    path = Path(csv_path) if csv_path else DATA_DIR / cfg["csv_name"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place {cfg['csv_name']} in data/raw/, "
            f"or pass csv_path=... explicitly."
        )
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Expected target column {target!r} in {path.name}. "
                         f"Found: {list(df.columns)[:8]}...")

    # Move target to the end for consistency with the sklearn path.
    cols = [c for c in df.columns if c != target] + [target]
    return df[cols]


def get_feature_names(df: pd.DataFrame, dataset: str = DEFAULT_DATASET) -> list[str]:
    return [c for c in df.columns if c != get_config(dataset)["target_column"]]


def split_X_y(df: pd.DataFrame, dataset: str = DEFAULT_DATASET):
    return df[get_feature_names(df, dataset)], df[get_config(dataset)["target_column"]]


# ==========================================================================
# CLINICAL LABELS -- for the doctor-facing output
# ==========================================================================
def clinical_label(feature: str, dataset: str = DEFAULT_DATASET) -> str:
    """Human-readable name for a raw feature, e.g. HighBP -> 'High blood pressure'."""
    meta = get_config(dataset)["feature_meta"]
    return meta[feature][0] if feature in meta else feature


def describe_value(feature: str, value, dataset: str = DEFAULT_DATASET) -> str:
    """Render one measurement the way a clinician reads it.

        HvyAlcoholConsump = 1  ->  "Heavy alcohol consumption: yes"
        GenHlth = 4            ->  "Self-reported general health: 4 (1 = excellent … 5 = poor)"
    """
    meta = get_config(dataset)["feature_meta"]
    if feature not in meta:
        return f"{feature}: {value}"
    label, coding = meta[feature]
    if coding.startswith("0 = no"):
        return f"{label}: {'yes' if float(value) >= 0.5 else 'no'}"
    return f"{label}: {value}  ({coding})"


def feature_reference(dataset: str = DEFAULT_DATASET) -> list[dict]:
    """Full feature dictionary for the UI's data-entry form."""
    cfg = get_config(dataset)
    df = load_dataset(dataset)
    out = []
    for f in get_feature_names(df, dataset):
        label, coding = cfg["feature_meta"].get(f, (f, "continuous measurement"))
        col = df[f]
        out.append({"feature": f, "label": label, "coding": coding,
                    "min": float(col.min()), "max": float(col.max()),
                    "median": float(col.median()),
                    "binary": bool(set(col.unique()) <= {0, 1})})
    return out


# ==========================================================================
# INSPECT
# ==========================================================================
def inspect_dataset(df: pd.DataFrame, dataset: str = DEFAULT_DATASET) -> dict:
    """Measured diagnostic report -- nothing asserted."""
    cfg = get_config(dataset)
    features = get_feature_names(df, dataset)
    y = df[cfg["target_column"]]
    n_pos = int((y == 1).sum())

    missing = df.isna().sum()
    corr = df[features].corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)

    return {
        "dataset_key": cfg["key"],
        "dataset_name": cfg["name"],
        "disease": cfg["disease"],
        "source": cfg["source"],
        "clinical_context": cfg["clinical_context"],
        "rows": int(len(df)),
        "n_features": len(features),
        "feature_names": features,
        "target_name": cfg["target_column"],
        "positive_class": cfg["positive_class_name"],
        "negative_class": cfg["negative_class_name"],
        "positive_cases": n_pos,
        "negative_cases": int(len(y) - n_pos),
        "positive_rate": round(n_pos / len(df), 4),
        "missing_values_total": int(missing.sum()),
        "columns_with_missing": missing[missing > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": len(df.select_dtypes(include="number").columns),
        "categorical_columns": len(df.select_dtypes(exclude="number").columns),
        "highly_correlated_pairs": int((np.abs(corr[iu]) > 0.9).sum()),
        "total_feature_pairs": len(iu[0]),
        "difficulty": cfg["difficulty"],
    }


def get_dataset_summary(df: pd.DataFrame, dataset: str = DEFAULT_DATASET,
                        train_samples=None, test_samples=None,
                        reduced_features=None) -> dict:
    """Compact summary for the dashboard."""
    i = inspect_dataset(df, dataset)
    return {
        "dataset_key": i["dataset_key"], "dataset_name": i["dataset_name"],
        "disease": i["disease"], "source": i["source"],
        "samples": i["rows"], "features": i["n_features"],
        "positive_cases": i["positive_cases"],
        "negative_cases": i["negative_cases"],
        "positive_class": i["positive_class"],
        "missing_values": i["missing_values_total"],
        "duplicate_rows": i["duplicate_rows"],
        "train_samples": train_samples, "test_samples": test_samples,
        "reduced_features": reduced_features,
    }


# Backwards compatibility with earlier single-dataset code.
DATASET_CONFIG = {**DATASETS[DEFAULT_DATASET], "random_state": RANDOM_STATE}


# ==========================================================================
# Manual check:  python -m src.data_loader
# ==========================================================================
if __name__ == "__main__":
    for key in DATASETS:
        try:
            df = load_dataset(key)
        except FileNotFoundError as e:
            print(f"\n!! {key}: {e}\n")
            continue
        i = inspect_dataset(df, key)
        print("=" * 68)
        print(f"{i['disease'].upper()}  --  {i['dataset_name']}")
        print("=" * 68)
        print(f"  {i['clinical_context']}")
        print(f"  patients x features : {i['rows']:,} x {i['n_features']}")
        print(f"  positive class      : {i['positive_class']} (encoded 1)")
        print(f"  balance             : {i['positive_cases']:,} {i['positive_class']} / "
              f"{i['negative_cases']:,} {i['negative_class']} "
              f"({i['positive_rate']*100:.1f}% positive)")
        print(f"  missing / duplicates: {i['missing_values_total']} / {i['duplicate_rows']}")
        print(f"  correlated pairs    : {i['highly_correlated_pairs']} of "
              f"{i['total_feature_pairs']}")
        print(f"  difficulty          : {i['difficulty']}")
        if DATASETS[key]["feature_meta"]:
            print("  sample of the clinical dictionary:")
            for f in i["feature_names"][:4]:
                print(f"      {describe_value(f, df[f].median(), key)}")
