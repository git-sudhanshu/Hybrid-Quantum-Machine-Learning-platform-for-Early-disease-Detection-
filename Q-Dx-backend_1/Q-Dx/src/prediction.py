"""
Q-Dx :: Clinician-facing prediction
===================================

Everything else in this codebase speaks in principal components. A doctor does
not. This module is the translation layer: real clinical measurements in, a
readable screening report out.

    report = predict_patient("diabetes", {
        "HighBP": 1, "HighChol": 1, "BMI": 34, "Age": 9, ...
    })

    report["risk_band"]        -> "High"
    report["recommendation"]   -> "Refer for confirmatory testing."
    report["top_factors"]      -> the measurements that pushed the result

WHAT THIS MODULE WILL NOT DO
----------------------------
It will not call its output a probability of disease. The VQC's score is a
rescaled expectation value and the SVM's is Platt-scaled -- neither is a
calibrated clinical risk, and presenting "73% chance of diabetes" to a doctor
would be a fabrication. The output is a RISK BAND plus the model's own score,
labelled as model confidence.

RISK BANDS ARE BUILT FROM THE OPERATING POINTS, NOT INVENTED
-------------------------------------------------------------
    Low       below the threshold that catches 95% of true cases
              -> even a deliberately cautious screen clears this patient
    Moderate  between that and the model's own decision threshold
              -> a cautious screen would flag them; the default would not
    High      at or above the model's decision threshold
    Very High well above it, in the top decile of training scores

So a band means something specific and auditable: it says which screening
policy would have flagged this patient.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_loader import (DEFAULT_DATASET, clinical_label, describe_value,
                          feature_reference, get_config, get_feature_names,
                          load_dataset)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DISCLAIMER = ("Research prototype. This is a screening indication, not a "
              "medical diagnosis, and must not replace evaluation by a "
              "qualified healthcare professional.")

RECOMMENDATIONS = {
    "Low":       "No action indicated by this screen. Follow routine guidelines.",
    "Moderate":  "Consider follow-up testing. A high-sensitivity screening "
                 "policy would flag this patient.",
    "High":      "Refer for confirmatory testing.",
    "Very High": "Refer for confirmatory testing. Model confidence is at the "
                 "top of its observed range.",
}


# ==========================================================================
# Form scaffolding for the UI
# ==========================================================================
def blank_patient(dataset: str = DEFAULT_DATASET) -> dict:
    """A pre-filled patient record using each measurement's population median.

    Gives the dashboard a sensible starting form rather than an empty one, so
    a doctor changes the three fields that matter instead of typing nineteen.
    """
    return {f["feature"]: f["median"] for f in feature_reference(dataset)}


def patient_form_spec(dataset: str = DEFAULT_DATASET) -> list[dict]:
    """Field definitions for the data-entry form: label, coding, range, type."""
    return feature_reference(dataset)


def describe_patient(patient: dict, dataset: str = DEFAULT_DATASET) -> list[str]:
    """Render a patient record in clinical language, for the report header."""
    return [describe_value(f, v, dataset) for f, v in patient.items()]


# ==========================================================================
# Artifact loading
# ==========================================================================
def load_artifacts(dataset: str = DEFAULT_DATASET) -> dict:
    """Load the fitted preprocessor and both trained models for one disease."""
    pre_path = MODELS_DIR / "classical" / f"preprocessor_{dataset}.joblib"
    svm_path = MODELS_DIR / "classical" / f"svm_{dataset}.joblib"
    vqc_path = MODELS_DIR / "quantum" / f"vqc_{dataset}.joblib"

    missing = [p.name for p in (pre_path, svm_path, vqc_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts for '{dataset}': {missing}. "
            f"Run `python run_pipeline.py` first."
        )
    return {"preprocessor": joblib.load(pre_path),
            "svm": joblib.load(svm_path),
            "vqc": joblib.load(vqc_path)}


# ==========================================================================
# Risk banding
# ==========================================================================
def risk_band(score: float, bands: dict) -> str:
    """Map a model score onto a named band using thresholds derived from the
    model's own operating points (see module docstring)."""
    if score >= bands["very_high"]:
        return "Very High"
    if score >= bands["high"]:
        return "High"
    if score >= bands["moderate"]:
        return "Moderate"
    return "Low"


def build_bands(scores_train, decision_threshold: float) -> dict:
    """Derive band edges from the model's threshold and its score distribution.

    `moderate` sits below the decision threshold, at the 25th percentile of
    scores -- the region a high-sensitivity policy would flag. `very_high` is
    the 90th percentile, i.e. the top decile of everything the model has seen.
    """
    s = np.asarray(scores_train, dtype=float)
    return {
        "moderate": float(min(np.percentile(s, 25), decision_threshold)),
        "high": float(decision_threshold),
        "very_high": float(max(np.percentile(s, 90), decision_threshold)),
    }


# ==========================================================================
# THE MAIN ENTRY POINT
# ==========================================================================
def predict_patient(dataset: str, patient: dict,
                    artifacts: dict | None = None,
                    model: str = "both") -> dict:
    """Screen one patient and return a clinician-readable report.

    patient : dict of raw clinical measurements, keyed by the dataset's own
              column names. Missing fields are filled with the population
              median and listed in the report, so a partially completed form
              still returns a result rather than an error -- with the gaps
              stated plainly.

    model   : "both" (default), "quantum" or "classical".
    """
    from .quantum_model import decision_values          # local import keeps
    # PennyLane out of the import path for classical-only callers.

    cfg = get_config(dataset)
    art = artifacts or load_artifacts(dataset)
    pre = art["preprocessor"]

    # --- assemble the feature row, in the exact training column order ------
    reference = load_dataset(dataset)
    columns = get_feature_names(reference, dataset)
    defaults = {f["feature"]: f["median"] for f in feature_reference(dataset)}

    filled_from_median, unknown = [], []
    row = {}
    for c in columns:
        if c in patient and patient[c] is not None:
            row[c] = float(patient[c])
        else:
            row[c] = defaults[c]
            filled_from_median.append(c)
    unknown = [k for k in patient if k not in columns]

    X = pd.DataFrame([row], columns=columns)
    X_reduced = pre["reducer"].transform(X)

    out = {
        "dataset": dataset,
        "disease": cfg["disease"],
        "positive_class": cfg["positive_class_name"],
        "patient_summary": describe_patient(row, dataset),
        "fields_defaulted": [clinical_label(c, dataset) for c in filled_from_median],
        "unrecognised_fields": unknown,
        "models": {},
        "disclaimer": DISCLAIMER,
    }

    # --- classical -------------------------------------------------------
    if model in ("both", "classical"):
        svm = art["svm"]
        score = float(svm["estimator"].predict_proba(X_reduced)[0, 1])
        bands = svm.get("bands") or {"moderate": 0.25, "high": 0.5, "very_high": 0.9}
        band = risk_band(score, bands)
        out["models"]["classical_svm"] = {
            "model": "Classical SVM (RBF)",
            "score": round(score, 4),
            "score_type": "Platt-scaled probability estimate",
            "decision_threshold": bands["high"],
            "prediction": cfg["positive_class_name"] if score >= bands["high"]
                          else cfg["negative_class_name"],
            "risk_band": band,
            "recommendation": RECOMMENDATIONS[band],
        }

    # --- quantum ---------------------------------------------------------
    if model in ("both", "quantum"):
        vqc = art["vqc"]
        raw = float(decision_values(vqc, X_reduced)[0])
        score = float((np.clip(raw, -1, 1) + 1) / 2)
        thr = float(vqc.get("threshold", 0.0))
        thr_p = float((np.clip(thr, -1, 1) + 1) / 2)
        bands = vqc.get("bands") or {"moderate": max(thr_p - .15, 0),
                                     "high": thr_p, "very_high": min(thr_p + .2, 1)}
        band = risk_band(score, bands)

        contributions = _quantum_factors(vqc, row, columns, dataset, pre)
        out["models"]["quantum_vqc"] = {
            "model": "Quantum VQC",
            "score": round(score, 4),
            "score_type": ("rescaled quantum expectation value -- model "
                           "confidence, NOT a calibrated probability"),
            "decision_threshold": round(thr_p, 4),
            "prediction": cfg["positive_class_name"] if raw >= thr
                          else cfg["negative_class_name"],
            "risk_band": band,
            "recommendation": RECOMMENDATIONS[band],
            "top_factors": contributions,
            "factor_caveat": (
                "These are ASSOCIATIONS the model learned, not causal clinical "
                "effects. Survey-derived datasets carry healthcare-access "
                "confounds -- for example 'cholesterol checked recently' is "
                "predictive because people who get tested are people already "
                "engaging with healthcare, not because testing causes disease."),
        }

    # --- agreement -------------------------------------------------------
    if model == "both":
        c = out["models"]["classical_svm"]["prediction"]
        q = out["models"]["quantum_vqc"]["prediction"]
        out["agreement"] = {
            "models_agree": c == q,
            "note": ("Both models reached the same conclusion." if c == q else
                     "The two models DISAGREE on this patient. Treat the result "
                     "as uncertain and prefer confirmatory testing."),
        }
    return out


def _quantum_factors(vqc, row: dict, columns, dataset, pre,
                     top_k: int = 4) -> list[dict]:
    """Which of THIS patient's measurements moved the result, by counterfactual.

    WHY NOT PROJECT PCA LOADINGS BACK
    ----------------------------------
    The obvious shortcut is to compute sensitivity in component space and
    multiply by the PCA loading matrix. It is fast, and it is misleading: with
    correlated survey features the signs smear across variables and the output
    cheerfully reports things like "eats fruit daily -> raises risk". A
    clinician reading that -- correctly -- stops trusting the system.

    So instead we ask the question directly. For each measurement we build a
    COUNTERFACTUAL version of this same patient with only that one value
    changed, push it through the entire fitted pipeline, and re-run the
    circuit:

        binary field      flipped (yes <-> no)
        continuous field  moved one population standard deviation

    influence = score(this patient) - score(counterfactual patient)

    Positive means the patient's actual value pushed the score UP. This is an
    end-to-end measurement of the deployed model, not a linear approximation
    of it -- and it is expressed in the only terms a doctor cares about:
    "if this were different, the result would be ___".

    STILL NOT CAUSAL. It describes the model's behaviour, not the disease.
    """
    from .quantum_model import decision_values

    ref = load_dataset(dataset)
    stats = {f["feature"]: f for f in feature_reference(dataset)}

    rows, meta = [], []
    for c in columns:
        alt = dict(row)
        info = stats[c]
        if info["binary"]:
            alt[c] = 0.0 if float(row[c]) >= 0.5 else 1.0
            change = f"{'no' if alt[c] == 0 else 'yes'}"
        else:
            sd = float(ref[c].std()) or 1.0
            alt[c] = float(row[c]) - sd
            change = f"{alt[c]:.1f}"
        rows.append([alt[k] for k in columns])
        meta.append((c, change))

    X_alt = pd.DataFrame(rows, columns=columns)
    alt_scores = decision_values(vqc, pre["reducer"].transform(X_alt))
    base = float(decision_values(vqc, pre["reducer"].transform(
        pd.DataFrame([[row[k] for k in columns]], columns=columns)))[0])

    influence = base - np.asarray(alt_scores, dtype=float)
    total = np.abs(influence).sum() or 1.0
    order = np.argsort(np.abs(influence))[::-1][:top_k]

    out = []
    for i in order:
        c, change = meta[i]
        out.append({
            "measurement": clinical_label(c, dataset),
            "raw_feature": c,
            "patient_value": row[c],
            "influence": round(float(influence[i]), 4),
            "share_percent": round(float(abs(influence[i]) / total * 100), 1),
            "direction": "raises risk" if influence[i] > 0 else "lowers risk",
            "counterfactual": (f"if {change} instead, score would move "
                               f"{'down' if influence[i] > 0 else 'up'} by "
                               f"{abs(float(influence[i]))/2:.3f}"),
        })
    return out


# ==========================================================================
# Human-readable rendering
# ==========================================================================
def format_report(report: dict) -> str:
    """Print-ready screening report."""
    L = []
    L.append("=" * 66)
    L.append(f"  Q-Dx SCREENING REPORT  --  {report['disease']}")
    L.append("=" * 66)

    L.append("\n  PATIENT")
    for line in report["patient_summary"][:8]:
        L.append(f"    {line}")
    if len(report["patient_summary"]) > 8:
        L.append(f"    ... and {len(report['patient_summary']) - 8} more measurements")
    if report["fields_defaulted"]:
        L.append(f"\n    NOT PROVIDED (population median used): "
                 f"{', '.join(report['fields_defaulted'][:5])}"
                 + (" ..." if len(report["fields_defaulted"]) > 5 else ""))

    L.append("\n  RESULT")
    for m in report["models"].values():
        L.append(f"    {m['model']:22s} {m['risk_band']:10s} "
                 f"(score {m['score']:.3f}, threshold {m['decision_threshold']:.3f})")
        L.append(f"      -> {m['recommendation']}")

    if "top_factors" in report["models"].get("quantum_vqc", {}):
        L.append("\n  LEADING FACTORS (quantum model)")
        for f in report["models"]["quantum_vqc"]["top_factors"]:
            L.append(f"    {f['share_percent']:5.1f}%  {f['measurement']}  "
                     f"-- {f['direction']}")
        L.append("    NOTE: model associations, not causal clinical effects.")

    if "agreement" in report:
        L.append(f"\n  {report['agreement']['note']}")

    L.append(f"\n  {report['disclaimer']}")
    L.append("=" * 66)
    return "\n".join(L)


# ==========================================================================
# Manual check:  python -m src.prediction
# ==========================================================================
if __name__ == "__main__":
    from .data_loader import DATASETS

    for key in DATASETS:
        try:
            art = load_artifacts(key)
        except FileNotFoundError as e:
            print(f"\n!! {key}: {e}\n")
            continue

        df = load_dataset(key)
        cols = get_feature_names(df, key)
        tgt = get_config(key)["target_column"]

        # one genuinely positive and one genuinely negative patient
        for want, tag in [(1, "known POSITIVE case"), (0, "known NEGATIVE case")]:
            row = df[df[tgt] == want].iloc[0]
            patient = {c: row[c] for c in cols}
            rep = predict_patient(key, patient, artifacts=art)
            print(f"\n### {tag} ({get_config(key)['disease']})")
            print(format_report(rep))
