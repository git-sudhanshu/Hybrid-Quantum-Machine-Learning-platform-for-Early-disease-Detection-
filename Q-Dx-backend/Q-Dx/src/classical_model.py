"""
Q-Dx :: Classical ML baseline
=============================

The model the quantum classifier is measured against. Its job is to be
GENUINELY STRONG. A weak baseline you beat proves nothing; a strong baseline
you match is a real result.

PRIMARY MODEL: RBF-kernel Support Vector Machine
------------------------------------------------
Why SVM rather than a neural network or gradient boosting:
  - It is the right tool for this data shape. 455 training patients with 4-30
    features is small-sample, moderate-dimension territory, exactly where
    kernel SVMs are strongest and where deep nets overfit.
  - The RBF kernel maps data into an implicit high-dimensional space, which
    makes it the closest CLASSICAL analogue to what a quantum feature map
    does. Comparing a VQC against an RBF-SVM is therefore a meaningful
    like-for-like question rather than a strawman.
  - It trains in milliseconds, so a 36-hour budget is never spent waiting.

SECONDARY: Random Forest, as an additional reference point only.

THE FAIRNESS RULE
-----------------
The headline comparison is SVM-on-4-PCA-components vs VQC-on-4-PCA-components
-- identical inputs, so any difference is attributable to the algorithm.

An SVM trained on all 30 features is also reported, clearly labelled as the
"classical ceiling". It is NOT the headline comparison: comparing a 30-feature
SVM against a 4-feature VQC would be comparing different information, not
different algorithms, and would rig the result against the quantum model.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .metrics import compute_metrics

RANDOM_STATE = 42


# ==========================================================================
# TRAINING
# ==========================================================================
def train_classical_model(X_train, y_train, model_type: str = "svm",
                          random_state: int = RANDOM_STATE,
                          **kwargs) -> dict:
    """Train a classical baseline. Returns a dict mirroring the quantum model's
    structure, so both sides of the comparison are handled identically.

    model_type : "svm" (default) or "rf"

    class_weight="balanced" is used rather than SMOTE. WDBC is 37% positive --
    mild imbalance that class weighting handles cleanly by penalising errors
    on the minority class more heavily. SMOTE synthesises fake patients, which
    for a 36-hour medical prototype adds risk and a story we would rather not
    have to defend.

    probability=True enables Platt-scaled probability estimates, which we need
    for ROC-AUC. Note this makes SVC fit an internal 5-fold cross-validation,
    so it is slower than a bare SVC -- that cost is real and is reported in
    training_time rather than hidden.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train).ravel()

    if model_type == "svm":
        params = dict(kernel="rbf", C=1.0, gamma="scale", probability=True,
                      class_weight="balanced", random_state=random_state)
        params.update(kwargs)
        estimator = SVC(**params)
        label = "Classical SVM (RBF)"
    elif model_type == "rf":
        params = dict(n_estimators=200, class_weight="balanced",
                      random_state=random_state, n_jobs=-1)
        params.update(kwargs)
        estimator = RandomForestClassifier(**params)
        label = "Classical Random Forest"
    else:
        raise ValueError(f"model_type must be 'svm' or 'rf', got {model_type!r}")

    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    return {
        "estimator": estimator,
        "model_type": model_type,
        "model_name": label,
        "params": params,
        "training_time": training_time,
        "n_train_samples": int(len(y_train)),
        "n_features": int(X_train.shape[1]),
        "random_state": random_state,
        "n_parameters": count_model_parameters(estimator, X_train.shape[1]),
    }


def count_model_parameters(estimator, n_features: int) -> int:
    """How many numbers the trained model actually stores.

    This feeds the project's model-compactness finding, so it must be honest:
      SVM -> support vectors (n_sv x n_features) + dual coefficients + intercept
      RF  -> total nodes across all trees
    A VQC stores n_layers x n_qubits x 2 rotation angles + 1 bias, and that
    count does NOT grow with the dataset -- which is the whole point of the
    comparison.
    """
    if isinstance(estimator, SVC):
        n_sv = int(estimator.n_support_.sum())
        return n_sv * n_features + n_sv + 1
    if isinstance(estimator, RandomForestClassifier):
        return int(sum(t.tree_.node_count for t in estimator.estimators_))
    return -1


# ==========================================================================
# INFERENCE
# ==========================================================================
def predict_classical_model(model: dict, X):
    return model["estimator"].predict(np.asarray(X, dtype=float))


def predict_proba_classical(model: dict, X):
    """P(malignant). Genuinely calibrated for the SVM via Platt scaling, unlike
    the quantum model's rescaled expectation value -- a difference worth being
    explicit about when the dashboard shows both."""
    return model["estimator"].predict_proba(np.asarray(X, dtype=float))[:, 1]


# ==========================================================================
# EVALUATION
# ==========================================================================
def evaluate_classical_model(model: dict, X_test, y_test,
                             threshold: float = 0.5) -> dict:
    """Evaluate on the held-out test set, using the SHARED metric functions so
    the numbers are computed identically to the quantum model's."""
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test).ravel()

    t0 = time.perf_counter()
    y_score = predict_proba_classical(model, X_test)
    inference_time = time.perf_counter() - t0

    y_pred = (y_score >= threshold).astype(int)

    result = compute_metrics(
        y_test, y_pred, y_score,
        model_name=model["model_name"],
        training_time=model["training_time"],
        inference_time=inference_time,
        n_train_samples=model["n_train_samples"],
        threshold=threshold,
    )
    result.update({
        "model_type": model["model_type"],
        "n_features": model["n_features"],
        "n_parameters": model["n_parameters"],
        "params": {k: str(v) for k, v in model["params"].items()},
        "predictions": y_pred.tolist(),
        "probabilities": y_score.tolist(),
    })
    return result


# ==========================================================================
# PERSISTENCE
# ==========================================================================
def save_classical_model(model: dict, path: str = "models/classical/svm_model.joblib") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_classical_model(path: str = "models/classical/svm_model.joblib") -> dict:
    return joblib.load(path)


# ==========================================================================
# Manual check:  python -m src.classical_model
# ==========================================================================
if __name__ == "__main__":
    from .preprocessing import prepare_data

    d = prepare_data(n_components=4)

    print("=" * 66)
    print("CLASSICAL BASELINES")
    print("=" * 66)

    print("\n--- FAIR COMPARISON INPUT: 4 PCA components (same as the VQC) ---")
    svm = train_classical_model(d["X_train_reduced"], d["y_train"], "svm")
    r_svm = evaluate_classical_model(svm, d["X_test_reduced"], d["y_test"])
    for k in ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]:
        print(f"    {k:12s}: {r_svm[k]:.4f}")
    print(f"    {'missed cancers':12s}: {r_svm['missed_cancers']}")
    print(f"    {'false alarms':12s}: {r_svm['false_alarms']}")
    print(f"    {'train time':12s}: {r_svm['training_time']:.4f}s")
    print(f"    {'parameters':12s}: {r_svm['n_parameters']} stored numbers")

    rf = train_classical_model(d["X_train_reduced"], d["y_train"], "rf")
    r_rf = evaluate_classical_model(rf, d["X_test_reduced"], d["y_test"])
    print(f"\n--- Random Forest reference (same 4 components) ---")
    print(f"    accuracy {r_rf['accuracy']:.4f}   auc {r_rf['roc_auc']:.4f}   "
          f"parameters {r_rf['n_parameters']}")

    print("\n--- CLASSICAL CEILING: all 30 features (NOT the headline) ---")
    svm30 = train_classical_model(d["X_train_full"], d["y_train"], "svm")
    r30 = evaluate_classical_model(svm30, d["X_test_full"], d["y_test"])
    print(f"    accuracy {r30['accuracy']:.4f}   auc {r30['roc_auc']:.4f}   "
          f"parameters {r30['n_parameters']}")
    print(f"    -> cost of compressing 30 features to 4: "
          f"{(r30['accuracy'] - r_svm['accuracy'])*100:+.2f} accuracy points")

    print(f"\nsaved: {save_classical_model(svm)}")
