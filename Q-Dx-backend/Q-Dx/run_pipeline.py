"""
Q-Dx :: Full backend pipeline
=============================

    python run_pipeline.py

One command runs the entire machine-learning engine and writes every artifact
the Streamlit frontend needs. Nothing else has to be executed by hand.

    dataset -> clean -> split -> scale -> PCA
                                   |
                    +--------------+--------------+
                    |                             |
              Classical SVM                 Quantum VQC
                    |                             |
                    +--------------+--------------+
                                   |
                    metrics / significance / operating points
                                   |
                            explainability
                                   |
                    results/*.json  +  models/*.joblib

Everything is seeded. Re-running produces identical numbers.
"""

import json
import time
from pathlib import Path

import numpy as np

from src.classical_model import (evaluate_classical_model, save_classical_model,
                                 train_classical_model)
from src.data_loader import get_dataset_summary, inspect_dataset, load_dataset
from src.explainability import (component_loadings, explain_classical_model,
                                explain_quantum_global,
                                explain_quantum_prediction)
from src.metrics import (bootstrap_ci, compare_models, compute_metrics,
                         format_comparison, mcnemar_test, operating_points)
from src.preprocessing import prepare_data, save_preprocessor
from src.quantum_model import (decision_values, get_model_metadata,
                               save_quantum_model, train_vqc)

N_COMPONENTS = 4
SEED = 42
EPOCHS = 40


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def main():
    t_start = time.perf_counter()
    Path("results").mkdir(exist_ok=True)

    # ================================================================
    banner("STEP 1/6  DATASET")
    df = load_dataset()
    info = inspect_dataset(df)
    print(f"  {info['dataset_name']}")
    print(f"  {info['rows']} patients x {info['n_features']} measurements")
    print(f"  {info['positive_cases']} malignant / {info['negative_cases']} benign "
          f"({info['positive_rate']*100:.1f}% positive)")
    print(f"  missing values: {info['missing_values_total']}   "
          f"duplicates: {info['duplicate_rows']}")
    print(f"  correlated feature pairs (|r|>0.9): "
          f"{info['highly_correlated_pairs']} of {info['total_feature_pairs']}")

    # ================================================================
    banner("STEP 2/6  PREPROCESSING")
    d = prepare_data(n_components=N_COMPONENTS)
    s = d["summary"]
    Xtr, Xte = d["X_train_reduced"], d["X_test_reduced"]
    ytr, yte = d["y_train"], d["y_test"]
    print(f"  split      : {s['train_samples']} train / {s['test_samples']} test "
          f"(stratified, seed={SEED})")
    print(f"  reduction  : {s['original_features']} -> {s['reduced_features']} "
          f"components, {s['variance_retained']*100:.2f}% variance retained")
    print(f"  leakage    : scaler and PCA fitted on training rows only")
    print(f"               test-set component means are "
          f"{np.round(Xte.mean(axis=0), 3)} (not forced to zero)")
    save_preprocessor(d)

    # ================================================================
    banner("STEP 3/6  CLASSICAL MODELS")
    svm = train_classical_model(Xtr, ytr, "svm")
    r_svm = evaluate_classical_model(svm, Xte, yte)
    print(f"  SVM  (4 components)  acc={r_svm['accuracy']:.4f}  "
          f"auc={r_svm['roc_auc']:.4f}  {r_svm['n_parameters']} stored numbers")

    rf = train_classical_model(Xtr, ytr, "rf")
    r_rf = evaluate_classical_model(rf, Xte, yte)
    print(f"  RF   (4 components)  acc={r_rf['accuracy']:.4f}  "
          f"auc={r_rf['roc_auc']:.4f}  {r_rf['n_parameters']} stored numbers")

    svm30 = train_classical_model(d["X_train_full"], ytr, "svm")
    r30 = evaluate_classical_model(svm30, d["X_test_full"], yte)
    r30["model"] = "Classical SVM (all 30 features)"
    print(f"  SVM  (all 30)        acc={r30['accuracy']:.4f}  "
          f"auc={r30['roc_auc']:.4f}   <- ceiling, NOT the headline comparison")
    save_classical_model(svm)

    # ================================================================
    banner("STEP 4/6  QUANTUM MODEL")
    vqc = train_vqc(Xtr, ytr, n_qubits=N_COMPONENTS, n_layers=3, epochs=EPOCHS,
                    batch_size=16, learning_rate=0.05, seed=SEED,
                    validation_fraction=0.2,
                    threshold_strategy="sensitivity_floor",
                    sensitivity_floor=0.90, verbose=False)
    meta = get_model_metadata(vqc)
    t0 = time.perf_counter()
    q_raw = decision_values(vqc, Xte)
    q_infer = time.perf_counter() - t0
    q_pred = (q_raw >= vqc["threshold"]).astype(int)
    q_score = (np.clip(q_raw, -1, 1) + 1) / 2

    r_vqc = compute_metrics(yte, q_pred, q_score, model_name="Quantum VQC",
                            training_time=vqc["training_time"],
                            inference_time=q_infer, n_train_samples=len(ytr),
                            threshold=vqc["threshold"])
    r_vqc["n_parameters"] = vqc["n_parameters"]
    print(f"  {meta['n_qubits']} qubits, {meta['n_layers']} layers, "
          f"depth {meta['depth']}, {meta['n_gates']} gates")
    print(f"  encoding: {meta['encoding']}   simulator: {meta['simulator']}")
    print(f"  acc={r_vqc['accuracy']:.4f}  auc={r_vqc['roc_auc']:.4f}  "
          f"{r_vqc['n_parameters']} stored numbers  "
          f"threshold={vqc['threshold']:+.4f}")
    save_quantum_model(vqc, "models/quantum/vqc_model.joblib")

    # ================================================================
    banner("STEP 5/6  COMPARISON AND SIGNIFICANCE")
    comparison = compare_models(r_svm, r_vqc)
    print(format_comparison(comparison))
    print(f"\n{'stored numbers':16s}{r_svm['n_parameters']:<21}{r_vqc['n_parameters']:<21}")
    print(f"{'missed cancers':16s}{r_svm['missed_cancers']:<21}{r_vqc['missed_cancers']:<21}")
    print(f"{'false alarms':16s}{r_svm['false_alarms']:<21}{r_vqc['false_alarms']:<21}")

    mc = mcnemar_test(yte, np.array(r_svm["predictions"]), q_pred, "SVM", "VQC")
    print(f"\n  McNemar p = {mc['p_value']:.4f}")
    print(f"  {mc['interpretation']}")

    cis = {}
    print(f"\n  95% bootstrap CIs      {'SVM':>24}{'VQC':>24}")
    for metric in ["accuracy", "sensitivity", "specificity", "roc_auc"]:
        a = bootstrap_ci(yte, np.array(r_svm["predictions"]),
                         np.array(r_svm["probabilities"]), metric)
        b = bootstrap_ci(yte, q_pred, q_score, metric)
        cis[metric] = {"SVM": a, "VQC": b}
        print(f"  {metric:20s}"
              f"{a['point_estimate']:.3f} [{a['ci_low']:.3f},{a['ci_high']:.3f}]".rjust(24)
              + f"{b['point_estimate']:.3f} [{b['ci_low']:.3f},{b['ci_high']:.3f}]".rjust(24))

    ops = {"SVM": operating_points(yte, np.array(r_svm["probabilities"])),
           "VQC": operating_points(yte, q_score)}
    print("\n  operating points (VQC): target sensitivity -> cost")
    for op in ops["VQC"]:
        if op["achievable"]:
            print(f"    {op['target_sensitivity']:.0%} -> catches "
                  f"{op['cancers_caught']}/{op['total_positives']} cancers, "
                  f"{op['false_alarms']} false alarms")

    # ================================================================
    banner("STEP 6/6  EXPLAINABILITY")
    loadings = component_loadings(d, top_k=5)
    for c in loadings["components"]:
        print(f"  {c['description']}")

    exp_svm = explain_classical_model(svm, Xte, yte)
    exp_vqc = explain_quantum_global(vqc, Xte)
    print(f"\n  SVM importance ranking : "
          f"{' > '.join(r['feature'] for r in exp_svm['ranking'])}")
    print(f"  VQC importance ranking : "
          f"{' > '.join(r['feature'] for r in exp_vqc['ranking'])}")
    if [r["feature"] for r in exp_svm["ranking"]] != [r["feature"] for r in exp_vqc["ranking"]]:
        print("  -> the two models weight the components differently, which is")
        print("     why they misclassify different patients.")

    example = explain_quantum_prediction(vqc, Xte[0])

    # ================================================================
    results = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": SEED, "n_components": N_COMPONENTS, "epochs": EPOCHS,
            "runtime_seconds": round(time.perf_counter() - t_start, 2),
        },
        "dataset": get_dataset_summary(df, s["train_samples"],
                                       s["test_samples"], N_COMPONENTS),
        "dataset_detail": {k: v for k, v in info.items() if k != "feature_names"},
        "preprocessing": s,
        "explained_variance_ratio": d["explained_variance_ratio"],
        "models": {
            "classical_svm": {k: v for k, v in r_svm.items()
                              if k not in ("predictions", "probabilities")},
            "classical_rf": {k: v for k, v in r_rf.items()
                             if k not in ("predictions", "probabilities")},
            "classical_svm_full": {k: v for k, v in r30.items()
                                   if k not in ("predictions", "probabilities")},
            "quantum_vqc": r_vqc,
        },
        "comparison": comparison,
        "significance": {"mcnemar": mc, "bootstrap_ci": cis},
        "operating_points": ops,
        "explainability": {
            "component_loadings": loadings,
            "classical_permutation_importance": exp_svm,
            "quantum_sensitivity_global": exp_vqc,
            "quantum_single_patient_example": example,
        },
        "quantum_circuit": {k: v for k, v in get_model_metadata(vqc).items()
                            if k != "history"},
        "training_history": vqc["history"],
        "disclaimer": ("Research prototype only. Predictions are not a medical "
                       "diagnosis and must not replace evaluation by a "
                       "qualified healthcare professional."),
    }

    with open("results/results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    banner("DONE")
    print(f"  total runtime: {time.perf_counter() - t_start:.1f}s")
    print("\n  artifacts written:")
    for p in ["results/results.json", "models/classical/svm_model.joblib",
              "models/classical/preprocessor.joblib",
              "models/quantum/vqc_model.joblib"]:
        print(f"    {p}   ({Path(p).stat().st_size/1024:.1f} KB)")
    print("\n  The frontend needs nothing else. results/results.json holds the")
    print("  comparison table, ROC curves, significance tests, operating points,")
    print("  explainability and circuit metadata.")


if __name__ == "__main__":
    main()
