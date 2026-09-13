"""
Q-Dx :: Classical vs Quantum comparison
=======================================

Run:  python run_comparison.py

The headline experiment. Both models see the IDENTICAL 4 PCA components and
the IDENTICAL held-out patients, and both are scored by the same functions in
src/metrics.py, so any difference is attributable to the algorithm and nothing
else.

Produces:
  1. the comparison table
  2. McNemar's test        -- is the difference real, or noise?
  3. bootstrap CIs         -- how much would these numbers wobble?
  4. operating points      -- the clinical decision-support view
  5. results/comparison.json for the dashboard
"""

import json
import time
from pathlib import Path

import numpy as np

from src.classical_model import (evaluate_classical_model,
                                 predict_proba_classical, train_classical_model)
from src.metrics import (bootstrap_ci, compare_models, compute_metrics,
                         format_comparison, mcnemar_test, operating_points)
from src.preprocessing import prepare_data
from src.quantum_model import decision_values, train_vqc

N_COMPONENTS = 4
SEED = 42


def main():
    print("=" * 70)
    print("Q-Dx :: CLASSICAL vs QUANTUM")
    print("=" * 70)

    # ---- data ---------------------------------------------------------
    d = prepare_data(n_components=N_COMPONENTS)
    s = d["summary"]
    Xtr, Xte = d["X_train_reduced"], d["X_test_reduced"]
    ytr, yte = d["y_train"], d["y_test"]
    print(f"\nDataset  : Wisconsin Breast Cancer (Diagnostic)")
    print(f"Samples  : {s['samples']}  ->  {s['train_samples']} train / "
          f"{s['test_samples']} test (stratified, seed={SEED})")
    print(f"Features : {s['original_features']} -> {s['reduced_features']} "
          f"PCA components ({s['variance_retained']*100:.2f}% variance)")
    print(f"Test set : {s['test_positives']} malignant / "
          f"{s['test_samples'] - s['test_positives']} benign")
    print("\nBoth models receive exactly these inputs. No exceptions.")

    # ---- classical ----------------------------------------------------
    print("\n" + "-" * 70)
    print("Training classical SVM ...")
    svm = train_classical_model(Xtr, ytr, "svm")
    r_svm = evaluate_classical_model(svm, Xte, yte)
    svm_score = np.array(r_svm["probabilities"])
    svm_pred = np.array(r_svm["predictions"])
    print(f"  done in {r_svm['training_time']:.4f}s  "
          f"({r_svm['n_parameters']} stored numbers)")

    # ---- quantum ------------------------------------------------------
    print("Training quantum VQC ...")
    vqc = train_vqc(Xtr, ytr, n_qubits=N_COMPONENTS, n_layers=3, epochs=40,
                    batch_size=16, learning_rate=0.05, seed=SEED,
                    validation_fraction=0.2,
                    threshold_strategy="sensitivity_floor",
                    sensitivity_floor=0.90, verbose=False)
    t0 = time.perf_counter()
    q_raw = decision_values(vqc, Xte)
    q_infer = time.perf_counter() - t0
    q_pred = (q_raw >= vqc["threshold"]).astype(int)
    q_score = (np.clip(q_raw, -1, 1) + 1) / 2
    n_q_params = vqc["n_parameters"]
    print(f"  done in {vqc['training_time']:.2f}s  "
          f"({n_q_params} stored numbers, threshold {vqc['threshold']:+.4f})")

    r_vqc = compute_metrics(yte, q_pred, q_score,
                            model_name="Quantum VQC",
                            training_time=vqc["training_time"],
                            inference_time=q_infer,
                            n_train_samples=len(ytr),
                            threshold=vqc["threshold"])
    r_vqc["n_parameters"] = n_q_params

    # ---- 1. comparison table ------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON  (held-out test set, identical inputs)")
    print("=" * 70)
    comparison = compare_models(r_svm, r_vqc)
    print(format_comparison(comparison))
    print(f"\n{'stored numbers':16s}{r_svm['n_parameters']:<18}{n_q_params:<18}")
    print(f"{'missed cancers':16s}{r_svm['missed_cancers']:<18}"
          f"{r_vqc['missed_cancers']:<18}")
    print(f"{'false alarms':16s}{r_svm['false_alarms']:<18}"
          f"{r_vqc['false_alarms']:<18}")

    # ---- 2. is the difference real? -----------------------------------
    print("\n" + "=" * 70)
    print("IS THE DIFFERENCE REAL?  McNemar's exact test")
    print("=" * 70)
    mc = mcnemar_test(yte, svm_pred, q_pred, "SVM", "VQC")
    print(f"  both correct        : {mc['both_correct']}")
    print(f"  both wrong          : {mc['both_wrong']}")
    print(f"  only SVM wrong      : {mc['only_SVM_wrong']}")
    print(f"  only VQC wrong      : {mc['only_VQC_wrong']}")
    print(f"  p-value             : {mc['p_value']:.4f}")
    print(f"\n  {mc['interpretation']}")

    # ---- 3. how much would these numbers wobble? ----------------------
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS  (2000 resamples)")
    print("=" * 70)
    print(f"{'metric':14s}{'SVM':>26}{'VQC':>26}")
    for metric in ["accuracy", "sensitivity", "specificity", "roc_auc"]:
        a = bootstrap_ci(yte, svm_pred, svm_score, metric)
        b = bootstrap_ci(yte, q_pred, q_score, metric)
        overlap = not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])
        mark = "" if overlap else "   <- intervals do NOT overlap"
        print(f"{metric:14s}"
              f"{a['point_estimate']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]".rjust(26)
              + f"{b['point_estimate']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]".rjust(26)
              + mark)
    print("\n  Overlapping intervals mean the two models are not distinguishable")
    print("  on that metric with this many test patients.")

    # ---- 4. clinical operating points ---------------------------------
    print("\n" + "=" * 70)
    print("CLINICAL OPERATING POINTS  (deliverable 4: threshold tuning)")
    print("=" * 70)
    for name, score in [("SVM", svm_score), ("VQC", q_score)]:
        print(f"\n  {name}:")
        print(f"    {'target sens':>12} {'achieved':>9} {'specificity':>12} "
              f"{'caught':>7} {'missed':>7} {'false alarms':>13}")
        for op in operating_points(yte, score, targets=(0.90, 0.95, 1.00)):
            if not op["achievable"]:
                print(f"    {op['target_sensitivity']:>12.0%}   not achievable")
                continue
            print(f"    {op['target_sensitivity']:>12.0%} {op['sensitivity']:>9.3f} "
                  f"{op['specificity']:>12.3f} {op['cancers_caught']:>7} "
                  f"{op['cancers_missed']:>7} {op['false_alarms']:>13}")
    print("\n  This is the decision-support view: a clinician picks the row,")
    print("  not the model. Catching every cancer is possible -- the table")
    print("  says what it costs in follow-up biopsies.")

    # ---- 5. save ------------------------------------------------------
    out = {
        "dataset": d["summary"],
        "n_components": N_COMPONENTS,
        "seed": SEED,
        "classical": {k: v for k, v in r_svm.items()
                      if k not in ("predictions", "probabilities", "roc_curve")},
        "quantum": {k: v for k, v in r_vqc.items() if k != "roc_curve"},
        "roc_curves": {"SVM": r_svm["roc_curve"], "VQC": r_vqc["roc_curve"]},
        "mcnemar": mc,
        "operating_points": {
            "SVM": operating_points(yte, svm_score),
            "VQC": operating_points(yte, q_score),
        },
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/comparison.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nwritten: results/comparison.json")


if __name__ == "__main__":
    main()
