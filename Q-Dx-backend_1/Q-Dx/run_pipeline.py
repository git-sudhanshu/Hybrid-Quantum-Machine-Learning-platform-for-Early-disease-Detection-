"""
Q-Dx :: Full backend pipeline -- all diseases
=============================================

    python run_pipeline.py                  # every disease
    python run_pipeline.py diabetes         # just one

Runs the complete engine for each supported disease and writes every artifact
the clinician-facing dashboard needs.

    dataset -> clean -> split -> scale -> PCA
                                   |
                    +--------------+--------------+
              Classical SVM                 Quantum VQC
                    +--------------+--------------+
                                   |
              metrics / significance / operating points
                                   |
                      explainability + risk bands
                                   |
        results/results_<disease>.json  +  models/**/<disease>.joblib

Everything is seeded; re-running reproduces identical numbers.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from src.classical_model import train_classical_model, evaluate_classical_model
from src.data_loader import (DATASETS, get_config, get_dataset_summary,
                             inspect_dataset, load_dataset)
from src.explainability import (component_loadings, explain_classical_model,
                                explain_quantum_global,
                                explain_quantum_prediction)
from src.metrics import (bootstrap_ci, compare_models, compute_metrics,
                         format_comparison, mcnemar_test, operating_points)
from src.prediction import build_bands
from src.preprocessing import prepare_data, save_preprocessor
from src.quantum_model import (decision_values, get_model_metadata,
                               save_quantum_model, train_vqc)

SEED = 42
N_COMPONENTS = 4
EPOCHS = 40
LARGE_SAMPLE_CAP = 20000     # cap for the "classical with more data" reference


def banner(text, ch="="):
    print("\n" + ch * 74)
    print(text)
    print(ch * 74)


def run_disease(key: str) -> dict:
    cfg = get_config(key)
    t_start = time.perf_counter()

    banner(f"{cfg['disease'].upper()}   ({cfg['name']})")

    # ---- 1. data ------------------------------------------------------
    df = load_dataset(key)
    info = inspect_dataset(df, key)
    print(f"  {info['clinical_context']}")
    print(f"  {info['rows']:,} patients x {info['n_features']} measurements  |  "
          f"{info['positive_cases']:,} {info['positive_class']} "
          f"({info['positive_rate']*100:.1f}%)")
    print(f"  missing {info['missing_values_total']}  duplicates "
          f"{info['duplicate_rows']}  correlated pairs "
          f"{info['highly_correlated_pairs']}/{info['total_feature_pairs']}")

    # ---- 2. preprocessing ---------------------------------------------
    d = prepare_data(key, n_components=N_COMPONENTS)
    s = d["summary"]
    Xtr, Xte = d["X_train_reduced"], d["X_test_reduced"]
    ytr, yte = d["y_train"], d["y_test"]
    print(f"\n  split      {s['train_samples']:,} train / {s['test_samples']:,} test"
          f"   {s['original_features']} -> {N_COMPONENTS} components "
          f"({s['variance_retained']*100:.1f}% variance)")
    if d["train_capped"]:
        print(f"  CAPPED     training subsampled {d['n_train_uncapped']:,} -> "
              f"{s['train_samples']:,} so the quantum simulator is tractable.")
        print(f"             BOTH models train on this subsample, so the "
              f"comparison stays fair.")
    save_preprocessor(d, f"models/classical/preprocessor_{key}.joblib")

    # ---- 3. classical --------------------------------------------------
    svm = train_classical_model(Xtr, ytr, "svm")
    r_svm = evaluate_classical_model(svm, Xte, yte)
    print(f"\n  SVM  acc={r_svm['accuracy']:.4f}  auc={r_svm['roc_auc']:.4f}  "
          f"{r_svm['n_parameters']:,} stored numbers  "
          f"{r_svm['training_time']:.2f}s")

    r_big = None
    if d["train_capped"]:
        n_big = min(LARGE_SAMPLE_CAP, len(d["y_train_all"]))
        idx = np.random.default_rng(SEED).choice(len(d["y_train_all"]), n_big,
                                                 replace=False)
        big = train_classical_model(d["X_train_reduced_all"][idx],
                                    np.asarray(d["y_train_all"])[idx], "svm")
        r_big = evaluate_classical_model(big, d["X_test_reduced_all"], yte)
        r_big["model"] = f"Classical SVM ({n_big:,} patients)"
        print(f"  SVM  with {n_big:,} patients: acc={r_big['accuracy']:.4f}  "
              f"auc={r_big['roc_auc']:.4f}  {r_big['n_parameters']:,} stored "
              f"numbers  {r_big['training_time']:.1f}s")

    # ---- 4. quantum ----------------------------------------------------
    batch = 16 if len(ytr) < 5000 else 64
    epochs = EPOCHS if len(ytr) < 5000 else 20
    vqc = train_vqc(Xtr, ytr, n_qubits=N_COMPONENTS, n_layers=3, epochs=epochs,
                    batch_size=batch, learning_rate=0.05, seed=SEED,
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
    print(f"  VQC  acc={r_vqc['accuracy']:.4f}  auc={r_vqc['roc_auc']:.4f}  "
          f"{r_vqc['n_parameters']} stored numbers  "
          f"{vqc['training_time']:.1f}s   ({meta['n_qubits']} qubits, "
          f"depth {meta['depth']})")

    # ---- risk bands, from each model's own score distribution ----------
    svm_train_scores = svm["estimator"].predict_proba(Xtr)[:, 1]
    svm["bands"] = build_bands(svm_train_scores, 0.5)
    q_train_raw = decision_values(vqc, Xtr)
    q_train_score = (np.clip(q_train_raw, -1, 1) + 1) / 2
    vqc["bands"] = build_bands(q_train_score,
                               float((np.clip(vqc["threshold"], -1, 1) + 1) / 2))

    import joblib
    Path("models/classical").mkdir(parents=True, exist_ok=True)
    joblib.dump(svm, f"models/classical/svm_{key}.joblib")
    save_quantum_model(vqc, f"models/quantum/vqc_{key}.joblib")

    # ---- 5. comparison and significance --------------------------------
    print()
    print(format_comparison(compare_models(r_svm, r_vqc)))
    print(f"\n  {'stored numbers':16s}{r_svm['n_parameters']:<21,}"
          f"{r_vqc['n_parameters']:<21,}")
    print(f"  {'missed cases':16s}{r_svm['missed_cancers']:<21}"
          f"{r_vqc['missed_cancers']:<21}")

    svm_pred = np.array(r_svm["predictions"])
    svm_score = np.array(r_svm["probabilities"])
    mc = mcnemar_test(yte, svm_pred, q_pred, "SVM", "VQC")
    print(f"\n  McNemar p = {mc['p_value']:.4g}  ->  "
          f"{'SIGNIFICANT' if mc['significant_at_0.05'] else 'not significant'}")

    n_boot = int(min(2000, max(400, 300000 // max(len(yte), 1))))
    cis = {}
    for metric in ["accuracy", "sensitivity", "specificity", "roc_auc"]:
        cis[metric] = {
            "SVM": bootstrap_ci(yte, svm_pred, svm_score, metric, n_boot=n_boot),
            "VQC": bootstrap_ci(yte, q_pred, q_score, metric, n_boot=n_boot),
        }
    a, b = cis["roc_auc"]["SVM"], cis["roc_auc"]["VQC"]
    print(f"  ROC-AUC 95% CI   SVM [{a['ci_low']:.4f}, {a['ci_high']:.4f}]   "
          f"VQC [{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
          + ("   <- no overlap" if a["ci_high"] < b["ci_low"]
             or b["ci_high"] < a["ci_low"] else ""))

    ops = {"SVM": operating_points(yte, svm_score),
           "VQC": operating_points(yte, q_score)}
    print("\n  VQC operating points:")
    for op in ops["VQC"]:
        if op["achievable"]:
            print(f"    catch {op['target_sensitivity']:.0%} -> "
                  f"{op['cancers_caught']:,}/{op['total_positives']:,} found, "
                  f"{op['false_alarms']:,} false alarms")

    # ---- 6. explainability ---------------------------------------------
    loadings = component_loadings(d, top_k=5)
    exp_svm = explain_classical_model(svm, Xte[:2000], yte[:2000])
    exp_vqc = explain_quantum_global(vqc, Xte, max_samples=100)
    print(f"\n  component structure:")
    for c in loadings["components"][:2]:
        print(f"    {c['description']}")
    print(f"  SVM ranks: {' > '.join(r['feature'] for r in exp_svm['ranking'])}"
          f"    VQC ranks: {' > '.join(r['feature'] for r in exp_vqc['ranking'])}")

    runtime = time.perf_counter() - t_start
    print(f"\n  [{cfg['disease']} complete in {runtime:.1f}s]")

    result = {
        "meta": {"dataset": key, "disease": cfg["disease"], "seed": SEED,
                 "n_components": N_COMPONENTS, "epochs": epochs,
                 "runtime_seconds": round(runtime, 2),
                 "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
        "dataset": get_dataset_summary(df, key, s["train_samples"],
                                       s["test_samples"], N_COMPONENTS),
        "dataset_detail": {k: v for k, v in info.items() if k != "feature_names"},
        "preprocessing": s,
        "training_cap": {"applied": d["train_capped"],
                         "cap": d["train_cap"],
                         "uncapped_train_size": d["n_train_uncapped"]},
        "explained_variance_ratio": d["explained_variance_ratio"],
        "models": {
            "classical_svm": {k: v for k, v in r_svm.items()
                              if k not in ("predictions", "probabilities")},
            "quantum_vqc": r_vqc,
            **({"classical_svm_large_sample":
                {k: v for k, v in r_big.items()
                 if k not in ("predictions", "probabilities")}} if r_big else {}),
        },
        "comparison": compare_models(r_svm, r_vqc),
        "significance": {"mcnemar": mc, "bootstrap_ci": cis},
        "operating_points": ops,
        "risk_bands": {"classical_svm": svm["bands"], "quantum_vqc": vqc["bands"]},
        "explainability": {
            "component_loadings": loadings,
            "classical_permutation_importance": exp_svm,
            "quantum_sensitivity_global": exp_vqc,
            "quantum_single_patient_example": explain_quantum_prediction(vqc, Xte[0]),
        },
        "quantum_circuit": {k: v for k, v in meta.items() if k != "history"},
        "training_history": vqc["history"],
        "disclaimer": ("Research prototype only. Predictions are not a medical "
                       "diagnosis and must not replace evaluation by a "
                       "qualified healthcare professional."),
    }

    Path("results").mkdir(exist_ok=True)
    with open(f"results/results_{key}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


def main():
    requested = sys.argv[1:] or list(DATASETS)
    for key in requested:
        if key not in DATASETS:
            print(f"Unknown dataset {key!r}. Available: {list(DATASETS)}")
            return

    t0 = time.perf_counter()
    all_results = {}
    for key in requested:
        try:
            all_results[key] = run_disease(key)
        except FileNotFoundError as e:
            print(f"\n  !! skipping {key}: {e}")

    # ---- cross-disease summary -----------------------------------------
    banner("ALL DISEASES")
    print(f"{'disease':22s}{'SVM AUC':>10}{'VQC AUC':>10}{'gap':>9}"
          f"{'McNemar p':>12}{'SVM params':>13}{'VQC params':>12}")
    for key, r in all_results.items():
        sv = r["models"]["classical_svm"]
        qv = r["models"]["quantum_vqc"]
        p = r["significance"]["mcnemar"]["p_value"]
        gap = (qv["roc_auc"] or 0) - (sv["roc_auc"] or 0)
        print(f"{r['meta']['disease']:22s}{sv['roc_auc']:>10.4f}"
              f"{qv['roc_auc']:>10.4f}{gap:>+9.4f}{p:>12.2e}"
              f"{sv['n_parameters']:>13,}{qv['n_parameters']:>12,}")

    with open("results/results.json", "w") as fh:
        json.dump({"diseases": all_results,
                   "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
                  fh, indent=2, default=str)

    print(f"\n  total runtime: {time.perf_counter() - t0:.1f}s")
    print("\n  artifacts:")
    for p in sorted(Path("results").glob("*.json")):
        print(f"    {p}  ({p.stat().st_size/1024:.0f} KB)")
    for p in sorted(Path("models").rglob("*.joblib")):
        print(f"    {p}  ({p.stat().st_size/1024:.0f} KB)")
    print("\n  Clinician-facing screening: see src/prediction.py "
          "-> predict_patient(disease, patient_dict)")


if __name__ == "__main__":
    main()
