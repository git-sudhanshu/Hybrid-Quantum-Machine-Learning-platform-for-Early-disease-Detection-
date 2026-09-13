"""
Q-Dx :: Quantum component smoke test  (MILESTONE 1)
===================================================

Run:  python quantum_test.py

Proves, end to end and with no real dataset required:
  1. the simulator initialises
  2. the 4-qubit circuit builds and has a real, non-trivial depth
  3. angle encoding accepts a generic reduced feature matrix
  4. training actually MOVES the circuit parameters
  5. prediction and evaluation run on UNSEEN test data

The dummy data here stands in for whatever the classical ML developer will
hand us after preprocessing + PCA. Nothing about the quantum module is
specific to it -- swap in the real X_train/X_test and it runs unchanged.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.quantum_circuit import (DEFAULT_N_LAYERS, build_quantum_device,
                                 initialize_parameters)
from src.quantum_model import (evaluate_vqc, get_model_metadata,
                               load_quantum_model, predict_proba_vqc,
                               predict_vqc, save_metrics_json,
                               save_quantum_model, train_vqc)

SEED = 42
N_QUBITS = 4
N_LAYERS = DEFAULT_N_LAYERS


def main():
    np.random.seed(SEED)

    # ---- PHASE 1: simulator -------------------------------------------
    dev = build_quantum_device(N_QUBITS)
    print("Quantum simulator initialized:", dev.name)
    print("Qubits:", N_QUBITS)

    # ---- PHASE 2-4: circuit -------------------------------------------
    w0 = initialize_parameters(N_QUBITS, N_LAYERS, seed=SEED)
    from src.quantum_circuit import get_quantum_circuit
    info = get_quantum_circuit(w0, np.zeros(N_QUBITS), N_QUBITS, N_LAYERS)
    print("Encoding:", info["encoding"])
    print("Variational layers:", N_LAYERS)
    print("Circuit depth:", info["depth"])
    print("Gates:", info["n_gates"])
    print("Trainable circuit parameters:", info["n_parameters"])
    print("\nCircuit:")
    print(info["diagram_text"])

    # ---- dummy "already reduced" feature matrix -----------------------
    # Stands in for the PCA output the classical developer will provide.
    X, y = make_classification(n_samples=300, n_features=N_QUBITS,
                               n_informative=3, n_redundant=1,
                               n_clusters_per_class=1, class_sep=1.5,
                               flip_y=0.05, random_state=SEED)

    # Stratified split so both classes appear in train AND test.
    # The test set is touched ONLY at evaluation time -- no leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    print(f"\nData: train={X_train.shape}, test={X_test.shape}, "
          f"features={X.shape[1]} -> {N_QUBITS} qubits")

    # ---- PHASE 5: training --------------------------------------------
    print("\nTraining started")
    model = train_vqc(X_train, y_train, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                      epochs=25, batch_size=16, learning_rate=0.05, seed=SEED,
                      validation_fraction=0.2,
                      threshold_strategy="sensitivity_floor",
                      sensitivity_floor=0.90)
    print("Training completed in %.2fs" % model["training_time"])
    print("Decision threshold (chosen on validation split): %+.4f"
          % model["threshold"])

    # ---- PHASE 6: did training actually change anything? ---------------
    print("Mean |change| in circuit parameters: %.4f" % model["mean_abs_param_change"])
    assert model["mean_abs_param_change"] > 1e-6, "Parameters did not move -- training failed."
    print("Learned bias: %.4f" % model["bias"])

    # ---- inference on unseen data --------------------------------------
    preds = predict_vqc(model, X_test)
    probs = predict_proba_vqc(model, X_test)
    print("\nPrediction (first 10):", preds[:10].tolist())
    print("True labels  (first 10):", y_test[:10].tolist())
    print("Scores       (first 5): ", np.round(probs[:5], 3).tolist())

    # ---- PHASE 8-9: evaluation on the held-out test set ----------------
    metrics = evaluate_vqc(model, X_test, y_test)
    print("\n--- QUANTUM VQC METRICS (held-out test set) ---")
    for k in ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]:
        v = metrics[k]
        print(f"  {k:12s}: {v:.4f}" if v is not None else f"  {k:12s}: n/a")
    print(f"  training_time : {metrics['training_time']:.3f}s")
    print(f"  inference_time: {metrics['inference_time']:.4f}s "
          f"({metrics['n_test_samples']} samples)")
    print("  confusion_matrix:", metrics["confusion_matrix"])

    # ---- persistence ----------------------------------------------------
    save_quantum_model(model, "models/quantum/vqc_smoketest.joblib")
    reloaded = load_quantum_model("models/quantum/vqc_smoketest.joblib")
    assert np.array_equal(predict_vqc(reloaded, X_test), preds), "Reload mismatch"
    print("\nModel saved + reloaded, predictions identical.")

    save_metrics_json(metrics, "results/quantum_metrics.json")
    print("Metrics written to results/quantum_metrics.json")

    meta = get_model_metadata(model)
    print("Frontend metadata keys:", sorted(meta.keys()))


if __name__ == "__main__":
    main()
