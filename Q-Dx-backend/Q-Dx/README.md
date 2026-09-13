# Q-Dx — Hybrid Quantum Machine Learning Platform for Early Disease Detection

**Smart India Hackathon 2026 · Problem Statement SIH26139 · Egreen Quanta**

A hybrid quantum-classical machine learning platform that detects breast cancer
from cell-nucleus measurements, and — more importantly — measures honestly
whether the quantum half is worth having.

---

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
python run_pipeline.py
```

Runs in about 85 seconds and writes everything the dashboard needs to
`results/results.json`.

If PowerShell blocks the activate script:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## What it does

```
Wisconsin Breast Cancer dataset (569 patients × 30 measurements)
        ↓  clean, de-duplicate
        ↓  stratified 80/20 split          ← the only split in the project
        ↓  impute + scale                  ← fitted on TRAIN only
        ↓  PCA → 4 components              ← fitted on TRAIN only
        │
        ├──────────────────────┬───────────────────────┐
        ↓                      ↓                       ↓
  Classical SVM          Quantum VQC            SVM on all 30
  (4 components)         (4 qubits)             (ceiling reference)
        │                      │
        └──────────┬───────────┘
                   ↓
     metrics · McNemar · bootstrap CIs · operating points
                   ↓
              explainability
                   ↓
          results/results.json  →  Streamlit dashboard
```

---

## Results

Held-out test set: 114 patients, 42 malignant. Both models receive the
identical 4 principal components.

| Metric | Classical SVM | Quantum VQC |
|---|---|---|
| Accuracy | 0.9561 | 0.9386 |
| Sensitivity | 0.9048 | 0.8571 |
| Specificity | 0.9861 | 0.9861 |
| Precision | 0.9744 | 0.9730 |
| F1 | 0.9383 | 0.9114 |
| ROC-AUC | 0.9970 | 0.9924 |
| Training time | 0.008 s | 21.7 s |
| Inference time | 0.0007 s | 0.011 s |
| **Stored parameters** | **491** | **25** |

**The difference is not statistically significant.** McNemar's exact test gives
**p = 0.625**, and every bootstrap confidence interval overlaps. We report this
rather than claiming a quantum win we cannot support.

### Three measured findings

**1. Adding qubits does nothing.** ROC-AUC stayed flat at ~0.993 from 2 qubits
to 8 while training cost grew six-fold. The model saturates at 2 qubits. This
is why we use 4 — measured, not assumed.

**2. Quantum model size is constant; the SVM's grows.** From 20 to 426 training
patients the SVM grew from 66 stored numbers to 471. The VQC stayed at exactly
25. A kernel SVM memorises support vectors; a variational circuit compresses
into fixed rotation angles.

**3. Threshold placement beat training.** Ten times more epochs changed nothing
— the loss floor and ROC-AUC were flat from epoch 10. Moving the decision
threshold, with no extra training, lifted sensitivity from 0.756 to 0.889.

---

## Clinical decision support

The platform exposes the sensitivity/specificity trade-off as a clinician's
choice rather than burying it in a default threshold:

| Target sensitivity | Cancers caught | Missed | False alarms |
|---|---|---|---|
| 90% | 38 / 42 | 4 | 1 |
| 95% | 41 / 42 | 1 | 3 |
| 100% | 42 / 42 | 0 | 9 |

Catching every cancer is achievable. The table states the price in follow-up
biopsies.

---

## Project structure

```
Q-Dx/
├── run_pipeline.py            full backend, one command
├── run_comparison.py          classical vs quantum only
├── quantum_test.py            quantum smoke test (synthetic data)
├── requirements.txt
├── src/
│   ├── data_loader.py         loading, inspection, dataset summary
│   ├── preprocessing.py       split, impute, scale, PCA, quantum interface
│   ├── feature_selection.py   ANOVA / mutual-information ranking (optional)
│   ├── classical_model.py     SVM and Random Forest baselines
│   ├── quantum_circuit.py     device, encoding, ansatz, measurement
│   ├── quantum_model.py       VQC training, inference, threshold selection
│   ├── metrics.py             shared metrics, McNemar, bootstrap, operating points
│   └── explainability.py      permutation importance + quantum sensitivity
├── models/
│   ├── classical/             svm_model.joblib, preprocessor.joblib
│   └── quantum/               vqc_model.joblib
└── results/
    └── results.json           everything the frontend consumes
```

---

## Integration

```python
from src.preprocessing import prepare_quantum_data
from src.quantum_model  import train_vqc, predict_vqc, evaluate_vqc
from src.classical_model import train_classical_model, evaluate_classical_model

X_train, X_test, y_train, y_test = prepare_quantum_data()

quantum  = train_vqc(X_train, y_train)
classical = train_classical_model(X_train, y_train, "svm")
```

`prepare_quantum_data()` returns a feature matrix whose width equals the qubit
count. `n_components` is configurable, so 6 or 8 qubits needs no code change.

For the dashboard, read `results/results.json` — it holds the comparison table,
both ROC curves, the significance tests, operating points, explainability and
circuit metadata. No model needs retraining at render time.

---

## Design decisions

**Dataset — Wisconsin Breast Cancer (Diagnostic).** Ships inside scikit-learn,
so there is no download to fail and every machine loads byte-identical data.
569 patients, 30 features, no missing values, no duplicates. 21 of 435 feature
pairs correlate above |r| = 0.9, which is genuine redundancy for PCA to exploit.

**The label is re-encoded.** scikit-learn ships this dataset as
`malignant=0, benign=1`. Left alone, "sensitivity" would measure how well the
model detects *healthy* people. `data_loader.py` flips it so **1 = malignant =
disease present**. Every metric depends on this.

**PCA to 4 components.** Angle encoding needs one qubit per feature. 30 qubits
is not simulable (2³⁰ amplitudes) and far beyond near-term hardware. The
dimensionality must fall before the quantum stage — and that necessity is what
makes the architecture genuinely hybrid rather than decorative. 4 components
retain 79.3% of the variance.

**Angle encoding (RY).** Needs only n qubits for n features, is shallow,
hardware-native and differentiable. Features are mapped into −π/2…+π/2 because
rotations wrap at 2π — without that, two very different patients could land on
the same quantum state.

**3 variational layers.** Deeper parameterised circuits hit barren plateaus,
where gradients vanish exponentially with depth and width. Weights initialise
small (σ = 0.1) for the same reason.

**SVM as the baseline.** The RBF kernel maps data into an implicit
high-dimensional space, making it the closest classical analogue to a quantum
feature map — a like-for-like comparison rather than a strawman.

**Class weighting, not SMOTE.** 37% positive is mild imbalance.
`class_weight="balanced"` handles it without synthesising fake patients.

---

## Data leakage prevention

The single most important correctness property in the project.

- One split, in `preprocessing.py`. Nothing else may split data.
- Duplicates dropped **before** the split, so no row appears in both halves.
- Imputer, encoder, scaler and PCA all live inside one sklearn `Pipeline`,
  fitted on training rows only.
- The VQC's decision threshold is chosen on a validation slice carved out of
  the training set — never on test data.
- Feature selection, when used, is fitted on training data only.

**Visible proof:** after preprocessing, training component means are exactly
`[0, 0, 0, 0]` while test means are `[-0.194, 0.249, 0.103, -0.224]`. Test
means are *not* forced to zero — the signature of a scaler that never saw them.

---

## Explainability

Two models, two different and non-interchangeable methods:

- **Classical SVM** — permutation importance (shuffle a column, measure the
  ROC-AUC drop).
- **Quantum VQC** — input-perturbation sensitivity (nudge one input, measure
  how far the circuit output moves).

**SHAP does not explain a quantum circuit.** It attributes output to inputs;
for the VQC those inputs are four principal components, and the attribution
says nothing about superposition or entanglement. We do not make that claim.

**Principal components are not biomarkers.** `component_loadings()` reports
which original measurements build each component, so the dashboard can say
*"PC1 (44.6% of variance) is dominated by mean concave points, mean concavity
and worst concave points"* rather than pretending a component is a biomarker.

Interestingly the two models rank the components differently — SVM
PC1>PC2>PC3>PC4, VQC PC1>PC3>PC2>PC4 — which is why they misclassify different
patients.

---

## Reproducibility

`random_state=42` throughout: split, PCA, SVM, circuit initialisation, batch
shuffling, bootstrap. The pipeline has produced identical metrics on Linux /
Python 3.11 and Windows / Python 3.14.

---

## Limitations

- **WDBC is an easy benchmark.** An RBF SVM reaches ROC-AUC ≈ 0.99 on only 4
  components, so there is little headroom for any model to distinguish itself.
- **114 test patients** is small. A three-patient difference moves accuracy by
  ~2.6 points, which is why significance testing is reported rather than raw
  point estimates.
- **The quantum model runs on a simulator**, not real hardware. The circuit is
  shallow enough to be hardware-plausible and the code supports the
  parameter-shift gradient rule that real devices require, but no QPU claim is
  made.
- **The VQC's probability output is not calibrated.** It is a monotone
  rescaling of an expectation value — fine for ranking and ROC-AUC, not a
  calibrated clinical risk percentage.
- **Training is ~2,500× slower** than the SVM. That is simulation overhead, not
  an algorithmic property, and it is reported rather than hidden.

---

## Medical disclaimer

This platform is intended for research and screening demonstration purposes
only. Predictions are **not** a medical diagnosis and must not replace
evaluation by a qualified healthcare professional.
