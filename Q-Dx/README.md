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

Two diseases, deliberately different in difficulty. Both models always receive
the identical inputs and are scored by the identical functions.

| | Breast cancer | Type 2 diabetes |
|---|---|---|
| Patients | 569 | 68,972 |
| Features | 30 continuous | 19 binary / ordinal |
| Classical SVM ROC-AUC | **0.9970** | 0.7865 |
| Quantum VQC ROC-AUC | 0.9924 | **0.7937** |
| Gap (VQC − SVM) | −0.0046 | **+0.0092** |
| Statistically significant? | **No** (McNemar p = 0.625) | **Yes** (paired bootstrap p < 0.001) |
| SVM stored parameters | 491 | 9,081 |
| VQC stored parameters | **25** | **25** |

### The finding

**On the easy dataset the models are tied. On the hard one, the quantum model
wins — and the win is real.**

A paired bootstrap of the ROC-AUC *difference* on diabetes (2,000 resamples of
the same 13,795 test patients) gives **+0.0092, 95% CI [+0.0059, +0.0125]**,
with the VQC ahead in **100% of resamples**. The interval excludes zero.

It reproduces independently: on identical training subsamples at n = 1,000 and
n = 3,000 across three seeds, the VQC beat the SVM in **6 of 6 runs** by a
consistent +0.011 AUC.

And the scale of it: a VQC trained on **1,000 patients in 5 seconds** (AUC
0.795) outperforms an SVM trained on all **55,177 patients in 320 seconds**
(AUC 0.782).

**The honest limits of that claim.** The advantage is in *ranking*, not
accuracy — at best-tuned thresholds accuracy ties (0.733 SVM vs 0.729 VQC).
+0.0092 AUC is a small effect, statistically solid rather than clinically
transformative. And the VQC trains on a 3,000-patient subsample because
simulating circuits on 55,177 patients is not tractable; the classical model
is trained on the same subsample so the comparison stays fair, and the
full-data SVM is reported separately.

### Three further measured findings

**1. Adding qubits does nothing.** ROC-AUC stayed flat at ~0.993 from 2 qubits
to 8 while training cost grew six-fold. The model saturates at 2 qubits. This
is why we use 4 — measured, not assumed.

**2. Quantum model size is constant; the SVM's grows.** From 20 to 426 training
patients the SVM grew from 66 stored numbers to 471; on diabetes it reaches
9,081. The VQC stays at exactly 25 in every case.

**3. Threshold placement beat training.** Ten times more epochs changed nothing.
Moving the decision threshold, with no extra training, lifted sensitivity from
0.756 to 0.889.

---

## Clinical decision support

Both models expose the sensitivity/specificity trade-off as a clinician's
choice rather than a hidden default. Breast cancer, quantum model:

| Target sensitivity | Cancers caught | Missed | False alarms |
|---|---|---|---|
| 90% | 38 / 42 | 4 | 1 |
| 95% | 41 / 42 | 1 | 3 |
| 100% | 42 / 42 | 0 | 9 |

### Screening one patient

```python
from src.prediction import predict_patient, blank_patient, format_report

patient = blank_patient("diabetes")           # population medians as defaults
patient.update({"HighBP": 1, "HighChol": 1, "BMI": 38, "Age": 11,
                "GenHlth": 4, "DiffWalk": 1, "PhysActivity": 0})

print(format_report(predict_patient("diabetes", patient)))
```

Returns a risk band (Low / Moderate / High / Very High), both models'
verdicts, whether they agree, and the measurements that moved the result —
computed by genuine counterfactual, not by projecting PCA loadings back.

Bands are derived from the model's own operating points, so each one means
something auditable: which screening policy would have flagged this patient.

**The output is never called a probability of disease.** The VQC's score is a
rescaled expectation value; presenting it as calibrated clinical risk would be
a fabrication.

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

**Datasets.** Two diseases, registered in `src/data_loader.py`; adding a third means one registry entry, not a pipeline edit.

**Wisconsin Breast Cancer (Diagnostic).** Ships inside scikit-learn,
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
