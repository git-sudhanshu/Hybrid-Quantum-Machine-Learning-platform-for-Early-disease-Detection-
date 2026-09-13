"""
Q-Dx :: Quantum circuit layer
=============================

This module owns EVERYTHING about the quantum circuit itself:
  - the simulator device
  - the feature encoding (classical vector -> quantum state)
  - the trainable variational ansatz
  - the measurement
  - circuit introspection (drawing, depth, parameter count)

It deliberately knows NOTHING about training, datasets or metrics.
Those live in quantum_model.py. Keeping the split means the circuit can be
swapped or re-tuned without touching the training/evaluation code.

------------------------------------------------------------------
DESIGN DECISIONS (defend these to the judges)
------------------------------------------------------------------
1. QUBIT COUNT = 4 (default)
   Angle encoding maps one feature to one qubit, so n_qubits == n_features
   after PCA. A 4-qubit state vector is 2^4 = 16 complex amplitudes, which
   simulates in microseconds. Every extra qubit DOUBLES the simulated state
   and, with parameter-shift gradients, also increases the number of circuit
   evaluations per gradient step. 4 qubits keeps a full training run in
   seconds, which is what makes an honest classical-vs-quantum comparison
   possible inside a 36-hour build.

2. ENCODING = ANGLE ENCODING (RY rotations)
   Each scaled feature x_i becomes the rotation angle of an RY gate on qubit i.
   Why: it needs only n qubits for n features (amplitude encoding needs
   log2(n) qubits but a costly state-preparation circuit that is not
   NISQ-friendly), it is shallow, hardware-native on real devices, and it is
   differentiable, so gradients flow back to the classical optimizer.
   Angles are periodic with period 2*pi, so inputs MUST be bounded --
   see `scale_features_to_angles`.

3. ANSATZ = strongly-entangling-style layers, depth 2-3
   Each layer applies trainable RY and RZ on every qubit (these two generate
   an arbitrary single-qubit rotation up to global phase) and then a ring of
   CNOTs. The CNOTs are what make the model genuinely quantum: without them
   the circuit factorises into 4 independent single-qubit classifiers and is
   no more expressive than a linear model on sin/cos features.
   Depth is kept at 2-3 because deep parameterised circuits hit barren
   plateaus (gradients that vanish exponentially with depth/width) and,
   on real hardware, decoherence.

4. MEASUREMENT = <PauliZ> on qubit 0
   Returns a single real number in [-1, +1]. A trainable bias is added in
   quantum_model.py, giving a scalar decision value for binary
   classification. Because qubit 0 is entangled with all the others, this
   single expectation value still depends on all four features.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

# --------------------------------------------------------------------------
# Defaults. Anything importing this module can override them per-call.
# --------------------------------------------------------------------------
DEFAULT_N_QUBITS = 4
DEFAULT_N_LAYERS = 3
DEFAULT_SIMULATOR = "default.qubit"
ENCODING_NAME = "Angle Encoding (RY)"


# ==========================================================================
# 1. DEVICE
# ==========================================================================
def build_quantum_device(n_qubits: int = DEFAULT_N_QUBITS,
                         simulator: str = DEFAULT_SIMULATOR,
                         shots: int | None = None):
    """Create the quantum device (simulator) the circuit will run on.

    shots=None  -> analytic mode. The simulator returns the exact expectation
                   value. Fast, noiseless, and the right default for training.
    shots=1000  -> sampling mode. Mimics a real quantum computer, where every
                   expectation value must be estimated from repeated
                   measurements ("shots") and therefore carries statistical
                   noise. Useful for a hardware-realism experiment later.

    Swapping to real hardware later means changing ONLY this function
    (e.g. a qml.device pointing at an IBM/Braket backend). Nothing else in
    the pipeline needs to know.
    """
    return qml.device(simulator, wires=n_qubits, shots=shots)


# ==========================================================================
# 2. FEATURE ENCODING
# ==========================================================================
def scale_features_to_angles(X, x_min=None, x_max=None, angle_range=pnp.pi):
    """Map raw feature values onto a safe rotation-angle range.

    Rotation gates are 2*pi periodic, so an unbounded feature would wrap
    around and make two very different patients look identical to the
    circuit. We linearly map each feature to [-angle_range/2, +angle_range/2]
    (default: [-pi/2, +pi/2]), which is a monotonic, non-wrapping window.

    LEAKAGE NOTE: x_min/x_max must be computed on the TRAINING SET ONLY and
    then reused for the test set. quantum_model.py enforces this -- it fits
    the bounds on X_train and passes them in when transforming X_test.

    Returns (X_scaled, x_min, x_max) so the caller can store the bounds.
    """
    X = pnp.array(X, requires_grad=False)
    if x_min is None or x_max is None:
        x_min = pnp.min(X, axis=0)
        x_max = pnp.max(X, axis=0)
    span = pnp.where((x_max - x_min) == 0, 1.0, (x_max - x_min))
    X_unit = (X - x_min) / span                      # -> roughly [0, 1]
    X_scaled = (X_unit - 0.5) * angle_range          # -> [-pi/2, +pi/2]
    return X_scaled, x_min, x_max


def angle_encoding(x, n_qubits: int):
    """Write the classical feature vector into the quantum state.

    feature_0 -> RY(x0) on qubit 0
    feature_1 -> RY(x1) on qubit 1
    ...

    This is the classical->quantum boundary of the whole Q-Dx pipeline.

    `x[..., i]` rather than `x[i]` so that a whole BATCH of samples,
    shape (n_samples, n_features), can be pushed through the circuit in one
    call. PennyLane calls this parameter broadcasting; it is a large speed-up
    over looping sample by sample.
    """
    for i in range(n_qubits):
        qml.RY(x[..., i], wires=i)


# ==========================================================================
# 3. VARIATIONAL ANSATZ
# ==========================================================================
def variational_layer(layer_weights, n_qubits: int):
    """One trainable layer: rotations on every qubit, then entanglement.

    layer_weights has shape (n_qubits, 2) -> [RY angle, RZ angle] per qubit.
    """
    for i in range(n_qubits):
        qml.RY(layer_weights[i, 0], wires=i)
        qml.RZ(layer_weights[i, 1], wires=i)

    # Ring of CNOTs: 0->1, 1->2, 2->3, 3->0.
    # This is what creates entanglement / feature correlations.
    if n_qubits > 1:
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])


# ==========================================================================
# 4. THE QNODE (circuit + measurement)
# ==========================================================================
def create_quantum_circuit(device,
                           n_qubits: int = DEFAULT_N_QUBITS,
                           n_layers: int = DEFAULT_N_LAYERS,
                           diff_method: str = "backprop"):
    """Assemble the full VQC as a differentiable PennyLane QNode.

    Structure:
        |0000>  ->  Angle Encoding  ->  [Variational layer] x n_layers  ->  <Z_0>

    diff_method:
      "backprop"       -> exact, fast gradients through the simulator.
                          Simulator-only; this is our training default.
      "parameter-shift"-> the hardware-compatible rule. Works on real QPUs
                          but needs 2 circuit evaluations per parameter.
                          Set this when doing a hardware-realism run.
    """
    @qml.qnode(device, diff_method=diff_method)
    def circuit(weights, x):
        angle_encoding(x, n_qubits)
        for layer in range(n_layers):
            variational_layer(weights[layer], n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


# ==========================================================================
# 5. PARAMETER INITIALISATION
# ==========================================================================
def initialize_parameters(n_qubits: int = DEFAULT_N_QUBITS,
                          n_layers: int = DEFAULT_N_LAYERS,
                          seed: int = 42,
                          scale: float = 0.1):
    """Small random initial weights, shape (n_layers, n_qubits, 2).

    Why SMALL (scale=0.1) rather than uniform over [0, 2*pi]?
    Near-zero angles keep the initial circuit close to the identity, which
    keeps the output away from the flat regions of the loss landscape
    (barren plateaus) where gradients vanish. Seeded for reproducibility.
    """
    rng = pnp.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=scale, size=(n_layers, n_qubits, 2))
    return pnp.array(weights, requires_grad=True)


def count_parameters(n_qubits: int = DEFAULT_N_QUBITS,
                     n_layers: int = DEFAULT_N_LAYERS) -> int:
    """Trainable circuit parameters (excludes the classical bias)."""
    return n_layers * n_qubits * 2


# ==========================================================================
# 6. INTROSPECTION -- what the frontend developer needs
# ==========================================================================
def _dig(obj, key, default=0):
    """Read `key` from either a dict-like or an attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_quantum_circuit(weights, x_example,
                        n_qubits: int = DEFAULT_N_QUBITS,
                        n_layers: int = DEFAULT_N_LAYERS,
                        simulator: str = DEFAULT_SIMULATOR) -> dict:
    """Return the REAL circuit diagram and metadata. Nothing is faked here --
    the drawing and the depth are extracted from the executed circuit.

    Returns a JSON-serialisable dict:
        diagram_text  : ASCII circuit drawing
        depth         : real circuit depth from qml.specs
        n_gates       : total gate count
        n_qubits, n_layers, n_parameters, encoding, simulator
    """
    dev = build_quantum_device(n_qubits, simulator)
    circuit = create_quantum_circuit(dev, n_qubits, n_layers)

    diagram_text = qml.draw(circuit, max_length=200)(weights, x_example)

    # qml.specs returns a CircuitSpecs object in PennyLane >= 0.43 and a plain
    # dict in older versions. Handle both so the module is not version-brittle.
    specs = qml.specs(circuit)(weights, x_example)
    spec_dict = specs.to_dict() if hasattr(specs, "to_dict") else dict(specs)
    resources = spec_dict.get("resources", specs)
    depth = int(_dig(resources, "depth", 0))
    n_gates = int(_dig(resources, "num_gates", _dig(resources, "gate_types_total", 0)))

    return {
        "diagram_text": diagram_text,
        "depth": depth,
        "n_gates": n_gates,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_parameters": count_parameters(n_qubits, n_layers),
        "encoding": ENCODING_NAME,
        "simulator": simulator,
        "measurement": "expval(PauliZ) on qubit 0",
        "entanglement": "CNOT ring",
    }


def save_circuit_figure(weights, x_example, path: str,
                        n_qubits: int = DEFAULT_N_QUBITS,
                        n_layers: int = DEFAULT_N_LAYERS,
                        simulator: str = DEFAULT_SIMULATOR) -> str:
    """Render the circuit to a PNG for the Streamlit frontend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = build_quantum_device(n_qubits, simulator)
    circuit = create_quantum_circuit(dev, n_qubits, n_layers)
    fig, _ = qml.draw_mpl(circuit, style="pennylane")(weights, x_example)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
