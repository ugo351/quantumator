# quantumator

**Quantic computation software** — a Python toolkit for simulating quantum circuits, gates, and algorithms on classical hardware.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Qubits](#qubits)
  - [Quantum Gates](#quantum-gates)
  - [Quantum Circuits](#quantum-circuits)
- [Examples](#examples)
  - [Bell State (Entanglement)](#bell-state-entanglement)
  - [Quantum Teleportation](#quantum-teleportation)
  - [Grover's Search Algorithm](#grovers-search-algorithm)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Quantumator** is an open-source quantum-computing simulator written in Python. It lets you build and run quantum circuits using a clean, expressive API — no quantum hardware required. Whether you are learning quantum computing fundamentals or prototyping novel algorithms, Quantumator provides the building blocks you need.

---

## Features

- 🧮 **Statevector simulation** — exact simulation of quantum states for arbitrary qubit counts
- 🔗 **Rich gate library** — Hadamard, Pauli (X/Y/Z), CNOT, Toffoli, SWAP, phase gates, and more
- 📊 **Measurement & sampling** — collapse states probabilistically and sample shot-level results
- ⚡ **Composable circuits** — build reusable sub-circuits and append them to larger programs
- 🐍 **Pure Python** — zero non-Python dependencies for the core engine; NumPy for numerics

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ugo351/quantumator.git
cd quantumator

# Install in editable mode (recommended for development)
pip install -e .

# Or install directly
pip install quantumator
```

---

## Quick Start

```python
from quantumator import QuantumCircuit, simulate

# Create a 2-qubit circuit
qc = QuantumCircuit(num_qubits=2)

# Apply a Hadamard gate to qubit 0 → put it in superposition
qc.h(0)

# Apply a CNOT gate (qubit 0 = control, qubit 1 = target) → entangle them
qc.cx(0, 1)

# Measure both qubits
qc.measure_all()

# Simulate 1 000 shots and print the counts
result = simulate(qc, shots=1000)
print(result.counts)
# Example output: {'00': 503, '11': 497}
```

---

## Core Concepts

### Qubits

A **qubit** is the fundamental unit of quantum information. Unlike a classical bit (0 or 1), a qubit can exist in a *superposition* of both states simultaneously until it is measured.

```python
from quantumator import Qubit, statevector

# A qubit starts in the |0⟩ state by default
q = Qubit()
print(statevector(q))
# [1.+0.j, 0.+0.j]  →  |0⟩ with probability 1
```

### Quantum Gates

Gates are unitary operations applied to qubits. Quantumator ships with all standard single- and multi-qubit gates.

```python
from quantumator import QuantumCircuit

qc = QuantumCircuit(num_qubits=1)

qc.h(0)   # Hadamard — creates an equal superposition of |0⟩ and |1⟩
qc.x(0)   # Pauli-X (NOT gate) — flips |0⟩ to |1⟩ and vice versa
qc.z(0)   # Pauli-Z — flips the phase of |1⟩
qc.s(0)   # S gate — applies a π/2 phase shift
qc.t(0)   # T gate — applies a π/4 phase shift
```

| Gate      | Symbol | Description                                  |
|-----------|--------|----------------------------------------------|
| Hadamard  | H      | Creates superposition                        |
| Pauli-X   | X      | Bit flip (quantum NOT)                       |
| Pauli-Y   | Y      | Bit + phase flip                             |
| Pauli-Z   | Z      | Phase flip                                   |
| CNOT      | CX     | Conditional bit flip (two qubits)            |
| Toffoli   | CCX    | Conditional bit flip (three qubits)          |
| SWAP      | SWAP   | Swaps the states of two qubits               |
| S / T     | S, T   | Phase rotation by π/2 and π/4               |

### Quantum Circuits

A `QuantumCircuit` is an ordered sequence of gate operations on a fixed register of qubits.

```python
from quantumator import QuantumCircuit

qc = QuantumCircuit(num_qubits=3)

# Build the circuit
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Visualise as ASCII diagram
print(qc.draw())
# q0: ─H─●───
# q1: ───X─●─
# q2: ─────X─
```

---

## Examples

### Bell State (Entanglement)

A Bell state is the simplest example of quantum entanglement — measuring one qubit instantly determines the other.

```python
from quantumator import QuantumCircuit, simulate

def bell_state():
    qc = QuantumCircuit(num_qubits=2)
    qc.h(0)   # Superposition on qubit 0
    qc.cx(0, 1)  # Entangle qubit 0 and qubit 1
    qc.measure_all()
    return qc

result = simulate(bell_state(), shots=2000)
print(result.counts)
# {'00': ~1000, '11': ~1000}  — never '01' or '10'
```

### Quantum Teleportation

Quantum teleportation transfers the state of one qubit to another using entanglement and classical communication.

```python
from quantumator import QuantumCircuit, simulate

def quantum_teleportation():
    """Teleport the state of qubit 0 to qubit 2."""
    qc = QuantumCircuit(num_qubits=3)

    # Prepare the state to teleport on qubit 0 (|+⟩ state)
    qc.h(0)

    # Create a Bell pair between qubits 1 and 2
    qc.h(1)
    qc.cx(1, 2)

    # Bell measurement on qubits 0 and 1
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0)
    qc.measure(1)

    # Classically controlled corrections on qubit 2
    qc.cx(1, 2)   # apply X if qubit-1 measurement is 1
    qc.cz(0, 2)   # apply Z if qubit-0 measurement is 1

    return qc

result = simulate(quantum_teleportation(), shots=1)
print(result.statevector)
```

### Grover's Search Algorithm

Grover's algorithm finds a marked item in an unsorted list of *N* items in O(√N) steps — a quadratic speedup over classical search.

```python
from quantumator import QuantumCircuit, simulate, grover_oracle

def grovers_algorithm(num_qubits: int, target: int):
    """Run Grover's search for `target` in a 2^num_qubits search space."""
    import math

    qc = QuantumCircuit(num_qubits=num_qubits)

    # Equal superposition over all states
    for q in range(num_qubits):
        qc.h(q)

    # Grover iterations: ≈ (π/4) * √(2^n) rounds
    iterations = int(math.pi / 4 * math.sqrt(2 ** num_qubits))
    for _ in range(iterations):
        # Oracle: flip the phase of the target state
        qc.append(grover_oracle(num_qubits, target))
        # Diffusion operator (inversion about average)
        for q in range(num_qubits):
            qc.h(q)
            qc.x(q)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for q in range(num_qubits):
            qc.x(q)
            qc.h(q)

    qc.measure_all()
    return qc

result = simulate(grovers_algorithm(num_qubits=3, target=5), shots=1000)
print(result.counts)
# {'101': ~1000}  — Grover's algorithm finds target=5 (binary 101) with high probability
```

---

## API Reference

### `QuantumCircuit(num_qubits)`

| Method | Signature | Description |
|---|---|---|
| `h` | `h(qubit)` | Hadamard gate |
| `x` | `x(qubit)` | Pauli-X (NOT) gate |
| `y` | `y(qubit)` | Pauli-Y gate |
| `z` | `z(qubit)` | Pauli-Z gate |
| `s` | `s(qubit)` | S (phase) gate |
| `t` | `t(qubit)` | T gate |
| `cx` | `cx(control, target)` | CNOT gate |
| `cz` | `cz(control, target)` | CZ gate |
| `ccx` | `ccx(c0, c1, target)` | Toffoli gate |
| `swap` | `swap(q0, q1)` | SWAP gate |
| `mcx` | `mcx(controls, target)` | Multi-controlled X |
| `measure` | `measure(qubit)` | Measure a single qubit |
| `measure_all` | `measure_all()` | Measure all qubits |
| `append` | `append(circuit)` | Append a sub-circuit |
| `draw` | `draw()` | Return ASCII diagram string |

### `simulate(circuit, shots=1024)`

Runs the circuit and returns a `Result` object with:

- `result.counts` — `dict[str, int]` mapping bitstring outcomes to shot counts
- `result.statevector` — `numpy.ndarray` of complex amplitudes (before measurement)
- `result.probabilities` — `dict[str, float]` mapping bitstrings to probabilities

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-new-gate
   ```
2. Make your changes and add tests under `tests/`.
3. Run the test suite:
   ```bash
   pytest tests/
   ```
4. Open a Pull Request describing your changes.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
