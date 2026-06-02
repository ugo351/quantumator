# Quantumator

**Quantumator** is a computational chemistry pipeline that takes a molecule (SMILES or pre-optimised `.xyz` geometry) and runs a fully automated sequence of quantum-chemistry and molecular-docking calculations:

```
SMILES / XYZ  →  DFT geometry optimisation  →  TD-DFT (UV-Vis)  →  Electron density  →  Molecular docking (Kd)
```

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Command-line (single molecule)](#command-line-single-molecule)
  - [Graphical interface](#graphical-interface)
  - [Python API](#python-api)
- [Pipeline steps in detail](#pipeline-steps-in-detail)
  - [Step 1 – DFT geometry optimisation](#step-1--dft-geometry-optimisation)
  - [Step 2 – TD-DFT UV-Vis spectrum](#step-2--td-dft-uv-vis-spectrum)
  - [Step 3 – Electron density analysis](#step-3--electron-density-analysis)
  - [Step 4 – Molecular docking & Kd estimation](#step-4--molecular-docking--kd-estimation)
- [Output files](#output-files)
- [Configuration reference](#configuration-reference)
- [Limitations](#limitations)
- [License](#license)

---

## Features

| Module | Capability |
|---|---|
| `psi4_calculator_fixed_y.py` | DFT geometry optimisation + TD-DFT (B3LYP/def2-SVP by default, configurable) |
| `electron_density_analysis_fixed.py` | SCF electron density, ESP, dual descriptor, HOMO/LUMO, Mulliken charges → `.cube` files |
| `docking_kd_pipeline.py` | AutoDock Vina docking of DFT-optimised ligands vs a peptide receptor, Kd estimation from ΔG |
| `pipeline_orchestrator.py` | Orchestrates all four steps, CLI & Python API, JSON + TXT reports |
| `pipeline_gui.py` | Tkinter GUI wrapping the full pipeline |

---

## Architecture

```
quantumator/
├── pipeline_orchestrator.py        # Main entry point (CLI + Python API)
├── pipeline_gui.py                 # Optional Tkinter GUI
├── psi4_calculator_fixed_y.py      # DFT / TD-DFT calculations (Psi4)
├── electron_density_analysis_fixed.py  # Electron density & electronic properties
└── docking_kd_pipeline.py          # AutoDock Vina docking pipeline
```

`pipeline_orchestrator.py` dynamically imports the three worker modules at runtime, so each module can also be executed standalone.

---

## Requirements

### Python packages

```
psi4              # quantum chemistry engine
rdkit             # cheminformatics (conda-forge)
meeko             # PDBQT preparation for AutoDock Vina
numpy
pandas
matplotlib
psutil            # optional – memory diagnostics
```

### External binaries

- **AutoDock Vina** ≥ 1.2 (`vina` on PATH, or supply the path with `--vina_exe`)

### Hardware

- ≥ 8 GB RAM (12 GB recommended for medium-sized organic molecules)
- Multi-core CPU (the pipeline auto-detects available cores, capped at 10 by default)

---

## Installation

```bash
# 1. Create a dedicated conda environment
conda create -n quantumator python=3.10
conda activate quantumator

# 2. Install Psi4 and RDKit from conda-forge
conda install -c conda-forge psi4 rdkit

# 3. Install remaining Python packages
pip install meeko numpy pandas matplotlib psutil

# 4. (Optional) Put the Vina binary on your PATH, or note its location
```

---

## Quick Start

### Command-line (single molecule)

```bash
python pipeline_orchestrator.py \
    --smiles "O=C(/C=C/c1ccccc1)NC" \
    --name "CAm" \
    --steps dft tddft density docking \
    --output_dir ./results/CAm \
    --peptide_mol path/to/peptide.mol \
    --vina_exe path/to/vina
```

Run only DFT + TD-DFT (no docking):

```bash
python pipeline_orchestrator.py \
    --smiles "O=C(/C=C/c1ccccc1)NC" \
    --name "CAm" \
    --steps dft tddft \
    --output_dir ./results/CAm
```

Skip optimisation and use a pre-existing `.xyz` file:

```bash
python pipeline_orchestrator.py \
    --smiles "O=C(/C=C/c1ccccc1)NC" \
    --name "CAm" \
    --steps dft tddft density \
    --output_dir ./results/CAm
    # set xyz_file in the PipelineConfig or pass it via the Python API
```

### Graphical interface

```bash
python pipeline_gui.py
```

The GUI exposes all pipeline options and streams the log output in real time.

### Python API

```python
from pipeline_orchestrator import MolecularPipeline, PipelineConfig

cfg = PipelineConfig(
    smiles="O=C(/C=C/c1ccccc1)NC",
    name="CAm",
    output_dir="./results/CAm",
    dft_functional="B3LYP",
    dft_basis="def2-SVP",
    memory="12 GB",
    threads=8,
    run_tddft=True,
    run_density=True,
    run_docking=False,
)

pipe = MolecularPipeline(cfg)
results = pipe.run()   # or pipe.run(steps=["dft", "tddft"])
print(results["tddft"]["lambda_max_nm"], "nm")
```

---

## Pipeline steps in detail

### Step 1 – DFT geometry optimisation

**Module:** `psi4_calculator_fixed_y.py`

- Converts a SMILES to a 3-D geometry (RDKit ETKDG), then runs a full DFT geometry optimisation with Psi4.
- Default level of theory: **B3LYP/def2-SVP**.
- **XYZ bypass:** if you already have an optimised geometry, pass `xyz_file=` in the config and only a single-point energy is computed (skipping the expensive optimisation step).
- Saves the optimised geometry as `<name>.xyz` in the output directory.

### Step 2 – TD-DFT UV-Vis spectrum

**Module:** `psi4_calculator_fixed_y.py`

- Runs a TD-DFT (or TDA) calculation on the DFT-optimised geometry.
- Extracts the λ_max (nm), excitation energy (eV), and oscillator strengths for all transitions.
- Solvent effects can be included via PCM (pass `--solvent`).
- Saves all transitions as `<name>_tddft_transitions.json`.

### Step 3 – Electron density analysis

**Module:** `electron_density_analysis_fixed.py`

- Runs a fresh SCF calculation on the optimised XYZ geometry.
- Generates **Gaussian cube files** (`.cube`) for:
  - Total electron density (`*_electron_density.cube`)
  - Electrostatic potential (`*_electrostatic_potential.cube`)
  - Dual descriptor for reactivity (`*_dual_descriptor.cube`)
  - HOMO / LUMO molecular orbitals
- Extracts electronic properties: HOMO, LUMO, HOMO–LUMO gap, chemical hardness η, chemical potential μ, electrophilicity ω, dipole moment.
- Computes **Mulliken atomic charges** and saves them alongside coordinates in `*_coordinates_and_charges.txt`.
- Results are cached: molecules with existing cube files are skipped automatically on re-run.

### Step 4 – Molecular docking & Kd estimation

**Module:** `docking_kd_pipeline.py`

- Prepares the DFT-optimised ligand (`.xyz` → PDBQT via RDKit + Meeko, preserving DFT geometry).
- Prepares the receptor (`.mol`, `.mol2`, or `.pdb`); large PDB structures (> 500 heavy atoms) are converted directly without MMFF.
- Runs **AutoDock Vina** for each ligand with configurable exhaustiveness and number of poses.
- Converts the binding free energy to a dissociation constant:  
  `Kd = exp(ΔG / RT)` with R = 1.987×10⁻³ kcal/(mol·K), T = 298.15 K
- Exports the best poses as individual `.sdf` files with RMSD convergence analysis.
- Saves results as `docking_results.csv` and `docking_results.png` (ranked bar charts of ΔG and pKd).
- Supports checkpointing: already-docked molecules are skipped on re-run.

---

## Output files

```
<output_dir>/
├── <name>.xyz                          # Optimised geometry
├── <name>_tddft_transitions.json       # All TD-DFT transitions
├── <name>_report.json                  # Full machine-readable report
├── <name>_properties.txt               # Human-readable summary
├── density/
│   ├── <name>_electron_density.cube
│   ├── <name>_electrostatic_potential.cube
│   ├── <name>_dual_descriptor.cube
│   └── <name>_coordinates_and_charges.txt
└── binding/
    ├── docking_results.csv
    ├── docking_results.png
    └── poses/
        └── <name>/
            ├── <name>_pose1.sdf
            ├── <name>_pose2.sdf
            └── ...
```

---

## Configuration reference

All options can be set via `PipelineConfig` (Python API) or the corresponding CLI flag.

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `smiles` | `--smiles` | *required* | Input SMILES string |
| `name` | `--name` | *required* | Molecule name (used for file naming) |
| `output_dir` | `--output_dir` | `./pipeline_results` | Root output directory |
| `dft_functional` | `--dft_functional` | `B3LYP` | DFT functional |
| `dft_basis` | `--dft_basis` | `def2-SVP` | Basis set |
| `memory` | `--memory` | `12 GB` | Memory allocated to Psi4 |
| `threads` | `--threads` | `10` | Number of CPU threads |
| `run_tddft` | `--steps tddft` | `True` | Enable TD-DFT step |
| `tddft_solvent` | `--solvent` | `None` | PCM solvent (e.g. `water`) |
| `tddft_use_tda` | *(API only)* | `True` | Use Tamm–Dancoff approximation |
| `run_density` | `--steps density` | `True` | Enable electron density step |
| `run_docking` | `--steps docking` | `False` | Enable docking step |
| `peptide_mol_file` | `--peptide_mol` | `None` | Receptor file (`.mol`/`.pdb`/`.mol2`) |
| `vina_exe` | `--vina_exe` | `vina` | Path to Vina binary |
| `vina_exhaustiveness` | `--vina_exhaustiveness` | `16` | Vina exhaustiveness (8–64) |
| `vina_n_poses` | `--vina_n_poses` | `100` | Number of docking poses |
| `vina_scoring` | `--vina_scoring` | `vina` | Scoring function (`vina` or `vinardo`) |
| `xyz_file` | *(API only)* | `None` | Pre-existing XYZ — skips optimisation |
| `skip_single_point` | *(API only)* | `False` | Skip SP energy in XYZ-bypass mode |

---

## Limitations

- Kd values from AutoDock Vina are **relative estimates** (reliable for ranking, not for absolute binding affinities). For quantitative Kd, complement with MD + MM-PBSA (GROMACS / OpenMM).
- The peptide/receptor 3-D conformation is taken *as-is* from the input file (rigid docking). Induced-fit effects are not modelled.
- TD-DFT λ_max values are sensitive to the choice of functional and solvent model.
- Cube files can be large (hundreds of MB per molecule at the default grid spacing).
