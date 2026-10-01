#!/usr/bin/env python3
"""
pipeline_gui.py  — v6
======================
Corrections :
  - flush() explicite avant restauration stdout/stderr (derniere ligne sans \n)
  - _log_buf : batch insert dans tkinter (un seul insert+see par tick)
  - LOG_BURST 100 (au lieu de 20)
  - Panneau droit Frame simple (pas de Canvas/scrollbar)
  - Spinner Canvas pilote par _tick() unique
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import multiprocessing as mp
import queue
import sys
import os
import io
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional
import math
import time

MSG_LOG    = "log"
MSG_STEP   = "step"
MSG_RESULT = "result"
MSG_DONE   = "done"
MSG_ERROR  = "error"


class QueueStream(io.TextIOBase):
    def __init__(self, q: queue.Queue, tag: str = "normal"):
        self._q   = q
        self._tag = tag
        self._buf = ""
        self._log_paths = []

    def set_log_path(self, path):
        self.set_log_paths([path] if path else [])

    def set_log_paths(self, paths):
        self.flush()
        self._log_paths = list(dict.fromkeys(Path(path) for path in paths if path))
        for log_path in self._log_paths:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=True)

    def _emit_line(self, line):
        self._q.put((MSG_LOG, (self._tag, line)))
        for log_path in self._log_paths:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(line + "\n")

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._emit_line(line)
        return len(text)

    def flush(self):
        # Vider le buffer même sans \n final (dernière ligne Psi4)
        if self._buf.strip():
            self._emit_line(self._buf)
            self._buf = ""


# =========================================================================
# Fonctions de pipeline — executees dans un processus separe (mp.Process)
# =========================================================================

from utils_paths import OutputLayout, _sanitize_mol_name


def _safe_script_path(path: Path, script_dir: str) -> Path:
    """Empêche le chargement de modules hors du répertoire de l'application."""
    allowed = Path(script_dir).resolve()
    resolved = path.resolve()
    if resolved.suffix != ".py":
        raise ValueError(f"Le chemin de module doit être un fichier .py : {path}")
    try:
        resolved.relative_to(allowed)
    except ValueError:
        raise ValueError(
            f"Chargement refusé : '{resolved}' est hors du répertoire "
            f"'{allowed}'."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Module introuvable : {resolved}")
    return resolved


def _load_batch_file(filepath: str):
    """Charge un CSV ou Excel. Retourne [(name, smiles, xyz_path), ...]."""
    filepath = filepath.strip()
    ext = Path(filepath).suffix.lower()
    rows = []

    if ext in (".xlsx", ".xls"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(filepath, read_only=True,
                                         data_only=True)
            ws = wb.active
            header_skipped = False
            for row in ws.iter_rows(values_only=True):
                if not header_skipped:
                    if (isinstance(row[0], str)
                            and row[0].strip().lower() in ("nom", "name", "molecule",
                                                            "molécule", "id")):
                        header_skipped = True
                        continue
                    header_skipped = True
                if row[0] is None or row[1] is None:
                    continue
                name = _sanitize_mol_name(str(row[0]).strip())
                smiles = str(row[1]).strip()
                xyz = str(row[2]).strip() if len(row) > 2 and row[2] else ""
                if name and smiles:
                    rows.append((name, smiles, xyz))
            wb.close()
        except ImportError:
            raise ImportError("openpyxl requis pour lire les fichiers Excel.\n"
                              "  pip install openpyxl")
    else:
        import csv
        with open(filepath, newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(2048)
            fh.seek(0)
            if "\t" in sample:
                sep = "\t"
            elif ";" in sample:
                sep = ";"
            else:
                sep = ","
            reader = csv.reader(fh, delimiter=sep)
            header_skipped = False
            for row in reader:
                if not row or len(row) < 2:
                    continue
                if not header_skipped:
                    if row[0].strip().lower() in ("nom", "name", "molecule",
                                                   "molécule", "id"):
                        header_skipped = True
                        continue
                    header_skipped = True
                name = _sanitize_mol_name(row[0].strip())
                smiles = row[1].strip()
                xyz = row[2].strip() if len(row) > 2 else ""
                if name and smiles:
                    rows.append((name, smiles, xyz))

    if not rows:
        raise ValueError(f"Aucune molecule trouvee dans {filepath}.\n"
                         "Format attendu : colonne 1 = Nom, colonne 2 = SMILES "
                         "(colonne 3 = chemin XYZ optionnel).")
    return rows


def _write_batch_summary(all_results, output_dir, _log):
    """Ecrit un CSV recapitulatif du batch."""
    import csv
    summary_dir = OutputLayout(output_dir).reports
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "batch_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Nom", "Energie_DFT_Ha", "Lambda_max_nm", "Fosc",
                     "dG_kcal_mol", "Kd_M", "Succes_DFT", "Succes_TDDFT",
                     "Succes_Docking", "Succes_xTB", "Energie_xTB_Ha",
                     "Convergence_xTB", "Dihedre_Cinnamique",
                     "Planarite_Cinnamique"])
        for name, res in all_results.items():
            dft   = res.get("dft", {})
            tddft = res.get("tddft", {})
            dock  = res.get("docking", {})
            xtb   = res.get("xtb", {})
            dihedrals = xtb.get("dihedrals", {})
            first_dihedral = dihedrals.get("dihedral_1", {})
            w.writerow([
                name,
                dft.get("energy_hartree", ""),
                tddft.get("lambda_max_nm", ""),
                tddft.get("max_oscillator_strength", ""),
                dock.get("best_dG_kcalmol", ""),
                dock.get("best_Kd_M", ""),
                dft.get("success", ""),
                tddft.get("success", ""),
                dock.get("success", ""),
                xtb.get("success", ""),
                xtb.get("energy_hartree", ""),
                xtb.get("converged", ""),
                first_dihedral.get("dihedral_deg", ""),
                xtb.get("planarity", ""),
            ])
    _log("info", f"Resume batch sauvegarde : {summary_path}")


def _write_receptor_ensemble_summary(receptor_results, output_dir, _log):
    """Summarize score stability across receptor conformations, not as ΔG_bind."""
    import csv
    import statistics

    molecule_names = sorted({
        molecule_name
        for _receptor_name, results in receptor_results
        for molecule_name in results
    })
    summaries = {}
    rows = []
    for molecule_name in molecule_names:
        entries = []
        for receptor_name, receptor_molecules in receptor_results:
            result = receptor_molecules.get(molecule_name, {})
            docking = result.get("docking", {})
            score = docking.get("best_dG_kcalmol")
            if score is not None:
                entries.append((receptor_name, float(score), result))

        scores = [entry[1] for entry in entries]
        pi_receptors = [
            receptor_name for receptor_name, _score, result in entries
            if result.get("docking", {}).get("pi_contact_geometry", {})
            .get("pi_geometry_candidates", 0) > 0
        ]
        summary = {
            "receptor_models": len(receptor_results),
            "successful_models": len(scores),
            "best_receptor": min(entries, key=lambda entry: entry[1])[0] if entries else "",
            "best_dG_kcalmol": min(scores) if scores else None,
            "mean_best_dG_kcalmol": statistics.fmean(scores) if scores else None,
            "std_best_dG_kcalmol": statistics.pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
            "pi_candidate_receptor_count": len(pi_receptors),
            "pi_candidate_receptors": ";".join(pi_receptors),
        }
        summaries[molecule_name] = summary
        rows.append([molecule_name, *summary.values()])

    summary_dir = OutputLayout(output_dir).reports
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "receptor_ensemble_summary.csv"
    columns = ["Molecule", "Receptor_models", "Successful_models", "Best_receptor",
               "Best_dG_kcal_mol", "Mean_dG_kcal_mol", "Std_dG_kcal_mol",
               "Pi_candidate_receptor_count", "Pi_candidate_receptors"]
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    _log("info", f"Resume ensemble recepteur sauvegarde : {summary_path}")
    return summaries


def _parse_dihedrals(text: str):
    dihedrals = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            indices = [int(value) for value in line.replace(",", " ").split()]
        except ValueError as exc:
            raise ValueError(f"Ligne de diedre {line_number}: indices entiers requis.") from exc
        if len(indices) != 4 or any(index < 1 for index in indices):
            raise ValueError(
                f"Ligne de diedre {line_number}: entrer exactement 4 indices positifs."
            )
        dihedrals.append([index - 1 for index in indices])
    return dihedrals


def _numbered_molecule_image(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES invalide.")
    AllChem.Compute2DCoords(molecule)
    atom_count = molecule.GetNumAtoms()
    width = max(1000, min(1900, atom_count * 20))
    height = max(680, min(1200, int(width * 0.66)))
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.padding = 0.08
    for atom in molecule.GetAtoms():
        options.atomLabels[atom.GetIdx()] = f"{atom.GetSymbol()}{atom.GetIdx() + 1}"
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()

    atom_rows = []
    for atom in molecule.GetAtoms():
        neighbors = ", ".join(
            f"{neighbor.GetIdx() + 1}:{neighbor.GetSymbol()}"
            for neighbor in atom.GetNeighbors()
        )
        atom_rows.append((atom.GetIdx() + 1, atom.GetSymbol(), neighbors))
    return drawer.GetDrawingText(), width, height, atom_rows


def _patch_step_fn(pipe, key, fn, _log, _step):
    """Wrap une etape du pipeline pour le suivi de progression."""
    _labels = {
        "init": "Initialisation",
        "xtb": "Optimisation xTB",
        "dft": "Optimisation DFT",
        "tddft": "TDDFT UV-Vis",
        "density": "Densite electronique",
        "docking": "Docking Vina",
    }
    def wrapped():
        label = _labels.get(key, key)
        _log("info", f"▶  {label}")
        _step(key, "running")
        t0 = time.time()
        try:
            r = fn()
            elapsed = time.time() - t0
            ok = r.get("success", True) if isinstance(r, dict) else True
            _step(key, "done" if ok else "error", f"{elapsed:.0f}s")
            _log("success" if ok else "error",
                 f"✓ {label} termine en {elapsed:.1f}s")
            if isinstance(r, dict):
                _log_step_results(key, r, _log)
            return r
        except Exception as e:
            elapsed = time.time() - t0
            _step(key, "error", str(e)[:50])
            _log("error",
                 f"✗ {label} echoue apres {elapsed:.1f}s : {e}")
            raise
    setattr(pipe, f"_step_{key}", wrapped)


def _log_step_results(key, results, _log):
    """Log un resume des resultats cles de chaque etape."""
    def _v(k, fmt=None):
        v = results.get(k)
        if v is None:
            return
        if fmt and isinstance(v, (int, float)):
            _log("info", f"    {k} = {fmt.format(v)}")
        else:
            _log("info", f"    {k} = {v}")

    if key == "dft":
        _v("energy_hartree", "{:.8f}")
        _v("n_atoms")
        _v("converged")
        _v("n_iterations")
    elif key == "tddft":
        _v("lambda_max_nm", "{:.1f}")
        _v("max_oscillator_strength", "{:.4f}")
        _v("n_states")
        _v("solvent")
    elif key == "density":
        _v("cube_file")
        _v("density_at_origin")
    elif key == "docking":
        _v("best_dG_kcalmol", "{:.2f}")
        _v("best_Kd_M", "{:.2e}")
        _v("n_poses")
        _v("n_runs")
        _v("best_seed")
        _v("mean_best_dG_kcalmol", "{:.2f}")
        _v("std_best_dG_kcalmol", "{:.2f}")
        geometry = results.get("pi_contact_geometry", {})
        if geometry:
            _log("info", f"    poses analysées = {geometry.get('poses_analyzed')}")
            _log("info", f"    géométries aromatiques candidates = "
                          f"{geometry.get('pi_geometry_candidates')}")
            _log("info", f"    rapport géométrique = {geometry.get('details_csv')}")
    elif key == "xtb":
        _v("converged")
        _v("energy_hartree", "{:.8f}")
        _v("n_iterations")
        _v("runtime_s", "{:.1f}")
        _v("process_logs_dir")
        if results.get("crest_conformers_xyz"):
            _log("info", f"    ensemble CREST (XYZ multi-frame) = {results['crest_conformers_xyz']}")
        if results.get("crest_energy_landscape_csv"):
            _log("info", f"    paysage energetique = {results['crest_energy_landscape_csv']}")
        if results.get("crest_energy_scan_csv"):
            _log("info", f"    scan CREST complet = {results['crest_energy_scan_csv']}")
        if results.get("crest_search_candidates_xyz"):
            _log("info", f"    candidats explorés CREST (XYZ) = {results['crest_search_candidates_xyz']}")
        if results.get("crest_search_scan_csv"):
            _log("info", f"    énergies des candidats CREST = {results['crest_search_scan_csv']}")
        if results.get("crest_conformer_landscape_3d_html"):
            _log("info", f"    paysage 3D CREST interactif = {results['crest_conformer_landscape_3d_html']}")
        if results.get("crest_dihedral_energy_absolute_html"):
            _log("info", f"    énergie selon le dièdre absolu = {results['crest_dihedral_energy_absolute_html']}")
        if results.get("crest_search_energy_distribution_csv"):
            _log("info", f"    distribution d'energies CREST (CSV) = {results['crest_search_energy_distribution_csv']}")
        if results.get("crest_refined_xyz_files"):
            _log("info", f"    conformeres raffines exportes = {len(results['crest_refined_xyz_files'])}")
        planarity = results.get("dihedrals", {})
        for name, dihedral in planarity.items():
            if name.startswith("dihedral_"):
                _log("info", f"    {name} = {dihedral.get('dihedral_deg', '?'):.1f} deg")
        if "all_planar" in planarity:
            _log("info", f"    planarity = {'OUI' if planarity['all_planar'] else 'NON'}")
        constrained = results.get("constrained", {})
        if constrained:
            _log("info", "    optimisation contrainte (séparée):")
            _log("info", f"      success = {constrained.get('success')}")
            _log("info", f"      energy_hartree = {constrained.get('energy_hartree')}")
            _log("info", f"      energy_difference_hartree = "
                          f"{constrained.get('energy_difference_hartree')}")
            _log("info", f"      optimized_xyz = {constrained.get('optimized_xyz')}")
    _known = {"success", "error", "energy_hartree", "n_atoms",
              "converged", "n_iterations", "lambda_max_nm",
              "max_oscillator_strength", "n_states", "solvent",
              "cube_file", "density_at_origin", "best_dG_kcalmol",
              "best_Kd_M", "n_poses", "n_runs", "best_seed",
              "mean_best_dG_kcalmol", "std_best_dG_kcalmol",
              "pi_contact_geometry", "runtime_s", "method", "opt_level",
              "optimized_xyz", "stdout_tail", "stderr_tail", "dihedrals",
              "planarity", "returncode", "conformers_refined",
              "crest_conformers_xyz", "crest_energy_landscape_csv",
              "crest_energy_scan_csv",
              "crest_search_candidates_xyz", "crest_search_scan_csv",
              "crest_conformer_landscape_3d_html",
              "crest_dihedral_energy_absolute_html",
              "crest_search_energy_distribution_csv",
              "crest_conformer_xyz_files", "crest_refined_xyz_files",
              "crest_stdout_tail", "process_logs_dir", "constrained"}
    extras = [(k, v) for k, v in results.items()
              if k not in _known and not isinstance(v, (dict, list, bytes))]
    for k, v in extras:
        _log("info", f"    {k} = {v}")


def _unique_dir(base_path: str) -> str:
    """Retourne un chemin unique : si base_path existe deja, ajoute _2, _3, etc."""
    p = Path(base_path)
    if not p.exists():
        return str(p)
    i = 2
    while True:
        candidate = p.parent / f"{p.name}_{i}"
        if not candidate.exists():
            return str(candidate)
        i += 1


def _run_pipeline(params: dict, q, stop_event, script_dir: str):
    """
    Point d'entree du processus fils (mp.Process).
    Tourne dans un processus SEPARE — aucune contention GIL avec la GUI.
    """
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = QueueStream(q, "normal")
    sys.stderr = QueueStream(q, "error")

    active_log: dict[str, Optional[Path]] = {"path": None}
    batch_log_path = None
    base_layout = None

    def _log(tag, text):
        q.put((MSG_LOG, (tag, text)))
        log_paths = [batch_log_path, active_log["path"]]
        for log_path in dict.fromkeys(path for path in log_paths if path is not None):
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"[{tag}] {text}\n")

    def _step(key, state, detail=""):
        q.put((MSG_STEP, (key, state, detail)))

    try:
        p = params
        base_output = p.get("output_dir", "./pipeline_results")
        Path(base_output).mkdir(parents=True, exist_ok=True)
        p["output_dir"] = base_output
        base_layout = OutputLayout(base_output).create()
        batch_log_path = base_layout.logs / "pipeline.log"
        batch_log_path.touch(exist_ok=True)
        active_log["path"] = batch_log_path
        if isinstance(sys.stdout, QueueStream):
            sys.stdout.set_log_paths([batch_log_path])
        if isinstance(sys.stderr, QueueStream):
            sys.stderr.set_log_paths([batch_log_path])

        orch_path = _safe_script_path(
            Path(p.get("orchestrator_path",
                       Path(script_dir) / "pipeline_orchestrator.py")),
            script_dir)
        if not orch_path.exists():
            raise FileNotFoundError(f"pipeline_orchestrator.py introuvable : {orch_path}")

        spec = importlib.util.spec_from_file_location("pipeline_orchestrator", orch_path)
        orch = importlib.util.module_from_spec(spec)
        sys.modules["pipeline_orchestrator"] = orch
        spec.loader.exec_module(orch)

        PipelineConfig    = orch.PipelineConfig
        MolecularPipeline = orch.MolecularPipeline

        # --- Construire la liste de molecules (mode simple ou batch) ---
        batch_file = p.get("batch_file", "").strip()
        molecules = []

        if batch_file:
            molecules = _load_batch_file(batch_file)
            _log("section", "━" * 50)
            _log("info",    f"MODE BATCH — {len(molecules)} molecules")
            _log("section", "━" * 50)
        else:
            molecules = [(_sanitize_mol_name(p["name"]), p["smiles"], p.get("xyz_file", ""))]

        steps = p["steps"]

        # --- Liste de récepteurs (multi-récepteur) ---
        receptor_files_raw = p.get("peptide_mol", [])
        if isinstance(receptor_files_raw, str):
            receptor_files_raw = [receptor_files_raw] if receptor_files_raw else []
        receptor_files = [r for r in receptor_files_raw if r]

        # --- Boîte unifiée si docking multi-récepteur ---
        unified_box = None
        if "docking" in steps and len(receptor_files) > 1:
            _log("section", "━" * 50)
            _log("info", f"MULTI-RECEPTEUR — {len(receptor_files)} récepteurs")
            _log("info", "Calcul de la boîte unifiée...")
            dock_path = _safe_script_path(
                Path(p.get("docking_file") or
                     str(Path(script_dir) / "docking_kd_pipeline.py")),
                script_dir)
            spec_dock = importlib.util.spec_from_file_location(
                "docking_kd_pipeline", dock_path)
            dock_mod = importlib.util.module_from_spec(spec_dock)
            spec_dock.loader.exec_module(dock_mod)
            unified_box = dock_mod.compute_unified_box(receptor_files)
            _log("success", "✓ Boîte unifiée calculée")
            _log("section", "━" * 50)

        # Si pas de docking, utiliser un seul passage sans récepteur
        if not receptor_files or "docking" not in steps:
            receptor_files = [None]

        t0_global = time.time()
        final_results = {}
        receptor_results = []
        molecule_output_dirs = {}
        grand_total = len(receptor_files) * len(molecules)
        global_idx = 0

        for rec_idx, receptor_file in enumerate(receptor_files, 1):
            receptor_name = Path(receptor_file).stem if receptor_file else ""
            multi_rec = len(receptor_files) > 1 or (
                len(receptor_files) == 1 and receptor_files[0] is not None)

            # Sous-dossier par récepteur si multi-récepteur
            if len(receptor_files) > 1:
                rec_output = str(Path(base_output) / f"Batch_{receptor_name}")
                Path(rec_output).mkdir(parents=True, exist_ok=True)
                _log("section", "━" * 50)
                _log("info", f"RECEPTEUR [{rec_idx}/{len(receptor_files)}] : "
                     f"{receptor_name}")
                _log("section", "━" * 50)
            else:
                rec_output = base_output

            all_results = {}

            for mol_idx, (mol_name, mol_smiles, mol_xyz) in enumerate(molecules, 1):
                global_idx += 1
                if stop_event.is_set():
                    _log("warning", "Arret demande par l'utilisateur.")
                    break

                if len(molecules) > 1 or len(receptor_files) > 1:
                    _log("section", "━" * 50)
                    header = f"[{global_idx}/{grand_total}]  {mol_name}"
                    if receptor_name:
                        header += f"  ⇄  {receptor_name}"
                    _log("info", header)
                    _log("section", "━" * 50)

                # Sous-dossier par molécule (suffixe _2, _3… si le nom existe déjà)
                out_dir = _unique_dir(str(Path(rec_output) / mol_name))
                mol_name_actual = Path(out_dir).name
                molecule_output_dirs[mol_name_actual] = Path(out_dir)
                mol_layout = OutputLayout(out_dir).create()
                active_log["path"] = mol_layout.logs / f"{mol_name_actual}_pipeline.log"
                if isinstance(sys.stdout, QueueStream):
                    sys.stdout.set_log_paths([batch_log_path, active_log["path"]])
                if isinstance(sys.stderr, QueueStream):
                    sys.stderr.set_log_paths([batch_log_path, active_log["path"]])

                _log("info", f"  SMILES        : {mol_smiles}")
                _log("info", f"  Etapes        : {', '.join(steps)}")
                _log("info", f"  Fonctionnelle : {p.get('dft_functional', 'B3LYP')}")
                _log("info", f"  Base          : {p.get('dft_basis', 'def2-SVP')}")
                if receptor_name:
                    _log("info", f"  Récepteur     : {receptor_name}")
                if mol_xyz:
                    _log("info", f"  XYZ bypass    : {mol_xyz}")
                _log("info", f"  Sortie        : {out_dir}")
                _log("section", "━" * 50)

                cfg_kwargs = dict(
                    smiles           = mol_smiles,
                    name             = mol_name_actual,
                    output_dir       = out_dir,
                    dft_functional   = p.get("dft_functional", "B3LYP"),
                    dft_basis        = p.get("dft_basis",      "def2-SVP"),
                    memory           = p.get("memory",         "12 GB"),
                    threads          = int(p.get("threads",    10)),
                    run_tddft        = "tddft"   in steps,
                    run_density      = "density" in steps,
                    run_docking      = "docking" in steps,
                    run_xtb          = "xtb" in steps,
                    xtb_exe          = p.get("xtb_exe", "xtb"),
                    xtb_method       = p.get("xtb_method", "GFN2"),
                    xtb_opt_level    = p.get("xtb_opt_level", "tight"),
                    xtb_charge       = int(p.get("xtb_charge", 0)),
                    xtb_multiplicity = int(p.get("xtb_multiplicity", 1)),
                    xtb_threads      = int(p.get("xtb_threads", 10)),
                    run_crest        = p.get("run_crest", False),
                    crest_exe        = p.get("crest_exe", "crest"),
                    crest_n_conformers = int(p.get("crest_n_conformers", 10)),
                    crest_use_wsl    = p.get("crest_use_wsl", False),
                    crest_wsl_exe    = p.get("crest_wsl_exe", "crest"),
                    crest_wsl_xtb_exe = p.get(
                        "crest_wsl_xtb_exe",
                        "/home/ugopasco/miniforge3/envs/crest_xtb/bin/xtb",
                    ),
                    xtb_use_optimized_geometry_for_dft = p.get(
                        "xtb_use_optimized_geometry_for_dft", True),
                    xtb_use_optimized_geometry_for_docking = p.get(
                        "xtb_use_optimized_geometry_for_docking", True),
                    xtb_dihedrals    = p.get("xtb_dihedrals", []),
                    xtb_planarity_threshold_deg = float(
                        p.get("xtb_planarity_threshold_deg", 5.0)),
                    xtb_run_constrained = p.get("xtb_run_constrained", False),
                    xtb_constraint_target_deg = float(
                        p.get("xtb_constraint_target_deg", 0.0)),
                    xtb_constraint_force_constant = float(
                        p.get("xtb_constraint_force_constant", 0.5)),
                    xtb_stop_after = p.get("xtb_stop_after", False),
                    vina_runs       = int(p.get("vina_runs", 3)),
                    vina_seed       = int(p.get("vina_seed", 42)),
                    tddft_solvent    = p.get("tddft_solvent")    or None,
                    tddft_functional = p.get("tddft_functional") or None,
                    tddft_basis      = p.get("tddft_basis")      or None,
                    tddft_use_tda    = p.get("tddft_use_tda", True),
                    tddft_n_states   = int(p.get("tddft_n_states", 4) or 4),
                    density_functional = p.get("density_functional", "B3LYP"),
                    density_basis      = p.get("density_basis",      "def2-SVP"),
                )
                # XYZ bypass
                if mol_xyz:
                    cfg_kwargs["xyz_file"] = mol_xyz
                    cfg_kwargs["skip_single_point"] = p.get("skip_single_point", False)

                if p.get("tddft_timeout"):
                    try:
                        cfg_kwargs["tddft_timeout_minutes"] = int(p["tddft_timeout"])
                    except Exception:
                        pass
                if "docking" in steps and receptor_file:
                    cfg_kwargs.update(
                        peptide_mol_file    = receptor_file,
                        vina_exe            = p.get("vina_exe", "vina"),
                        vina_exhaustiveness = int(p.get("vina_exhaustiveness", 16)),
                        vina_n_poses        = int(p.get("vina_n_poses", 100)),
                        vina_scoring        = p.get("vina_scoring", "vina"),
                    )
                    if unified_box is not None:
                        cfg_kwargs["box_override"] = unified_box
                for attr, skey in [("psi4_calc_file", "psi4_calc_file"),
                                   ("elec_dens_file",  "elec_dens_file"),
                                   ("docking_file",    "docking_file")]:
                    if p.get(skey):
                        cfg_kwargs[attr] = p[skey]

                cfg  = PipelineConfig(**cfg_kwargs)

                _step("init", "running")
                _log("info", "Creation du pipeline et structure 3D...")
                pipe = MolecularPipeline(cfg)
                _step("init", "done", "OK")
                _log("success", "✓ Pipeline initialise")

                # Patcher les etapes principales pour suivi detaille
                for skey in ["xtb", "dft", "tddft", "density", "docking"]:
                    if hasattr(pipe, f"_step_{skey}"):
                        _patch_step_fn(pipe, skey, getattr(pipe, f"_step_{skey}"),
                                       _log, _step)

                for skey in ["xtb", "dft", "tddft", "density", "docking"]:
                    if skey not in steps:
                        _step(skey, "skip")

                t0 = time.time()
                results = pipe.run(steps=steps)
                t_total = time.time() - t0

                _log("section", "━" * 50)
                _log("success", f"{mol_name_actual} termine en {t_total:.1f}s")
                _log("section", "━" * 50)

                all_results[mol_name_actual] = results

                # Reset indicators pour la prochaine molecule
                if global_idx < grand_total:
                    for skey in ["init", "xtb", "dft", "tddft", "density", "docking"]:
                        _step(skey, "pending")

            if stop_event.is_set():
                break

            # Résumé batch par récepteur
            if len(molecules) > 1:
                _write_batch_summary(all_results, rec_output, _log)

            final_results = all_results
            receptor_results.append((receptor_name or f"receptor_{rec_idx}", all_results))

        if "docking" in steps and len(receptor_results) > 1:
            ensemble_summaries = _write_receptor_ensemble_summary(
                receptor_results, base_output, _log
            )
            for molecule_name, summary in ensemble_summaries.items():
                if molecule_name in final_results:
                    final_results[molecule_name]["docking_ensemble"] = summary
                    _log(
                        "info",
                        f"Ensemble {molecule_name}: "
                        f"{summary['successful_models']}/{summary['receptor_models']} "
                        f"récepteurs; ΔG moyen = {summary['mean_best_dG_kcalmol']}; "
                        f"écart-type = {summary['std_best_dG_kcalmol']}; "
                        f"π-candidats = {summary['pi_candidate_receptor_count']}",
                    )

        if base_layout is not None:
            active_log["path"] = base_layout.logs / "pipeline.log"
            if isinstance(sys.stdout, QueueStream):
                sys.stdout.set_log_paths([batch_log_path])
            if isinstance(sys.stderr, QueueStream):
                sys.stderr.set_log_paths([batch_log_path])

        # Résumé global
        elapsed_global = time.time() - t0_global
        if grand_total > 1:
            _log("section", "━" * 50)
            _log("success", f"BATCH TERMINE — {global_idx}/{grand_total} "
                   f"combinaisons en {elapsed_global:.1f}s")
            _log("section", "━" * 50)

        # Copier le fichier batch source dans le dossier de resultats
        if batch_file:
            try:
                import shutil
                src = Path(batch_file)
                if base_layout is not None:
                    dst_dir = base_layout.reports
                elif molecule_output_dirs:
                    last_molecule_dir = next(reversed(molecule_output_dirs.values()))
                    dst_dir = OutputLayout(last_molecule_dir).reports
                else:
                    dst_dir = Path(base_output)
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                if src.exists() and src.resolve() != dst.resolve():
                    shutil.copy2(str(src), str(dst))
                    _log("info", f"Copie du datasheet : {dst}")
            except Exception as e:
                _log("warning", f"Impossible de copier le datasheet : {e}")

        # Envoyer le dernier résultat pour l'affichage
        last_mol = list(final_results.keys())[-1] if final_results else None
        q.put((MSG_RESULT, final_results.get(last_mol, {}) if last_mol else {}))

    except Exception as exc:
        q.put((MSG_ERROR, f"{exc}\n\n{traceback.format_exc()}"))
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if isinstance(sys.stdout, QueueStream):
            sys.stdout.set_log_path(None)
        if isinstance(sys.stderr, QueueStream):
            sys.stderr.set_log_path(None)
        sys.stdout = old_out
        sys.stderr = old_err
        q.put((MSG_DONE, None))


COLORS = {
    "bg":           "#1e1e2e",
    "panel":        "#2a2a3e",
    "card":         "#313149",
    "accent":       "#7c6af7",
    "accent_hover": "#9a8bff",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "error":        "#f38ba8",
    "info":         "#89dceb",
    "text":         "#cdd6f4",
    "text_dim":     "#6c7086",
    "border":       "#45475a",
    "step_pending": "#6c7086",
    "step_running": "#f9e2af",
    "step_done":    "#a6e3a1",
    "step_error":   "#f38ba8",
    "step_skip":    "#45475a",
}
FONT_MONO   = ("Consolas", 10)
FONT_NORMAL = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_SMALL  = ("Segoe UI", 9)


class SpinnerWidget(tk.Canvas):
    """Fancy atomic-orbital spinner: 3 tilted elliptical orbits + pulsing core."""
    SIZE       = 36
    ORBIT_R    = 14      # semi-major axis
    N_ORBITS   = 3       # number of electron orbits
    N_TRAIL    = 6       # trail dots per orbit
    CORE_R_MIN = 2.0
    CORE_R_MAX = 4.0
    # Accent palette (purple → cyan → green) for each orbit
    ORBIT_COLORS = [
        (0x7c, 0x6a, 0xf7),   # accent purple
        (0x89, 0xdc, 0xeb),   # info cyan
        (0xa6, 0xe3, 0xa1),   # success green
    ]

    def __init__(self, parent, **kw):
        kw.setdefault("width",  self.SIZE)
        kw.setdefault("height", self.SIZE)
        kw.setdefault("bg",     COLORS["bg"])
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, **kw)
        self._frame    = 0
        self._visible  = False

    def show(self):
        self._visible = True
        self._draw()

    def hide(self):
        self._visible = False
        self.delete("all")

    def advance(self):
        if not self._visible:
            return
        self._frame += 1
        self._draw()

    def _draw(self):
        self.delete("all")
        cx = cy = self.SIZE / 2
        t = self._frame * 0.10   # continuous time parameter

        # --- Pulsing core glow (two layers) ---
        pulse = 0.5 + 0.5 * math.sin(t * 2.5)
        core_r = self.CORE_R_MIN + (self.CORE_R_MAX - self.CORE_R_MIN) * pulse

        # outer glow
        gr = core_r * 2.2
        glow_alpha = 0.15 + 0.10 * pulse
        rg = int(0x7c * glow_alpha + 0x2a * (1 - glow_alpha))
        gg = int(0x6a * glow_alpha + 0x2a * (1 - glow_alpha))
        bg = int(0xf7 * glow_alpha + 0x3e * (1 - glow_alpha))
        self.create_oval(cx - gr, cy - gr, cx + gr, cy + gr,
                         fill=f"#{rg:02x}{gg:02x}{bg:02x}", outline="")

        # bright core
        ca = 0.6 + 0.4 * pulse
        rc = int(0x7c * ca + 0xcf * (1 - ca))
        gc = int(0x6a * ca + 0xd0 * (1 - ca))
        bc = int(0xf7 * ca + 0xf4 * (1 - ca))
        self.create_oval(cx - core_r, cy - core_r, cx + core_r, cy + core_r,
                         fill=f"#{rc:02x}{gc:02x}{bc:02x}", outline="")

        # --- Orbiting electron trails ---
        tilt_angles = [0, 60, 120]   # degrees – tilt of orbit plane
        speed_mults = [1.0, -0.75, 0.55]  # direction & speed variety

        for k in range(self.N_ORBITS):
            tilt = math.radians(tilt_angles[k])
            base_r, base_g, base_b = self.ORBIT_COLORS[k]
            speed = speed_mults[k]

            for j in range(self.N_TRAIL):
                # position along orbit (trail fades behind)
                angle = t * speed * 3.0 - j * 0.28
                # ellipse: semi-major = ORBIT_R, semi-minor = ORBIT_R * 0.38
                ex = self.ORBIT_R * math.cos(angle)
                ey = self.ORBIT_R * 0.38 * math.sin(angle)
                # rotate by tilt
                rx = ex * math.cos(tilt) - ey * math.sin(tilt)
                ry = ex * math.sin(tilt) + ey * math.cos(tilt)
                sx, sy = cx + rx, cy + ry

                # fade: head bright, tail dim
                alpha = (self.N_TRAIL - j) / self.N_TRAIL
                alpha = alpha ** 1.5   # sharper falloff
                dr = int(base_r * alpha + 0x1e * (1 - alpha))
                dg = int(base_g * alpha + 0x1e * (1 - alpha))
                db = int(base_b * alpha + 0x2e * (1 - alpha))
                dot_r = max(1.0, 2.5 * alpha)
                self.create_oval(sx - dot_r, sy - dot_r,
                                 sx + dot_r, sy + dot_r,
                                 fill=f"#{dr:02x}{dg:02x}{db:02x}", outline="")


class ParamEntry(tk.Frame):
    def __init__(self, parent, label: str, default: str = "",
                 width: int = 28, tooltip: str = "", **kwargs):
        super().__init__(parent, bg=COLORS["panel"], **kwargs)
        if label:
            tk.Label(self, text=label, font=FONT_SMALL,
                     bg=COLORS["panel"], fg=COLORS["text_dim"],
                     anchor="w").pack(anchor="w")
        self.var = tk.StringVar(value=default)
        e = tk.Entry(self, textvariable=self.var, font=FONT_MONO,
                     width=width, bg=COLORS["card"], fg=COLORS["text"],
                     insertbackground=COLORS["text"], relief="flat", bd=4,
                     highlightthickness=1,
                     highlightcolor=COLORS["accent"],
                     highlightbackground=COLORS["border"])
        e.pack(fill="x")
        if tooltip:
            self._add_tooltip(e, tooltip)

    def get(self) -> str: return self.var.get().strip()
    def set(self, v: str): self.var.set(v)

    def _add_tooltip(self, widget, text):
        tip = None
        def enter(e):
            nonlocal tip
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            tk.Label(tip, text=text, font=FONT_SMALL,
                     bg="#313149", fg=COLORS["text"],
                     relief="flat", padx=6, pady=4,
                     wraplength=280).pack()
        def leave(e):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)


class StepIndicator(tk.Frame):
    ICONS = {"pending": "○", "running": "◉", "done": "✓",
             "error": "✗", "skip": "—"}
    CKEYS = {"pending": "step_pending", "running": "step_running",
             "done": "step_done", "error": "step_error", "skip": "step_skip"}

    def __init__(self, parent, label: str, **kwargs):
        super().__init__(parent, bg=COLORS["panel"], **kwargs)
        self._icon = tk.Label(self, text="○", font=("Segoe UI", 13), width=2,
                              bg=COLORS["panel"], fg=COLORS["step_pending"])
        self._icon.pack(side="left")
        self._name = tk.Label(self, text=label, font=FONT_NORMAL,
                              bg=COLORS["panel"], fg=COLORS["step_pending"])
        self._name.pack(side="left", padx=(4, 12))
        self._detail = tk.Label(self, text="", font=FONT_SMALL,
                                bg=COLORS["panel"], fg=COLORS["text_dim"])
        self._detail.pack(side="left")

    def set_state(self, state: str, detail: str = ""):
        col = COLORS.get(self.CKEYS.get(state, "step_pending"), COLORS["text_dim"])
        self._icon.config(text=self.ICONS.get(state, "?"), fg=col)
        self._name.config(fg=col)
        self._detail.config(text=f"— {detail[:40]}" if detail else "")


class PipelineGUI(tk.Tk):

    TICK_MS    = 50
    LOG_BURST  = 500    # messages max depiles par tick
    LOG_MAX    = 8000   # lignes max dans la console (trim oldest)

    def __init__(self):
        super().__init__()
        self.title("Molecular Pipeline — SMILES > DFT > TDDFT > Density > Docking")
        self.configure(bg=COLORS["bg"])
        self.geometry("1360x900")
        self.minsize(1120, 720)

        self._queue      = mp.Queue()
        self._stop_event = mp.Event()
        self._process: Optional[mp.Process] = None
        self._running = False
        self._log_buf: list = []   # lignes (tag, texte) a inserer en batch

        _s = ttk.Style(self)
        _s.theme_use("clam")
        _s.configure("Vertical.TScrollbar",
                      background=COLORS["border"],
                      troughcolor=COLORS["panel"],
                      arrowcolor=COLORS["text_dim"],
                      borderwidth=0, relief="flat")

        self._build_ui()
        self._tick()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=COLORS["bg"])
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="Molecular Pipeline", font=FONT_TITLE,
                 bg=COLORS["bg"], fg=COLORS["accent"]).pack(side="left")
        tk.Label(hdr, text="  SMILES > DFT > TDDFT > Densite > Docking",
                 font=FONT_NORMAL, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(side="left")

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=10)

        left = tk.Frame(body, bg=COLORS["bg"], width=500)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        self._build_params(left)

        right = tk.Frame(body, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._build_right(right)

    def _build_params(self, parent):
        canvas = tk.Canvas(parent, bg=COLORS["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=COLORS["bg"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind("<Enter>", lambda e: canvas.bind_all(
            "<MouseWheel>",
            lambda ev: canvas.yview_scroll(-1 * (ev.delta // 120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        f = inner

        self._sep(f, "Molecule")
        self.e_smiles = ParamEntry(f, "SMILES *", tooltip="SMILES canonique (obligatoire sauf mode batch)")
        self.e_smiles.pack(fill="x", pady=2)
        self.e_name = ParamEntry(f, "Nom *", default="MOL")
        self.e_name.pack(fill="x", pady=2)

        self._sep(f, "Mode batch (CSV / Excel)")
        self.e_batch = self._file_row(f, "Fichier batch (col1=Nom, col2=SMILES)",
                                       self._browse_batch)
        tk.Label(f, text="  Si renseigne, SMILES/Nom ci-dessus sont ignores",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")

        self._sep(f, "Etapes")
        sf = tk.Frame(f, bg=COLORS["panel"])
        sf.pack(fill="x", pady=4)
        self._step_vars = {}
        self._step_checkbuttons = {}
        self._steps_before_xtb_stop = None
        for key, lbl, default, disabled in [
            ("xtb",     "Optimisation xTB",          True,  False),
            ("dft",     "Optimisation DFT",          True,  False),
            ("tddft",   "TDDFT — lambda max UV-Vis",  False, False),
            ("density", "Densite electronique",       False, False),
            ("docking", "Docking / Kd (Vina)",        False, False),
        ]:
            v = tk.BooleanVar(value=default)
            self._step_vars[key] = v
            checkbox = tk.Checkbutton(
                sf, text=lbl, variable=v, font=FONT_NORMAL,
                bg=COLORS["panel"], fg=COLORS["text"],
                activebackground=COLORS["panel"],
                activeforeground=COLORS["accent"],
                selectcolor=COLORS["card"],
                state="disabled" if disabled else "normal",
            )
            if key == "xtb":
                checkbox.config(command=self._on_xtb_step_toggle)
            else:
                checkbox.config(command=self._on_step_selection_changed)
            checkbox.pack(anchor="w", pady=1)
            self._step_checkbuttons[key] = checkbox

        self._settings_host = tk.Frame(f, bg=COLORS["bg"])
        self._settings_host.pack(fill="x")
        self._step_panels = {}

        def make_panel(key, title):
            panel = tk.Frame(self._settings_host, bg=COLORS["bg"])
            self._step_panels[key] = panel
            self._sep(panel, title)
            return panel

        geometry_panel = make_panel("geometry", "Structure initiale")
        self.e_xyz = self._file_row(
            geometry_panel, "Fichier XYZ d'entrée", self._browse_xyz
        )
        tk.Label(geometry_panel, text="  Utilisé comme géométrie de départ.",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")

        xtb_panel = make_panel("xtb", "Paramètres xTB et analyse géométrique")
        self.e_xtb_exe = self._file_row(
            xtb_panel, "Executable xtb.exe", self._browse_xtb, default="xtb"
        )
        row_xtb = tk.Frame(xtb_panel, bg=COLORS["bg"]); row_xtb.pack(fill="x", pady=2)
        self.e_xtb_method = ParamEntry(row_xtb, "Methode", "GFN2", 8)
        self.e_xtb_method.pack(side="left", padx=(0, 6))
        self.e_xtb_level = ParamEntry(row_xtb, "Optimisation", "tight", 10)
        self.e_xtb_level.pack(side="left", padx=(0, 6))
        self.e_xtb_charge = ParamEntry(row_xtb, "Charge", "0", 6)
        self.e_xtb_charge.pack(side="left", padx=(0, 6))
        self.e_xtb_mult = ParamEntry(row_xtb, "Multiplicite", "1", 8)
        self.e_xtb_mult.pack(side="left")
        self.e_xtb_threads = ParamEntry(xtb_panel, "Threads xTB", "10", 8)
        self.e_xtb_threads.pack(anchor="w", pady=2)
        self.crest_var = tk.BooleanVar(value=False)
        tk.Checkbutton(xtb_panel, text="Recherche de conformeres CREST", variable=self.crest_var,
                   font=FONT_NORMAL, bg=COLORS["bg"], fg=COLORS["text"],
                   activebackground=COLORS["bg"], selectcolor=COLORS["card"]
                   ).pack(anchor="w")
        self.crest_wsl_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            xtb_panel, text="Lancer CREST via WSL", variable=self.crest_wsl_var,
            font=FONT_NORMAL, bg=COLORS["bg"], fg=COLORS["text"],
            activebackground=COLORS["bg"], selectcolor=COLORS["card"],
        ).pack(anchor="w")
        self.e_crest_wsl_exe = ParamEntry(
            xtb_panel, "Executable CREST (chemin Linux)",
            "/mnt/c/Users/ugo.pasco/bin/crest", width=52,
        )
        self.e_crest_wsl_exe.pack(fill="x", pady=2)
        self.e_crest_wsl_xtb_exe = ParamEntry(
            xtb_panel, "Executable xTB dans WSL",
            "/home/ugopasco/miniforge3/envs/crest_xtb/bin/xtb", width=52,
        )
        self.e_crest_wsl_xtb_exe.pack(fill="x", pady=2)
        self.e_crest_exe = self._file_row(xtb_panel, "Executable crest", self._browse_crest,
                           default="crest")
        self.e_crest_n = ParamEntry(xtb_panel, "Nombre de conformeres CREST", "10", 8)
        self.e_crest_n.pack(anchor="w", pady=2)
        self.xtb_stop_after_var = tk.BooleanVar(value=True)
        self.xtb_stop_after_check = tk.Checkbutton(
            xtb_panel,
            text="Arrêter le pipeline après xTB (ne pas lancer DFT ni les étapes suivantes)",
            variable=self.xtb_stop_after_var,
            font=FONT_NORMAL, bg=COLORS["bg"], fg=COLORS["text"],
            activebackground=COLORS["bg"], selectcolor=COLORS["card"],
            command=self._on_xtb_stop_after_toggle,
        )
        self.xtb_stop_after_check.pack(anchor="w", pady=(4, 0))

        tk.Label(xtb_panel,
             text="Par défaut, la géométrie xTB est raffinée par DFT.",
                 font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["text_dim"]
                 ).pack(anchor="w", pady=(3, 2))
        tk.Button(xtb_panel, text="Afficher molecule et indices",
              font=FONT_NORMAL, bg=COLORS["card"], fg=COLORS["text"],
              activebackground=COLORS["border"], relief="flat",
              command=self._show_atom_index_preview
              ).pack(anchor="w", pady=(2, 5))
        tk.Label(xtb_panel, text="Indices des dièdres, un par ligne (atomes numerotes a partir de 1)",
             font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["text_dim"]
             ).pack(anchor="w")
        self.xtb_dihedrals_text = tk.Text(xtb_panel, height=3, width=34, font=FONT_MONO,
                          bg=COLORS["card"], fg=COLORS["text"],
                          insertbackground=COLORS["text"], relief="flat")
        self.xtb_dihedrals_text.pack(fill="x", pady=2)
        self.xtb_dihedrals_text.insert("1.0", "")
        self.e_xtb_threshold = ParamEntry(xtb_panel, "Seuil de planarite (degres)", "5.0", 8)
        self.e_xtb_threshold.pack(anchor="w", pady=2)
        self.xtb_constrain_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            xtb_panel,
            text="Tester aussi une optimisation plane contrainte (séparée)",
            variable=self.xtb_constrain_var,
            font=FONT_NORMAL, bg=COLORS["bg"], fg=COLORS["text"],
            activebackground=COLORS["bg"], selectcolor=COLORS["card"],
        ).pack(anchor="w", pady=(4, 0))
        row_constraint = tk.Frame(xtb_panel, bg=COLORS["bg"])
        row_constraint.pack(fill="x", pady=2)
        self.e_xtb_target = ParamEntry(row_constraint, "Cible (deg)", "0", 8)
        self.e_xtb_target.pack(side="left", padx=(0, 6))
        self.e_xtb_force = ParamEntry(row_constraint, "Constante de force", "0.5", 12)
        self.e_xtb_force.pack(side="left")
        tk.Label(
            xtb_panel,
            text="Le calcul libre reste intact; cibles planes: 0 ou 180 deg.",
            font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["text_dim"],
        ).pack(anchor="w")

        dft_panel = make_panel("dft", "Paramètres DFT")
        self.skip_sp_var = tk.BooleanVar(value=False)
        tk.Checkbutton(dft_panel, text="Skip single-point (bypass sans calcul d'energie)",
                       variable=self.skip_sp_var, font=FONT_NORMAL,
                   bg=COLORS["bg"], fg=COLORS["text"],
                   activebackground=COLORS["bg"],
                       activeforeground=COLORS["accent"],
                       selectcolor=COLORS["card"]
                       ).pack(anchor="w", pady=(2, 0))
        tk.Label(dft_panel, text="  Option utile avec un XYZ fourni",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")
        r = tk.Frame(dft_panel, bg=COLORS["bg"]); r.pack(fill="x", pady=2)
        self.e_func  = ParamEntry(r, "Fonctionnelle", "B3LYP",    12,
                                  "Ex: B3LYP, CAM-B3LYP, PBE0")
        self.e_func.pack(side="left", padx=(0, 6))
        self.e_basis = ParamEntry(r, "Base",           "def2-SVP", 12,
                                  "Ex: def2-SVP, def2-TZVP, 6-31G*")
        self.e_basis.pack(side="left")
        r2 = tk.Frame(dft_panel, bg=COLORS["bg"]); r2.pack(fill="x", pady=2)
        self.e_memory  = ParamEntry(r2, "Memoire Psi4", "12 GB", 10)
        self.e_memory.pack(side="left", padx=(0, 6))
        self.e_threads = ParamEntry(r2, "Threads", "10", 6)
        self.e_threads.pack(side="left")
        self.e_psi4_file = self._file_row(
            dft_panel, "Module Psi4",
            lambda: self._browse_py(self.e_psi4_file),
        )

        tddft_panel = make_panel("tddft", "Paramètres TDDFT UV-Vis")
        self.e_solvent = ParamEntry(tddft_panel, "Solvant", default="ACN",
                                    tooltip="Ex: ACN, methanol, dmso, thf, cyclohexane — vide=gaz")
        self.e_solvent.pack(fill="x", pady=2)
        # Option TDA (Tamm-Dancoff Approximation) — cochée par défaut (~2x plus rapide)
        tda_frame = tk.Frame(tddft_panel, bg=COLORS["bg"]); tda_frame.pack(fill="x", pady=2)
        self.tda_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tda_frame, text="TDA (Tamm-Dancoff) — plus rapide",
                       variable=self.tda_var, font=FONT_NORMAL,
                       bg=COLORS["bg"], fg=COLORS["text"],
                       activebackground=COLORS["bg"],
                       activeforeground=COLORS["accent"],
                       selectcolor=COLORS["card"]).pack(anchor="w")
        r3 = tk.Frame(tddft_panel, bg=COLORS["bg"]); r3.pack(fill="x", pady=2)
        self.e_tddft_func = ParamEntry(r3, "Fonctionnelle", "B3LYP", 14,
                                       "Vide = meme que DFT")
        self.e_tddft_func.pack(side="left", padx=(0, 6))
        self.e_tddft_basis = ParamEntry(r3, "Base", "", 12,
                                        "Vide = meme que DFT")
        self.e_tddft_basis.pack(side="left", padx=(0, 6))
        self.e_tddft_timeout = ParamEntry(r3, "Timeout", "", 6,
                                          "Minutes, vide = adaptatif")
        self.e_tddft_timeout.pack(side="left", padx=(0, 6))
        self.e_tddft_nstates = ParamEntry(r3, "États", "4", 4,
                                          "Nombre d'états excités TD-DFT (4 = rapide, 10 = complet)")
        self.e_tddft_nstates.pack(side="left")

        density_panel = make_panel("density", "Paramètres densité électronique")
        r_dens = tk.Frame(density_panel, bg=COLORS["bg"]); r_dens.pack(fill="x", pady=2)
        self.e_dens_func = ParamEntry(r_dens, "Fonctionnelle", "B3LYP", 14,
                                      "Fonctionnelle pour calcul SCF densite")
        self.e_dens_func.pack(side="left", padx=(0, 6))
        self.e_dens_basis = ParamEntry(r_dens, "Base", "def2-SVP", 12,
                                       "Base pour calcul SCF densite")
        self.e_dens_basis.pack(side="left")
        self.e_dens_file = self._file_row(
            density_panel, "Module analyse densité",
            lambda: self._browse_py(self.e_dens_file),
        )

        docking_panel = make_panel("docking", "Docking AutoDock Vina")
        _default_vina = str(Path(__file__).parent / "vina_1.2.7_win.exe")

        # Multi-récepteur : liste de fichiers
        tk.Label(docking_panel, text="Fichier(s) recepteur (.mol / .pdb)",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(anchor="w", pady=(4, 0))
        r_recep = tk.Frame(docking_panel, bg=COLORS["bg"]); r_recep.pack(fill="x", pady=2)
        self.lb_receptors = tk.Listbox(r_recep, height=4,
                                       bg=COLORS["card"], fg=COLORS["text"],
                                       selectbackground=COLORS["accent"],
                                       font=FONT_SMALL, relief="flat")
        self.lb_receptors.pack(side="left", fill="x", expand=True)
        r_recep_btns = tk.Frame(r_recep, bg=COLORS["bg"])
        r_recep_btns.pack(side="left", padx=(4, 0))
        tk.Button(r_recep_btns, text="+", font=FONT_BOLD, width=3,
                  bg=COLORS["card"], fg=COLORS["text"],
                  activebackground=COLORS["accent"],
                  relief="flat", command=self._add_receptors).pack(pady=(0, 2))
        tk.Button(r_recep_btns, text="−", font=FONT_BOLD, width=3,
                  bg=COLORS["card"], fg=COLORS["text"],
                  activebackground=COLORS["accent"],
                  relief="flat", command=self._remove_receptor).pack()

        self.e_vina_exe = self._file_row(docking_panel, "Executable Vina", self._browse_vina,
                                          default=_default_vina)
        r4 = tk.Frame(docking_panel, bg=COLORS["bg"]); r4.pack(fill="x", pady=2)
        self.e_vina_exh     = ParamEntry(r4, "Exhaustivite", "34",   6)
        self.e_vina_exh.pack(side="left", padx=(0, 6))
        self.e_vina_poses   = ParamEntry(r4, "Nb poses",     "1000",  6)
        self.e_vina_poses.pack(side="left", padx=(0, 6))
        self.e_vina_scoring = ParamEntry(r4, "Scoring",      "vinardo", 8)
        self.e_vina_scoring.pack(side="left")
        self.e_vina_runs = ParamEntry(
            docking_panel, "Recherches globales indépendantes", "3", 8,
            "Seeds Vina différentes; le peptide entier reste dans la boîte.",
        )
        self.e_vina_runs.pack(anchor="w", pady=2)
        self.e_dock_file = self._file_row(
            docking_panel, "Module docking Vina",
            lambda: self._browse_py(self.e_dock_file),
        )

        self._panel_order = ("geometry", "xtb", "dft", "tddft", "density", "docking")
        if self.xtb_stop_after_var.get():
            self._on_xtb_stop_after_toggle()
        else:
            self._refresh_step_panels()

        self._sep(f, "Dossier de sortie")
        self.e_outdir = self._file_row(f, "", self._browse_outdir,
                                       default=r"C:\Users\ugo.pasco\Documents\2A\Data finale MD\Quantumator\Results")

    def _show_atom_index_preview(self):
        batch_file = self.e_batch.get().strip()
        batch_dir = Path(batch_file).resolve().parent if batch_file else None
        if batch_file:
            try:
                molecules = _load_batch_file(batch_file)
            except Exception as exc:
                messagebox.showerror("Indices atomiques", str(exc))
                return
        else:
            molecules = [(self.e_name.get() or "Molecule",
                          self.e_smiles.get().strip(),
                          self.e_xyz.get().strip())]

        window = tk.Toplevel(self)
        window.title("Structure et indices atomiques")
        window.geometry("1450x850")
        window.minsize(900, 600)

        selector_frame = tk.Frame(window, bg=COLORS["bg"])
        selector_frame.pack(fill="x", padx=8, pady=8)
        tk.Label(selector_frame, text="Molecule", font=FONT_BOLD,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(side="left", padx=(0, 8))
        choices = [f"{name} | {smiles[:55]}" for name, smiles, _ in molecules]
        selected = ttk.Combobox(selector_frame, values=choices,
                                 state="readonly", width=78)
        selected.pack(side="left", fill="x", expand=True)
        selected.current(0)

        body = tk.Frame(window, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        image_panel = tk.Frame(body, bg=COLORS["panel"])
        image_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        canvas = tk.Canvas(image_panel, bg="white", highlightthickness=0)
        image_ybar = ttk.Scrollbar(image_panel, orient="vertical", command=canvas.yview)
        image_xbar = ttk.Scrollbar(image_panel, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=image_ybar.set,
                         xscrollcommand=image_xbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        image_ybar.grid(row=0, column=1, sticky="ns")
        image_xbar.grid(row=1, column=0, sticky="ew")
        image_panel.rowconfigure(0, weight=1)
        image_panel.columnconfigure(0, weight=1)

        table_panel = tk.Frame(body, bg=COLORS["panel"], width=410)
        table_panel.pack(side="right", fill="y")
        table_panel.pack_propagate(False)
        tree = ttk.Treeview(table_panel, columns=("index", "atom", "neighbors"),
                            show="headings", height=28)
        tree.heading("index", text="Indice")
        tree.heading("atom", text="Atome")
        tree.heading("neighbors", text="Voisins (indice:atome)")
        tree.column("index", width=58, anchor="center", stretch=False)
        tree.column("atom", width=70, anchor="center", stretch=False)
        tree.column("neighbors", width=255, anchor="w")
        tree_ybar = ttk.Scrollbar(table_panel, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_ybar.set)
        tree.pack(side="left", fill="both", expand=True)
        tree_ybar.pack(side="right", fill="y")

        def render_selected(_event=None):
            molecule_index = selected.current()
            if molecule_index < 0:
                molecule_index = 0
            name, smiles, xyz_file = molecules[molecule_index]
            canvas.delete("all")
            tree.delete(*tree.get_children())
            window._atom_preview_image = None

            if xyz_file:
                xyz_path = Path(xyz_file)
                if not xyz_path.is_absolute() and batch_dir:
                    xyz_path = batch_dir / xyz_path
                try:
                    xyz_lines = [line for line in xyz_path.read_text(
                        encoding="utf-8").splitlines() if line.strip()]
                    atom_count = int(xyz_lines[0])
                    atom_lines = xyz_lines[2:2 + atom_count]
                    if len(atom_lines) != atom_count:
                        raise ValueError("Fichier XYZ incomplet.")
                    for index, atom_line in enumerate(atom_lines, 1):
                        symbol = atom_line.split()[0]
                        tree.insert("", "end", values=(index, symbol, ""))
                    canvas.create_text(
                        24, 24, anchor="nw", fill="#222222",
                        font=FONT_NORMAL,
                        text=(f"{name}\nIndices basés sur l'ordre des atomes "
                              f"dans le fichier XYZ ({atom_count} atomes).\n\n"
                              "Le XYZ ne contient pas de connectivité pour dessiner "
                              "la molécule sans réattribuer les atomes."),
                    )
                    canvas.configure(scrollregion=canvas.bbox("all"))
                except (OSError, ValueError, IndexError) as exc:
                    canvas.create_text(24, 24, anchor="nw", fill="#a00000",
                                       font=FONT_NORMAL,
                                       text=f"Impossible de lire {xyz_path}: {exc}")
                return

            try:
                png, width, height, atom_rows = _numbered_molecule_image(smiles)
                import base64
                image = tk.PhotoImage(data=base64.b64encode(png).decode("ascii"))
                window._atom_preview_image = image
                canvas.create_image(0, 0, image=image, anchor="nw")
                canvas.configure(scrollregion=(0, 0, width, height))
                for row in atom_rows:
                    tree.insert("", "end", values=row)
            except Exception as exc:
                canvas.create_text(24, 24, anchor="nw", fill="#a00000",
                                   font=FONT_NORMAL,
                                   text=f"Impossible de dessiner {name}: {exc}")

        selected.bind("<<ComboboxSelected>>", render_selected)
        render_selected()

    def _on_xtb_step_toggle(self):
        xtb_selected = self._step_vars["xtb"].get()
        self.xtb_stop_after_check.config(state="normal" if xtb_selected else "disabled")
        if not xtb_selected and self.xtb_stop_after_var.get():
            self.xtb_stop_after_var.set(False)
            self._on_xtb_stop_after_toggle()
        self._refresh_step_panels()

    def _on_xtb_stop_after_toggle(self):
        downstream = ("dft", "tddft", "density", "docking")
        if self.xtb_stop_after_var.get():
            if self._steps_before_xtb_stop is None:
                self._steps_before_xtb_stop = {
                    key: self._step_vars[key].get() for key in downstream
                }
            for key in downstream:
                self._step_vars[key].set(False)
                self._step_checkbuttons[key].config(state="disabled")
        else:
            if self._steps_before_xtb_stop is not None:
                for key in downstream:
                    self._step_vars[key].set(self._steps_before_xtb_stop[key])
            self._steps_before_xtb_stop = None
            for key in downstream:
                self._step_checkbuttons[key].config(state="normal")
        self._refresh_step_panels()

    def _on_step_selection_changed(self):
        self._refresh_step_panels()

    def _refresh_step_panels(self):
        selected = {key: variable.get() for key, variable in self._step_vars.items()}
        visible = {
            "geometry": selected["xtb"] or selected["dft"] or selected["docking"],
            "xtb": selected["xtb"],
            "dft": selected["dft"] and not self.xtb_stop_after_var.get(),
            "tddft": selected["tddft"] and not self.xtb_stop_after_var.get(),
            "density": selected["density"] and not self.xtb_stop_after_var.get(),
            "docking": selected["docking"] and not self.xtb_stop_after_var.get(),
        }
        for panel in self._step_panels.values():
            panel.pack_forget()
        for key in self._panel_order:
            if visible[key]:
                self._step_panels[key].pack(fill="x", pady=2)

    def _build_right(self, parent):
        ind_f = tk.Frame(parent, bg=COLORS["panel"])
        ind_f.pack(fill="x", pady=(0, 6))
        tk.Label(ind_f, text="  Progression", font=FONT_BOLD,
                 bg=COLORS["panel"], fg=COLORS["text_dim"]
                 ).pack(anchor="w", padx=8, pady=(6, 2))
        row_ind = tk.Frame(ind_f, bg=COLORS["panel"])
        row_ind.pack(anchor="w", padx=8, pady=(0, 6))
        self._indicators: dict = {}
        for key, lbl in [("init", "Init"), ("xtb", "xTB"), ("dft", "DFT"),
                         ("tddft", "TDDFT"), ("density", "Densite"),
                         ("docking", "Docking")]:
            ind = StepIndicator(row_ind, lbl)
            ind.pack(side="left", padx=(0, 8))
            self._indicators[key] = ind

        btn_f = tk.Frame(parent, bg=COLORS["bg"])
        btn_f.pack(fill="x", pady=(0, 6))
        self.btn_run = tk.Button(
            btn_f, text="  Lancer le pipeline",
            font=FONT_BOLD, padx=18, pady=8,
            bg=COLORS["accent"], fg="white",
            activebackground=COLORS["accent_hover"],
            relief="flat", cursor="hand2",
            command=self._on_run)
        self.btn_run.pack(side="left", padx=(0, 8))
        self.btn_stop = tk.Button(
            btn_f, text="  Arreter",
            font=FONT_BOLD, padx=14, pady=8,
            bg=COLORS["error"], fg="white",
            activebackground="#e06c75",
            relief="flat", cursor="hand2",
            state="disabled", command=self._on_stop)
        self.btn_stop.pack(side="left", padx=(0, 10))
        tk.Button(btn_f, text="Ouvrir dossier",
                  font=FONT_BOLD, padx=10, pady=8,
                  bg=COLORS["card"], fg=COLORS["text"],
                  activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  command=self._open_folder).pack(side="left")

        self.spinner = SpinnerWidget(btn_f)
        self.spinner.pack(side="left", padx=(12, 0))

        self.lbl_status = tk.Label(btn_f, text="", font=FONT_SMALL,
                                   bg=COLORS["bg"], fg=COLORS["text_dim"])
        self.lbl_status.pack(side="right", padx=8)

        log_hdr = tk.Frame(parent, bg=COLORS["panel"])
        log_hdr.pack(fill="x")
        tk.Label(log_hdr, text="  Console", font=FONT_BOLD,
                 bg=COLORS["panel"], fg=COLORS["text_dim"]
                 ).pack(side="left", padx=8, pady=4)
        for lbl, cmd in [("Effacer", self._clear_log), ("Copier", self._copy_log)]:
            tk.Button(log_hdr, text=lbl, font=FONT_SMALL,
                      bg=COLORS["panel"], fg=COLORS["text_dim"],
                      activebackground=COLORS["border"],
                      relief="flat", cursor="hand2",
                      command=cmd).pack(side="right", padx=4)

        self.log_box = scrolledtext.ScrolledText(
            parent, font=FONT_MONO, wrap="none",
            bg=COLORS["card"], fg=COLORS["text"],
            insertbackground=COLORS["text"],
            selectbackground=COLORS["accent"],
            relief="flat", bd=0,
            state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(0, 6))

        # Scrollbar horizontale pour wrap=none
        xsb = ttk.Scrollbar(parent, orient="horizontal",
                             command=self.log_box.xview)
        xsb.pack(fill="x")
        self.log_box.configure(xscrollcommand=xsb.set)

        for tag, fg in [("normal",  COLORS["text"]),
                        ("success", COLORS["success"]),
                        ("error",   COLORS["error"]),
                        ("warning", COLORS["warning"]),
                        ("info",    COLORS["info"]),
                        ("section", COLORS["accent"])]:
            self.log_box.tag_config(tag, foreground=fg)
        self.log_box.tag_config("section", font=("Consolas", 10, "bold"))

        self._build_results(parent)

    def _build_results(self, parent):
        rf = tk.Frame(parent, bg=COLORS["panel"])
        rf.pack(fill="x", pady=(4, 0))
        tk.Label(rf, text="  Resultats cles", font=FONT_BOLD,
                 bg=COLORS["panel"], fg=COLORS["text_dim"]
                 ).pack(anchor="w", padx=8, pady=(6, 2))
        grid = tk.Frame(rf, bg=COLORS["panel"])
        grid.pack(fill="x", padx=8, pady=(0, 8))

        def cell(lbl, row, col):
            frm = tk.Frame(grid, bg=COLORS["card"])
            frm.grid(row=row, column=col, padx=3, pady=2, sticky="ew")
            grid.columnconfigure(col, weight=1)
            tk.Label(frm, text=lbl, font=FONT_SMALL,
                     bg=COLORS["card"], fg=COLORS["text_dim"]
                     ).pack(anchor="w", padx=6, pady=(4, 0))
            v = tk.Label(frm, text="—", font=FONT_BOLD,
                         bg=COLORS["card"], fg=COLORS["text"])
            v.pack(anchor="w", padx=6, pady=(0, 4))
            return v

        self.r_energy = cell("Energie DFT (Ha)", 0, 0)
        self.r_lambda = cell("lambda max (nm)",  0, 1)
        self.r_fosc   = cell("Force osc.",       0, 2)
        self.r_dg     = cell("DeltaG kcal/mol",  0, 3)
        self.r_dg.config(wraplength=190, justify="left")
        self.r_kd     = cell("Kd (M)",           0, 4)
        self.r_xtb    = cell("Resume xTB",        1, 0)
        self.r_xtb.config(wraplength=240, justify="left")

    # =========================================================================
    # Tick global unique
    # =========================================================================

    def _tick(self):
        # 1. Avancer le spinner
        if self._running:
            self.spinner.advance()

        # 2. Depiler la queue (max LOG_BURST messages)
        processed = 0
        try:
            while processed < self.LOG_BURST:
                msg_type, payload = self._queue.get_nowait()
                processed += 1
                if msg_type == MSG_LOG:
                    self._append_log(*payload)
                elif msg_type == MSG_STEP:
                    key = payload[0]
                    if key in self._indicators:
                        self._indicators[key].set_state(payload[1], payload[2])
                elif msg_type == MSG_RESULT:
                    self._show_results(payload)
                elif msg_type == MSG_DONE:
                    self._on_done()
                elif msg_type == MSG_ERROR:
                    self._append_log("error", f"ERREUR CRITIQUE :\n{payload}")
                    self._on_done()
        except queue.Empty:
            pass

        # 3. Si le processus est mort sans MSG_DONE, forcer le nettoyage
        if (self._running and self._process is not None
                and not self._process.is_alive() and processed == 0):
            self._on_done()

        # 4. Flush du buffer log en un seul appel tkinter
        self._flush_log()

        # 5. Replanifier
        self.after(self.TICK_MS, self._tick)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_run(self):
        if not self._validate():
            return
        if self._process and self._process.is_alive():
            return
        for ind in self._indicators.values():
            ind.set_state("pending")
        for w in [self.r_energy, self.r_lambda, self.r_fosc, self.r_dg, self.r_kd, self.r_xtb]:
            w.config(text="—", fg=COLORS["text"])

        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._running = True
        self.spinner.show()
        self.lbl_status.config(text="Calcul en cours...")

        steps = [k for k in ["xtb", "dft", "tddft", "density", "docking"]
                 if self._step_vars[k].get()]
        if self.xtb_stop_after_var.get():
            steps = ["xtb"]
        dihedrals = _parse_dihedrals(
            self.xtb_dihedrals_text.get("1.0", "end")
        )
        params = {
            "steps":              steps,
            "xtb_exe":            self.e_xtb_exe.get() or "xtb",
            "xtb_method":         self.e_xtb_method.get() or "GFN2",
            "xtb_opt_level":      self.e_xtb_level.get() or "tight",
            "xtb_charge":         self.e_xtb_charge.get() or "0",
            "xtb_multiplicity":   self.e_xtb_mult.get() or "1",
            "xtb_threads":        self.e_xtb_threads.get() or "10",
            "run_crest":          self.crest_var.get(),
            "crest_exe":          self.e_crest_exe.get() or "crest",
            "crest_n_conformers": self.e_crest_n.get() or "10",
            "crest_use_wsl":      self.crest_wsl_var.get(),
            "crest_wsl_exe":      self.e_crest_wsl_exe.get() or "crest",
            "crest_wsl_xtb_exe":  self.e_crest_wsl_xtb_exe.get() or (
                "/home/ugopasco/miniforge3/envs/crest_xtb/bin/xtb"
            ),
            "xtb_dihedrals":      dihedrals,
            "xtb_planarity_threshold_deg": self.e_xtb_threshold.get() or "5.0",
            "xtb_run_constrained": self.xtb_constrain_var.get(),
            "xtb_constraint_target_deg": self.e_xtb_target.get() or "0",
            "xtb_constraint_force_constant": self.e_xtb_force.get() or "0.5",
            "xtb_stop_after":    self.xtb_stop_after_var.get(),
            "smiles":             self.e_smiles.get(),
            "name":               self.e_name.get(),
            "output_dir":         self.e_outdir.get()       or "./pipeline_results",
            "dft_functional":     self.e_func.get()         or "B3LYP",
            "dft_basis":          self.e_basis.get()        or "def2-SVP",
            "memory":             self.e_memory.get()       or "12 GB",
            "threads":            self.e_threads.get()      or "10",
            "tddft_solvent":      self.e_solvent.get(),
            "tddft_functional":   self.e_tddft_func.get(),
            "tddft_basis":        self.e_tddft_basis.get(),
            "tddft_timeout":      self.e_tddft_timeout.get(),
            "tddft_use_tda":      self.tda_var.get(),
            "tddft_n_states":     self.e_tddft_nstates.get() or "4",
            "density_functional": self.e_dens_func.get()    or "B3LYP",
            "density_basis":      self.e_dens_basis.get()   or "def2-SVP",
            "peptide_mol":        list(self.lb_receptors.get(0, "end")),
            "vina_exe":           self.e_vina_exe.get(),
            "vina_exhaustiveness": self.e_vina_exh.get()    or "34",
            "vina_n_poses":       self.e_vina_poses.get()   or "1000",
            "vina_scoring":       self.e_vina_scoring.get() or "vinardo",
            "vina_runs":          self.e_vina_runs.get()    or "3",
            "vina_seed":          "42",
            "skip_single_point":  self.skip_sp_var.get(),
            "xyz_file":           self.e_xyz.get(),
            "batch_file":         self.e_batch.get(),
            "psi4_calc_file":     self.e_psi4_file.get(),
            "elec_dens_file":     self.e_dens_file.get(),
            "docking_file":       self.e_dock_file.get(),
        }
        self._stop_event.clear()
        self._process = mp.Process(
            target=_run_pipeline,
            args=(params, self._queue, self._stop_event,
                  str(Path(__file__).parent)),
            daemon=True,
        )
        self._process.start()

    def _on_stop(self):
        self._stop_event.set()
        self._append_log("warning",
            "Arret demande — terminaison du processus...")
        self._flush_log()
        self.btn_stop.config(state="disabled")
        # Terminaison forcee du processus apres 5 secondes
        self.after(5000, self._force_stop)

    def _force_stop(self):
        """Termine le processus de force s'il est encore actif."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._append_log("warning", "Processus termine de force.")
            self._flush_log()

    def _on_done(self):
        self._running = False
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.spinner.hide()
        self.lbl_status.config(
            text=f"Termine — {datetime.now().strftime('%H:%M:%S')}")

    # =========================================================================
    # Log — buffer batch
    # =========================================================================

    def _append_log(self, tag: str, text: str):
        """Determine le tag et ajoute au buffer — PAS d'appel tkinter ici."""
        tl = text.lower()
        if any(x in tl for x in ["erreur", "error", "echec", "failed", "traceback"]):
            tag = "error"
        elif any(x in tl for x in ["ok ", "converge", "succes", "reussi"]):
            tag = "success"
        elif any(x in tl for x in ["attention", "warning", "fallback"]):
            tag = "warning"
        elif text.lstrip().startswith(("=", "─", "━")):
            tag = "section"
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_buf.append((tag, f"[{ts}] {text}\n"))

    def _flush_log(self):
        """Insere toutes les lignes bufferisees de maniere optimisee."""
        if not self._log_buf:
            return
        box = self.log_box
        box.config(state="normal")

        # Regrouper les lignes consecutives de meme tag en un seul insert
        batch = self._log_buf
        i = 0
        while i < len(batch):
            tag = batch[i][0]
            parts = [batch[i][1]]
            j = i + 1
            while j < len(batch) and batch[j][0] == tag:
                parts.append(batch[j][1])
                j += 1
            box.insert("end", "".join(parts), tag)
            i = j
        self._log_buf.clear()

        # Trim : supprimer les lignes les plus anciennes si on depasse le max
        total = int(box.index("end-1c").split(".")[0])
        if total > self.LOG_MAX:
            excess = total - self.LOG_MAX
            box.delete("1.0", f"{excess}.0")

        box.see("end")
        box.config(state="disabled")

    # =========================================================================
    # Utilitaires
    # =========================================================================

    def _validate(self) -> bool:
        if not any(variable.get() for variable in self._step_vars.values()):
            messagebox.showerror("Etapes", "Selectionnez au moins une etape du pipeline.")
            return False
        if (self._step_vars["xtb"].get()
                and not self.xtb_stop_after_var.get()
                and not self._step_vars["dft"].get()):
            messagebox.showerror(
                "Etapes",
                "Avec xTB, activez DFT pour continuer ou cochez 'Arreter apres xTB'.",
            )
            return False
        if (self._step_vars["tddft"].get() or self._step_vars["density"].get()) \
                and not self._step_vars["dft"].get():
            messagebox.showerror(
                "Etapes", "TDDFT et densite electronique necessitent l'etape DFT."
            )
            return False
        try:
            if self._step_vars["xtb"].get():
                if self.e_xtb_method.get().strip().upper() not in ("GFN1", "GFN2"):
                    raise ValueError("La methode xTB doit etre GFN1 ou GFN2.")
                if self.e_xtb_level.get().strip().lower() not in (
                        "normal", "tight", "verytight"):
                    raise ValueError("Le niveau xTB doit etre normal, tight ou verytight.")
                int(self.e_xtb_charge.get())
                if int(self.e_xtb_mult.get()) < 1 or int(self.e_xtb_threads.get()) < 1:
                    raise ValueError("Multiplicite et threads doivent etre positifs.")
                if int(self.e_crest_n.get()) < 1:
                    raise ValueError("Le nombre de conformeres CREST doit etre positif.")
                if float(self.e_xtb_threshold.get()) < 0:
                    raise ValueError("Le seuil de planarite ne peut pas etre negatif.")
            dihedral_indices = _parse_dihedrals(
                self.xtb_dihedrals_text.get("1.0", "end")
            )
            if self.xtb_constrain_var.get():
                if not self._step_vars["xtb"].get():
                    raise ValueError("Activez l'etape xTB pour lancer le test contraint.")
                if not dihedral_indices:
                    raise ValueError("Entrez au moins un diedre a contraindre.")
                target = float(self.e_xtb_target.get())
                if min(abs(target), abs(180.0 - abs(target))) > 1e-6:
                    raise ValueError("Une cible plane doit etre 0 ou 180 degres.")
                if float(self.e_xtb_force.get()) <= 0:
                    raise ValueError("La constante de force doit etre positive.")
        except ValueError as exc:
            messagebox.showerror("Parametres xTB", str(exc))
            return False
        batch_file = self.e_batch.get()
        if batch_file:
            # Mode batch : verifier que le fichier existe
            if not os.path.isfile(batch_file):
                messagebox.showerror("Batch", f"Fichier introuvable :\n{batch_file}")
                return False
        else:
            # Mode simple : nom obligatoire, SMILES obligatoire SAUF si XYZ fourni
            if not self.e_name.get():
                messagebox.showerror("Champ manquant", "Le nom est obligatoire.")
                return False
            if not self.e_smiles.get() and not self.e_xyz.get():
                messagebox.showerror("Champ manquant",
                    "Le SMILES est obligatoire\n"
                    "(sauf si un fichier XYZ est fourni pour bypass).")
                return False
        if self._step_vars["docking"].get():
            receptors = list(self.lb_receptors.get(0, "end"))
            if not receptors:
                messagebox.showerror("Docking", "Au moins un fichier recepteur requis (.mol ou .pdb).")
                return False
            for rp in receptors:
                if not os.path.isfile(rp):
                    messagebox.showerror("Docking", f"Fichier recepteur introuvable :\n{rp}")
                    return False
            if not self.e_vina_exe.get():
                messagebox.showerror("Docking",
                                     "Chemin de l'executable Vina requis.")
                return False
            try:
                if int(self.e_vina_runs.get()) < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Docking", "Le nombre de recherches globales doit etre un entier positif."
                )
                return False
            scoring = (self.e_vina_scoring.get() or "vina").strip().lower()
            if scoring not in ("vina", "vinardo"):
                messagebox.showerror(
                    "Docking",
                    f"Scoring '{scoring}' non support\u00e9.\n\n"
                    "Fonctions disponibles :\n"
                    "  \u2022 vina    (standard, rapide)\n"
                    "  \u2022 vinardo (am\u00e9lior\u00e9, recommand\u00e9 pour aromatiques)\n\n"
                    "'ad4' n\u00e9cessite AutoGrid4 (cartes d'affinit\u00e9\n"
                    "pr\u00e9-calcul\u00e9es) qui n'est pas inclus.")
                return False
        return True

    def _show_results(self, results: dict):
        dft   = results.get("dft",     {})
        tddft = results.get("tddft",   {})
        dock  = results.get("docking", {})
        xtb   = results.get("xtb",     {})
        e = dft.get("energy_hartree")
        self.r_energy.config(
            text=f"{e:.8f}" if isinstance(e, float) else "—",
            fg=COLORS["success"] if isinstance(e, float) else COLORS["text_dim"])
        lmax = tddft.get("lambda_max_nm")
        self.r_lambda.config(
            text=f"{lmax:.1f}" if isinstance(lmax, float) else "—",
            fg=COLORS["success"] if isinstance(lmax, float) else COLORS["text_dim"])
        fosc = tddft.get("max_oscillator_strength")
        self.r_fosc.config(
            text=f"{fosc:.4f}" if isinstance(fosc, float) else "—")
        dg = dock.get("best_dG_kcalmol")
        ensemble = results.get("docking_ensemble", {})
        if ensemble and ensemble.get("mean_best_dG_kcalmol") is not None:
            dg_text = (
                f"Best {dg:.2f}" if isinstance(dg, (int, float)) else "Best —"
            )
            dg_text += (
                f" | Ens {ensemble['mean_best_dG_kcalmol']:.2f} "
                f"± {ensemble['std_best_dG_kcalmol']:.2f} "
                f"({ensemble['successful_models']}/{ensemble['receptor_models']})"
            )
        else:
            dg_text = f"{dg:.2f}" if isinstance(dg, (int, float)) else "—"
        self.r_dg.config(
            text=dg_text,
            fg=COLORS["success"] if isinstance(dg, float) and dg < -4
               else COLORS["warning"] if isinstance(dg, float)
               else COLORS["text_dim"])
        kd = dock.get("best_Kd_M")
        self.r_kd.config(text=f"{kd:.2e}" if isinstance(kd, float) else "—")
        if xtb:
            energy = xtb.get("energy_hartree")
            parts = ["OK" if xtb.get("converged") else "NON CONVERGE"]
            if energy is not None:
                parts.append(f"{energy:.6f} Ha")
            if xtb.get("n_iterations") is not None:
                parts.append(f"{xtb['n_iterations']} iter")
            if xtb.get("time_s") is not None:
                parts.append(f"{xtb['time_s']:.1f} s")
            dihedral = xtb.get("dihedrals", {}).get("dihedral_1")
            if dihedral:
                parts.append(f"D1 {dihedral['dihedral_deg']:.1f} deg")
            if xtb.get("planarity") is not None:
                parts.append("PLANAIRE" if xtb["planarity"] else "NON PLANAIRE")
            summary_text = " | ".join(parts)
            constrained = xtb.get("constrained", {})
            if constrained:
                if constrained.get("success"):
                    constrained_energy = constrained.get("energy_hartree")
                    delta_energy = constrained.get("energy_difference_hartree")
                    constrained_status = "Contrainte OK"
                    constrained_planarity = constrained.get("planarity", {}).get("all_planar")
                    if constrained_planarity is not None:
                        constrained_status += (
                            " | PLANAIRE" if constrained_planarity else " | NON PLANAIRE"
                        )
                    if constrained_energy is not None:
                        constrained_status += f" | E {constrained_energy:.6f} Ha"
                    if delta_energy is not None:
                        constrained_status += f" | dE {delta_energy:+.6f} Ha"
                    summary_text += "\n" + constrained_status
                else:
                    summary_text += f"\nContrainte echec: {constrained.get('error', '?')}"
            self.r_xtb.config(
                text=summary_text,
                fg=COLORS["success"] if xtb.get("success") else COLORS["error"],
            )
        else:
            self.r_xtb.config(text="—", fg=COLORS["text_dim"])

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _copy_log(self):
        self.clipboard_clear()
        self.clipboard_append(self.log_box.get("1.0", "end"))

    def _open_folder(self):
        path = os.path.abspath(self.e_outdir.get() or "./pipeline_results")
        if os.path.isdir(path):
            import subprocess, platform
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])

    def _sep(self, parent, text):
        frm = tk.Frame(parent, bg=COLORS["bg"])
        frm.pack(fill="x", pady=(10, 2))
        tk.Label(frm, text=text.upper(), font=("Segoe UI", 8, "bold"),
                 bg=COLORS["bg"], fg=COLORS["accent"]).pack(side="left")
        tk.Frame(frm, bg=COLORS["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=6)

    def _file_row(self, parent, label, cmd, default="") -> ParamEntry:
        row = tk.Frame(parent, bg=COLORS["bg"])
        row.pack(fill="x", pady=2)
        e = ParamEntry(row, label, default=default, width=22)
        e.pack(side="left", fill="x", expand=True)
        tk.Button(row, text="...", font=FONT_BOLD, width=3,
                  bg=COLORS["card"], fg=COLORS["text"],
                  activebackground=COLORS["accent"],
                  relief="flat", command=cmd).pack(side="left", padx=(4, 0))
        return e

    def _add_receptors(self):
        _receptor_dir = str(Path(__file__).parent / "binding_receptor")
        files = filedialog.askopenfilenames(
            title="Fichier(s) recepteur (.mol / .pdb)",
            initialdir=_receptor_dir if os.path.isdir(_receptor_dir) else None,
            filetypes=[("Receptor files", "*.mol *.pdb *.mol2 *.sdf"),
                       ("Mol files", "*.mol"),
                       ("PDB files", "*.pdb"),
                       ("All", "*.*")])
        existing = set(self.lb_receptors.get(0, "end"))
        for f in files:
            if f not in existing:
                self.lb_receptors.insert("end", f)

    def _remove_receptor(self):
        sel = self.lb_receptors.curselection()
        for i in reversed(sel):
            self.lb_receptors.delete(i)

    def _browse_vina(self):
        p = filedialog.askopenfilename(
            title="Executable Vina",
            filetypes=[("Executables", "*.exe"), ("All", "*.*")])
        if p: self.e_vina_exe.set(p)

    def _browse_xtb(self):
        p = filedialog.askopenfilename(
            title="Executable xTB",
            filetypes=[("Executables", "*.exe"), ("All", "*.*")])
        if p: self.e_xtb_exe.set(p)

    def _browse_crest(self):
        p = filedialog.askopenfilename(
            title="Executable CREST",
            filetypes=[("Executables", "*.exe"), ("All", "*.*")])
        if p: self.e_crest_exe.set(p)

    def _browse_py(self, entry):
        p = filedialog.askopenfilename(
            title="Module Python",
            filetypes=[("Python", "*.py"), ("All", "*.*")])
        if p: entry.set(p)

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Dossier de sortie")
        if p: self.e_outdir.set(p)

    def _browse_batch(self):
        p = filedialog.askopenfilename(
            title="Fichier batch CSV / Excel",
            filetypes=[("CSV / Excel", "*.csv *.xlsx *.xls"), ("All", "*.*")])
        if p: self.e_batch.set(p)

    def _browse_xyz(self):
        p = filedialog.askopenfilename(
            title="Fichier XYZ (geometrie pre-optimisee)",
            filetypes=[("XYZ files", "*.xyz"), ("All", "*.*")])
        if p: self.e_xyz.set(p)


def main():
    mp.freeze_support()
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()