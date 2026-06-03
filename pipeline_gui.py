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

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._q.put((MSG_LOG, (self._tag, line)))
        return len(text)

    def flush(self):
        # Vider le buffer même sans \n final (dernière ligne Psi4)
        if self._buf.strip():
            self._q.put((MSG_LOG, (self._tag, self._buf)))
            self._buf = ""


# =========================================================================
# Fonctions de pipeline — executees dans un processus separe (mp.Process)
# =========================================================================

from utils_paths import _sanitize_mol_name


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
    summary_path = Path(output_dir) / "batch_summary.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Nom", "Energie_DFT_Ha", "Lambda_max_nm", "Fosc",
                     "dG_kcal_mol", "Kd_M", "Succes_DFT", "Succes_TDDFT",
                     "Succes_Docking"])
        for name, res in all_results.items():
            dft   = res.get("dft", {})
            tddft = res.get("tddft", {})
            dock  = res.get("docking", {})
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
            ])
    _log("info", f"Resume batch sauvegarde : {summary_path}")


def _patch_step_fn(pipe, key, fn, _log, _step):
    """Wrap une etape du pipeline pour le suivi de progression."""
    _labels = {
        "init": "Initialisation",
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
    _known = {"success", "error", "energy_hartree", "n_atoms",
              "converged", "n_iterations", "lambda_max_nm",
              "max_oscillator_strength", "n_states", "solvent",
              "cube_file", "density_at_origin", "best_dG_kcalmol",
              "best_Kd_M", "n_poses"}
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

    def _log(tag, text):
        q.put((MSG_LOG, (tag, text)))

    def _step(key, state, detail=""):
        q.put((MSG_STEP, (key, state, detail)))

    try:
        p = params
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

        # --- Dossier de sortie (toujours le même) ---
        base_output = p.get("output_dir", "./pipeline_results")
        Path(base_output).mkdir(parents=True, exist_ok=True)
        p["output_dir"] = base_output

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
                    tddft_solvent    = p.get("tddft_solvent")    or None,
                    tddft_functional = p.get("tddft_functional") or None,
                    tddft_basis      = p.get("tddft_basis")      or None,
                    tddft_use_tda    = p.get("tddft_use_tda", True),
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
                for skey in ["dft", "tddft", "density", "docking"]:
                    if hasattr(pipe, f"_step_{skey}"):
                        _patch_step_fn(pipe, skey, getattr(pipe, f"_step_{skey}"),
                                       _log, _step)

                for skey in ["tddft", "density", "docking"]:
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
                    for skey in ["init", "dft", "tddft", "density", "docking"]:
                        _step(skey, "pending")

            if stop_event.is_set():
                break

            # Résumé batch par récepteur
            if len(molecules) > 1:
                _write_batch_summary(all_results, rec_output, _log)

            final_results = all_results

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
                dst = Path(base_output) / src.name
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
        self.geometry("1160x860")
        self.minsize(920, 680)

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

        left = tk.Frame(body, bg=COLORS["bg"], width=380)
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
        for key, lbl, default, disabled in [
            ("dft",     "Optimisation DFT",          True,  True),
            ("tddft",   "TDDFT — lambda max UV-Vis",  False, False),
            ("density", "Densite electronique",       False, False),
            ("docking", "Docking / Kd (Vina)",        False, False),
        ]:
            v = tk.BooleanVar(value=default)
            self._step_vars[key] = v
            tk.Checkbutton(sf, text=lbl, variable=v, font=FONT_NORMAL,
                           bg=COLORS["panel"], fg=COLORS["text"],
                           activebackground=COLORS["panel"],
                           activeforeground=COLORS["accent"],
                           selectcolor=COLORS["card"],
                           state="disabled" if disabled else "normal"
                           ).pack(anchor="w", pady=1)
        tk.Label(sf, text="  * DFT toujours en premier", font=FONT_SMALL,
                 bg=COLORS["panel"], fg=COLORS["text_dim"]).pack(anchor="w")

        self._sep(f, "Parametres DFT")
        self.e_xyz = self._file_row(f, "Fichier XYZ (bypass optim. geom.)",
                                     self._browse_xyz)
        tk.Label(f, text="  Si renseigne, saute l'optimisation geometrique DFT",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")
        self.skip_sp_var = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Skip single-point (bypass sans calcul d'energie)",
                       variable=self.skip_sp_var, font=FONT_NORMAL,
                       bg=COLORS["bg"], fg=COLORS["text"],
                       activebackground=COLORS["bg"],
                       activeforeground=COLORS["accent"],
                       selectcolor=COLORS["card"]
                       ).pack(anchor="w", pady=(2, 0))
        tk.Label(f, text="  Uniquement avec bypass XYZ — saute le calcul single-point Psi4",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")
        r = tk.Frame(f, bg=COLORS["bg"]); r.pack(fill="x", pady=2)
        self.e_func  = ParamEntry(r, "Fonctionnelle", "B3LYP",    12,
                                  "Ex: B3LYP, CAM-B3LYP, PBE0")
        self.e_func.pack(side="left", padx=(0, 6))
        self.e_basis = ParamEntry(r, "Base",           "def2-SVP", 12,
                                  "Ex: def2-SVP, def2-TZVP, 6-31G*")
        self.e_basis.pack(side="left")
        r2 = tk.Frame(f, bg=COLORS["bg"]); r2.pack(fill="x", pady=2)
        self.e_memory  = ParamEntry(r2, "Memoire Psi4", "12 GB", 10)
        self.e_memory.pack(side="left", padx=(0, 6))
        self.e_threads = ParamEntry(r2, "Threads", "10", 6)
        self.e_threads.pack(side="left")

        self._sep(f, "Parametres TDDFT")
        self.e_solvent = ParamEntry(f, "Solvant", default="ACN",
                                    tooltip="Ex: ACN, methanol, dmso, thf, cyclohexane — vide=gaz")
        self.e_solvent.pack(fill="x", pady=2)
        # Option TDA (Tamm-Dancoff Approximation) — cochée par défaut (~2x plus rapide)
        tda_frame = tk.Frame(f, bg=COLORS["bg"]); tda_frame.pack(fill="x", pady=2)
        self.tda_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tda_frame, text="TDA (Tamm-Dancoff) — plus rapide",
                       variable=self.tda_var, font=FONT_NORMAL,
                       bg=COLORS["bg"], fg=COLORS["text"],
                       activebackground=COLORS["bg"],
                       activeforeground=COLORS["accent"],
                       selectcolor=COLORS["card"]).pack(anchor="w")
        r3 = tk.Frame(f, bg=COLORS["bg"]); r3.pack(fill="x", pady=2)
        self.e_tddft_func = ParamEntry(r3, "Fonctionnelle", "CAM-B3LYP", 14,
                                       "Vide = meme que DFT")
        self.e_tddft_func.pack(side="left", padx=(0, 6))
        self.e_tddft_basis = ParamEntry(r3, "Base", "", 12,
                                        "Vide = meme que DFT")
        self.e_tddft_basis.pack(side="left", padx=(0, 6))
        self.e_tddft_timeout = ParamEntry(r3, "Timeout", "", 6,
                                          "Minutes, vide = adaptatif")
        self.e_tddft_timeout.pack(side="left")

        self._sep(f, "Parametres Densite")
        r_dens = tk.Frame(f, bg=COLORS["bg"]); r_dens.pack(fill="x", pady=2)
        self.e_dens_func = ParamEntry(r_dens, "Fonctionnelle", "B3LYP", 14,
                                      "Fonctionnelle pour calcul SCF densite")
        self.e_dens_func.pack(side="left", padx=(0, 6))
        self.e_dens_basis = ParamEntry(r_dens, "Base", "def2-SVP", 12,
                                       "Base pour calcul SCF densite")
        self.e_dens_basis.pack(side="left")

        self._sep(f, "Docking AutoDock Vina")
        _base_dir = str(Path(__file__).parent)
        _default_vina = str(Path(__file__).parent / "vina_1.2.7_win.exe")

        # Multi-récepteur : liste de fichiers
        tk.Label(f, text="Fichier(s) recepteur (.mol / .pdb)",
                 font=FONT_SMALL, bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(anchor="w", pady=(4, 0))
        r_recep = tk.Frame(f, bg=COLORS["bg"]); r_recep.pack(fill="x", pady=2)
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

        self.e_vina_exe = self._file_row(f, "Executable Vina",        self._browse_vina,
                                          default=_default_vina)
        r4 = tk.Frame(f, bg=COLORS["bg"]); r4.pack(fill="x", pady=2)
        self.e_vina_exh     = ParamEntry(r4, "Exhaustivite", "34",   6)
        self.e_vina_exh.pack(side="left", padx=(0, 6))
        self.e_vina_poses   = ParamEntry(r4, "Nb poses",     "1000",  6)
        self.e_vina_poses.pack(side="left", padx=(0, 6))
        self.e_vina_scoring = ParamEntry(r4, "Scoring",      "vinardo", 8)
        self.e_vina_scoring.pack(side="left")

        self._sep(f, "Modules (optionnel)")
        self.e_psi4_file = self._file_row(
            f, "psi4_calculator_fixed_y.py",
            lambda: self._browse_py(self.e_psi4_file))
        self.e_dens_file = self._file_row(
            f, "electron_density_analysis_fixed.py",
            lambda: self._browse_py(self.e_dens_file))
        self.e_dock_file = self._file_row(
            f, "docking_kd_pipeline.py",
            lambda: self._browse_py(self.e_dock_file))

        self._sep(f, "Dossier de sortie")
        self.e_outdir = self._file_row(f, "", self._browse_outdir,
                                       default=r"C:\Users\ugo.pasco\Documents\2A\Data finale MD\Quantumator\Results")

    def _build_right(self, parent):
        ind_f = tk.Frame(parent, bg=COLORS["panel"])
        ind_f.pack(fill="x", pady=(0, 6))
        tk.Label(ind_f, text="  Progression", font=FONT_BOLD,
                 bg=COLORS["panel"], fg=COLORS["text_dim"]
                 ).pack(anchor="w", padx=8, pady=(6, 2))
        row_ind = tk.Frame(ind_f, bg=COLORS["panel"])
        row_ind.pack(anchor="w", padx=8, pady=(0, 6))
        self._indicators: dict = {}
        for key, lbl in [("init", "Init"), ("dft", "DFT"),
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
        self.r_kd     = cell("Kd (M)",           0, 4)

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
        for w in [self.r_energy, self.r_lambda, self.r_fosc, self.r_dg, self.r_kd]:
            w.config(text="—", fg=COLORS["text"])

        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._running = True
        self.spinner.show()
        self.lbl_status.config(text="Calcul en cours...")

        steps = ["dft"] + [k for k in ["tddft", "density", "docking"]
                           if self._step_vars[k].get()]
        params = {
            "steps":              steps,
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
            "density_functional": self.e_dens_func.get()    or "B3LYP",
            "density_basis":      self.e_dens_basis.get()   or "def2-SVP",
            "peptide_mol":        list(self.lb_receptors.get(0, "end")),
            "vina_exe":           self.e_vina_exe.get(),
            "vina_exhaustiveness": self.e_vina_exh.get()    or "34",
            "vina_n_poses":       self.e_vina_poses.get()   or "1000",
            "vina_scoring":       self.e_vina_scoring.get() or "vinardo",
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
        self.r_dg.config(
            text=f"{dg:.2f}" if isinstance(dg, float) else "—",
            fg=COLORS["success"] if isinstance(dg, float) and dg < -4
               else COLORS["warning"] if isinstance(dg, float)
               else COLORS["text_dim"])
        kd = dock.get("best_Kd_M")
        self.r_kd.config(text=f"{kd:.2e}" if isinstance(kd, float) else "—")

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