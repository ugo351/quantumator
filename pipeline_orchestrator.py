#!/usr/bin/env python3
"""
pipeline_orchestrator.py
========================
Pipeline complet : SMILES -> Optimisation DFT -> (TDDFT) -> (Densité élec.) -> (Docking/Kd)

Étapes :
  1. [OBLIGATOIRE] Optimisation DFT (Psi4, B3LYP/def2-SVP par défaut)
  2. [Optionnel]   Calcul TDDFT - lambda max / spectre UV-Vis
  3. [Optionnel]   Analyse densité électronique - fichiers .cube, ESP, charges Mulliken
  4. [Optionnel]   Docking moléculaire & estimation Kd (AutoDock Vina)

Usage CLI :
  python pipeline_orchestrator.py \
      --smiles "O=C(/C=C/c1ccccc1)NC" \
      --name "CAm" \
      --steps dft tddft density docking \
      --peptide_mol path/to/peptide.mol \
      --output_dir ./results

Utilisation programmatique :
  from pipeline_orchestrator import MolecularPipeline, PipelineConfig
  cfg = PipelineConfig(smiles="...", name="CAm", output_dir="./results")
  pipe = MolecularPipeline(cfg)
  results = pipe.run(steps=["dft", "tddft", "density"])
"""

import os
import sys
import json
import argparse
import time
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import psi4

# ─────────────────────────────────────────────────────────────────────────────
# Import dynamique des trois modules
# ─────────────────────────────────────────────────────────────────────────────

def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Impossible de trouver le module : {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_DEFAULT_DIR = Path(__file__).parent

_PSI4_CALC_FILE  = _DEFAULT_DIR / "psi4_calculator_fixed_y.py"
_ELEC_DENS_FILE  = _DEFAULT_DIR / "electron_density_analysis_fixed.py"
_DOCKING_FILE    = _DEFAULT_DIR / "docking_kd_pipeline.py"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class PipelineConfig:
    def __init__(
        self,
        smiles: str,
        name: str,
        output_dir: str = "./pipeline_results",
        dft_functional: str = "B3LYP",
        dft_basis: str = "def2-SVP",
        memory: str = "12 GB",
        threads: int = 10,
        run_tddft: bool = True,
        tddft_solvent: Optional[str] = None,
        tddft_timeout_minutes: Optional[int] = None,
        tddft_functional: Optional[str] = None,
        tddft_basis: Optional[str] = None,
        tddft_use_tda: bool = True,
        run_density: bool = True,
        density_functional: str = "B3LYP",
        density_basis: str = "def2-SVP",
        run_docking: bool = False,
        peptide_mol_file: Optional[str] = None,
        box_override: Optional[tuple] = None,
        vina_exe: str = "vina",
        vina_exhaustiveness: int = 16,
        vina_n_poses: int = 100,
        vina_scoring: str = "vina",
        xyz_file: Optional[str] = None,
        skip_single_point: bool = False,
        psi4_calc_file: Optional[str] = None,
        elec_dens_file: Optional[str] = None,
        docking_file: Optional[str] = None,
    ):
        self.smiles = smiles
        self.name = name
        self.output_dir = Path(output_dir)
        self.dft_functional = dft_functional
        self.dft_basis = dft_basis
        self.memory = memory
        self.threads = threads
        self.run_tddft = run_tddft
        self.tddft_solvent = tddft_solvent
        self.tddft_timeout_minutes = tddft_timeout_minutes
        self.tddft_functional = tddft_functional
        self.tddft_basis = tddft_basis
        self.tddft_use_tda = tddft_use_tda
        self.run_density = run_density
        self.density_functional = density_functional
        self.density_basis = density_basis
        self.run_docking = run_docking
        self.peptide_mol_file = peptide_mol_file
        self.box_override = box_override
        self.vina_exe = vina_exe
        self.vina_exhaustiveness = vina_exhaustiveness
        self.vina_n_poses = vina_n_poses
        self.vina_scoring = vina_scoring
        self.xyz_file = Path(xyz_file) if xyz_file else None
        self.skip_single_point = skip_single_point
        self.psi4_calc_file = Path(psi4_calc_file or _PSI4_CALC_FILE)
        self.elec_dens_file = Path(elec_dens_file or _ELEC_DENS_FILE)
        self.docking_file   = Path(docking_file   or _DOCKING_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MolecularPipeline:
    AVAILABLE_STEPS = ["dft", "tddft", "density", "docking"]

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.results: Dict[str, Any] = {}
        self.xyz_path: Optional[Path] = None
        self._dft_atoms = None
        self._dft_coords = None
        self._log_lines: List[str] = []
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        print(msg)                          # plus de f"[{ts}] {msg}"
        self._log_lines.append(msg)


    def _section(self, title: str):
        sep = "=" * 70
        self._log(sep)
        self._log(f"  {title}")
        self._log(sep)

    # ── Chargement modules ────────────────────────────────────────────────────

    def _load_modules(self, steps: List[str]):
        self._psi4_mod    = None
        self._density_mod = None
        self._docking_mod = None
        if any(s in steps for s in ["dft", "tddft"]):
            self._log("Chargement psi4_calculator...")
            self._psi4_mod = _load_module("psi4_calculator", str(self.cfg.psi4_calc_file))
        if "density" in steps:
            self._log("Chargement electron_density_analysis...")
            self._density_mod = _load_module("electron_density_analysis", str(self.cfg.elec_dens_file))
        if "docking" in steps:
            self._log("Chargement docking_kd_pipeline...")
            self._docking_mod = _load_module("docking_kd_pipeline", str(self.cfg.docking_file))

    # =========================================================================
    # ÉTAPE 1 : DFT (OBLIGATOIRE)
    # =========================================================================

    def _step_dft(self) -> Dict:
        t0 = time.time()
        import psi4
        psi4.core.clean_options()
        
        calc = self._psi4_mod.Psi4MolecularCalculator(
            memory=self.cfg.memory,
            threads=self.cfg.threads,
        )
        if self._psi4_mod is None:
            raise RuntimeError("Module psi4_calculator non chargé.")

        calc = self._psi4_mod.Psi4MolecularCalculator(
            memory=self.cfg.memory,
            threads=self.cfg.threads,
        )

        self._log(f"Molécule   : {self.cfg.name}")
        self._log(f"SMILES     : {self.cfg.smiles}")
        self._log(f"Méthode    : {self.cfg.dft_functional}/{self.cfg.dft_basis}")

        # --- Bypass : utiliser un fichier XYZ pré-existant ---
        if self.cfg.xyz_file and not self.cfg.xyz_file.exists():
            raise FileNotFoundError(
                f"Fichier XYZ introuvable : {self.cfg.xyz_file}\n"
                f"Vérifiez le chemin dans votre fichier batch (colonne 3)."
            )
        if self.cfg.xyz_file and self.cfg.xyz_file.exists():
            self._log(f"XYZ BYPASS : {self.cfg.xyz_file}")
            self._log("Optimisation géométrique sautée — géométrie chargée depuis le fichier XYZ.")

            # Charger le XYZ et construire la molécule Psi4
            xyz_content = self.cfg.xyz_file.read_text(encoding="utf-8")
            lines = [l for l in xyz_content.strip().splitlines() if l.strip()]
            try:
                n_atoms = int(lines[0].strip())
            except (ValueError, IndexError):
                raise ValueError(
                    f"Le fichier XYZ '{self.cfg.xyz_file}' n'est pas au format XYZ valide.\n"
                    f"Première ligne lue : {lines[0]!r}\n"
                    f"Format attendu : première ligne = nombre d'atomes (entier), "
                    f"deuxième ligne = commentaire, puis une ligne par atome : 'Symbol X Y Z'."
                )
            # Construire un bloc XYZ pour Psi4
            geom_lines = []
            atoms = []
            coords = []
            for line in lines[2:2 + n_atoms]:
                parts = line.split()
                sym = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                geom_lines.append(f"  {sym}  {x:.8f}  {y:.8f}  {z:.8f}")
                atoms.append(sym)
                coords.append((x, y, z))

            mol_psi4 = None
            sp_energy = None

            # Calcul single-point pour obtenir l'énergie
            if self.cfg.skip_single_point:
                self._log("Single-point SKIP — pas de calcul d'énergie (option bypass).")
            else:
                import psi4
                mol_input = f"0 1\n" + "\n".join(geom_lines) + "\nunits angstrom\nno_reorient\nno_com\nsymmetry c1\n"
                mol_psi4 = psi4.geometry(mol_input)
                mol_psi4.update_geometry()
                self._log(f"Calcul single-point {self.cfg.dft_functional}/{self.cfg.dft_basis}...")
                try:
                    sp_energy = psi4.energy(f"{self.cfg.dft_functional}/{self.cfg.dft_basis}",
                                             molecule=mol_psi4)
                except Exception as e:
                    self._log(f"Single-point échoué : {e} — énergie non disponible")
                    sp_energy = None

            # Copier le XYZ dans le dossier de sortie
            self.xyz_path = self.cfg.output_dir / f"{self.cfg.name}.xyz"
            import shutil
            shutil.copy2(str(self.cfg.xyz_file), str(self.xyz_path))

            self._dft_atoms  = atoms
            self._dft_coords = coords

            elapsed = time.time() - t0
            dft_res = {
                "success":        True,
                "functional":     self.cfg.dft_functional,
                "basis":          self.cfg.dft_basis,
                "energy_hartree": sp_energy,
                "n_atoms":        n_atoms,
                "xyz_file":       str(self.xyz_path),
                "xyz_bypass":     True,
                "time_s":         round(elapsed, 2),
                "mol_psi4":       mol_psi4,
            }
            if sp_energy is not None:
                self._log(f"Énergie SP   : {sp_energy:.8f} Hartree")
            self._log(f"Temps DFT    : {elapsed:.1f} s (bypass, pas d'optimisation)")

            self.results["dft"] = dft_res
            return dft_res

        # --- Mode normal : optimisation DFT complète ---
        # Créer la molécule Psi4
        mol_psi4, xyz_file_used, xyz_content = calc.setup_molecule(
            self.cfg.smiles, self.cfg.name
        )

        # FIX : Forcer C1 + orientation fixe pour éviter "Point group changed!"
        try:
            mol_psi4.reset_point_group("c1")
            mol_psi4.fix_orientation(True)
            mol_psi4.fix_com(True)
            mol_psi4.update_geometry()
            self._log("Symétrie forcée C1 (fix_orientation=True, fix_com=True)")
        except Exception as sym_err:
            self._log(f"Info symétrie (non critique) : {sym_err}")

        # Optimisation géométrique
        opt_result = calc.optimize_geometry(
            mol_psi4,
            self.cfg.dft_functional,
            self.cfg.dft_basis,
            self.cfg.name,
        )

        converged  = opt_result.get("converged", False)
        opt_energy = opt_result.get("optimized_energy", None)

        if not converged:
            self._log("AVERTISSEMENT : Optimisation non convergée — géométrie initiale conservée.")

        # Sauvegarder le XYZ optimisé
        import psi4
        n = mol_psi4.natom()
        BOHR_TO_ANG = 0.529177
        xyz_lines = [str(n), f"Molecule {self.cfg.name} | DFT {self.cfg.dft_functional}/{self.cfg.dft_basis}"]
        for i in range(n):
            sym = mol_psi4.symbol(i)
            x = mol_psi4.x(i) * BOHR_TO_ANG
            y = mol_psi4.y(i) * BOHR_TO_ANG
            z = mol_psi4.z(i) * BOHR_TO_ANG
            xyz_lines.append(f"{sym:4s}  {x:12.8f}  {y:12.8f}  {z:12.8f}")

        self.xyz_path = self.cfg.output_dir / f"{self.cfg.name}.xyz"
        self.xyz_path.write_text("\n".join(xyz_lines), encoding="utf-8")
        self._log(f"Géométrie sauvegardée : {self.xyz_path}")

        self._dft_atoms  = [mol_psi4.symbol(i) for i in range(n)]
        self._dft_coords = [
            (mol_psi4.x(i) * BOHR_TO_ANG,
             mol_psi4.y(i) * BOHR_TO_ANG,
             mol_psi4.z(i) * BOHR_TO_ANG)
            for i in range(n)
        ]

        elapsed = time.time() - t0
        dft_res = {
            "success":        converged,
            "functional":     self.cfg.dft_functional,
            "basis":          self.cfg.dft_basis,
            "energy_hartree": opt_energy,
            "n_atoms":        n,
            "xyz_file":       str(self.xyz_path),
            "time_s":         round(elapsed, 2),
            "mol_psi4":       mol_psi4,
        }

        self._log(f"Optimisation : {'CONVERGÉE' if converged else 'NON CONVERGÉE (géométrie initiale)'}")
        if opt_energy is not None:
            self._log(f"Énergie DFT  : {opt_energy:.8f} Hartree")
        self._log(f"Temps DFT    : {elapsed:.1f} s")

        self.results["dft"] = dft_res
        return dft_res

    # =========================================================================
    # ÉTAPE 2 : TDDFT (optionnel)
    # =========================================================================

    def _step_tddft(self) -> Dict:
        t0 = time.time()
        import psi4
        psi4.core.clean_options()

        if self._psi4_mod is None:
            raise RuntimeError("Module psi4_calculator non chargé.")

        calc = self._psi4_mod.Psi4MolecularCalculator(
            memory=self.cfg.memory,
            threads=self.cfg.threads,
        )

        self._log(f"Solvant            : {self.cfg.tddft_solvent or 'phase gazeuse'}")
        self._log(f"Fonctionnelle TDDFT: {self.cfg.tddft_functional or 'auto'}")
        self._log(f"Base TDDFT         : {self.cfg.tddft_basis or self.cfg.dft_basis}")
        self._log(f"TDA (Tamm-Dancoff) : {'oui' if self.cfg.tddft_use_tda else 'non (full TD-DFT)'}")

        # Réutiliser la géométrie optimisée par la DFT si disponible
        dft_mol = self.results.get("dft", {}).get("mol_psi4")
        if dft_mol is not None:
            self._log("Géométrie DFT optimisée réutilisée pour TDDFT")
        else:
            self._log("ATTENTION : pas de géométrie DFT — reconstruction depuis SMILES")

        result = calc.calculate_lambda_max(
            smiles=self.cfg.smiles,
            name=self.cfg.name,
            optimize=False,
            solvent=self.cfg.tddft_solvent,
            functional=self.cfg.tddft_functional,
            geometry_functional_override=self.cfg.dft_functional,
            geometry_basis_override=self.cfg.dft_basis,
            mol_psi4=dft_mol,
            timeout_minutes=self.cfg.tddft_timeout_minutes,
            tddft_basis=self.cfg.tddft_basis or self.cfg.dft_basis,
            use_tda=self.cfg.tddft_use_tda,
        )
        elapsed = time.time() - t0
        tddft_res = {
            "success":                 result.get("calculation_successful", False),
            "lambda_max_nm":           result.get("lambda_max_nm"),
            "lambda_max_energy_eV":    result.get("lambda_max_energy_eV"),
            "max_oscillator_strength": result.get("max_oscillator_strength"),
            "all_transitions":         result.get("all_transitions", []),
            "tddft_method":            result.get("tddft_method"),
            "tddft_basis":             result.get("tddft_basis"),
            "solvent":                 self.cfg.tddft_solvent,
            "time_s":                  round(elapsed, 2),
            "raw":                     result,
        }

        if tddft_res["success"]:
            lmax = tddft_res["lambda_max_nm"]
            fosc = tddft_res["max_oscillator_strength"]
            if lmax: self._log(f"lambda max : {lmax:.1f} nm")
            if fosc is not None: self._log(f"f(osc)     : {fosc:.6f}")
        else:
            self._log(f"TDDFT échoué : {result.get('error', '?')}")

        self._log(f"Temps TDDFT : {elapsed:.1f} s")

        trans_file = self.cfg.output_dir / f"{self.cfg.name}_tddft_transitions.json"
        with open(trans_file, "w", encoding="utf-8") as fh:
            json.dump(tddft_res["all_transitions"], fh, indent=2, default=str)

        self.results["tddft"] = tddft_res
        return tddft_res

    # =========================================================================
    # ÉTAPE 3 : Densité électronique (optionnel)
    # =========================================================================

    def _step_density(self) -> Dict:
        t0 = time.time()

        if self._density_mod is None:
            raise RuntimeError("Module electron_density_analysis non chargé.")

        dens = self._density_mod
        dens.setup_psi4(memory=self.cfg.memory, threads=self.cfg.threads,
                        functional=self.cfg.density_functional,
                        basis=self.cfg.density_basis)

        self._log(f"Fonctionnelle densité : {self.cfg.density_functional}")
        self._log(f"Base densité          : {self.cfg.density_basis}")

        mol_out_dir = self.cfg.output_dir / "density"
        mol_out_dir.mkdir(parents=True, exist_ok=True)

        if self.xyz_path is None or not self.xyz_path.exists():
            raise FileNotFoundError("Fichier XYZ introuvable — l'étape DFT doit précéder.")

        geometry, charge, multiplicity, atoms, coords = dens.read_xyz_geometry(
            str(self.xyz_path), self.cfg.name
        )

        if geometry is None:
            raise RuntimeError("Échec de lecture du fichier XYZ.")

        self._log("Calcul SCF pour génération des cubes...")
        wfn, energy = dens.calculate_electron_density_robust(
            geometry, charge, multiplicity, self.cfg.name,
            functional=self.cfg.density_functional,
            basis=self.cfg.density_basis,
        )
        if wfn is None:
            raise RuntimeError("Calcul SCF densité échoué.")

        self._log("Génération fichiers .cube (densité, ESP, dual descriptor)...")
        density_cube = dens.generate_electron_density_cube(
            wfn, self.cfg.name, str(mol_out_dir), atoms=atoms, coords=coords
        )

        atomic_charges = dens.calculate_atomic_charges(wfn, atoms, self.cfg.name)

        # Extraire propriétés électroniques (HOMO/LUMO, dipôle, etc.)
        electronic_props = {}
        if hasattr(dens, 'extract_electronic_properties'):
            electronic_props = dens.extract_electronic_properties(wfn, self.cfg.name)
            for k, v in electronic_props.items():
                self._log(f"  {k}: {v}")

        coord_file = dens.save_coordinates_and_charges(
            atoms, coords, atomic_charges, energy,
            self.cfg.name, str(mol_out_dir),
        )

        elapsed = time.time() - t0
        density_res = {
            "success":          density_cube is not None,
            "density_cube":     str(density_cube) if density_cube else None,
            "output_dir":       str(mol_out_dir),
            "energy_hartree":   energy,
            "mulliken_charges": dict(zip(atoms, atomic_charges or [])),
            "coord_file":       str(coord_file) if coord_file else None,
            "time_s":           round(elapsed, 2),
            **electronic_props,
        }

        self._log(f"Cube densité  : {density_res['density_cube'] or 'non généré'}")
        self._log(f"Temps densité : {elapsed:.1f} s")

        self.results["density"] = density_res
        return density_res

    # =========================================================================
    # ÉTAPE 4 : Docking & Kd (optionnel)
    # =========================================================================

    def _step_docking(self) -> Dict:
        t0 = time.time()

        if self._docking_mod is None:
            raise RuntimeError("Module docking_kd_pipeline non chargé.")

        dock = self._docking_mod

        if not self.cfg.peptide_mol_file:
            raise ValueError("peptide_mol_file requis pour le docking (--peptide_mol).")
        if not os.path.isfile(self.cfg.peptide_mol_file):
            raise FileNotFoundError(f"Fichier peptide introuvable : {self.cfg.peptide_mol_file}")

        dock_out = self.cfg.output_dir / "binding"
        dock_out.mkdir(parents=True, exist_ok=True)

        dock.VINA_EXE            = self.cfg.vina_exe
        dock.VINA_EXHAUSTIVENESS = self.cfg.vina_exhaustiveness
        dock.VINA_N_POSES        = self.cfg.vina_n_poses
        dock.VINA_SCORING        = self.cfg.vina_scoring
        dock.VINA_N_POSES_EXPORT = min(5, self.cfg.vina_n_poses)

        self._log("Préparation récepteur...")
        receptor_file, center, size, receptor_mol = dock.prepare_receptor(
            self.cfg.peptide_mol_file, str(dock_out),
            box_override=self.cfg.box_override
        )

        if self.xyz_path is None or not self.xyz_path.exists():
            raise FileNotFoundError("XYZ ligand introuvable — étape DFT requise.")

        print("\n" + "=" * 60)
        print("ÉTAPE 2/4 — Préparation du ligand")
        print("=" * 60)
        print(f"  Ligand     : {self.cfg.name}")
        print(f"  Source XYZ : {self.xyz_path}")


        lig_mol = dock.read_xyz_to_mol(str(self.xyz_path))

        if lig_mol is None:
            raise RuntimeError("Échec lecture XYZ ligand.")

        # Copier le XYZ du ligand dans le dossier docking (seul export)
        import shutil
        ligand_xyz_dst = dock_out / f"{self.cfg.name}.xyz"
        shutil.copy2(str(self.xyz_path), str(ligand_xyz_dst))
        self._log(f"XYZ ligand : {ligand_xyz_dst}")

        # PDBQT interne pour Vina
        work_dir = dock_out / "_work"
        work_dir.mkdir(exist_ok=True)
        pdbqt_path = work_dir / f"{self.cfg.name}.pdbqt"
        dock.mol_to_ligand_pdbqt(lig_mol, str(pdbqt_path))
        ligand_files = {self.cfg.name: str(pdbqt_path)}

        self._log("Lancement docking Vina...")
        results_raw = dock.run_docking(
            receptor_file,
            ligand_files,
            center,
            size,
            str(dock_out),
        )

        df = dock.analyze_results(results_raw, str(dock_out))

        elapsed = time.time() - t0

        if len(df) > 0:
            best = df.iloc[0]
            self._log(f"Colonnes DataFrame docking : {list(df.columns)}")

            # FIX : détection robuste du nom de colonne (varie selon version)
            dg_col  = next((c for c in df.columns
                            if "dg" in c.lower() or "g_kcal" in c.lower()), None)
            kd_col  = next((c for c in df.columns
                            if "kd_m" in c.lower() or c.lower() == "kdm"), None)
            pkd_col = next((c for c in df.columns
                            if "pkd" in c.lower()), None)

            best_dg  = float(best[dg_col])  if dg_col  else None
            best_kd  = float(best[kd_col])  if kd_col  else None
            best_pkd = float(best[pkd_col]) if pkd_col else None

            # Compter les poses exportées
            n_poses_col = next((c for c in df.columns
                                if "n_poses" in c.lower()), None)
            n_poses_val = int(best[n_poses_col]) if n_poses_col else None

            docking_res = {
                "success":         True,
                "best_dG_kcalmol": best_dg,
                "best_Kd_M":       best_kd,
                "best_pKd":        best_pkd,
                "n_poses":         n_poses_val,
                "csv_file":        str(dock_out / "docking_results.csv"),
                "plot_file":       str(dock_out / "docking_results.png"),
                "poses_dir":       str(dock_out / "poses"),
                "time_s":          round(elapsed, 2),
                "df_columns":      list(df.columns),
            }
            if best_dg  is not None: self._log(f"DeltaG (meilleure pose) : {best_dg:.2f} kcal/mol")
            if best_kd  is not None: self._log(f"Kd estimé               : {best_kd:.2e} M")
            if best_pkd is not None: self._log(f"pKd                     : {best_pkd:.2f}")
            self._log(f"CSV résultats : {dock_out / 'docking_results.csv'}")
            self._log(f"Graphique     : {dock_out / 'docking_results.png'}")
            self._log(f"Poses SDF     : {dock_out / 'poses'}")
            if n_poses_val is not None:
                self._log(f"Poses trouvées : {n_poses_val}")
        else:
            docking_res = {"success": False, "time_s": round(elapsed, 2)}
            self._log("Docking échoué ou aucun résultat.")

        self._log(f"Temps docking : {elapsed:.1f} s")
        self.results["docking"] = docking_res
        return docking_res

    # =========================================================================
    # Rapport
    # =========================================================================

    def _save_report(self, steps_done: List[str]):
        # ── JSON (complet, machine-readable) ──
        report = {
            "molecule":   self.cfg.name,
            "smiles":     self.cfg.smiles,
            "date":       datetime.now().isoformat(),
            "steps_done": steps_done,
            "results":    {k: {kk: vv for kk, vv in v.items()
                               if kk not in ("mol_psi4", "raw", "all_transitions",
                                             "mulliken_charges")}
                           for k, v in self.results.items()},
        }

        json_path = self.cfg.output_dir / f"{self.cfg.name}_report.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        # ── TXT (résumé lisible) ──
        txt_path = self.cfg.output_dir / f"{self.cfg.name}_properties.txt"
        W = 70
        lines = [
            "=" * W,
            f"  PROPRIÉTÉS CALCULÉES — {self.cfg.name}",
            "=" * W,
            f"  SMILES           : {self.cfg.smiles}",
            f"  Date             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Étapes réalisées : {', '.join(steps_done)}",
            "=" * W,
        ]

        # ── DFT ──
        dft = self.results.get("dft", {})
        if dft:
            lines += ["", "─" * W, "  DFT — Optimisation de géométrie", "─" * W]
            lines.append(f"  Fonctionnelle      : {dft.get('functional', '?')}")
            lines.append(f"  Base               : {dft.get('basis', '?')}")
            lines.append(f"  Nombre d'atomes    : {dft.get('n_atoms', '?')}")
            e = dft.get("energy_hartree")
            if e is not None:
                lines.append(f"  Énergie (Hartree)  : {e:.10f}")
                lines.append(f"  Énergie (eV)       : {e * 27.2114:.6f}")
                lines.append(f"  Énergie (kcal/mol) : {e * 627.509:.4f}")
            conv = dft.get("success")
            if conv is not None:
                lines.append(f"  Convergence        : {'OUI' if conv else 'NON'}")
            if dft.get("xyz_bypass"):
                lines.append(f"  Mode               : Bypass XYZ (single-point)")
            lines.append(f"  Temps              : {dft.get('time_s', '?')} s")

        # ── TDDFT ──
        tddft = self.results.get("tddft", {})
        if tddft:
            lines += ["", "─" * W, "  TDDFT — Propriétés spectroscopiques", "─" * W]
            lines.append(f"  Méthode            : {tddft.get('tddft_method', '?')}")
            lines.append(f"  Base               : {tddft.get('tddft_basis', '?')}")
            lines.append(f"  Solvant            : {tddft.get('solvent') or 'phase gazeuse'}")
            lmax = tddft.get("lambda_max_nm")
            if lmax is not None:
                lines.append(f"  λ max              : {lmax:.1f} nm")
            e_ev = tddft.get("lambda_max_energy_eV")
            if e_ev is not None:
                lines.append(f"  Énergie excitation : {e_ev:.4f} eV")
            fosc = tddft.get("max_oscillator_strength")
            if fosc is not None:
                lines.append(f"  Force d'oscillateur: {fosc:.6f}")
            lines.append(f"  Convergence        : {'OUI' if tddft.get('success') else 'NON'}")
            lines.append(f"  Temps              : {tddft.get('time_s', '?')} s")

            # Table des transitions
            trans = tddft.get("all_transitions", [])
            if trans:
                lines += ["", f"  Transitions ({len(trans)} états) :"]
                lines.append(f"  {'État':<6} {'λ (nm)':>10} {'E (eV)':>10} {'f(osc)':>10}")
                lines.append(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
                for i, t in enumerate(trans, 1):
                    if isinstance(t, dict):
                        wl = t.get("wavelength_nm") or t.get("lambda_nm", 0)
                        ev = t.get("energy_eV", 0)
                        f  = t.get("oscillator_strength", t.get("fosc", 0))
                    elif isinstance(t, (list, tuple)) and len(t) >= 3:
                        ev, f, wl = t[0], t[1], t[2]
                    else:
                        continue
                    lines.append(f"  S{i:<5} {wl:>10.2f} {ev:>10.4f} {f:>10.6f}")

        # ── Densité ──
        dens = self.results.get("density", {})
        if dens:
            lines += ["", "─" * W, "  DENSITÉ ÉLECTRONIQUE & PROPRIÉTÉS", "─" * W]
            lines.append(f"  Énergie SCF        : {dens.get('energy_hartree', '?')}")
            lines.append(f"  Cube densité       : {'OUI' if dens.get('success') else 'NON'}")
            lines.append(f"  Dossier            : density/")
            # Propriétés électroniques
            homo = dens.get("homo_eV")
            lumo = dens.get("lumo_eV")
            gap  = dens.get("homo_lumo_gap_eV")
            if homo is not None:
                lines.append(f"  HOMO               : {homo:.4f} eV")
            if lumo is not None:
                lines.append(f"  LUMO               : {lumo:.4f} eV")
            if gap is not None:
                lines.append(f"  Gap HOMO-LUMO      : {gap:.4f} eV")
            eta = dens.get("hardness_eV")
            mu  = dens.get("chemical_potential_eV")
            omega = dens.get("electrophilicity_eV")
            if eta is not None:
                lines.append(f"  Dureté chimique η  : {eta:.4f} eV")
            if mu is not None:
                lines.append(f"  Potentiel chim. μ  : {mu:.4f} eV")
            if omega is not None:
                lines.append(f"  Électrophilicité ω : {omega:.4f} eV")
            dip = dens.get("dipole_debye")
            if dip is not None:
                dx = dens.get('dipole_x', 0)
                dy = dens.get('dipole_y', 0)
                dz = dens.get('dipole_z', 0)
                lines.append(f"  Moment dipolaire   : {dip:.4f} Debye")
                lines.append(f"    composantes (x,y,z) : ({dx:.4f}, {dy:.4f}, {dz:.4f})")
            ne = dens.get("n_electrons")
            no = dens.get("n_orbitals")
            if ne is not None:
                lines.append(f"  Nb électrons       : {ne}")
            if no is not None:
                lines.append(f"  Nb orbitales       : {no}")
            charges = dens.get("mulliken_charges", {})
            if charges:
                lines.append(f"  Charges Mulliken ({len(charges)} atomes) :")
                for atom, q in charges.items():
                    lines.append(f"    {atom:<4} : {q:+.4f}")
            lines.append(f"  Temps              : {dens.get('time_s', '?')} s")

        # ── Docking ──
        dock = self.results.get("docking", {})
        if dock:
            lines += ["", "─" * W, "  DOCKING — Affinité de liaison", "─" * W]
            dg = dock.get("best_dG_kcalmol")
            if dg is not None:
                lines.append(f"  ΔG (meilleur pose) : {dg:.2f} kcal/mol")
            kd = dock.get("best_Kd_M")
            if kd is not None:
                lines.append(f"  Kd estimé          : {kd:.2e} M")
            pkd = dock.get("best_pKd")
            if pkd is not None:
                lines.append(f"  pKd                : {pkd:.2f}")
            np_ = dock.get("n_poses")
            if np_ is not None:
                lines.append(f"  Nombre de poses    : {np_}")
            lines.append(f"  Convergence        : {'OUI' if dock.get('success') else 'NON'}")
            lines.append(f"  Dossier            : binding/")
            lines.append(f"  Temps              : {dock.get('time_s', '?')} s")

        # ── Pied de page ──
        lines += ["", "=" * W, f"  Fichiers exportés dans : {self.cfg.output_dir}", "=" * W, ""]

        txt_path.write_text("\n".join(lines), encoding="utf-8")

        self._log(f"Rapport JSON  : {json_path}")
        self._log(f"Propriétés    : {txt_path}")
        return json_path, txt_path

    # =========================================================================
    # Point d'entrée
    # =========================================================================

    def run(self, steps: Optional[List[str]] = None) -> Dict:
        if steps is None:
            steps = ["dft"]
            if self.cfg.run_tddft:   steps.append("tddft")
            if self.cfg.run_density: steps.append("density")
            if self.cfg.run_docking: steps.append("docking")

        steps = [s.lower().strip() for s in steps]
        unknown = set(steps) - set(self.AVAILABLE_STEPS)
        if unknown:
            raise ValueError(f"Étapes inconnues : {unknown}")

        if "dft" not in steps:
            steps = ["dft"] + steps
        steps = sorted(steps, key=lambda s: self.AVAILABLE_STEPS.index(s))

        self._section(f"PIPELINE MOLECULAIRE - {self.cfg.name}")
        self._log(f"Étapes : {' -> '.join(steps)}")

        self._load_modules(steps)

        steps_done = []
        t_total = time.time()

        _step_labels = {
            "dft":     "ÉTAPE 1 - Optimisation DFT",
            "tddft":   "ÉTAPE 2 - Calcul TDDFT (lambda max / UV-Vis)",
            "density": "ÉTAPE 3 - Analyse densité électronique",
            "docking": "ÉTAPE 4 - Docking moléculaire & estimation Kd",
        }

        try:
            for step in steps:
                self._section(_step_labels.get(step, step.upper()))   # ← ICI dans run()
                if step == "dft": self._step_dft()
                elif step == "tddft": self._step_tddft()
                elif step == "density": self._step_density()
                elif step == "docking": self._step_docking()
                steps_done.append(step)

        except Exception as exc:
            self._log(f"ERREUR a l'etape '{steps[len(steps_done)]}' : {exc}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self._section("RÉSUMÉ FINAL")
            total_time = time.time() - t_total
            self._log(f"Étapes réalisées : {steps_done}")
            self._log(f"Temps total      : {total_time:.1f} s ({total_time/60:.1f} min)")
            self._save_report(steps_done)

        return self.results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline SMILES -> DFT -> TDDFT -> Densite -> Docking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smiles",  required=True)
    p.add_argument("--name",    required=True)
    p.add_argument("--output_dir", default="./pipeline_results")
    p.add_argument("--steps", nargs="+", default=["dft"],
                   choices=MolecularPipeline.AVAILABLE_STEPS)
    p.add_argument("--dft_functional", default="B3LYP")
    p.add_argument("--dft_basis",      default="def2-SVP")
    p.add_argument("--memory",         default="12 GB")
    p.add_argument("--threads",        type=int, default=10)
    p.add_argument("--solvent",        default=None)
    p.add_argument("--tddft_timeout",  type=int, default=None)
    p.add_argument("--tddft_functional", default=None)
    p.add_argument("--peptide_mol",    default=None)
    p.add_argument("--vina_exe",       default="vina")
    p.add_argument("--vina_exhaustiveness", type=int, default=16)
    p.add_argument("--vina_n_poses",   type=int, default=100)
    p.add_argument("--vina_scoring",   default="vina", choices=["vina", "vinardo"])
    p.add_argument("--psi4_calc_file", default=None)
    p.add_argument("--elec_dens_file", default=None)
    p.add_argument("--docking_file",   default=None)
    return p


def main():
    args = build_parser().parse_args()
    cfg = PipelineConfig(
        smiles=args.smiles,
        name=args.name,
        output_dir=args.output_dir,
        dft_functional=args.dft_functional,
        dft_basis=args.dft_basis,
        memory=args.memory,
        threads=args.threads,
        run_tddft   = "tddft"   in args.steps,
        run_density = "density" in args.steps,
        run_docking = "docking" in args.steps,
        tddft_solvent=args.solvent,
        tddft_timeout_minutes=args.tddft_timeout,
        tddft_functional=args.tddft_functional,
        peptide_mol_file=args.peptide_mol,
        vina_exe=args.vina_exe,
        vina_exhaustiveness=args.vina_exhaustiveness,
        vina_n_poses=args.vina_n_poses,
        vina_scoring=args.vina_scoring,
        psi4_calc_file=args.psi4_calc_file,
        elec_dens_file=args.elec_dens_file,
        docking_file=args.docking_file,
    )
    MolecularPipeline(cfg).run(steps=args.steps)


if __name__ == "__main__":
    main()