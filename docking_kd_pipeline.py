#!/usr/bin/env python3
"""
=================================================================
Pipeline de Docking Moléculaire — Estimation du Kd
Petites molécules (DFT-optimisées) vs Peptide 13hS3'
Outil : AutoDock Vina (open source)
=================================================================

Théorie :
    ΔG = RT·ln(Kd)  =>  Kd = exp(ΔG / RT)
    avec R = 1.987×10⁻³ kcal/(mol·K), T = 298.15 K

Limites :
    - Vina est paramétré pour des interactions protéine-ligand.
      Pour un peptide court, les valeurs de Kd sont des estimations
      relatives (classement fiable, valeurs absolues approximatives).
    - La conformation 3D du peptide est générée in silico (ETKDG),
      pas expérimentale.
    - Pour des Kd absolus, compléter par MD + MM-PBSA (GROMACS/OpenMM).

Dépendances :
    conda install -c conda-forge rdkit numpy pandas matplotlib
    pip install meeko
    + binaire vina_1.2.7_win.exe (déjà dans Downloads)

Usage (dans l'env rdkit_env) :
    conda activate rdkit_env
    python docking_kd_pipeline.py
=================================================================
"""

import os
import sys
import glob
import time
import shutil
import warnings
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
MOLECULES_DIR = r"C:\Users\ugo.pasco\Documents\2A\Data finale MD\electron_density_results"
PEPTIDE_FILE  = r"C:\Users\ugo.pasco\Documents\2A\Data finale MD\docking_results\receptors\13hS3'-HOCCAm.mol"
OUTPUT_DIR    = r"C:\Users\ugo.pasco\Documents\2A\Data finale MD\docking_results"
VINA_EXE      = str(Path(__file__).resolve().parent / "vina_1.2.7_win.exe")

VINA_EXHAUSTIVENESS = 32    # Plus élevé = plus fiable mais plus lent (8-64)
VINA_N_POSES        = 1000  # Nombre de poses générées
VINA_N_POSES_EXPORT = 5     # Nombre de meilleures poses à exporter en SDF
VINA_SCORING        = "vinardo"  # Scoring function (vina, vinardo)
BOX_PADDING         = 5.0  # Å de marge autour du récepteur (5 Å adapté peptide/pi-pi)

MAX_LIGANDS_TEST    = None  # Limiter le nombre de ligands (None = tous)

R_KCAL = 1.987e-3  # kcal/(mol·K)
T_KELVIN = 298.15   # K

# ============================================================
# IMPORTS CHIMIE (vérification au lancement)
# ============================================================
def check_imports():
    """Vérifie que toutes les dépendances sont installées."""
    missing = []
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
    except ImportError:
        missing.append("rdkit  →  conda install -c conda-forge rdkit")
    try:
        import meeko
    except ImportError:
        missing.append("meeko  →  pip install meeko")
    if not os.path.isfile(VINA_EXE):
        missing.append(f"vina binary not found at: {VINA_EXE}")

    # Séparer les erreurs critiques (rdkit/meeko) des avertissements (vina path)
    critical = [m for m in missing if "vina binary" not in m]
    warnings = [m for m in missing if "vina binary" in m]

    if warnings:
        for w in warnings:
            print(f"[WARNING] {w}  (sera surchargé par l'interface)")

    if critical:
        print("╔══════════════════════════════════════════════════════╗")
        print("║  DÉPENDANCES MANQUANTES                             ║")
        print("╠══════════════════════════════════════════════════════╣")
        for m in critical:
            print(f"║  {m:<52} ║")
        print("╚══════════════════════════════════════════════════════╝")
        sys.exit(1)

check_imports()

import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate


# ============================================================
# UTILITAIRES
# ============================================================

def extract_protein_from_pdb(pdb_path, output_path=None):
    """
    Extrait uniquement la protéine (records ATOM) d'un fichier PDB,
    en éliminant les ligands (HETATM), eaux (HOH), ions métalliques, etc.
    Conserve les headers utiles (CRYST1, REMARK) et les connectivités protéiques.

    Args:
        pdb_path:    chemin du fichier PDB original
        output_path: chemin de sortie (défaut = _protein.pdb dans le même dossier)

    Returns:
        chemin du fichier PDB nettoyé
    """
    if output_path is None:
        stem = Path(pdb_path).stem
        output_path = str(Path(pdb_path).parent / f"{stem}_protein.pdb")

    kept_prefixes = ("HEADER", "TITLE ", "COMPND", "SOURCE", "KEYWDS",
                     "EXPDTA", "AUTHOR", "CRYST1", "SCALE", "ORIG",
                     "REMARK", "DBREF", "SEQRES", "ATOM  ", "TER   ",
                     "MODEL ", "ENDMDL", "END   ", "END")

    removed_het = set()
    n_atom = 0
    n_removed = 0

    with open(pdb_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = line[:6].ljust(6)

            if record == "HETATM":
                res_name = line[17:20].strip()
                removed_het.add(res_name)
                n_removed += 1
                continue

            if record.rstrip() in ("CONECT", "MASTER"):
                continue

            if record == "ATOM  ":
                n_atom += 1

            fout.write(line)

    print(f"  PDB nettoyé : {output_path}")
    print(f"    Atomes protéine conservés : {n_atom}")
    print(f"    HETATM supprimés          : {n_removed}")
    if removed_het:
        print(f"    Résidus retirés           : {', '.join(sorted(removed_het))}")

    return output_path


def _read_receptor_file(receptor_path):
    """
    Lit un fichier récepteur (.mol, .mol2, .pdb) et retourne un RDKit Mol 3D.
    Pour les PDB de grosse protéine, utilise rdkit.Chem.MolFromPDBFile.
    """
    ext = Path(receptor_path).suffix.lower()

    if ext == ".pdb":
        mol = Chem.MolFromPDBFile(receptor_path, removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError(f"Impossible de lire le PDB : {receptor_path}")
        # Sanitization partielle (évite les erreurs de valence sur les métaux)
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                ),
            )
        except Exception:
            pass
        return mol

    elif ext in (".mol", ".sdf"):
        mol = Chem.MolFromMolFile(receptor_path, removeHs=False, sanitize=True)
        if mol is None:
            raise ValueError(f"Impossible de lire le fichier : {receptor_path}")
        return mol

    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(receptor_path, removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError(f"Impossible de lire le Mol2 : {receptor_path}")
        return mol

    else:
        raise ValueError(f"Format non supporté : {ext}\n"
                         f"Formats acceptés : .mol, .sdf, .pdb, .mol2")


def _pdb_to_rigid_pdbqt(pdb_path, output_path):
    """
    Convertit un PDB nettoyé en PDBQT rigide pour Vina.
    Pour les grosses protéines, on convertit directement les lignes ATOM
    en format PDBQT avec types AD4 déduits de l'élément et du contexte.
    """
    # Types AD4 pour métaux / halogènes
    ad4_special = {
        "ZN": "Zn", "CA": "Ca", "FE": "Fe", "MG": "Mg",
        "MN": "Mn", "CU": "Cu", "CL": "Cl", "BR": "Br",
        "F":  "F",  "I":  "I",  "P":  "P",
    }

    def _get_ad4_type(element, atom_name):
        """Déduit le type AD4 d'un atome PDB."""
        el = element.upper()
        if el in ad4_special:
            return ad4_special[el]
        if el == "C":
            return "A" if atom_name.strip() in ("CG", "CD1", "CD2", "CE1",
                "CE2", "CZ", "CH2", "CG2") else "C"
        if el == "N":
            # N backbone ou N-H → N (donneur), N accepteur pur → NA
            name = atom_name.strip()
            if name in ("N", "NE", "NH1", "NH2", "NZ", "NE1", "NE2", "ND1", "ND2"):
                return "N"
            return "NA"
        if el == "O":
            return "OA"
        if el == "S":
            return "SA"
        if el == "H":
            return "HD"
        return el[:2]

    lines_out = []
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            padded = line.rstrip("\n").ljust(80)
            element = padded[76:78].strip().upper()
            atom_name = padded[12:16]
            if not element:
                name_s = atom_name.strip()
                element = name_s.lstrip("0123456789")[:2].upper()
                if len(element) > 1 and element not in ad4_special:
                    element = element[0]

            ad4_type = _get_ad4_type(element, atom_name)

            # Format PDBQT : cols 1-54 coords, 55-60 occ, 61-66 bfac,
            # 67-76 charge (10 chars), 77 space, 78-79 atom type (2 chars)
            pdbqt_line = (
                f"{padded[:54]}"          # record..coords (54 chars)
                f"{0.00:6.2f}"            # occupancy  (6 chars, cols 55-60)
                f"{0.00:6.2f}"            # tempFactor (6 chars, cols 61-66)
                f"{0.000:10.3f}"          # charge     (10 chars, cols 67-76)
                f" {ad4_type:>2s}"        # space + type (2 chars, cols 78-79)
            )
            lines_out.append(pdbqt_line.rstrip())

    lines_out.append("END")

    with open(output_path, "wb") as f:
        f.write("\n".join(lines_out).encode())

    print(f"  Atomes dans le PDBQT : {len(lines_out) - 1}")
    return output_path


def mol_to_rigid_pdbqt(mol, output_path):
    """
    Convertit un RDKit Mol (3D, avec H) en PDBQT rigide (récepteur).
    Utilise meeko pour assigner les types d'atomes AD4 et charges,
    puis supprime l'arbre torsionnel pour obtenir un récepteur rigide.
    """
    preparator = MoleculePreparation(
        rigid_macrocycles=True,
        hydrate=False,
    )
    mol_setups = preparator.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])

    if not is_ok:
        raise RuntimeError(f"Meeko receptor preparation failed: {error_msg}")

    # Garder uniquement les lignes ATOM/HETATM pour un récepteur rigide
    rigid_lines = []
    for line in pdbqt_string.split("\n"):
        stripped = line.rstrip("\r")
        if stripped.startswith(("ATOM", "HETATM")):
            rigid_lines.append(stripped)
    rigid_lines.append("END")

    with open(output_path, "wb") as f:
        f.write("\n".join(rigid_lines).encode())

    return output_path


def mol_to_ligand_pdbqt(mol, output_path):
    """
    Convertit un RDKit Mol (3D, avec H) en PDBQT ligand (avec torsions).
    Conserve les hydrogènes non-polaires pour garder la géométrie DFT intacte.
    Rigidifie les doubles liaisons et les amides pour éviter des torsions
    aberrantes (dièdres de 90° sur C=C, C=O d'amide, etc.).
    """
    # --- Pré-calcul des charges Gasteiger ---
    # Gasteiger peut diverger (NaN/Inf) sur P(V), S(VI), etc.
    # On les calcule ici et on remplace les valeurs non finies par 0.0
    # pour que Meeko puisse écrire le PDBQT sans erreur.
    import math
    AllChem.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
        q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if not math.isfinite(q):
            atom.SetDoubleProp("_GasteigerCharge", 0.0)

    preparator = MoleculePreparation(
        merge_these_atom_types=(),      # Ne fusionner aucun H → géométrie fidèle
        rigid_macrocycles=True,         # Éviter de tordre les macrocycles
        flexible_amides=False,          # Amides non-rotables (liaison C-N conjuguée)
        rigidify_bonds_smarts=[
            "[!#1]=[!#1]",             # Toute double liaison (C=C, C=N, C=O...)
            "[#6]#[#7]",              # C≡N (triple liaison nitrile)
            "[c]-[CX3]=[CX3]",        # Ar—C=C : conjugaison cycle-vinyle
            "[CX3]=[CX3]-[CX3]=O",    # C=C—C=O : conjugaison vinyle-carbonyle
            "[CX3]=[CX3]-[CX3]=N",    # C=C—C=N : conjugaison vinyle-imine
            "[c]-[CX3](=O)",           # Ar—C=O : conjugaison aryle-carbonyle
            "[CX3]=[CX3]-[CX3]=[CX3]", # C=C—C=C : diène conjugué (CDCAm)
            "[CX3]=[CX3]-[#6]#[#7]",  # C=C—C≡N : vinyle-nitrile conjugué
            "[c]-[CX3]=[CX3]-[CX3]",  # Ar—C=C—C : chaîne conjuguée étendue
        ],
        rigidify_bonds_indices=[
            (0, 1),                    # [!#1]=[!#1] : double liaison
            (0, 1),                    # [#6]#[#7] : triple liaison
            (0, 1),                    # [c]-[CX3]=[CX3] : Ar—C
            (1, 2),                    # [CX3]=[CX3]-[CX3]=O : C=C—C(=O)
            (1, 2),                    # [CX3]=[CX3]-[CX3]=N : C=C—C(=N)
            (0, 1),                    # [c]-[CX3](=O) : Ar—C(=O)
            (1, 2),                    # [CX3]=[CX3]-[CX3]=[CX3] : liaison simple du diène
            (1, 2),                    # [CX3]=[CX3]-[#6]#[#7] : C=C—C≡N
            (2, 3),                    # [c]-[CX3]=[CX3]-[CX3] : C=C—C dans chaîne étendue
        ],
    )
    mol_setups = preparator.prepare(mol)

    # Meeko recalcule Gasteiger en interne → re-corriger les NaN/Inf
    # sur le MoleculeSetup AVANT l'écriture PDBQT.
    setup = mol_setups[0]
    for atom in setup.atoms:
        if atom.is_dummy:
            continue
        if not math.isfinite(atom.charge):
            atom.charge = 0.0

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)

    if not is_ok:
        raise RuntimeError(f"Meeko ligand preparation failed: {error_msg}")

    # Écrire en binaire pour éviter les doubles \r\r\n sur Windows
    with open(output_path, "wb") as f:
        f.write(pdbqt_string.replace("\r", "").encode())

    return output_path


def get_box_parameters(mol):
    """
    Calcule le centre et la taille de la boîte de docking
    à partir des coordonnées 3D du récepteur.
    """
    conf = mol.GetConformer()
    coords = np.array([
        list(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])
    center = coords.mean(axis=0)
    size = (coords.max(axis=0) - coords.min(axis=0)) + 2 * BOX_PADDING
    # Vina requiert une taille minimale
    size = np.maximum(size, 22.0)
    return center, size


def compute_unified_box(receptor_mol_files):
    """
    Calcule une boîte de docking unifiée englobant tous les récepteurs.
    Élimine le biais de taille de boîte lors de la comparaison de récepteurs
    de tailles différentes (ex: naked vs taggé).

    Args:
        receptor_mol_files: liste de chemins vers les fichiers récepteur (.mol, .pdb, ...)

    Returns:
        (center, size) : centre et taille de la boîte unifiée (np.array)
    """
    from rdkit import Chem
    all_min = np.array([np.inf, np.inf, np.inf])
    all_max = np.array([-np.inf, -np.inf, -np.inf])

    for mol_file in receptor_mol_files:
        mol = _read_receptor_file(mol_file)
        mol = Chem.AddHs(mol, addCoords=True)
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i))
                           for i in range(mol.GetNumAtoms())])
        all_min = np.minimum(all_min, coords.min(axis=0))
        all_max = np.maximum(all_max, coords.max(axis=0))

    center = (all_min + all_max) / 2.0
    size = (all_max - all_min) + 2 * BOX_PADDING
    size = np.maximum(size, 22.0)

    print(f"  Boîte unifiée ({len(receptor_mol_files)} récepteurs) :")
    print(f"    Centre : ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) Å")
    print(f"    Taille : ({size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}) Å")
    return center, size


# ============================================================
# LECTURE ET PRÉPARATION MOLÉCULES
# ============================================================


def read_xyz_to_mol(xyz_path):
    """
    Lit un fichier .xyz et retourne un RDKit Mol avec les liaisons
    déterminées automatiquement (rdDetermineBonds).
    Préserve fidèlement la géométrie DFT originale.
    Gère le cas où DetermineBonds produit des fragments multiples.
    """
    raw_mol = Chem.MolFromXYZFile(xyz_path)
    if raw_mol is None:
        raise ValueError(f"Cannot read XYZ: {xyz_path}")

    def _try_determine_bonds(raw, charge=0):
        """Tente de déterminer les liaisons, retourne le mol ou None."""
        mol = Chem.RWMol(raw)
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except Exception:
            try:
                rdDetermineBonds.DetermineConnectivity(mol)
                try:
                    rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
                except Exception:
                    pass
            except Exception:
                return None
        return mol

    mol = _try_determine_bonds(raw_mol, charge=0)
    if mol is None:
        raise ValueError(f"Impossible de déterminer les liaisons : {xyz_path}")

    # --- Vérifier le nombre de fragments ---
    frags = Chem.GetMolFrags(mol)
    if len(frags) > 1:
        print(f"  ⚠ {len(frags)} fragments détectés — tentative de correction...")

        # Stratégie 1 : essayer avec charge=-1 et +1
        for alt_charge in [-1, 1]:
            mol_alt = _try_determine_bonds(raw_mol, charge=alt_charge)
            if mol_alt is not None and len(Chem.GetMolFrags(mol_alt)) == 1:
                print(f"  ✓ Corrigé avec charge={alt_charge}")
                mol = mol_alt
                break
        else:
            # Stratégie 2 : connecter manuellement les fragments les plus proches
            print(f"  → Connexion manuelle des fragments par distance...")
            conf = mol.GetConformer()
            while len(Chem.GetMolFrags(mol)) > 1:
                frags = Chem.GetMolFrags(mol)
                # Trouver la paire d'atomes inter-fragments la plus proche
                best_dist = float('inf')
                best_pair = None
                for i in frags[0]:
                    pi = conf.GetAtomPosition(i)
                    for frag_idx in range(1, len(frags)):
                        for j in frags[frag_idx]:
                            pj = conf.GetAtomPosition(j)
                            d = pi.Distance(pj)
                            if d < best_dist:
                                best_dist = d
                                best_pair = (i, j)
                if best_pair is None or best_dist > 3.0:
                    # Distance > 3 Å, c'est vraiment 2 molécules séparées
                    break
                print(f"    Liaison ajoutée : atomes {best_pair[0]}-{best_pair[1]} "
                      f"(d={best_dist:.2f} Å)")
                mol.AddBond(best_pair[0], best_pair[1], Chem.BondType.SINGLE)

            if len(Chem.GetMolFrags(mol)) > 1:
                # Stratégie 3 : garder le plus gros fragment
                frags = Chem.GetMolFrags(mol)
                biggest = max(frags, key=len)
                print(f"  ⚠ Toujours {len(frags)} fragments — conservation du "
                      f"plus gros ({len(biggest)} atomes)")
                mol = Chem.RWMol(Chem.MolFragmentToMol(mol.GetMol(), biggest))
            else:
                print(f"  ✓ Molécule reconnectée (1 fragment)")

    # --- Sanitization partielle ---
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            ),
        )
    except Exception:
        pass

    # --- Assigner la stéréochimie depuis les coordonnées 3D DFT ---
    mol_out = mol.GetMol()
    try:
        Chem.AssignStereochemistryFrom3D(mol_out)
    except Exception:
        pass

    return mol_out


# ============================================================
# ÉTAPE 1 : PRÉPARER LE RÉCEPTEUR (PEPTIDE)
# ============================================================
def prepare_receptor(mol_file, output_dir, box_override=None):
    """
    Convertit un fichier récepteur (.mol, .pdb, .mol2) en récepteur PDBQT 3D.
    Pour les PDB, extrait automatiquement la protéine si nécessaire.
    Returns:
        (pdbqt_rigid, center, size, receptor_mol)
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 1/4 — Préparation du récepteur")
    print("=" * 60)

    ext = Path(mol_file).suffix.lower()
    is_pdb = (ext == ".pdb")

    # --- Pour les PDB : nettoyage automatique (retirer HETATM) ---
    if is_pdb:
        print("  Format PDB détecté — extraction de la protéine...")
        clean_pdb = os.path.join(output_dir, "_work",
                                 Path(mol_file).stem + "_protein.pdb")
        os.makedirs(os.path.dirname(clean_pdb), exist_ok=True)
        mol_file = extract_protein_from_pdb(mol_file, clean_pdb)

    # --- Lecture du fichier récepteur ---
    mol = _read_receptor_file(mol_file)

    # Vérifier qu'il y a bien des coordonnées 3D
    if mol.GetNumConformers() == 0:
        raise ValueError("Le fichier récepteur ne contient pas de conformère 3D")

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    is_3d = np.any(coords[:, 2] != 0.0)  # z != 0 → c'est bien 3D
    print(f"  Coordonnées 3D détectées : {'OUI ✓' if is_3d else 'NON (2D !)'}")

    n_heavy = mol.GetNumHeavyAtoms()
    print(f"  Atomes lourds  : {n_heavy}")

    # Pour les grosses protéines PDB, on convertit directement le PDB en PDBQT
    # via meeko sans MMFF (trop lent / impossible sur >500 atomes lourds).
    if is_pdb and n_heavy > 500:
        print(f"  Grosse protéine détectée ({n_heavy} atomes lourds)")
        print("  → Conversion directe PDB → PDBQT (pas de MMFF)")

        receptor_work_dir = os.path.join(output_dir, "_work")
        os.makedirs(receptor_work_dir, exist_ok=True)

        pdbqt_file = os.path.join(receptor_work_dir, "receptor.pdbqt")
        _pdb_to_rigid_pdbqt(mol_file, pdbqt_file)
        print(f"  PDBQT récepteur rigide : {pdbqt_file}")

        center, size = get_box_parameters(mol)
        print(f"  Boîte de docking :")
        print(f"    Centre : ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) Å")
        print(f"    Taille : ({size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}) Å")

        return pdbqt_file, center, size, mol

    # --- Petites molécules / peptides : pipeline classique ---
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.ExactMolWt(mol)
    print(f"  Formule brute : {formula}")
    print(f"  Masse exacte  : {mw:.2f} Da")

    # Ajouter les H avec coordonnées 3D positionnées
    mol = Chem.AddHs(mol, addCoords=True)
    print(f"  Atomes totaux (+ H) : {mol.GetNumAtoms()}")

    # ----------------------------------------------------------------
    # Optimisation MMFF94s — STRATÉGIE CONSERVATIVE
    # ----------------------------------------------------------------
    # Le fichier .mol contient des coordonnées 3D fiables (OpenBabel).
    # On fixe TOUS les atomes lourds pour préserver cette géométrie :
    #   - liaisons C=C, C=O, C≡N restent parfaitement planes
    #   - angles COOH, amides, cycles aromatiques intacts
    #   - backbone peptidique immobile
    # Seuls les H ajoutés par AddHs sont optimisés (placement correct
    # des O-H, N-H, C-H sans perturber le squelette).
    # ----------------------------------------------------------------
    fixed_full = set()
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() != 1:
            fixed_full.add(i)

    n_heavy = len(fixed_full)
    n_h = mol.GetNumAtoms() - n_heavy
    print(f"  Atomes lourds (fixés)  : {n_heavy}")
    print(f"  Hydrogènes (libres)    : {n_h}")
    print("  Optimisation MMFF94s (atomes lourds fixés, H libres)...", end=" ")
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        for idx in fixed_full:
            ff.AddFixedPoint(idx)
        ff.Minimize(maxIts=2000)
        print("OK")
    except Exception as e:
        print(f"échoué ({e}), géométrie gardée telle quelle")


    # Dossier interne pour le PDBQT (nécessaire à Vina)
    receptor_work_dir = os.path.join(output_dir, "_work")
    os.makedirs(receptor_work_dir, exist_ok=True)

    # Sauvegarder le XYZ du récepteur (seul export pour visualisation)
    xyz_file = os.path.join(output_dir, "receptor_peptide.xyz")
    Chem.MolToXYZFile(mol, xyz_file)
    print(f"  XYZ récepteur : {xyz_file}")

    # Calcul de la boîte de docking
    if box_override is not None:
        center, size = box_override
        print(f"  Boîte de docking (UNIFIÉE multi-récepteur) :")
    else:
        center, size = get_box_parameters(mol)
        print(f"  Boîte de docking :")
    print(f"    Centre : ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) Å")
    print(f"    Taille : ({size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}) Å")

    # Récepteur entièrement rigide
    receptor_work_dir = os.path.join(output_dir, "_work")
    os.makedirs(receptor_work_dir, exist_ok=True)
    pdbqt_file = os.path.join(receptor_work_dir, "receptor_peptide.pdbqt")
    mol_to_rigid_pdbqt(mol, pdbqt_file)
    print(f"  PDBQT récepteur rigide : {pdbqt_file}")

    return pdbqt_file, center, size, mol


# ============================================================
# ÉTAPE 2 : PRÉPARER LES LIGANDS (PETITES MOLÉCULES)
# ============================================================
def prepare_ligands(molecules_dir, output_dir):
    """Convertit les fichiers .xyz des petites molécules en PDBQT ligands."""
    print("\n" + "=" * 60)
    print("ÉTAPE 2/4 — Préparation des ligands")
    print("=" * 60)

    ligands_dir = os.path.join(output_dir, "ligands")
    os.makedirs(ligands_dir, exist_ok=True)

    ligand_files = {}
    errors = []

    subdirs = sorted([
        d for d in os.listdir(molecules_dir)
        if os.path.isdir(os.path.join(molecules_dir, d))
        and d != "ipso_plots"
    ])

    # Limiter le nombre de ligands pour les tests
    if MAX_LIGANDS_TEST is not None:
        subdirs = subdirs[:MAX_LIGANDS_TEST]
        print(f"  ⚠ Mode test : limité aux {MAX_LIGANDS_TEST} premiers ligands")

    for i, subdir in enumerate(subdirs, 1):
        subdir_path = os.path.join(molecules_dir, subdir)

        # Chercher geom.xyz ou tout .xyz
        xyz_file = os.path.join(subdir_path, "geom.xyz")
        if not os.path.exists(xyz_file):
            xyz_candidates = glob.glob(os.path.join(subdir_path, "*.xyz"))
            if xyz_candidates:
                xyz_file = xyz_candidates[0]
            else:
                print(f"  [{i}/{len(subdirs)}] {subdir}: IGNORÉ (pas de .xyz)")
                continue

        pdbqt_file = os.path.join(ligands_dir, f"{subdir}.pdbqt")

        # Supprimer l'ancien PDBQT pour forcer la regénération
        # (nécessaire après correction du champ de force)
        if os.path.exists(pdbqt_file):
            os.remove(pdbqt_file)

        try:
            mol = read_xyz_to_mol(xyz_file)

            # Sauvegarder le XYZ du ligand (seul export pour visualisation)
            xyz_out = os.path.join(ligands_dir, f"{subdir}.xyz")
            Chem.MolToXYZFile(mol, xyz_out)

            mol_to_ligand_pdbqt(mol, pdbqt_file)
            ligand_files[subdir] = pdbqt_file
            n_atoms = mol.GetNumAtoms()
            print(f"  [{i}/{len(subdirs)}] {subdir}: OK ({n_atoms} atomes)")
        except Exception as e:
            errors.append((subdir, str(e)))
            print(f"  [{i}/{len(subdirs)}] {subdir}: ERREUR — {e}")

    print(f"\n  Ligands préparés : {len(ligand_files)}/{len(subdirs)}")
    if errors:
        print(f"  Erreurs : {len(errors)}")
        for name, err in errors:
            print(f"    - {name}: {err}")

    return ligand_files


# ============================================================
# ÉTAPE 3 : DOCKING AUTODOCK VINA
# ============================================================
def parse_vina_output(output_text):
    """Parse la sortie texte de Vina CLI pour extraire les énergies."""
    energies = []
    in_table = False
    for line in output_text.split("\n"):
        line = line.strip()
        if line.startswith("-----+"):
            in_table = True
            continue
        if in_table:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    energies.append(affinity)
                except (ValueError, IndexError):
                    if energies:  # fin du tableau
                        break
    return energies

_AD4_TO_ELEM = {
    'A':  ' C', 'C':  ' C',
    'NA': ' N', 'N':  ' N',
    'OA': ' O', 'O':  ' O',
    'SA': ' S', 'S':  ' S',
    'HD': ' H', 'H':  ' H',
    'Cl': 'Cl', 'CL': 'Cl',
    'Br': 'Br', 'BR': 'Br',
    'F':  ' F', 'I':  ' I',
    'P':  ' P', 'Zn': 'Zn',
    'Fe': 'Fe', 'Mg': 'Mg',
    'Ca': 'Ca', 'Mn': 'Mn',
    'Cu': 'Cu',
}

def _sanitize_pdbqt_line(line):
    """Remplace le type AD4 (col 77-79) par le symbole élément PDB."""
    if not line.startswith(('ATOM', 'HETATM')):
        return line
    padded = line.ljust(80)
    ad4 = padded[77:79].strip()
    elem = _AD4_TO_ELEM.get(ad4, ad4)
    return padded[:76] + f'{elem:>2s}' + padded[78:].rstrip()


def pdbqt_poses_to_sdf(pdbqt_file, poses_dir, name, energies, n_poses=5):
    """
    Extrait les N meilleures poses d'un PDBQT Vina.
    """
    import re
    import tempfile
    from rdkit import RDLogger as _RL

    mol_dir = os.path.join(poses_dir, name)
    os.makedirs(mol_dir, exist_ok=True)

    # ── 1. Parsing MODEL par MODEL ───────────────────────────────────
    with open(pdbqt_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    ligand_per_pose      = []
    ligand_topo_per_pose = []

    for idx_m, m in enumerate(re.finditer(r'MODEL\s+\d+\n(.*?)ENDMDL', content, re.DOTALL)):
        lig_all, lig_atoms = [], []
        for line in m.group(1).splitlines():
            s = line.rstrip('\r')
            if s.startswith('BEGIN_RES') or s.startswith('END_RES'):
                continue
            lig_all.append(s)
            if s.startswith(('ATOM', 'HETATM')):
                lig_atoms.append(s)
        if lig_atoms:
            ligand_per_pose.append(lig_atoms)
            ligand_topo_per_pose.append(lig_all)

    n_from_file = len(ligand_per_pose)

    # ── 2 + 3. Topologie ET géométrie via meeko (toutes les poses) ───
    n_export = min(n_poses, n_from_file, len(energies))

    tmp_pdbqt = tempfile.NamedTemporaryFile(
        mode='w', suffix='.pdbqt', delete=False, encoding='utf-8')
    for idx_pose in range(n_export):
        sanitized = [_sanitize_pdbqt_line(l) for l in ligand_topo_per_pose[idx_pose]]
        tmp_pdbqt.write(f"MODEL {idx_pose + 1}\n")
        tmp_pdbqt.write("\n".join(sanitized))
        tmp_pdbqt.write("\nENDMDL\n")
    tmp_pdbqt.close()
    topo_file = tmp_pdbqt.name

    _lg = _RL.logger()
    _lg.setLevel(_RL.CRITICAL)
    try:
        pdbqt_mol_all = PDBQTMolecule.from_file(topo_file, skip_typing=True)
        mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol_all)
    finally:
        _lg.setLevel(_RL.WARNING)
        try:
            os.unlink(topo_file)
        except OSError:
            pass

    if not mol_list or mol_list[0] is None:
        raise RuntimeError("Impossible de convertir le PDBQT en RDKit Mol")

    mol_ref  = mol_list[0]
    n_atoms  = mol_ref.GetNumAtoms()
    n_conf   = mol_ref.GetNumConformers()
    n_export = min(n_export, n_conf)
    conf_ids = [c.GetId() for c in mol_ref.GetConformers()][:n_export]

    # ── 4. RMSD entre poses ──────────────────────────────────────────
    heavy_ids = [i for i in range(n_atoms)
                 if mol_ref.GetAtomWithIdx(i).GetAtomicNum() != 1]

    def _rmsd(ca_id, cb_id):
        ca, cb = mol_ref.GetConformer(ca_id), mol_ref.GetConformer(cb_id)
        s = sum(
            (ca.GetAtomPosition(i).x - cb.GetAtomPosition(i).x)**2 +
            (ca.GetAtomPosition(i).y - cb.GetAtomPosition(i).y)**2 +
            (ca.GetAtomPosition(i).z - cb.GetAtomPosition(i).z)**2
            for i in heavy_ids
        )
        return (s / len(heavy_ids))**0.5 if heavy_ids else 0.0

    rmsd_matrix = [[0.0] * n_export for _ in range(n_export)]
    for i in range(n_export):
        for j in range(i + 1, n_export):
            v = _rmsd(conf_ids[i], conf_ids[j])
            rmsd_matrix[i][j] = rmsd_matrix[j][i] = v
    rmsd_vs_best = [rmsd_matrix[i][0] for i in range(n_export)]

    # ── 5. Export SDF par pose ───────────────────────────────────────
    n_written = 0
    for i in range(n_export):
        pose_mol = Chem.RWMol(mol_ref)
        for cid in [c.GetId() for c in pose_mol.GetConformers()]:
            if cid != conf_ids[i]:
                pose_mol.RemoveConformer(cid)
        rank = i + 1
        dg   = energies[i]
        kd   = np.exp(dg / (R_KCAL * T_KELVIN))
        pose_mol.SetProp('_Name',        f'{name}_pose{rank}')
        pose_mol.SetProp('Pose',          str(rank))
        pose_mol.SetProp('dG_kcal_mol',  f'{dg:.2f}')
        pose_mol.SetProp('Kd_M',         f'{kd:.2e}')
        pose_mol.SetProp('RMSD_vs_best', f'{rmsd_vs_best[i]:.2f}')
        sdf_path = os.path.join(mol_dir, f'{name}_pose{rank}.sdf')
        writer = Chem.SDWriter(sdf_path)
        writer.write(pose_mol)
        writer.close()
        n_written += 1

    # ── 6. Résumé convergence ────────────────────────────────────────
    print(f'\n  {"─"*50}')
    print(f'  Convergence des {n_export} meilleures poses ({name}) :')
    print(f'  {"Pose":>6} {"ΔG (kcal/mol)":>14} {"RMSD vs #1 (Å)":>16}')
    for i in range(n_export):
        print(f'  {"#"+str(i+1):>6} {energies[i]:>14.2f} {rmsd_vs_best[i]:>16.2f}')

    pairwise = [rmsd_matrix[i][j]
                for i in range(n_export)
                for j in range(i + 1, n_export)]
    if pairwise:
        mean_r, max_r = np.mean(pairwise), np.max(pairwise)
        verdict = ("✓ CONVERGENCE FORTE"    if mean_r < 1.0 else
                   "◯ CONVERGENCE MODÉRÉE"  if mean_r < 2.0 else
                   "⚠ CONVERGENCE FAIBLE")
        print(f'  RMSD moyen inter-poses : {mean_r:.2f} Å (max : {max_r:.2f} Å)')
        print(f'  {verdict}')
    print(f'  {"─"*50}')
    print(f'  Dossier : {os.path.abspath(mol_dir)}')
    print(f'  Poses   : {n_written} fichier(s) SDF')

    return n_written


def run_docking(receptor_file, ligand_files, center, size, output_dir):
    """Exécute le docking Vina (CLI) pour chaque ligand contre le peptide."""
    print("\n" + "=" * 60)
    print("ÉTAPE 3/4 — Docking AutoDock Vina (CLI)")
    print("=" * 60)

    _scoring = VINA_SCORING.strip().lower() if isinstance(VINA_SCORING, str) else "vina"
    if _scoring not in ("vina", "vinardo"):
        raise ValueError(
            f"Scoring '{VINA_SCORING}' non supporté. "
            f"Utilisez 'vina' ou 'vinardo'. "
            f"('ad4' nécessite AutoGrid4 non inclus.)")

    print(f"  Binaire Vina : {VINA_EXE}")
    print(f"  Scoring      : {VINA_SCORING}")
    print(f"  Exhaustivité : {VINA_EXHAUSTIVENESS}")
    print(f"  Poses par ligand : {VINA_N_POSES}")
    print(f"  Nombre de ligands : {len(ligand_files)}")

    poses_dir = os.path.join(output_dir, "poses")
    os.makedirs(poses_dir, exist_ok=True)

    work_dir = os.path.join(output_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    checkpoint_file = os.path.join(output_dir, "docking_checkpoint.csv")
    already_docked  = set()
    ckpt_df         = None
    if os.path.exists(checkpoint_file):
        ckpt_df    = pd.read_csv(checkpoint_file)
        valid_mask = ckpt_df["dG_kcal_mol"].notna()
        ckpt_df    = ckpt_df[valid_mask].reset_index(drop=True)
        already_docked = set(ckpt_df["Molecule"].values)
        print(f"  Checkpoint trouvé : {len(already_docked)} molécules déjà dockées (valides)")

    results        = []
    total          = len(ligand_files)
    t_start_global = time.time()

    for i, (name, ligand_file) in enumerate(ligand_files.items(), 1):

        if name in already_docked:
            row = ckpt_df[ckpt_df["Molecule"] == name].iloc[0]
            results.append(row.to_dict())
            print(f"  [{i}/{total}] {name}: checkpoint ✓ "
                  f"(ΔG = {row['dG_kcal_mol']:.2f} kcal/mol)")
            continue

        t_start = time.time()
        print(f"  [{i}/{total}] {name}...", end=" ", flush=True)

        try:
            pose_file = os.path.join(work_dir, f"{name}_docked.pdbqt")

            cmd = [VINA_EXE,
                "--receptor", receptor_file,
                "--ligand", ligand_file,
                "--center_x",      f"{center[0]:.3f}",
                "--center_y",      f"{center[1]:.3f}",
                "--center_z",      f"{center[2]:.3f}",
                "--size_x",        f"{size[0]:.3f}",
                "--size_y",        f"{size[1]:.3f}",
                "--size_z",        f"{size[2]:.3f}",
                "--out", pose_file,
                "--exhaustiveness", str(VINA_EXHAUSTIVENESS),
                "--num_modes", str(VINA_N_POSES),
                "--scoring", VINA_SCORING,
                "--energy_range", "3"]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=6000,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Vina exit code {proc.returncode}: {proc.stderr[:200]}")

            energies = parse_vina_output(proc.stdout)
            if not energies:
                raise RuntimeError("Aucune énergie trouvée dans la sortie Vina")

            best_dg = energies[0]
            kd      = np.exp(best_dg / (R_KCAL * T_KELVIN))
            pkd     = -np.log10(kd) if kd > 0 else float("nan")
            elapsed = time.time() - t_start

            result = {
                "Molecule":     name,
                "dG_kcal_mol":  round(best_dg, 2),
                "Kd_M":         kd,
                "pKd":          round(pkd, 2),
                "n_poses":      len(energies),
                "time_s":       round(elapsed, 1),
            }
            results.append(result)

            try:
                n_exported = pdbqt_poses_to_sdf(
                    pose_file, poses_dir, name, energies,
                    n_poses=VINA_N_POSES_EXPORT)


            except Exception as export_err:
                n_exported = 0
                print(f"  ⚠ Export SDF échoué pour {name}: {export_err}")

            print(f"ΔG = {best_dg:.2f} kcal/mol | "
                  f"Kd = {kd:.2e} M | "
                  f"pKd = {pkd:.1f} | "
                  f"poses exportées: {n_exported} | "
                  f"{elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"ERREUR ({elapsed:.1f}s) — {e}")
            results.append({
                "Molecule":    name,
                "dG_kcal_mol": None,
                "Kd_M":        None,
                "pKd":         None,
                "n_poses":     0,
                "time_s":      round(elapsed, 1),
            })

        pd.DataFrame(results).to_csv(checkpoint_file, index=False)

    elapsed_total = time.time() - t_start_global
    print(f"\n  Temps total de docking : {elapsed_total / 60:.1f} min")
    print(f"  Dossier des poses      : {os.path.abspath(poses_dir)}")

    return results


# ============================================================
# ÉTAPE 4 : ANALYSE ET VISUALISATION
# ============================================================
def analyze_results(results, output_dir):
    """Génère le tableau récapitulatif, CSV et graphiques."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("ÉTAPE 4/4 — Analyse des résultats")
    print("=" * 60)

    df = pd.DataFrame(results)
    df = df.dropna(subset=["dG_kcal_mol"])
    df = df.sort_values("dG_kcal_mol")
    df = df.reset_index(drop=True)

    # ---- CSV ----
    csv_file = os.path.join(output_dir, "docking_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"  CSV : {csv_file}")

    # ---- Statistiques ----
    print(f"\n  Molécules dockées : {len(df)}")
    if df.empty:
        print("  ⚠ Aucune molécule dockée avec succès — vérifiez les logs Vina ci-dessus.")
        return df

    print(f"  ΔG moyen  : {df['dG_kcal_mol'].mean():.2f} kcal/mol")
    print(f"  ΔG min    : {df['dG_kcal_mol'].min():.2f} kcal/mol "
          f"({df.iloc[0]['Molecule']})")
    print(f"  ΔG max    : {df['dG_kcal_mol'].max():.2f} kcal/mol "
          f"({df.iloc[-1]['Molecule']})")

    # ---- Graphiques ----
    n_mol = len(df)
    fig_height = max(8, n_mol * 0.32)

    fig, axes = plt.subplots(1, 2, figsize=(18, fig_height))

    # Couleurs par affinité
    def color_dg(dg):
        if dg < -6:
            return "#1abc9c"   # Forte affinité
        elif dg < -4:
            return "#2ecc71"   # Bonne
        elif dg < -3:
            return "#f39c12"   # Modérée
        else:
            return "#e74c3c"   # Faible

    colors = [color_dg(dg) for dg in df["dG_kcal_mol"]]

    # --- Plot 1 : ΔG ---
    ax = axes[0]
    y_pos = range(n_mol)
    ax.barh(y_pos, df["dG_kcal_mol"], color=colors, edgecolor="gray",
            linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Molecule"], fontsize=7)
    ax.set_xlabel("ΔG (kcal/mol)", fontsize=11)
    ax.set_title("Énergie libre de liaison (AutoDock Vina)", fontsize=12)
    ax.invert_yaxis()
    ax.axvline(x=-6, color="gray", linestyle="--", alpha=0.4, label="ΔG = -6")
    ax.axvline(x=-4, color="gray", linestyle=":", alpha=0.4, label="ΔG = -4")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)

    # --- Plot 2 : pKd ---
    ax2 = axes[1]
    ax2.barh(y_pos, df["pKd"], color=colors, edgecolor="gray",
             linewidth=0.3, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df["Molecule"], fontsize=7)
    ax2.set_xlabel("pKd  (−log₁₀ Kd)", fontsize=11)
    ax2.set_title("Affinité de liaison (pKd)", fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "docking_results.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graphique : {plot_file}")

    # ---- Tableau console ----
    print("\n" + "=" * 72)
    print(f"{'Rang':<5} {'Molécule':<25} {'ΔG (kcal/mol)':>14} "
          f"{'Kd (M)':>14} {'pKd':>8}")
    print("-" * 72)
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        kd_str = f"{row['Kd_M']:.2e}"
        print(f"{rank:<5} {row['Molecule']:<25} {row['dG_kcal_mol']:>14.2f} "
              f"{kd_str:>14} {row['pKd']:>8.2f}")
    print("-" * 72)

    best = df.iloc[0]
    print(f"\n  ★ Meilleur binder : {best['Molecule']}")
    print(f"    ΔG = {best['dG_kcal_mol']:.2f} kcal/mol")
    print(f"    Kd = {best['Kd_M']:.2e} M")
    print(f"    pKd = {best['pKd']:.2f}")

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("╔════════════════════════════════════════════════════════╗")
    print("║  PIPELINE DE DOCKING — Estimation du Kd              ║")
    print("║  Petites molécules (DFT) vs Peptide 13hS3'           ║")
    print("║  Outil : AutoDock Vina (open source)                 ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n  Répertoire molécules : {MOLECULES_DIR}")
    print(f"  Peptide              : {PEPTIDE_FILE}")
    print(f"  Résultats            : {OUTPUT_DIR}")

    # Étape 1 : Récepteur
    receptor_file, center, size, receptor_mol = prepare_receptor(PEPTIDE_FILE, OUTPUT_DIR)
    

    # Étape 2 : Ligands
    ligand_files = prepare_ligands(MOLECULES_DIR, OUTPUT_DIR)

    if not ligand_files:
        print("\n  ERREUR : aucun ligand n'a pu être préparé.")
        sys.exit(1)

    # Étape 3 : Docking
    results = run_docking(receptor_file, ligand_files, center, size, OUTPUT_DIR)

    # Étape 4 : Analyse
    df = analyze_results(results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  PIPELINE TERMINÉ")
    print(f"  Tous les résultats dans : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
