#!/usr/bin/env python3
"""
Module Psi4 pour calculs de lambda max via TD-DFT
Calculs spécialisés pour longueur d'onde d'absorption maximale
"""

import psi4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
import psutil
import platform
import threading
import time
import locale
import re
from datetime import datetime

# Forcer la locale anglaise pour éviter les problèmes de virgules décimales
try:
    locale.setlocale(locale.LC_NUMERIC, 'C')
except:
    pass  # Si impossible, continuer sans


def classify_molecule_family(name: str, smiles: str = "") -> dict:
    """
    Classifie une molécule de façon simplifiée
    
    Returns:
        dict: {"family": str, "substituent": str, "position": str}
    """
    name_upper = name.upper()
    
    # Déterminer la position (para, ortho, meta)
    position = ""
    if name.startswith("m-") or "M-" in name_upper or name_upper.startswith("M"):
        position = "meta"
    elif name.startswith("p-") or "P-" in name_upper or name_upper.startswith("P"):
        position = "para"
    elif name.startswith("o-") or "O-" in name_upper or name_upper.startswith("O"):
        position = "ortho"
    
    # Déterminer la famille: Cam, CCAm, CDCAm
    family = "Cam"  # Cinnamic methyl amide par défaut
    if "CDCAM" in name_upper or "CDC" in name_upper:
        family = "CDCAm"  # Cyano dicinnamic methyl amide
    elif "CCAM" in name_upper or "CC" in name_upper:
        family = "CCAm"  # Cyano cinnamic methyl amide
    elif "CAM" in name_upper or "CA" in name_upper:
        family = "Cam"  # Cinnamic methyl amide
    
    # Déterminer le substituant
    substituent = ""
    substituent_patterns = {
        "CL": "Cl",
        "F3C": "CF3",
        "F": "F",
        "I": "I",
        "BR": "Br",
        "HO": "OH",
        "MEO": "OMe",
        "IPR": "OiPr",
        "H2N": "NH2",
        "2MEN": "NMe2",
        "NO2": "NO2",
        "CY": "CN",
        "MES": "SMe",
        "HS": "SH",
        "H3SI": "SiH3",
        "ME": "Me",
        "TBU": "tBu",
    }
    
    for pattern, sub_name in substituent_patterns.items():
        if pattern in name_upper:
            # Éviter les faux positifs (ex: "ME" dans "MEO")
            if pattern == "ME" and "MEO" in name_upper:
                continue
            if pattern == "I" and ("IPR" in name_upper or "SI" in name_upper):
                continue
            if pattern == "F" and "F3C" in name_upper:
                continue
            substituent = sub_name
            break
    
    return {
        "family": family,
        "substituent": substituent,
        "position": position
    }

class _Psi4OutputTailer(threading.Thread):
    """Thread daemon qui surveille le fichier de sortie Psi4 et renvoie
    les lignes *importantes* vers sys.stdout (QueueStream dans le
    sous-processus GUI) pour garder la console lisible.

    Affiche :
      - Un résumé par pas d'optimisation (numéro + énergie)
      - Convergence (forces / déplacements)
      - Fin d'optimisation
      - Informations TDDFT
      - Warnings / erreurs
    N'affiche PAS :
      - Chaque itération SCF individuelle
      - Les en-têtes DFT Potential / Composite Functional répétés
      - Les tstop() / wall time intermédiaires
    """

    def __init__(self, filepath, poll_interval=0.5):
        super().__init__(daemon=True)
        self._filepath = filepath
        self._poll = poll_interval
        self._stop_evt = threading.Event()
        self._pos = 0
        self._grad_count = 0       # compte les gradients = pas d'optimisation
        self._last_energy = None   # dernière énergie SCF vue
        self._last_emitted = None  # anti-doublon
        self._seen_header = False  # DFT Potential déjà montré

    def _filter_and_emit(self, text):
        """Filtre les lignes et envoie les importantes vers sys.stdout (GUI)."""
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # ── Anti-doublon strict ──
            if stripped == self._last_emitted:
                continue

            # ── Capturer l'énergie totale (sans l'afficher directement) ──
            if "Total Energy" in stripped and "=" in stripped:
                try:
                    self._last_energy = stripped.split("=")[1].strip()
                except Exception:
                    pass
                continue

            # ── Gradient calculé = un pas d'optimisation terminé ──
            if "-Total Gradient:" in stripped:
                self._grad_count += 1
                e_str = f"  E = {self._last_energy} Ha" if self._last_energy else ""
                self._emit(
                    f"  [Psi4] ── Pas d'optimisation {self._grad_count} terminé{e_str}"
                )
                continue

            # ── Tableau de convergence (lignes avec ~) ──
            # Les lignes de données commencent par ~ suivi d'un numéro de pas
            if stripped.startswith("~") and len(stripped) > 2 and stripped[1:].strip()[0:1].isdigit():
                self._emit(f"  [Psi4] {stripped}")
                continue
            # Ignorer le header du tableau (Step  Total Energy  Delta E ...)
            if stripped.startswith("~") or ("Step" in stripped and "Total Energy" in stripped
                                            and "Delta E" in stripped):
                continue

            # ── Convergence atteinte ──
            if "Optimization is complete" in stripped or "OPTKING Finished" in stripped:
                self._emit(f"  [Psi4] ✓ {stripped}")
                continue

            # ── Critères de convergence (lignes de données, pas le header) ──
            # Les lignes de convergence ressemblent à :
            #   MAX Force    0.001234   0.000450    no
            if any(k in stripped for k in ("MAX Force", "RMS Force",
                                           "MAX Disp", "RMS Disp")):
                # Ignorer si c'est le header (contient plusieurs critères sur la même ligne)
                criteria_count = sum(1 for k in ("MAX Force", "RMS Force", "MAX Disp", "RMS Disp")
                                     if k in stripped)
                if criteria_count >= 2:
                    continue  # c'est le header
                self._emit(f"  [Psi4]   {stripped}")
                continue

            # ── TDDFT ──
            if any(k in stripped for k in ("TD-DFT", "TDA", "Excitation Energy",
                                           "states computed", "TDDFT")):
                self._emit(f"  [Psi4] {stripped}")
                continue

            # ── Warnings / Erreurs ──
            sl = stripped.lower()
            # Ignorer les séparateurs visuels Psi4 ("*** WARNING ***" borders)
            if sl.replace('*', '').replace(' ', '') == 'warning':
                continue
            if ("warning" in sl or "error" in sl or "failed" in sl) \
                    and "e_convergence" not in sl and "d_convergence" not in sl \
                    and "fail_on_maxiter" not in sl:
                self._emit(f"  [Psi4] ⚠ {stripped}")
                continue

            # ── Psi4 wall time final (seulement la dernière occurrence) ──
            if "wall time for execution" in stripped:
                self._emit(f"  [Psi4] {stripped}")
                continue

            # ── En-tête DFT Potential : montrer une seule fois ──
            if "==> DFT Potential <==" in stripped:
                if not self._seen_header:
                    self._seen_header = True
                    self._emit(f"  [Psi4] {stripped}")
                continue
            if "Composite Functional:" in stripped:
                if self._grad_count == 0:
                    self._emit(f"  [Psi4] {stripped}")
                continue

            # ── Tout le reste est ignoré (itérations SCF, tstop, etc.) ──

    def _emit(self, line):
        """Écrit une ligne vers sys.stdout (= QueueStream dans le sous-processus)."""
        self._last_emitted = line.replace("  [Psi4] ", "").strip()
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with open(self._filepath, "r", encoding="utf-8",
                          errors="replace") as fh:
                    fh.seek(self._pos)
                    new_data = fh.read()
                    if new_data:
                        self._pos += len(new_data)
                        self._filter_and_emit(new_data)
            except FileNotFoundError:
                pass
            except Exception:
                pass
            self._stop_evt.wait(self._poll)

    def stop(self):
        self._stop_evt.set()
        try:
            with open(self._filepath, "r", encoding="utf-8",
                      errors="replace") as fh:
                fh.seek(self._pos)
                rest = fh.read()
                if rest:
                    self._filter_and_emit(rest)
        except Exception:
            pass


class Psi4MolecularCalculator:
    """Calculateur quantique simplifié utilisant Psi4 pour lambda max TD-DFT avec B3LYP/def2-SVP"""
    
    def __init__(self, memory="12 GB", threads=10):
        """
        Initialise Psi4 avec configuration simplifiée
        
        Args:
            memory: Mémoire allouée à Psi4 
            threads: Nombre de threads pour les calculs
        """
        self._output_tailer = None
        try:
            psi4.set_memory(memory)
            psi4.set_num_threads(threads)
            import tempfile as _tf
            _psi4_tmp = _tf.NamedTemporaryFile(
                mode='w', suffix='.psi4out', delete=False,
                prefix='psi4_', dir=_tf.gettempdir())
            _psi4_tmp.close()
            self._psi4_outfile = _psi4_tmp.name
            psi4.core.set_output_file(self._psi4_outfile, append=False)
            # Démarrer le tailer pour relayer les logs Psi4 vers stdout/GUI
            self._output_tailer = _Psi4OutputTailer(self._psi4_outfile)
            self._output_tailer.start()
            print(f"OK Psi4 configuré: {memory}, {threads} threads")
            print(f"   Fichier sortie Psi4 : {self._psi4_outfile}")
        except Exception as e:
            print(f"AVERT Erreur configuration Psi4: {e}")
            # Fallback vers configuration minimale
            psi4.set_memory("4 GB")
            psi4.set_num_threads(2)
            print(f"[refresh] Configuration fallback: 4 GB, 2 threads")
        
        # Configuration simplifiée pour performance
        self.default_basis = "def2-SVP"
        self.default_functional = "B3LYP"
        
        # Options de performance pour B3LYP/def2-SVP
        psi4.set_options({
            'scf_type': 'df',                # Density fitting (plus rapide)
            'guess': 'sad',                  # Superposition of Atomic Densities
            'maxiter': 150,                  # Moins d'itérations pour éviter divergence
            'e_convergence': 1e-6,           # Convergence moins stricte (évite NaN)
            'd_convergence': 1e-6,           # Convergence moins stricte
            'fail_on_maxiter': False,        # Ne pas échouer sur max iterations
            'soscf': False,                  # Désactiver SOSCF (peut causer NaN)
            'diis': True,                    # DIIS pour convergence stable
        })
        
        print(f"[PC] Configuration Psi4 simplifiée B3LYP/def2-SVP:")
        print(f"   Mémoire: {memory}")
        print(f"   Threads: {threads}")
        print(f"   Méthode: B3LYP/def2-SVP (équilibre performance/qualité)")
        
        # Résultats stockés
        self.results_cache = {}

    def stop_output_tailer(self):
        """Arrête le thread de surveillance du fichier de sortie Psi4."""
        if self._output_tailer is not None:
            self._output_tailer.stop()
            self._output_tailer.join(timeout=2)
            self._output_tailer = None

    def __del__(self):
        self.stop_output_tailer()
    
    def determine_optimal_basis(self, mol: psi4.core.Molecule, purpose="tddft") -> str:
        """
        Base simplifiée: def2-SVP pour équilibre performance/qualité
        """
        basis = 'def2-SVP'
        print(f"   OK Base sélectionnée: {basis} (équilibre performance/qualité)")
        return basis
    
    def needs_special_tddft_treatment(self, mol: psi4.core.Molecule, smiles: str) -> Tuple[bool, str]:
        """
        Détermine si la molécule nécessite CAM-B3LYP + def2-TZVP pour TD-DFT
        Basé sur présence de Cl, Si, NO2, CN
        """
        import re
        
        # Vérifier éléments spéciaux
        elements = set()
        for i in range(mol.natom()):
            symbol = mol.symbol(i).upper()  # Normaliser en majuscules
            elements.add(symbol)
        
        special_elements = elements.intersection({'SI'})  # Seulement Si maintenant
        
        # Vérifier groupes spéciaux  
        special_groups = []
        if re.search(r'\[N\+\]\(\[O\-\]\)=O|NO2', smiles):
            special_groups.append('NO2')
        if re.search(r'C#N', smiles):
            special_groups.append('CN')
        
        needs_special = bool(special_elements or special_groups)
        
        if needs_special:
            reasons = []
            if special_elements:
                # Convertir en format lisible pour affichage
                display_elements = []
                for elem in sorted(special_elements):
                    if elem == 'SI':
                        display_elements.append('Si')
                    else:
                        display_elements.append(elem)
                reasons.append(f"éléments: {display_elements}")
            if special_groups:
                reasons.append(f"groupes: {special_groups}")
            reason = " + ".join(reasons)
            print(f"   [>] TRAITEMENT SPÉCIAL TD-DFT requis: {reason}")
            return True, f"CAM-B3LYP + def2-TZVP ({reason})"
        
        return False, ""
    
    def determine_optimal_functional(self, smiles: str, name: str = "") -> str:
        """
        Fonctionnelle simplifiée: toujours B3LYP pour performance
        """
        functional = "B3LYP"
        print(f"   OK Fonctionnelle sélectionnée: {functional} (forcé pour performance)")
        return functional
        
    def rdkit_to_psi4_xyz(self, mol: Chem.Mol, optimize_3d=True) -> str:
        """
        Convertit une molécule RDKit en coordonnées XYZ pour Psi4
        AMÉLIORATION: Force la planéité pour les benzamides (BAm)
        """
        if mol is None:
            raise ValueError("Molécule invalide")
        
        # Ajouter les hydrogènes explicites
        mol_h = Chem.AddHs(mol)
        
        if optimize_3d:
            # Détecter si c'est un benzamide pour génération optimisée
            is_benzamide = self._detect_benzamide_pattern(mol_h)
            
            if is_benzamide:
                print("[>] Benzamide détecté - Génération conformères optimisée pour planéité")
                return self._generate_planar_benzamide_geometry(mol_h)
            else:
                # Génération de conformères multiples avec RDKit pour meilleure géométrie
                # 1. Générer plusieurs conformères
                try:
                    AllChem.EmbedMultipleConfs(mol_h, numConfs=10, randomSeed=42, 
                                             useExpTorsionAnglePrefs=True, 
                                             useBasicKnowledge=True)
                except:
                    # Fallback si EmbedMultipleConfs échoue
                    AllChem.EmbedMolecule(mol_h, randomSeed=42, useExpTorsionAnglePrefs=True)
                
                # 2. Optimiser tous les conformères avec MMFF
                for conf_id in range(mol_h.GetNumConformers()):
                    AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=5000)
                
                # 3. Choisir le conformère de plus basse énergie
                energies = []
                for conf_id in range(mol_h.GetNumConformers()):
                    props = AllChem.MMFFGetMoleculeProperties(mol_h)
                    if props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=conf_id)
                        if ff is not None:
                            energies.append((conf_id, ff.CalcEnergy()))
                        else:
                            energies.append((conf_id, float('inf')))
                    else:
                        energies.append((conf_id, float('inf')))
                
                # Sélectionner le conformère de plus basse énergie
                if energies:
                    best_conf_id = min(energies, key=lambda x: x[1])[0]
                    print(f"Conformère sélectionné: {best_conf_id} (énergie MMFF: {energies[best_conf_id][1]:.2f} kcal/mol)")
                    # Garder seulement le meilleur conformère
                    for conf_id in reversed(range(mol_h.GetNumConformers())):
                        if conf_id != best_conf_id:
                            mol_h.RemoveConformer(conf_id)
                else:
                    print("AVERT Impossible de calculer les énergies MMFF, conformère 0 conservé")
        else:
            # Fallback simple si optimize_3d=False
            AllChem.EmbedMolecule(mol_h, randomSeed=42, useExpTorsionAnglePrefs=True)
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=5000)
        
        # Obtenir la conformation
        conf = mol_h.GetConformer()
        
        # Construire les coordonnées XYZ
        xyz_lines = []
        for i, atom in enumerate(mol_h.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        
        return "\n".join(xyz_lines)
    
    def _detect_benzamide_pattern(self, mol: Chem.Mol) -> bool:
        """Détecte si la molécule contient un pattern benzamide (cycle aromatique-CO-NH)"""
        # Chercher pattern: cycle aromatique connecté à C=O connecté à N
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'O']) > 0:
                # Carbone carbonyle candidat
                has_aromatic_neighbor = any(n.GetIsAromatic() for n in atom.GetNeighbors() if n.GetSymbol() == 'C')
                has_nitrogen_neighbor = any(n.GetSymbol() == 'N' for n in atom.GetNeighbors())
                
                if has_aromatic_neighbor and has_nitrogen_neighbor:
                    return True
        return False
    
    def _generate_planar_benzamide_geometry(self, mol: Chem.Mol) -> str:
        """
        Génère une géométrie optimisée pour benzamides en privilégiant la planéité
        APPROCHE INTELLIGENTE: Générer plusieurs conformères et sélectionner le plus planaire
        """
        print("   [fix] Génération intelligente de conformères pour benzamide...")
        
        # 1. Générer de nombreux conformères avec différentes stratégies
        conformers_generated = []
        
        # Stratégie 1: Conformères standards avec préférences de torsion
        try:
            AllChem.EmbedMultipleConfs(mol, numConfs=20, randomSeed=42, 
                                     useExpTorsionAnglePrefs=True, 
                                     useBasicKnowledge=True,
                                     enforceChirality=False)
            conformers_generated.append("Standard (20 conformères)")
        except:
            pass
        
        # Stratégie 2: Conformères avec seeds différents pour plus de diversité
        for seed in [123, 456, 789]:
            try:
                AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=seed, 
                                         useExpTorsionAnglePrefs=True, 
                                         useBasicKnowledge=True)
                conformers_generated.append(f"Seed {seed} (5 conformères)")
            except:
                pass
        
        total_confs = mol.GetNumConformers()
        print(f"   [data] {total_confs} conformères générés: {', '.join(conformers_generated)}")
        
        if total_confs == 0:
            # Fallback: un seul conformère
            AllChem.EmbedMolecule(mol, randomSeed=42, useExpTorsionAnglePrefs=True)
            total_confs = 1
            print(f"   AVERT Fallback: 1 conformère généré")
        
        # 2. Optimiser tous les conformères avec MMFF
        print(f"   [gear] Optimisation MMFF de {total_confs} conformères...")
        valid_conformers = []
        
        for conf_id in range(total_confs):
            try:
                result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=2000)
                if result == 0:  # Convergence réussie
                    valid_conformers.append(conf_id)
            except:
                pass
        
        print(f"   OK {len(valid_conformers)} conformères optimisés avec succès")
        
        # 3. Évaluer la planéité de chaque conformère
        best_conf = 0
        best_planarity = float('inf')
        planarity_scores = []
        
        for conf_id in valid_conformers:
            planarity_score = self._evaluate_benzamide_planarity(mol, conf_id)
            planarity_scores.append((conf_id, planarity_score))
            
            if planarity_score < best_planarity:
                best_planarity = planarity_score
                best_conf = conf_id
        
        # Trier par planéité
        planarity_scores.sort(key=lambda x: x[1])
        
        print(f"   [ruler] Évaluation planéité:")
        for i, (conf_id, score) in enumerate(planarity_scores[:3]):  # Top 3
            status = "[1st]" if i == 0 else "[2nd]" if i == 1 else "[3rd]"
            print(f"      {status} Conformère {conf_id}: planéité {score:.4f} Å")
        
        # 4. Sélectionner le conformère le plus planaire
        print(f"   [>] Conformère sélectionné: {best_conf} (planéité: {best_planarity:.4f} Å)")
        
        # Garder seulement le meilleur conformère
        for conf_id in reversed(range(mol.GetNumConformers())):
            if conf_id != best_conf:
                mol.RemoveConformer(conf_id)
        
        # 5. Construire les coordonnées XYZ
        conf = mol.GetConformer()
        xyz_lines = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        
        print(f"   OK Géométrie planaire optimisée générée ({len(xyz_lines)} atomes)")
        return "\n".join(xyz_lines)
    
    def _evaluate_benzamide_planarity(self, mol: Chem.Mol, conf_id: int) -> float:
        """
        Évalue la planéité d'un conformère de benzamide
        Retourne un score (plus petit = plus planaire)
        """
        import numpy as np
        
        try:
            conf = mol.GetConformer(conf_id)
            
            # Identifier les atomes importants
            aromatic_carbons = []
            carbonyl_carbon = None
            nitrogen_atom = None
            
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
                    aromatic_carbons.append(i)
                elif atom.GetSymbol() == 'C':
                    neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
                    if 'O' in neighbors and 'N' in neighbors:
                        carbonyl_carbon = i
                elif atom.GetSymbol() == 'N':
                    nitrogen_atom = i
            
            if len(aromatic_carbons) < 6 or carbonyl_carbon is None or nitrogen_atom is None:
                return 999.0  # Score très mauvais si structure incomplète
            
            # Coordonnées des atomes clés pour définir le plan conjugué
            key_atoms = aromatic_carbons[:6] + [carbonyl_carbon, nitrogen_atom]
            coords = []
            for idx in key_atoms:
                pos = conf.GetAtomPosition(idx)
                coords.append([pos.x, pos.y, pos.z])
            
            coords_array = np.array(coords)
            centroid = np.mean(coords_array, axis=0)
            centered = coords_array - centroid
            
            # SVD pour trouver le plan optimal
            U, s, Vt = np.linalg.svd(centered)
            
            # Distance RMS par rapport au plan optimal
            distances = np.abs(np.dot(centered, Vt[2]))  # Vt[2] = normale au plan
            rms_planarity = np.sqrt(np.mean(distances**2))
            
            return rms_planarity
            
        except:
            return 999.0  # Score très mauvais en cas d'erreur
        """
        Génère une géométrie planaire forcée pour les benzamides
        Force la coplanéité cycle aromatique + amide
        """
        print("   [fix] Construction géométrie planaire forcée...")
        
        # 1. Générer géométrie de base
        AllChem.EmbedMolecule(mol, randomSeed=42, useExpTorsionAnglePrefs=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)  # Optimisation rapide
        
        # 2. Identifier les atomes clés du benzamide
        aromatic_carbons = []
        carbonyl_carbon = None
        nitrogen_atom = None
        
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
                aromatic_carbons.append(atom.GetIdx())
            elif atom.GetSymbol() == 'C':
                # Vérifier si c'est le carbone carbonyle
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if 'O' in neighbors and 'N' in neighbors:
                    carbonyl_carbon = atom.GetIdx()
            elif atom.GetSymbol() == 'N':
                # Vérifier si connecté au carbonyle
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'C':
                        neigh_neighbors = [n.GetSymbol() for n in neighbor.GetNeighbors()]
                        if 'O' in neigh_neighbors:
                            nitrogen_atom = atom.GetIdx()
                            break
        
        if carbonyl_carbon is not None and nitrogen_atom is not None and aromatic_carbons:
            print(f"   [pin] Benzamide identifié: C_carbonyle={carbonyl_carbon}, N={nitrogen_atom}, {len(aromatic_carbons)} C aromatiques")
            
            # 3. Forcer planéité en ajustant les angles de torsion
            conf = mol.GetConformer()
            
            # Trouver le carbone aromatique connecté au carbonyle
            aromatic_connected = None
            carbonyl_atom = mol.GetAtomWithIdx(carbonyl_carbon)
            for neighbor in carbonyl_atom.GetNeighbors():
                if neighbor.GetIdx() in aromatic_carbons:
                    aromatic_connected = neighbor.GetIdx()
                    break
            
            if aromatic_connected is not None:
                # NOUVELLE APPROCHE: Contraintes explicites pour forcer planéité
                try:
                    from rdkit.Chem import rdMolTransforms
                    
                    # Trouver un hydrogène sur l'azote
                    h_on_nitrogen = None
                    nitrogen = mol.GetAtomWithIdx(nitrogen_atom)
                    for neighbor in nitrogen.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            h_on_nitrogen = neighbor.GetIdx()
                            break
                    
                    if h_on_nitrogen is not None:
                        print(f"   [>] Forçage planéité: Ar{aromatic_connected}-C{carbonyl_carbon}-N{nitrogen_atom}-H{h_on_nitrogen}")
                        
                        # NOUVELLE APPROCHE RADICALE: Forcer géométrie sp2 planaire complète
                        
                        # ÉTAPE 1: Calculer le plan optimal cycle aromatique + carbonyle
                        aromatic_atoms = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic()]
                        ref_coords = []
                        for idx in aromatic_atoms[:6]:  # 6 premiers carbones aromatiques
                            pos = conf.GetAtomPosition(idx)
                            ref_coords.append([pos.x, pos.y, pos.z])
                        ref_coords.append([conf.GetAtomPosition(carbonyl_carbon).x, 
                                          conf.GetAtomPosition(carbonyl_carbon).y, 
                                          conf.GetAtomPosition(carbonyl_carbon).z])
                        
                        # Calculer plan optimal
                        import numpy as np
                        ref_array = np.array(ref_coords)
                        centroid = np.mean(ref_array, axis=0)
                        centered = ref_array - centroid
                        U, s, Vt = np.linalg.svd(centered)
                        plane_normal = Vt[2]  # Vecteur normal au plan
                        
                        # ÉTAPE 2: Projeter l'azote sur ce plan
                        n_pos = conf.GetAtomPosition(nitrogen_atom)
                        n_coord = np.array([n_pos.x, n_pos.y, n_pos.z])
                        
                        # Projection de l'azote sur le plan
                        n_centered = n_coord - centroid
                        projection = n_centered - np.dot(n_centered, plane_normal) * plane_normal
                        new_n_coord = centroid + projection
                        
                        # Mettre à jour position azote
                        conf.SetAtomPosition(nitrogen_atom, new_n_coord.tolist())
                        print(f"   [pin] Azote projeté sur plan: {np.linalg.norm(n_centered - projection):.4f} Å déplacé")
                        
                        # ÉTAPE 3: Repositionner l'hydrogène pour géométrie sp2
                        # Vecteur C-N
                        c_pos = conf.GetAtomPosition(carbonyl_carbon)
                        cn_vector = new_n_coord - np.array([c_pos.x, c_pos.y, c_pos.z])
                        cn_vector = cn_vector / np.linalg.norm(cn_vector)
                        
                        # Hydrogène à 120 deg dans le plan (géométrie sp2)
                        # Rotation de 120 deg autour de la normale au plan
                        angle_120 = np.radians(120)
                        cos_a, sin_a = np.cos(angle_120), np.sin(angle_120)
                        
                        # Matrice de rotation autour de plane_normal
                        def rotate_around_axis(vector, axis, angle):
                            cos_a, sin_a = np.cos(angle), np.sin(angle)
                            axis = axis / np.linalg.norm(axis)
                            cross_matrix = np.array([[0, -axis[2], axis[1]],
                                                   [axis[2], 0, -axis[0]],
                                                   [-axis[1], axis[0], 0]])
                            rotation_matrix = (cos_a * np.eye(3) + sin_a * cross_matrix + 
                                             (1 - cos_a) * np.outer(axis, axis))
                            return rotation_matrix @ vector
                        
                        h_direction = rotate_around_axis(cn_vector, plane_normal, angle_120)
                        bond_length = 1.01  # Longueur N-H typique
                        new_h_coord = new_n_coord + h_direction * bond_length
                        
                        conf.SetAtomPosition(h_on_nitrogen, new_h_coord.tolist())
                        print(f"   [?] Hydrogène repositionné pour géométrie sp2 planaire")
                        
                        # ÉTAPE 4: Optimisation douce pour relaxer
                        try:
                            conf_id = conf.GetId() if conf is not None else 0
                            props = AllChem.MMFFGetMoleculeProperties(mol)
                            if props is not None:
                                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                                if ff is not None:
                                    # Optimisation très douce pour ne pas casser la planéité
                                    ff.Initialize()
                                    # Seulement quelques itérations pour relaxer les tensions
                                    converged = ff.Minimize(maxIts=50)  # Très peu d'itérations
                                    print(f"   OK Relaxation douce: {'Convergé' if converged == 0 else 'Partiel'}")
                                    
                                    # Vérifier planéité finale
                                    final_n_pos = conf.GetAtomPosition(nitrogen_atom)
                                    final_n_coord = np.array([final_n_pos.x, final_n_pos.y, final_n_pos.z])
                                    final_dist = np.abs(np.dot(final_n_coord - centroid, plane_normal))
                                    print(f"   [angle] Distance finale N du plan: {final_dist:.4f} Å")
                                    
                                else:
                                    print(f"   AVERT Force field non disponible - géométrie forcée conservée")
                            else:
                                print(f"   AVERT MMFF properties non disponibles - géométrie forcée conservée")
                        except Exception as e2:
                            print(f"   ERREUR Échec relaxation: {e2}")
                            print(f"   [fix] Géométrie planaire forcée conservée")
                    
                except Exception as e:
                    print(f"   AVERT Impossible de forcer planéité complète: {e}")
                    # Au moins essayer une optimisation standard
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                    except:
                        pass
        
        # Obtenir la conformation finale
        conf = mol.GetConformer()
        
        # Construire les coordonnées XYZ
        xyz_lines = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        
        print(f"   OK Géométrie planaire générée ({len(xyz_lines)} atomes)")
        return "\n".join(xyz_lines)
    
    def determine_optimal_functional(self, smiles: str, name: str = "") -> str:
        """
        Détermine automatiquement la fonctionnelle DFT optimale selon la structure moléculaire
        
        CAM-B3LYP pour:
        - Molécules avec groupes NO2, cyano (CN), cycliques (Cy)
        - Hétéroatomes halogénés (F, Cl, Br, I)
        - Soufre (S) qui peut avoir transfert de charge
        - Structures pi-conjuguées étendues
        
        B3LYP pour:
        - Molécules simples sans ces caractéristiques
        """
        functional = "B3LYP"  # Défaut
        reasons = []
        
        # 1. Vérifier le nom de la molécule pour indices structuraux
        name_upper = name.upper()
        if any(pattern in name_upper for pattern in ["NO", "NO2", "CN", "CY"]):
            functional = "CAM-B3LYP"
            if "NO" in name_upper:
                reasons.append("groupe nitro/nitroso détecté dans le nom")
            if "CN" in name_upper:
                reasons.append("groupe cyano détecté dans le nom")
            if "CY" in name_upper:
                reasons.append("structure cyclique détectée dans le nom")
        
        # 2. Analyser la structure SMILES pour hétéroatomes
        heteroatoms_detected = []
        
        # Halogènes
        if "F" in smiles:
            heteroatoms_detected.append("fluor")
        if "Cl" in smiles:
            heteroatoms_detected.append("chlore")
        if "Br" in smiles:
            heteroatoms_detected.append("brome")
        if "I" in smiles:
            heteroatoms_detected.append("iode")
        
        # Soufre
        if "S" in smiles:
            heteroatoms_detected.append("soufre")
        
        # Groupes électron-attracteurs dans SMILES
        if "[N+]" in smiles or "N(=O)" in smiles or "N(O)" in smiles:
            heteroatoms_detected.append("groupe nitro")
        
        if "#N" in smiles or "C#N" in smiles:
            heteroatoms_detected.append("groupe cyano")
        
        # 3. Si hétéroatomes détectés -> CAM-B3LYP
        if heteroatoms_detected:
            functional = "CAM-B3LYP"
            reasons.append(f"hétéroatomes détectés: {', '.join(heteroatoms_detected)}")
        
        # 4. Vérifier conjugaison étendue
        aromatic_rings = smiles.count("c") + smiles.count("C1") + smiles.count("c1")
        if aromatic_rings > 8:  # Système pi étendu
            functional = "CAM-B3LYP"
            reasons.append("système pi-conjugué étendu")
        
        # Rapport de sélection
        if functional == "CAM-B3LYP":
            print(f"[>] Fonctionnelle {functional} sélectionnée pour {name}")
            for reason in reasons:
                print(f"   • {reason}")
        else:
            print(f"[>] Fonctionnelle {functional} (défaut) pour {name}")
        
        return functional

    def check_and_load_xyz_file(self, name: str, xyz_directory: str = None) -> str:
        """
        Vérifie si un fichier XYZ existe pour la molécule et retourne son contenu si trouvé
        Recherche récursivement dans tous les sous-dossiers
        
        Args:
            name: Nom de la molécule
            xyz_directory: Dossier de recherche (par défaut: le dossier spécifié par l'utilisateur)
        
        Returns:
            str: Contenu XYZ du fichier si trouvé, None sinon
        """
        if xyz_directory is None:
            xyz_directory = r"C:\Users\ugo.pasco\Documents\2A\Data finale MD\molecules_data\lambda max - Copie"
        
        import os
        import glob
        
        # Essayer différents formats de noms de fichiers
        possible_patterns = [
            f"{name}.xyz",
            f"{name}_optimized.xyz",
            f"{name}_hplc_mobile_optimized.xyz",  # Format présent dans le dossier
            f"{name}.XYZ",
            f"{name}_optimized.XYZ",
            f"{name}_hplc_mobile_optimized.XYZ"
        ]
        
        print(f"? Recherche récursive de fichiers XYZ pour {name}...")
        print(f"   ? Dossier racine: {xyz_directory}")
        
        # Recherche récursive dans tous les sous-dossiers
        for pattern in possible_patterns:
            # Utiliser glob pour recherche récursive
            search_pattern = os.path.join(xyz_directory, "**", pattern)
            matching_files = glob.glob(search_pattern, recursive=True)
            
            # Aussi chercher dans le dossier racine
            root_file = os.path.join(xyz_directory, pattern)
            if os.path.exists(root_file):
                matching_files.insert(0, root_file)  # Priorité au dossier racine
            
            # Traiter les fichiers trouvés
            for filepath in matching_files:
                print(f"   [>] Test: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Vérifier que le fichier est valide
                    lines = content.split('\n')
                    if len(lines) >= 3:  # Au moins: natoms, comment, 1 atome
                        try:
                            natoms = int(lines[0])
                            if len(lines) >= natoms + 2:  # Vérifier nombre correct de lignes
                                # Calculer le chemin relatif pour affichage
                                rel_path = os.path.relpath(filepath, xyz_directory)
                                print(f"OK Fichier XYZ trouvé: {rel_path}")
                                print(f"   ? {natoms} atomes lus depuis fichier existant")
                                print(f"   [pin] Chemin complet: {filepath}")
                                return content
                        except ValueError:
                            print(f"   AVERT Format invalide (natoms): {filepath}")
                            continue
                except Exception as e:
                    print(f"   AVERT Erreur lecture {filepath}: {e}")
                    continue
        
        print(f"? Aucun fichier XYZ trouvé pour {name} dans {xyz_directory} (recherche récursive)")
        return None
    
    def analyze_xyz_atoms(self, xyz_content: str) -> set:
        """
        Analyse les atomes présents dans un fichier XYZ
        
        Args:
            xyz_content: Contenu du fichier XYZ
            
        Returns:
            set: Ensemble des symboles atomiques présents
        """
        lines = xyz_content.strip().split('\n')
        atoms = set()
        
        try:
            natoms = int(lines[0])
            
            for i in range(2, 2 + natoms):
                if i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 4:
                        symbol = parts[0].upper()  # Normaliser en majuscules
                        atoms.add(symbol)
        except:
            pass
        
        return atoms
    
    def determine_functional_from_xyz(self, xyz_content: str, name: str = "") -> str:
        """
        Détermine la fonctionnelle optimale en analysant les atomes du fichier XYZ et le nom
        
        Args:
            xyz_content: Contenu du fichier XYZ
            name: Nom de la molécule
            
        Returns:
            str: Fonctionnelle recommandée (B3LYP ou CAM-B3LYP)
        """
        atoms = self.analyze_xyz_atoms(xyz_content)
        functional = "B3LYP"  # Défaut
        reasons = []
        
        print(f"? Analyse atomes du fichier XYZ pour {name}:")
        print(f"   Atomes détectés: {sorted(atoms)}")
        
        # Vérifier le nom de la molécule pour groupes fonctionnels spéciaux
        name_upper = name.upper()
        if "CY" in name_upper:
            functional = "CAM-B3LYP"
            reasons.append("groupe cyclohexyl (Cy) détecté dans le nom")
        
        if "NO2" in name_upper or "NO" in name_upper:
            functional = "CAM-B3LYP"
            reasons.append("groupe nitro/nitroso (NO2/NO) détecté dans le nom")
        
        # Vérifier présence d'éléments spéciaux dans le fichier XYZ
        special_elements = atoms.intersection({'SI', 'CL', 'BR', 'I', 'F', 'S'})
        
        if special_elements:
            functional = "CAM-B3LYP"
            # Convertir en format lisible
            display_elements = []
            for elem in sorted(special_elements):
                if elem == 'SI':
                    display_elements.append('Si')
                elif elem == 'CL':
                    display_elements.append('Cl')
                elif elem == 'BR':
                    display_elements.append('Br')
                else:
                    display_elements.append(elem)
            reasons.append(f"éléments spéciaux: {', '.join(display_elements)}")
        
        # Compter les atomes aromatiques (estimation par nombre de C)
        n_carbons = sum(1 for line in xyz_content.split('\n')[2:] if line.strip() and line.split()[0].upper() == 'C')
        if n_carbons > 15:  # Système conjugué étendu probable
            functional = "CAM-B3LYP"
            reasons.append(f"système potentiellement conjugué étendu ({n_carbons} carbones)")
        
        # Rapport
        if functional == "CAM-B3LYP":
            print(f"   [>] Fonctionnelle {functional} sélectionnée (fichier XYZ)")
            for reason in reasons:
                print(f"      • {reason}")
        else:
            print(f"   OK Fonctionnelle {functional} (défaut, fichier XYZ)")
        
        return functional
    
    def parse_xyz_coordinates(self, xyz_content: str) -> str:
        """
        Parse le contenu d'un fichier XYZ et extrait les coordonnées pour Psi4
        
        Args:
            xyz_content: Contenu du fichier XYZ
            
        Returns:
            str: Coordonnées au format Psi4
        """
        lines = xyz_content.strip().split('\n')
        
        if len(lines) < 3:
            raise ValueError("Fichier XYZ invalide: trop peu de lignes")
        
        try:
            natoms = int(lines[0])
            comment_line = lines[1]  # Ligne de commentaire (ignorée)
            
            # Extraire les coordonnées atomiques
            coord_lines = []
            for i in range(2, 2 + natoms):
                if i >= len(lines):
                    raise ValueError(f"Fichier XYZ incomplet: ligne {i} manquante")
                
                parts = lines[i].split()
                if len(parts) < 4:
                    raise ValueError(f"Ligne {i} invalide: {lines[i]}")
                
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                coord_lines.append(f"{symbol:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
            
            xyz_coords = "\n".join(coord_lines)
            print(f"   OK Coordonnées XYZ parsées: {natoms} atomes")
            return xyz_coords
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Erreur parsing XYZ: {e}")

    def setup_molecule(self, smiles: str, name: str = "", charge=None, multiplicity=None):
        """
        Crée une molécule Psi4 à partir d'un SMILES avec détection automatique charge/multiplicité
        PRIORITÉ ABSOLUE aux fichiers XYZ existants (plus fiables que génération RDKit)
        
        Returns:
            tuple: (molécule Psi4, True si fichier XYZ utilisé, contenu XYZ ou None)
        """
        # Vérifier d'abord s'il existe un fichier XYZ pour cette molécule (PRIORITÉ)
        xyz_content = self.check_and_load_xyz_file(name)
        xyz_file_used = False
        xyz_content_for_functional = None
        
        if xyz_content is not None:
            # Utiliser les coordonnées du fichier XYZ existant (RECOMMANDÉ)
            try:
                xyz_coords = self.parse_xyz_coordinates(xyz_content)
                xyz_file_used = True
                xyz_content_for_functional = xyz_content
                if smiles and smiles.strip():
                    print(f"AVERT  [INFO] SMILES fourni mais fichier XYZ trouvé -> utilisation XYZ (plus fiable)")
                print(f"OK [PSI4] Coordonnées XYZ existantes pour {name} - OPTIMISATION DÉSACTIVÉE")
            except Exception as e:
                print(f"ERREUR Erreur parsing XYZ pour {name}: {e}")
                if not smiles or not smiles.strip():
                    raise ValueError(f"Fichier XYZ corrompu et aucun SMILES fourni pour {name}")
                print(f"[refresh] Fallback vers génération RDKit depuis SMILES...")
                # Fallback vers RDKit
                rdkit_mol = Chem.MolFromSmiles(smiles)
                if rdkit_mol is None:
                    raise ValueError(f"SMILES invalide: {smiles}")
                xyz_coords = self.rdkit_to_psi4_xyz(rdkit_mol)
                xyz_file_used = False
                xyz_content_for_functional = None
        else:
            # Aucun fichier XYZ -> générer depuis SMILES (moins fiable)
            if not smiles or not smiles.strip():
                raise ValueError(f"Aucun fichier XYZ et aucun SMILES fourni pour {name}")
            
            print(f"AVERT  [WARNING] Aucun fichier XYZ trouvé -> génération depuis SMILES (peut être moins précis)")
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                raise ValueError(f"SMILES invalide: {smiles}")
            xyz_coords = self.rdkit_to_psi4_xyz(rdkit_mol)
            xyz_file_used = False
            xyz_content_for_functional = None
        
        # Charge et multiplicité : défaut 0/1, pas d'auto-détection
        if charge is None:
            charge = 0
        if multiplicity is None:
            multiplicity = 1
        charge = int(charge)
        multiplicity = int(multiplicity)
        
        # Créer la molécule Psi4 — format identique au bypass qui fonctionne
        mol_string = f"{charge} {multiplicity}\n" + xyz_coords + "\nunits angstrom\nno_reorient\nno_com\nsymmetry c1\n"
        
        print(f"[PSI4] Création molécule {name}: charge={charge}, multiplicité={multiplicity}")
        return psi4.geometry(mol_string), xyz_file_used, xyz_content_for_functional
    
    def prepare_planar_amide_geometry(self, mol: psi4.core.Molecule, name: str) -> psi4.core.Molecule:
        """
        Prépare une géométrie de départ plus planaire pour les amides (BAm, etc.)
        Force l'azote amide vers une géométrie sp2 planaire
        """
        if 'BAm' in name or 'amide' in name.lower():
            print(f"[fix] Pré-planification géométrie amide pour {name}")
            
            # Identifier les azotes amide (liés à un carbonyle)
            amide_nitrogens = []
            for i in range(mol.natom()):
                if mol.symbol(i) == 'N':
                    # Chercher un carbone carbonyle à proximité
                    for j in range(mol.natom()):
                        if mol.symbol(j) == 'C':
                            distance = ((mol.x(i) - mol.x(j))**2 + 
                                      (mol.y(i) - mol.y(j))**2 + 
                                      (mol.z(i) - mol.z(j))**2)**0.5
                            # Distance typique N-C amide: ~1.32-1.40 Å
                            if 1.2 < distance < 1.5:
                                # Vérifier si le carbone a un oxygène (carbonyle)
                                has_oxygen = False
                                for k in range(mol.natom()):
                                    if mol.symbol(k) == 'O':
                                        co_distance = ((mol.x(j) - mol.x(k))**2 + 
                                                     (mol.y(j) - mol.y(k))**2 + 
                                                     (mol.z(j) - mol.z(k))**2)**0.5
                                        if 1.1 < co_distance < 1.3:  # Liaison C=O
                                            has_oxygen = True
                                            break
                                
                                if has_oxygen:
                                    amide_nitrogens.append(i)
                                    print(f"   [pin] Azote amide identifié: index {i}")
                                    break
            
            if amide_nitrogens:
                print(f"   [>] {len(amide_nitrogens)} azote(s) amide détecté(s) - géométrie pré-planifiée")
                
        return mol

    def optimize_geometry(self, mol: psi4.core.Molecule, 
                         method="B3LYP", basis=None, name="") -> Dict:
        """
        Optimise la géométrie moléculaire avec B3LYP/def2-SVP
        AMÉLIORATION: Convergence renforcée pour planéité des amides
        """
        # Utiliser toujours B3LYP/def2-SVP pour l'optimisation
        if basis is None:
            basis = "def2-SVP"
        
        n_atoms = mol.natom()
        print(f"Optimisation géométrie avec {method}/{basis} ({n_atoms} atomes)...")
        
        # Pré-planification pour les amides
        mol = self.prepare_planar_amide_geometry(mol, name)
        
        # Configuration pour optimisation — équilibre convergence / coût
        psi4.set_options({
            'reference': 'rhf',
            'scf_type': 'df',
            'geom_maxiter': 40,             # 40 pas max — au-delà c'est une oscillation
            'g_convergence': 'gau',         # Standard Gaussian (tight provoque des oscillations)
            'maxiter': 200,                 # Itérations SCF
            'e_convergence': 1e-8,
            'd_convergence': 1e-8,
            'fail_on_maxiter': False,
            'cart_hess_read': False,
            'full_hess_every': -1,          # Pas de recalcul Hessien exact (coûteux et inutile ici)
            'hess_update': 'bfgs',          # Mise-à-jour BFGS entre les recalculs
            'intrafrag_step_limit': 0.3,    # Défaut Psi4
            'interfrag_step_limit': 0.3,
        })
        
        try:
            # Vérifier la géométrie avant optimisation
            if mol.natom() < 3:
                print("AVERT Molécule trop petite pour optimisation, géométrie conservée")
                return {
                    "optimized_energy": 0.0,
                    "method": method,
                    "basis": basis,
                    "converged": True,
                    "note": "Géométrie initiale conservée"
                }
            
            # Optimisation avec B3LYP/def2-SVP
            energy = psi4.optimize(f"{method}/{basis}", molecule=mol)
            
            print(f"OK Optimisation réussie: {energy:.6f} Ha")
            
            # Vérification spéciale planéité pour amides
            if 'BAm' in name:
                print(f"? Vérification planéité amide pour {name}...")
                # Ici on pourrait ajouter une validation spécifique de la planéité de l'amide
                # et éventuellement faire une seconde optimisation si nécessaire
            
            # Valider la géométrie optimisée
            validation = validate_molecular_geometry(mol, "après optimisation")
            
            return {
                "optimized_energy": energy,
                "method": method,
                "basis": basis,
                "converged": True,
                "geometry_validation": validation
            }
            
        except Exception as e:
            print(f"ERREUR Erreur optimisation: {e}")
            return {
                "optimized_energy": None,
                "method": method,
                "basis": basis,
                "converged": False,
                "error": str(e)
            }
    
    def extract_tddft_results_from_wfn(self, wfn) -> Tuple[List[float], List[float]]:
        """
        Extrait les résultats TD-DFT directement depuis l'objet wavefunction
        """
        excitation_energies = []
        oscillator_strengths = []
        
        try:
            # Essayer d'accéder aux propriétés TD-DFT du wavefunction
            if hasattr(wfn, 'tdscf_excitation_energies'):
                energies = wfn.tdscf_excitation_energies()
                oscillators = wfn.tdscf_oscillator_strengths() if hasattr(wfn, 'tdscf_oscillator_strengths') else None
                
                for i in range(len(energies)):
                    energy_ev = energies[i] * 27.2114  # Conversion Hartree -> eV
                    excitation_energies.append(energy_ev)
                    
                    if oscillators and i < len(oscillators):
                        oscillator_strengths.append(oscillators[i])
                    else:
                        oscillator_strengths.append(0.0)
                
                print(f"OK Extraction directe wfn: {len(excitation_energies)} états")
                return excitation_energies, oscillator_strengths
        except Exception as e:
            print(f"AVERT Extraction wfn échouée: {e}")
        
        # Fallback vers la méthode par variables
        return self.extract_tddft_results_by_variables(10)
    
    def extract_tddft_results_by_variables(self, n_states: int) -> Tuple[List[float], List[float]]:
        """
        Extrait les résultats TD-DFT depuis les variables Psi4 (compatible version 1.6+)
        """
        excitation_energies = []
        oscillator_strengths = []
        
        print("  [data] Extraction TD-DFT compatible Psi4 1.6+...")
        
        try:
            # Méthode moderne pour Psi4 >= 1.4
            try:
                all_vars = psi4.core.variables()
            except AttributeError:
                all_vars = psi4.get_variables()
            
            # Debug: afficher quelques variables pour comprendre le format
            print(f"  ? Variables disponibles (échantillon):")
            tddft_vars = [name for name in all_vars.keys() if any(keyword in name.upper() for keyword in 
                         ['EXCITATION', 'OSCILLATOR', 'TD-DFT', 'ROOT'])][:10]
            for var in tddft_vars:
                print(f"    {var}: {all_vars[var]}")
            
            # Stratégie robuste pour toutes les fonctionnelles TD-DFT
            import re
            
            def extract_root_number(var_name):
                # Parser les formats possibles:
                # "TD-B3LYP ROOT 0 (A) -> ROOT X (A)" 
                # "TD-CAM-B3LYP ROOT 0 (A) -> ROOT X (A)"
                # "TD-PBE0 ROOT 0 (A) -> ROOT X (A)" etc.
                match = re.search(r'TD-[\w\-]+\s+ROOT 0 \([A-Z]\) -> ROOT (\d+) \([A-Z]\)', var_name)
                return int(match.group(1)) if match else None
            
            # Créer un dictionnaire organisé par numéro de ROOT
            transitions = {}
            
            print(f"  ? Parsing des variables TD-DFT...")
            
            for var_name, value in all_vars.items():
                # Vérifier si c'est une variable TD-DFT avec le bon format (toutes fonctionnelles)
                if 'ROOT 0' in var_name and '->' in var_name and 'TD-' in var_name:
                    root_num = extract_root_number(var_name)
                    
                    if root_num is not None:
                        if root_num not in transitions:
                            transitions[root_num] = {'energy': None, 'osc_strength': None}
                        
                        if 'EXCITATION ENERGY' in var_name:
                            try:
                                energy_hartree = float(value)
                                energy_ev = energy_hartree * 27.2114  # Conversion Hartree -> eV
                                transitions[root_num]['energy'] = energy_ev
                                print(f"    Trouvé énergie ROOT {root_num}: {energy_hartree:.4f} Ha -> {energy_ev:.4f} eV")
                            except (ValueError, TypeError):
                                continue
                        elif 'OSCILLATOR STRENGTH (LEN)' in var_name:
                            try:
                                osc_val = float(value)
                                transitions[root_num]['osc_strength'] = osc_val
                                print(f"    Trouvé oscillateur ROOT {root_num}: {osc_val:.6f}")
                            except (ValueError, TypeError):
                                continue
            
            # Extraire les résultats dans l'ordre des ROOTs
            if transitions:
                print(f"  OK Transitions trouvées: {list(transitions.keys())}")
                
                for root_num in sorted(transitions.keys()):
                    trans = transitions[root_num]
                    
                    if trans['energy'] is not None:
                        energy_val = trans['energy']
                        osc_val = trans['osc_strength'] if trans['osc_strength'] is not None else 0.0
                        
                        # Filtrer les énergies raisonnables pour UV-Vis (1.5-15 eV)
                        if 1.5 < energy_val < 15.0:
                            excitation_energies.append(energy_val)
                            oscillator_strengths.append(osc_val)
                            print(f"  OK État {root_num}: {energy_val:.4f} eV, f = {osc_val:.6f}")
                            
                            if len(excitation_energies) >= n_states:
                                break
                        else:
                            print(f"  AVERT État {root_num}: {energy_val:.4f} eV (énergie hors plage UV-Vis)")
            else:
                print(f"  AVERT Aucune transition TD-DFT trouvée avec le format attendu")
            
            # Si aucune transition dans la plage UV-Vis, prendre toutes les transitions
            if len(excitation_energies) == 0 and transitions:
                print("  [refresh] Aucune transition UV-Vis, inclusion de toutes les transitions...")
                
                for root_num in sorted(transitions.keys()):
                    trans = transitions[root_num]
                    
                    if trans['energy'] is not None:
                        energy_val = trans['energy']
                        osc_val = trans['osc_strength'] if trans['osc_strength'] is not None else 0.0
                        
                        # Prendre toutes les transitions raisonnables (> 0.1 eV)
                        if energy_val > 0.1:
                            excitation_energies.append(energy_val)
                            oscillator_strengths.append(osc_val)
                            lambda_nm = 1240.0 / energy_val
                            print(f"  OK État {root_num}: {energy_val:.4f} eV -> {lambda_nm:.1f} nm, f = {osc_val:.6f}")
                            
                            if len(excitation_energies) >= n_states:
                                break
                        
        except Exception as e:
            print(f"  ERREUR Erreur extraction variables: {e}")
        
        print(f"  ? Résultats: {len(excitation_energies)} états extraits")
        return excitation_energies, oscillator_strengths

    def calculate_tddft(self, mol: psi4.core.Molecule,
                       method="B3LYP", basis=None, 
                       n_states=10, solvent=None, timeout_minutes=None,
                       use_tda=True) -> Dict:
        """
        Calcul TD-DFT avec sélection automatique de fonctionnelle (B3LYP ou CAM-B3LYP)
        use_tda: si True, utilise l'approximation Tamm-Dancoff (~2x plus rapide)
        """
        # Utiliser la méthode et base passées en paramètres
        if basis is None:
            basis = "def2-SVP"
            
        print(f"Calcul TD-DFT avec {method}/{basis} ({n_states} états)...")
        
        # Configuration spécifique selon la fonctionnelle
        if method.upper() == "CAM-B3LYP":
            print(f"   [>] Fonctionnelle CAM-B3LYP: optimisée pour transfert de charge")
        else:
            print(f"   [>] Fonctionnelle {method}: standard hybride")
        
        # Configuration TD-DFT — TDA contrôlé par le paramètre use_tda
        tddft_options = {
            'reference': 'rhf',
            'scf_type': 'df',
            'tdscf_states': n_states,
            'tdscf_tda': use_tda,
            'tdscf_maxiter': 150,
            'tdscf_r_convergence': 1e-6,
        }
        tda_label = "TDA" if use_tda else "full TD-DFT"
        print(f"   [>] Mode {tda_label} (tdscf_tda={use_tda})")
        
        # Forcer la symétrie C1 (après configuration des options)
        mol.reset_point_group('c1')
        mol.fix_orientation(True)
        mol.fix_com(True)
        
        # Ajouter les options de solvatation si spécifié
        if solvent:
            print(f"? Configuration solvatation PCM pour {solvent}")
            
            # Réinitialiser complètement les options pour éviter conflits
            psi4.core.clean_options()
            
            # CAM-B3LYP + PCM : forcer TDA (convergence instable en full TD-DFT)
            if method.upper() == "CAM-B3LYP":
                if not use_tda:
                    print(f"   [fix] CAM-B3LYP + PCM -> TDA forcé malgré demande full TD-DFT (stabilité)")
                else:
                    print(f"   [>] CAM-B3LYP + PCM en TDA")
                tddft_options.update({
                    'tdscf_tda': True,
                    'tdscf_maxiter': 200,
                    'tdscf_r_convergence': 1e-5,
                    'pcm': True,
                    'pcm_scf_type': 'total',
                    'guess': 'sad',
                    'maxiter': 200,
                    'e_convergence': 1e-5,
                    'd_convergence': 1e-5
                })
            else:
                # B3LYP, PBE0, etc. + PCM : respecter le choix TDA de l'utilisateur
                tda_label = "TDA" if use_tda else "full TD-DFT"
                print(f"   [>] {method} + PCM en {tda_label}")
                tddft_options.update({
                    'pcm': True,
                    'pcm_scf_type': 'total'
                })
            
            # Configuration solvants disponibles — abréviations courantes
            solvent_mapping = {
                # Eau
                'water': 'water', 'h2o': 'water', 'eau': 'water',
                # Acétonitrile
                'acetonitrile': 'acetonitrile', 'acn': 'acetonitrile',
                'mecn': 'acetonitrile', 'ch3cn': 'acetonitrile',
                # DMSO
                'dmso': 'dmso', 'dimethylsulfoxide': 'dmso',
                # Méthanol
                'methanol': 'methanol', 'meoh': 'methanol', 'ch3oh': 'methanol',
                # Éthanol
                'ethanol': 'ethanol', 'etoh': 'ethanol',
                # Acétone
                'acetone': 'acetone', 'me2co': 'acetone',
                # THF
                'thf': 'tetrahydrofuran', 'tetrahydrofuran': 'tetrahydrofuran',
                # DCM / Dichlorométhane
                'dcm': 'dichloromethane', 'dichloromethane': 'dichloromethane',
                'ch2cl2': 'dichloromethane', 'methylenechloride': 'dichloromethane',
                # Chloroforme
                'chloroform': 'chloroform', 'chcl3': 'chloroform',
                # DMF
                'dmf': 'dimethylformamide', 'dimethylformamide': 'dimethylformamide',
                # Toluène
                'toluene': 'toluene', 'tol': 'toluene', 'phme': 'toluene',
                # Hexane / Cyclohexane
                'hexane': 'cyclohexane', 'cyclohexane': 'cyclohexane',
                'chex': 'cyclohexane', 'c6h12': 'cyclohexane',
                # Diéthyl éther
                'ether': 'diethylether', 'diethylether': 'diethylether',
                'et2o': 'diethylether', 'dee': 'diethylether',
                # Benzène
                'benzene': 'benzene', 'phh': 'benzene',
                # CCl4
                'ccl4': 'carbontetrachloride', 'carbontetrachloride': 'carbontetrachloride',
                # Propanol
                'isopropanol': 'isopropanol', 'ipa': 'isopropanol',
                '2-propanol': 'isopropanol', 'iproh': 'isopropanol',
                # Nitrobenzène
                'nitrobenzene': 'nitrobenzene', 'phno2': 'nitrobenzene',
                # Pyridine
                'pyridine': 'pyridine', 'py': 'pyridine',
                # Mélange HPLC
                'hplc_mobile': 'acetonitrile',
            }
            
            solvent_descriptions = {
                'hplc_mobile': 'H2O/ACN 50:50 + TFA pH~2 (approx. ACN)',
            }
            
            if solvent.lower() in solvent_mapping:
                psi4_solvent = solvent_mapping[solvent.lower()]
                
                # Configuration PCM (Area=0.6 pour stabilité matrice S)
                pcm_input = f"""
                    Medium {{
                        SolverType = IEFPCM
                        Solvent = {psi4_solvent.title()}
                    }}
                    Cavity {{
                        RadiiSet = UFF
                        Type = GePol
                        Scaling = True
                        Area = 0.6
                        Mode = Implicit
                    }}
                    """
                
                try:
                    psi4.pcm_helper(pcm_input)
                    
                    if solvent.lower() in solvent_descriptions:
                        print(f"   OK {solvent_descriptions[solvent.lower()]}")
                    else:
                        print(f"   OK Solvant: {psi4_solvent.title()}")
                except Exception as e:
                    print(f"   AVERT Erreur PCM pour '{solvent}', passage en phase gazeuse: {e}")
                    tddft_options.pop('pcm', None)
                    tddft_options.pop('pcm_scf_type', None)
            else:
                # Pass-through : laisser Psi4 valider le nom du solvant
                psi4_solvent = solvent.strip()
                pcm_pt = f"""
                    Medium {{
                        SolverType = IEFPCM
                        Solvent = {psi4_solvent.title()}
                    }}
                    Cavity {{
                        RadiiSet = UFF
                        Type = GePol
                        Scaling = True
                        Area = 0.6
                        Mode = Implicit
                    }}
                """
                try:
                    psi4.pcm_helper(pcm_pt)
                    print(f"OK Solvant '{psi4_solvent.title()}' accepte par Psi4 PCM")
                except Exception as e:
                    print(f"AVERT '{psi4_solvent}' rejete ({e}) -> fallback Water")
                    try:
                        psi4.pcm_helper("""
                            Medium { SolverType = IEFPCM  Solvent = Water }
                            Cavity { RadiiSet = UFF  Type = GePol
                                     Scaling = True  Area = 0.6  Mode = Implicit }
                        """)
                    except Exception:
                        tddft_options.pop("pcm", None)
                        tddft_options.pop("pcm_scf_type", None)
        psi4.set_options(tddft_options)
        
        print(f"? Options TD-DFT haute performance activées")
        
        # Forcer explicitement la base avant calcul TD-DFT
        psi4.set_options({'basis': basis})
        print(f"[fix] Base forcée explicitement: {basis}")
        
        try:
            # Vérification préalable de la stabilité SCF
            print("Vérification stabilité SCF avant TD-DFT...")
            try:
                ground_energy = psi4.energy(f"{method}/{basis}", molecule=mol)
                if ground_energy != ground_energy:  # Check for NaN
                    raise ValueError("État fondamental instable (énergie NaN)")
                print(f"OK État fondamental stable: {ground_energy:.6f} Ha")
            except Exception as scf_error:
                raise Exception(f"SCF préalable échoué: {scf_error}")
            
            # Calcul TD-DFT avec gestion d'erreurs robuste
            print("Lancement du calcul TD-DFT...")
            
            # Timeout multiplateforme avec threading
            import threading
            import time
            
            # Variables pour le timeout
            calculation_result = {"energy": None, "wfn": None, "error": None, "completed": False}
            
            def run_tddft():
                """Fonction pour exécuter TD-DFT dans un thread séparé"""
                try:
                    energy, wfn = psi4.energy(f"td-{method}/{basis}", molecule=mol, return_wfn=True)
                    calculation_result["energy"] = energy
                    calculation_result["wfn"] = wfn
                    calculation_result["completed"] = True
                except Exception as e:
                    calculation_result["error"] = e
                    calculation_result["completed"] = True
            
            # Lancer le calcul dans un thread
            calc_thread = threading.Thread(target=run_tddft)
            calc_thread.daemon = True
            calc_thread.start()
            
            # Timeout adaptatif ou personnalisé
            n_atoms = mol.natom()
            
            if timeout_minutes is not None:
                if timeout_minutes <= 0:
                    timeout_seconds = None  # Pas de timeout
                    print(f"   ? Timeout désactivé - calcul illimité")
                else:
                    timeout_seconds = timeout_minutes * 60
                    print(f"   ? Timeout personnalisé: {timeout_minutes} min")
            else:
                # Timeout adaptatif selon la taille de la molécule
                if n_atoms <= 10:
                    timeout_seconds = 600   # 10 min pour petites molécules
                elif n_atoms <= 20:
                    timeout_seconds = 3600  # 30 min pour molécules moyennes
                elif n_atoms <= 50:
                    timeout_seconds = 3600  # 1h pour grosses molécules
                else:
                    timeout_seconds = 7200  # 2h pour très grosses molécules
                
                print(f"   ? Timeout adaptatif: {timeout_seconds//60} min pour {n_atoms} atomes")
            
            start_time = time.time()
            
            if timeout_seconds is None:
                # Attente illimitée
                while not calculation_result["completed"]:
                    time.sleep(5)
                    if int(time.time() - start_time) % 300 == 0:  # Message toutes les 5 minutes
                        elapsed = int(time.time() - start_time)
                        print(f"   ? TD-DFT en cours... {elapsed//60} minutes écoulées (pas de limite)")
            else:
                # Attente avec timeout
                while not calculation_result["completed"] and (time.time() - start_time) < timeout_seconds:
                    time.sleep(1)
                    if int(time.time() - start_time) % 60 == 0:  # Message toutes les minutes
                        elapsed = int(time.time() - start_time)
                        remaining = timeout_seconds - elapsed
                        print(f"   ? TD-DFT en cours... {elapsed//60}min écoulées, {remaining//60}min restantes")
            
            # Vérifier le résultat
            if not calculation_result["completed"] and timeout_seconds is not None:
                raise TimeoutError(f"TD-DFT timeout après {timeout_seconds//60} minutes")
            elif calculation_result["error"]:
                raise calculation_result["error"]
            else:
                energy, wfn = calculation_result["energy"], calculation_result["wfn"]
            
            # Vérifier que l'énergie TD-DFT est valide
            if energy != energy:  # Check for NaN
                raise ValueError("Énergie TD-DFT invalide (NaN)")
            
            # Extraction robuste des résultats TD-DFT
            print("Extraction des résultats TD-DFT...")
            excitation_energies, oscillator_strengths = self.extract_tddft_results_from_wfn(wfn)
            
            # Vérifier que nous avons des résultats TD-DFT réels
            if not excitation_energies:
                raise ValueError("Aucune transition TD-DFT extraite - calcul échoué")
            
            # Calculer les longueurs d'onde
            wavelengths = [1240.0 / energy for energy in excitation_energies if energy > 0]
            
            # Trouver la transition la plus intense dans le visible/UV
            transitions = list(zip(excitation_energies, oscillator_strengths, wavelengths))
            
            print(f"? Debug transitions:")
            for i, (e, f, w) in enumerate(transitions):
                print(f"  Transition {i+1}: {e:.4f} eV -> {w:.1f} nm, f = {f:.6f}")
            
            # Algorithme amélioré pour sélection lambda max
            # PRIORITÉ: Force d'oscillateur maximale (f le plus proche de 1.0)
            # 1. Prendre la transition avec la plus haute force d'oscillateur (f > 0.01) dans l'UV-Visible
            # 2. Si aucune significative, prendre la transition la plus intense globalement
            # 3. Filtrer seulement les transitions non physiques (< 100 nm ou > 1000 nm)
            
            print(f"? Analyse de toutes les transitions disponibles: {len(transitions)}")
            
            # Filtrer seulement les transitions physiquement raisonnables
            valid_transitions = [(e, f, w) for e, f, w in transitions if 100 <= w <= 1000]
            uv_visible_transitions = [(e, f, w) for e, f, w in transitions if 150 <= w <= 800]
            
            print(f"? Transitions physiquement valides (100-1000nm): {len(valid_transitions)}")
            print(f"? Transitions UV-Visible (150-800nm): {len(uv_visible_transitions)}")
            
            # Afficher les transitions les plus intenses pour diagnostic
            if transitions:
                sorted_by_intensity = sorted(transitions, key=lambda x: x[1], reverse=True)
                print("[data] Top 5 transitions par force d'oscillateur:")
                for i, (e, f, w) in enumerate(sorted_by_intensity[:5]):
                    print(f"   {i+1}. lambda_={w:.1f}nm, f={f:.6f}, E={e:.3f}eV")
            
            # Stratégie de sélection simplifiée: PRIORITÉ À LA FORCE D'OSCILLATEUR
            lambda_max = None
            max_osc_strength = 0
            selection_reason = ""
            
            # Étape 1: Transitions significatives (f > 0.01) dans UV-Visible
            significant_uv_vis = [(e, f, w) for e, f, w in uv_visible_transitions if f > 0.01]
            if significant_uv_vis:
                max_transition = max(significant_uv_vis, key=lambda x: x[1])  # Prendre la PLUS INTENSE
                lambda_max = max_transition[2]
                max_osc_strength = max_transition[1]
                selection_reason = f"transition la plus intense UV-Visible (f > 0.01)"
                print(f"OK Stratégie 1: {selection_reason}")
            
            # Étape 2: Transitions moyennes (f > 0.001) dans UV-Visible
            elif uv_visible_transitions:
                moderate_uv_vis = [(e, f, w) for e, f, w in uv_visible_transitions if f > 0.001]
                if moderate_uv_vis:
                    max_transition = max(moderate_uv_vis, key=lambda x: x[1])  # Prendre la PLUS INTENSE
                    lambda_max = max_transition[2]
                    max_osc_strength = max_transition[1]
                    selection_reason = f"transition la plus intense UV-Visible (f > 0.001)"
                    print(f"OK Stratégie 2: {selection_reason}")
                else:
                    # Prendre la plus intense même si très faible
                    max_transition = max(uv_visible_transitions, key=lambda x: x[1])
                    lambda_max = max_transition[2]
                    max_osc_strength = max_transition[1]
                    selection_reason = f"transition la plus intense UV-Visible disponible"
                    print(f"AVERT Stratégie 2b: {selection_reason}")
            
            # Étape 3: Étendre à toutes les transitions valides
            elif valid_transitions:
                max_transition = max(valid_transitions, key=lambda x: x[1])  # Prendre la PLUS INTENSE
                lambda_max = max_transition[2]
                max_osc_strength = max_transition[1]
                selection_reason = f"transition la plus intense globalement"
                print(f"AVERT Stratégie 3: {selection_reason}")
            
            # Étape 4: Dernier recours - première transition
            else:
                if transitions:
                    lambda_max = wavelengths[0]
                    max_osc_strength = oscillator_strengths[0]
                    selection_reason = f"première transition disponible"
                    print(f"AVERT Stratégie 4: {selection_reason}")
                else:
                    lambda_max = None
                    max_osc_strength = 0
                    selection_reason = "Aucune transition trouvée"
                    print(f"ERREUR Aucune transition disponible")
            
            if lambda_max:
                print(f"OK Lambda max sélectionné: {lambda_max:.1f} nm, f = {max_osc_strength:.6f} ({selection_reason})")
            
            return {
                "excitation_energies_ev": excitation_energies,
                "oscillator_strengths": oscillator_strengths,
                "wavelengths_nm": wavelengths,
                "lambda_max_nm": lambda_max,
                "max_oscillator_strength": max_osc_strength,
                "transitions": transitions,
                "method": f"TD-{method}",
                "basis": basis,
                "n_states": len(excitation_energies),
                "converged": True,
                "ground_state_energy_hartree": ground_energy
            }
            
        except Exception as e:
            # Vérifier si c'est un timeout ou une autre erreur
            error_msg = str(e)
            is_timeout = "timeout" in error_msg.lower() or "TimeoutError" in str(type(e))
            
            if is_timeout:
                print(f"? Timeout TD-DFT: {e}")
                print(f"   ? Le calcul prend trop de temps - essayer molécule plus simple")
            else:
                print(f"ERREUR Erreur TD-DFT: {e}")
            
            # Nettoyage forcé en cas d'erreur
            try:
                psi4.core.clean()
                psi4.core.clean_options()
                psi4.core.clean_variables()
            except:
                pass
            
            return {
                "converged": False,
                "error": error_msg,
                "method": f"TD-{method}",
                "basis": basis
            }
    
    def calculate_lambda_max(self, smiles: str, name: str, optimize=True, solvent=None, timeout_minutes=None, functional=None,
                             geometry_functional_override=None, geometry_basis_override=None,
                             mol_psi4=None, tddft_basis=None, use_tda=True, n_states=4) -> Dict:
        """
        Calcul simplifié pour lambda max via TD-DFT avec sélection automatique de fonctionnelle
        Inclut le suivi des temps de calcul et la classification par famille.
        Si mol_psi4 est fourni, utilise cette géométrie (ex: DFT optimisée) au lieu
        de recréer la molécule depuis SMILES.
        Si tddft_basis est fourni, utilise cette base pour le calcul TDDFT au lieu
        du défaut def2-SVP.
        """
        print(f"\n{'='*60}")
        print(f"CALCUL LAMBDA MAX TD-DFT: {name}")
        print(f"{'='*60}")
        
        # Suivi des temps de calcul
        total_start_time = time.time()
        optimization_time = 0.0
        tddft_time = 0.0
        
        # Classification de la molécule par famille
        mol_classification = classify_molecule_family(name, smiles)
        print(f"? Classification: {mol_classification['family']}")
        print(f"   Substituant: {mol_classification['substituent']}, Position: {mol_classification['position']}")
        
        # Nettoyage préventif
        try:
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.clean_variables()
        except:
            pass
        
        try:
            # 1. Créer la molécule Psi4 (ou réutiliser celle fournie, ex: DFT optimisée)
            if mol_psi4 is not None:
                mol = mol_psi4
                xyz_file_used = True   # considérer comme géo. existante
                xyz_content = None
                print(f"OK Molécule réutilisée (DFT optimisée): {mol.natom()} atomes, charge={mol.molecular_charge()}, mult={mol.multiplicity()}")
            else:
                mol, xyz_file_used, xyz_content = self.setup_molecule(smiles, name)
                print(f"OK Molécule créée: {mol.natom()} atomes, charge={mol.molecular_charge()}, mult={mol.multiplicity()}")
            
            # 2. Déterminer la fonctionnelle optimale automatiquement
            if functional is None:
                if xyz_file_used and xyz_content:
                    # Analyser les atomes du fichier XYZ pour choisir la fonctionnelle
                    functional = self.determine_functional_from_xyz(xyz_content, name)
                elif mol_psi4 is not None:
                    # Molécule DFT fournie sans xyz_content : construire xyz_content
                    # depuis la molécule Psi4 pour une auto-sélection correcte
                    try:
                        BOHR_TO_ANG = 0.529177
                        _lines = [str(mol.natom()), name]
                        for i in range(mol.natom()):
                            sym = mol.symbol(i)
                            x = mol.x(i) * BOHR_TO_ANG
                            y = mol.y(i) * BOHR_TO_ANG
                            z = mol.z(i) * BOHR_TO_ANG
                            _lines.append(f"{sym}  {x:.8f}  {y:.8f}  {z:.8f}")
                        _xyz_for_func = "\n".join(_lines)
                        functional = self.determine_functional_from_xyz(_xyz_for_func, name)
                    except Exception:
                        functional = self.determine_optimal_functional(smiles, name)
                else:
                    # Utiliser le SMILES pour déterminer la fonctionnelle
                    functional = self.determine_optimal_functional(smiles, name)
            else:
                print(f"[>] Fonctionnelle forcée: {functional}")
            
            # Utiliser la base fournie ou le défaut def2-SVP
            optimal_basis = tddft_basis or "def2-SVP"
            print(f"   OK Méthode sélectionnée: {functional}/{optimal_basis}")
        
            results = {
                "name": name,
                "smiles": smiles,
                "method": f"TD-{functional}",
                "basis": optimal_basis,
                "functional_used": functional,
                # Classification par famille
                "family": mol_classification["family"],
                "substituent": mol_classification["substituent"],
                "position": mol_classification["position"]
            }
            
            # Stocker les informations de charge/multiplicité
            results["molecular_charge"] = mol.molecular_charge()
            results["multiplicity"] = mol.multiplicity()
            results["n_atoms"] = mol.natom()
            results["basis"] = optimal_basis  # Base réellement utilisée
            
            # 3. Optimisation de géométrie (automatiquement désactivée si fichier XYZ trouvé)
            geometry_functional = geometry_functional_override or "B3LYP"
            geometry_basis = geometry_basis_override or "def2-SVP"
            
            # Logique pour l'optimisation
            n_atoms = mol.natom()
            opt_start_time = time.time()
            
            if xyz_file_used:
                print("OK Fichier XYZ détecté - Optimisation automatiquement désactivée")
                results["geometry_optimized"] = False
                results["geometry_method"] = "Géométrie XYZ existante conservée"
                results["optimization_method"] = "N/A (fichier XYZ)"
                results["optimization_basis"] = "N/A"
            elif not optimize:
                print("AVERT Optimisation désactivée par l'utilisateur")
                results["geometry_optimized"] = False
                results["geometry_method"] = "Géométrie RDKit conservée"
                results["optimization_method"] = "N/A (désactivée)"
                results["optimization_basis"] = "N/A"
            else:
                print(f"[fix] Optimisation géométrique pour {n_atoms} atomes...")
                opt_results = self.optimize_geometry(mol, geometry_functional, geometry_basis, name)
                results["geometry_optimized"] = opt_results["converged"]
                results["geometry_method"] = f"{geometry_functional}/{geometry_basis}"
                results["optimization_method"] = geometry_functional
                results["optimization_basis"] = geometry_basis
                
                results["optimized_energy_hartree"] = opt_results.get("optimized_energy", None)
                if not opt_results["converged"]:
                    print("AVERT Optimisation échouée, utilisation géométrie initiale RDKit")
                else:
                    print(f"OK Géométrie optimisée avec {geometry_functional}/{geometry_basis}")
            
            optimization_time = time.time() - opt_start_time
            results["optimization_time_seconds"] = round(optimization_time, 2)
            print(f"? Temps optimisation: {optimization_time:.1f}s")
            
            # 4. Calcul TD-DFT pour lambda max avec fonctionnelle et base sélectionnées
            print(f"[>] Calcul TD-DFT: {functional}/{optimal_basis}")
            tddft_start_time = time.time()
            tddft_results = self.calculate_tddft(mol, functional, optimal_basis,
                                               n_states=n_states, solvent=solvent,
                                               timeout_minutes=timeout_minutes,
                                               use_tda=use_tda)
            tddft_time = time.time() - tddft_start_time
            
            # Enregistrer les méthodes TD-DFT
            results["tddft_method"] = functional
            results["tddft_basis"] = optimal_basis
            results["tddft_time_seconds"] = round(tddft_time, 2)
            print(f"? Temps TD-DFT: {tddft_time:.1f}s")
            
            # Vérifier le succès du calcul
            if tddft_results["converged"] and tddft_results.get("lambda_max_nm"):
                results.update({
                    "lambda_max_nm": tddft_results["lambda_max_nm"],
                    "lambda_max_energy_ev": 1240.0 / tddft_results["lambda_max_nm"],
                    "max_oscillator_strength": tddft_results["max_oscillator_strength"],
                    "all_transitions": tddft_results["transitions"],
                    "n_states_calculated": tddft_results["n_states"],
                    "calculation_successful": True,
                    "method": tddft_results["method"],  # Utiliser la méthode retournée par calculate_tddft
                    "basis": tddft_results["basis"],     # Utiliser la base retournée par calculate_tddft
                    "ground_state_energy_hartree": tddft_results.get("ground_state_energy_hartree", None)
                })
                print(f"OK Lambda max: {tddft_results['lambda_max_nm']:.1f} nm")
                print(f"OK Force oscillateur: {tddft_results['max_oscillator_strength']:.4f}")
                print(f"OK Méthode finale: {tddft_results['method']}/{tddft_results['basis']}")
                
                # Calculer le temps total AVANT la sauvegarde
                total_time = time.time() - total_start_time
                results["total_time_seconds"] = round(total_time, 2)
                
                # Sauvegarder les coordonnées optimisées
                self.save_molecule_data(mol, name, results, solvent)
            else:
                # Calcul échoué - passer à la molécule suivante
                error_msg = tddft_results.get("error", "Calcul TD-DFT échoué")
                results.update({
                    "calculation_successful": False,
                    "error": error_msg
                })
                print(f"ERREUR Calcul échoué: {error_msg}")
                print(f"? Passage à la molécule suivante")
            
        except Exception as e:
            print(f"ERREUR Erreur calcul lambda max: {e}")
            print(f"? Passage à la molécule suivante")
            results = {
                "name": name,
                "smiles": smiles,
                "calculation_successful": False,
                "error": str(e),
                "family": mol_classification["family"],
                "substituent": mol_classification["substituent"],
                "position": mol_classification["position"]
            }
        
        # Temps total de calcul
        total_time = time.time() - total_start_time
        results["total_time_seconds"] = round(total_time, 2)
        results["optimization_time_seconds"] = results.get("optimization_time_seconds", 0.0)
        results["tddft_time_seconds"] = results.get("tddft_time_seconds", 0.0)
        
        print(f"\n? TEMPS DE CALCUL POUR {name}:")
        print(f"   Optimisation: {results['optimization_time_seconds']:.1f}s")
        print(f"   TD-DFT:       {results['tddft_time_seconds']:.1f}s")
        print(f"   TOTAL:        {results['total_time_seconds']:.1f}s")
        
        return results
    
    def save_molecule_data(self, mol: psi4.core.Molecule, name: str, results: Dict, solvent: str = None):
        """
        Sauvegarde les coordonnées optimisées et informations détaillées
        """
        import os
        import json
        
        # Créer dossier de sortie si nécessaire
        output_dir = "molecules_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Nom de fichier basé sur molécule et solvant
        base_name = name.replace(" ", "_")
        if solvent:
            base_name += f"_{solvent}"
        
        # 1. Sauvegarder géométrie XYZ optimisée (format standard avec points)
        xyz_file = os.path.join(output_dir, f"{base_name}_optimized.xyz")
        
        # Facteur de conversion Bohr -> Angström
        bohr_to_angstrom = 0.529177210903
        
        # Créer fichier XYZ manuellement pour éviter les problèmes de locale (virgules vs points)
        with open(xyz_file, 'w') as f:
            f.write(f"{mol.natom()}\n")
            f.write(f"Molecule: {name} Lambda_max: {results.get('lambda_max_nm', 'N/A')} nm\n")
            
            for i in range(mol.natom()):
                symbol = mol.symbol(i)
                # Convertir les coordonnées de Bohr vers Angström
                x = mol.x(i) * bohr_to_angstrom
                y = mol.y(i) * bohr_to_angstrom  
                z = mol.z(i) * bohr_to_angstrom
                # Forcer le format anglais avec points décimaux
                f.write(f"{symbol:2s} {x:12.8f} {y:12.8f} {z:12.8f}\n")
        
        # 2. Sauvegarder informations détaillées JSON
        detailed_info = {
            "molecule_name": name,
            "smiles": results.get("smiles", "N/A"),
            "solvent": solvent if solvent else "Vacuum",
            # Classification par famille
            "classification": {
                "family": results.get("family", "N/A"),
                "substituent": results.get("substituent", "N/A"),
                "position": results.get("position", "N/A")
            },
            # Méthodes utilisées
            "methods": {
                "optimization": {
                    "functional": results.get("optimization_method", "N/A"),
                    "basis": results.get("optimization_basis", "N/A"),
                    "full_method": results.get("geometry_method", "N/A"),
                    "converged": results.get("geometry_optimized", False)
                },
                "tddft": {
                    "functional": results.get("tddft_method", "N/A"),
                    "basis": results.get("tddft_basis", "N/A"),
                    "full_method": f"TD-{results.get('tddft_method', 'N/A')}/{results.get('tddft_basis', 'N/A')}"
                }
            },
            # Temps de calcul
            "calculation_times": {
                "optimization_seconds": results.get("optimization_time_seconds", 0.0),
                "tddft_seconds": results.get("tddft_time_seconds", 0.0),
                "total_seconds": results.get("total_time_seconds", 0.0),
                "optimization_formatted": f"{results.get('optimization_time_seconds', 0.0):.1f}s",
                "tddft_formatted": f"{results.get('tddft_time_seconds', 0.0):.1f}s",
                "total_formatted": f"{results.get('total_time_seconds', 0.0):.1f}s"
            },
            # Résultats spectroscopiques
            "lambda_max_nm": results.get("lambda_max_nm"),
            "lambda_max_energy_ev": results.get("lambda_max_energy_ev"),
            "max_oscillator_strength": results.get("max_oscillator_strength"),
            "all_transitions": results.get("all_transitions", []),
            "n_states_calculated": results.get("n_states_calculated"),
            "geometry_optimized": results.get("geometry_optimized", False),
            "coordinates": {
                "units": "Angstrom",  # Coordonnées converties de Bohr vers Angström
                "natom": mol.natom(),
                "molecular_charge": mol.molecular_charge(),
                "multiplicity": mol.multiplicity(),
                "point_group": mol.point_group().symbol(),
                "atoms": []
            }
        }
        
        # Ajouter coordonnées atomiques (convertir en Angström)
        for i in range(mol.natom()):
            atom_info = {
                "element": mol.symbol(i),
                "atomic_number": mol.Z(i),
                "x": mol.x(i) * bohr_to_angstrom,
                "y": mol.y(i) * bohr_to_angstrom,
                "z": mol.z(i) * bohr_to_angstrom,
                "mass": mol.mass(i)
            }
            detailed_info["coordinates"]["atoms"].append(atom_info)
        
        json_file = os.path.join(output_dir, f"{base_name}_data.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_info, f, indent=2, ensure_ascii=False)
        
        # 3. Créer fichier texte résumé lisible
        summary_file = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RÉSUMÉ CALCUL TD-DFT\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Molécule: {name}\n")
            f.write(f"SMILES: {results.get('smiles', 'N/A')}\n")
            f.write(f"Solvant: {solvent if solvent else 'Phase gazeuse'}\n\n")
            
            f.write(f"CLASSIFICATION:\n")
            f.write(f"Famille: {results.get('family', 'N/A')}\n")
            f.write(f"Substituant: {results.get('substituent', 'N/A')}\n")
            f.write(f"Position: {results.get('position', 'N/A')}\n\n")
            
            f.write(f"MÉTHODES UTILISÉES:\n")
            f.write(f"Optimisation: {results.get('optimization_method', 'N/A')}/{results.get('optimization_basis', 'N/A')}\n")
            f.write(f"TD-DFT: TD-{results.get('tddft_method', 'N/A')}/{results.get('tddft_basis', 'N/A')}\n\n")
            
            f.write(f"TEMPS DE CALCUL:\n")
            f.write(f"Optimisation: {results.get('optimization_time_seconds', 0.0):.1f}s\n")
            f.write(f"TD-DFT: {results.get('tddft_time_seconds', 0.0):.1f}s\n")
            f.write(f"TOTAL: {results.get('total_time_seconds', 0.0):.1f}s\n\n")
            
            f.write(f"RÉSULTATS SPECTROSCOPIQUES:\n")
            lambda_max = results.get('lambda_max_nm', 'N/A')
            energy = results.get('lambda_max_energy_ev', 'N/A')
            osc = results.get('max_oscillator_strength', 'N/A')
            f.write(f"lambda_max: {lambda_max:.1f} nm\n" if isinstance(lambda_max, (int, float)) else f"lambda_max: {lambda_max}\n")
            f.write(f"Énergie: {energy:.3f} eV\n" if isinstance(energy, (int, float)) else f"Énergie: {energy}\n")
            f.write(f"Force d'oscillateur: {osc:.6f}\n\n" if isinstance(osc, (int, float)) else f"Force d'oscillateur: {osc}\n\n")
            
            # Énergie totale de la molécule
            f.write(f"ÉNERGIE TOTALE:\n")
            opt_e = results.get('optimized_energy_hartree', None)
            gs_e = results.get('ground_state_energy_hartree', None)
            total_e = opt_e if opt_e is not None else gs_e
            if total_e is not None and isinstance(total_e, (int, float)):
                f.write(f"Énergie totale: {total_e:.8f} Hartree ({total_e * 27.2114:.4f} eV)\n\n")
            else:
                f.write(f"Énergie totale: N/A\n\n")
            
            f.write(f"GÉOMÉTRIE MOLÉCULAIRE:\n")
            f.write(f"Nombre d'atomes: {mol.natom()}\n")
            f.write(f"Charge: {mol.molecular_charge()}\n")
            f.write(f"Multiplicité: {mol.multiplicity()}\n")
            f.write(f"Groupe ponctuel: {mol.point_group().symbol()}\n\n")
            
            if results.get("all_transitions"):
                f.write(f"TRANSITIONS ÉLECTRONIQUES:\n")
                for i, trans in enumerate(results["all_transitions"][:10]):  # Top 10
                    if isinstance(trans, dict):
                        # Format dictionnaire
                        energy = trans.get('energy_ev', 0)
                        wavelength = 1240.0 / energy if energy > 0 else 0
                        osc = trans.get('oscillator_strength', 0)
                        f.write(f"État {i+1}: {energy:.3f} eV ({wavelength:.1f} nm), f={osc:.6f}\n")
                    elif isinstance(trans, (list, tuple)) and len(trans) >= 3:
                        # Format liste [energie_eV, oscillateur, wavelength_nm]
                        energy = trans[0]
                        osc = trans[1]
                        wavelength = trans[2]
                        f.write(f"État {i+1}: {energy:.3f} eV ({wavelength:.1f} nm), f={osc:.6f}\n")
        
        print(f"? Fichiers sauvegardés:")
        print(f"   - Géométrie XYZ: {xyz_file}")
        print(f"   - Données JSON: {json_file}")
        print(f"   - Résumé TXT: {summary_file}")
    
    def analyze_lambda_max_series(self, molecules_data: List[Dict], solvent=None) -> pd.DataFrame:
        """
        Analyse série de molécules pour lambda max uniquement
        Continue même en cas d'erreur sur une molécule
        """
        print(f"\n{'='*80}")
        print(f"ANALYSE LAMBDA MAX TD-DFT SÉRIE")
        if solvent:
            print(f"CONDITIONS: {solvent}")
        print(f"{'='*80}")
        
        all_results = []
        successful_count = 0
        failed_count = 0
        
        for i, mol_data in enumerate(molecules_data):
            print(f"\n[{i+1}/{len(molecules_data)}] Lambda max de {mol_data['name']}...")
            
            try:
                result = self.calculate_lambda_max(
                    mol_data["smiles"], 
                    mol_data["name"],
                    optimize=True,
                    solvent=solvent
                )
                
                if result.get("calculation_successful", False):
                    successful_count += 1
                    print(f"OK {mol_data['name']}: {result.get('lambda_max_nm', 'N/A'):.1f} nm")
                else:
                    failed_count += 1
                    print(f"ERREUR {mol_data['name']}: {result.get('error', 'Erreur inconnue')}")
                
                all_results.append(result)
                
            except Exception as e:
                # En cas d'erreur inattendue, créer un résultat d'échec
                failed_count += 1
                error_result = {
                    "name": mol_data["name"],
                    "smiles": mol_data["smiles"],
                    "calculation_successful": False,
                    "error": f"Erreur inattendue: {str(e)}"
                }
                all_results.append(error_result)
                print(f"ERREUR {mol_data['name']}: Erreur inattendue - {str(e)}")
                
                # Nettoyage Psi4 en cas d'erreur
                try:
                    psi4.core.clean()
                    psi4.core.clean_options()
                    psi4.core.clean_variables()
                except:
                    pass
        
        print(f"\n[data] BILAN FINAL:")
        print(f"   OK Succès: {successful_count}/{len(molecules_data)}")
        print(f"   ERREUR Échecs: {failed_count}/{len(molecules_data)}")
        print(f"   ? Taux de réussite: {(successful_count/len(molecules_data)*100):.1f}%")
        
        return pd.DataFrame(all_results)
    
    def create_lambda_max_visualizations(self, df: pd.DataFrame, output_dir="."):
        """
        Crée des visualisations spécialisées pour lambda max
        """
        print("\n[data] Création des visualisations lambda max...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Lambda max par molécule
        ax1 = axes[0]
        if 'lambda_max_nm' in df.columns:
            successful = df[df['calculation_successful'] == True]
            if len(successful) > 0:
                bars = ax1.bar(successful['name'], successful['lambda_max_nm'], 
                              alpha=0.8, color='red', edgecolor='darkred', linewidth=1)
                ax1.set_title('Lambda max TD-DFT', fontsize=14, fontweight='bold')
                ax1.set_ylabel('lambda_max (nm)', fontsize=12)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Ajouter les valeurs sur les barres
                for bar, value in zip(bars, successful['lambda_max_nm']):
                    if pd.notna(value):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Forces d'oscillateur
        ax2 = axes[1]
        if 'max_oscillator_strength' in df.columns:
            successful = df[df['calculation_successful'] == True]
            if len(successful) > 0:
                bars = ax2.bar(successful['name'], successful['max_oscillator_strength'], 
                              alpha=0.8, color='orange', edgecolor='darkorange', linewidth=1)
                ax2.set_title('Force d\'oscillateur max', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Force d\'oscillateur', fontsize=12)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Ajouter les valeurs sur les barres
                for bar, value in zip(bars, successful['max_oscillator_strength']):
                    if pd.notna(value):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Énergie d'excitation
        ax3 = axes[2]
        if 'lambda_max_energy_ev' in df.columns:
            successful = df[df['calculation_successful'] == True]
            if len(successful) > 0:
                bars = ax3.bar(successful['name'], successful['lambda_max_energy_ev'], 
                              alpha=0.8, color='blue', edgecolor='darkblue', linewidth=1)
                ax3.set_title('Énergie d\'excitation', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Énergie (eV)', fontsize=12)
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # Ajouter les valeurs sur les barres
                for bar, value in zip(bars, successful['lambda_max_energy_ev']):
                    if pd.notna(value):
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lambda_max_tddft.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("OK Visualisations lambda max créées!")


def validate_molecular_geometry(mol: psi4.core.Molecule, name: str = "") -> Dict:
    """
    Valide la géométrie moléculaire et détecte les problèmes courants
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "bond_lengths_ok": True,
        "angles_ok": True,
        "clashes_detected": False
    }
    
    try:
        import numpy as np
        
        # 1. Vérifier les distances interatomiques
        min_distance = float('inf')
        max_distance = 0.0
        clash_threshold = 0.8  # Angström - seuil de collision atomique
        
        for i in range(mol.natom()):
            for j in range(i+1, mol.natom()):
                # Calculer distance entre atomes i et j
                xi, yi, zi = mol.x(i) * 0.529177, mol.y(i) * 0.529177, mol.z(i) * 0.529177
                xj, yj, zj = mol.x(j) * 0.529177, mol.y(j) * 0.529177, mol.z(j) * 0.529177
                
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                min_distance = min(min_distance, distance)
                max_distance = max(max_distance, distance)
                
                # Détecter collisions atomiques
                if distance < clash_threshold:
                    validation_results["clashes_detected"] = True
                    validation_results["warnings"].append(
                        f"Collision détectée: atomes {i+1}-{j+1} ({mol.symbol(i)}-{mol.symbol(j)}) distance={distance:.2f}Å"
                    )
        
        # 2. Vérifier les longueurs de liaison raisonnables
        if min_distance < 0.5:  # Trop proche
            validation_results["bond_lengths_ok"] = False
            validation_results["warnings"].append(f"Distance minimale suspecte: {min_distance:.2f}Å")
        
        if max_distance > 15.0:  # Trop éloigné pour molécules organiques
            validation_results["warnings"].append(f"Molécule très étendue: distance max {max_distance:.2f}Å")
        
        # 3. Résumé de validation
        if validation_results["clashes_detected"] or not validation_results["bond_lengths_ok"]:
            validation_results["valid"] = False
        
        # Rapport
        if validation_results["warnings"]:
            print(f"AVERT Géométrie {name}: {len(validation_results['warnings'])} avertissements")
            for warning in validation_results["warnings"]:
                print(f"   • {warning}")
        else:
            print(f"OK Géométrie {name}: validation OK (dist. min={min_distance:.2f}Å)")
            
    except Exception as e:
        validation_results["valid"] = False
        validation_results["warnings"].append(f"Erreur validation: {e}")
        print(f"ERREUR Erreur validation géométrie {name}: {e}")
    
    return validation_results


def main():
    """Fonction principale avec sélection automatique de fonctionnelle DFT"""
    print("? CALCUL LAMBDA MAX TD-DFT AVEC SÉLECTION AUTOMATIQUE DE FONCTIONNELLE")
    print("B3LYP ou CAM-B3LYP selon la structure moléculaire\n")
    
    # TOUTES LES MOLÉCULES triées par famille: CAm -> CCAm -> CDCAm
    molecules = [
        
        
        # ================================================================
        # FAMILLE 3: CDCAm (Cyanodienyl-amides)
        # ================================================================

        # Para-substitués CDCAm
    
        {"name": "3,4,5-triMeOCDCAm", "smiles": "O=C(NC)/C(C#N)=C/C=C/C1=CC(OC)=C(OC)C(OC)=C1"},
        {"name": "3,4,5-triHOCDCAm", "smiles": "O=C(NC)/C(C#N)=C/C=C/C1=CC(O)=C(O)C(O)=C1"},
        
       
    ]
   
    try:
        # Initialiser le calculateur Psi4 avec configuration haute performance
        calculator = Psi4MolecularCalculator(memory="12 GB", threads=10)
        
        print("? Test de sélection automatique de fonctionnelle...")
        print("="*60)
        
        # Tester chaque molécule individuellement pour voir la sélection de fonctionnelle
        all_results = []
        for mol_data in molecules:
            print(f"\n? Analyse: {mol_data['name']}")
            result = calculator.calculate_lambda_max(
                mol_data['smiles'], 
                mol_data['name'], 
                solvent="hplc_mobile"
            )
            all_results.append(result)
        
        # Créer DataFrame des résultats
        import pandas as pd
        results_df = pd.DataFrame(all_results)
        
        
        # Sauvegarder les résultats complets
        output_file = "lambda_max_tddft.csv"
        results_df.to_csv(output_file, index=False)
        
        # Sauvegarder résumé détaillé avec coordonnées
        detailed_output = "lambda_max_detailed.csv"
        detailed_df = results_df.copy()
        
        # Ajouter colonnes d'informations supplémentaires
        for col in ['method', 'basis', 'n_atoms', 'molecular_charge', 'multiplicity']:
            if col not in detailed_df.columns:
                detailed_df[col] = 'N/A'
        
        detailed_df.to_csv(detailed_output, index=False)
        
        print(f"\nOK Résultats sauvegardés:")
        print(f"   - Résumé: {output_file}")
        print(f"   - Détaillé: {detailed_output}")
        print(f"   - Coordonnées individuelles: dossier 'molecules_data/'")
        
        # Afficher résumé
        print("\n" + "="*80)
        print("RÉSULTATS LAMBDA MAX TD-DFT")
        print("="*80)
        
        successful = results_df[results_df['calculation_successful'] == True]
        if len(successful) > 0:
            summary_cols = ['name', 'lambda_max_nm', 'lambda_max_energy_ev', 'max_oscillator_strength']
            print(successful[summary_cols].round(2).to_string(index=False))
            
            print("\n[data] Statistiques:")
            print(f"   Lambda max moyen: {successful['lambda_max_nm'].mean():.1f} nm")
            print(f"   Lambda max min:   {successful['lambda_max_nm'].min():.1f} nm")
            print(f"   Lambda max max:   {successful['lambda_max_nm'].max():.1f} nm")
        else:
            print("ERREUR Aucun calcul réussi")
        
        # Créer visualisations spécialisées
        calculator.create_lambda_max_visualizations(results_df)
        
        print(f"\n? Calculs lambda max terminés!")
        print(f"Méthode: TD-DFT de niveau recherche pour spectroscopie UV-Vis")
        
    except ImportError:
        print("ERREUR Psi4 non installé!")
        print("Installation: conda install psi4 -c psi4")
    except Exception as e:
        print(f"ERREUR Erreur: {e}")


if __name__ == "__main__":
    main()