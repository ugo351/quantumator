#!/usr/bin/env python3
"""
Script d'analyse de densité électronique avec Psi4
Utilise des géométries .xyz pré-optimisées (HPLC)
Génère des fichiers .cube et .txt pour chaque molécule
"""

import os
import locale

# Forcer la locale C AVANT d'importer Psi4 pour éviter les virgules
# décimales dans les fichiers .cube (bug locale française Windows)
os.environ['LC_NUMERIC'] = 'C'
os.environ['LC_ALL'] = 'C'
try:
    locale.setlocale(locale.LC_ALL, 'C')
except Exception:
    pass

import psi4
import multiprocessing
import shutil
import time
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None

def diagnose_calculation_stability():
    """Diagnostic de stabilité pour les calculs Psi4"""
    print("🔍 DIAGNOSTIC STABILITÉ CALCULS PSI4")
    print("="*50)
    
    if psutil:
        # Vérifications mémoire
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        print(f"Mémoire disponible: {available_gb:.1f} GB")
        
        if available_gb < 6:
            print("⚠️ ATTENTION: Mémoire insuffisante (<6 GB)")
            return False
    else:
        print("Info: psutil non disponible, diagnostic limité")
    
    return True

def setup_psi4(memory=None, threads=None, functional=None, basis=None):
    """Configuration de Psi4 avec gestion robuste.
    
    Parameters
    ----------
    memory : str, optional
        Mémoire à allouer (ex: '8 GB'). Si None, utilise '6 GB'.
    threads : int, optional
        Nombre de threads. Si None, détection automatique (max 4).
    functional : str, optional
        Fonctionnelle DFT. Défaut 'B3LYP'.
    basis : str, optional
        Base. Défaut 'def2-SVP'.
    """
    basis_to_use = basis or 'def2-SVP'
    func_to_use  = functional or 'B3LYP'
    
    # Diagnostic de stabilité
    if not diagnose_calculation_stability():
        print("⚠️ Conditions système non optimales détectées")
    
    # Détection automatique du nombre de cœurs
    num_cores = multiprocessing.cpu_count()
    mem_to_use = memory if memory else '12 GB'
    threads_to_use = threads if threads else min(num_cores, 10)
    
    print(f"Détection automatique: {num_cores} coeurs disponibles")
    print(f"Configuration Psi4: {threads_to_use} threads, {mem_to_use} memoire")
    print(f"Fonctionnelle: {func_to_use}, Base: {basis_to_use}")
    
    # Configuration robuste pour calculs avec charges
    try:
        psi4.set_memory(mem_to_use)
        psi4.set_num_threads(threads_to_use)
        print(f"✅ Psi4 configuré avec succès")
    except Exception as e:
        print(f"⚠️ Erreur configuration Psi4: {e}")
        # Fallback configuration
        psi4.set_memory('4 GB')
        psi4.set_num_threads(2)
        print(f"🔄 Configuration fallback: 4 GB, 2 threads")
    
    psi4.core.set_output_file('psi4_output.log', False)
    
    # Options robustes
    psi4.set_options({
        'basis': basis_to_use,
        'scf_type': 'df',
        'reference': 'rhf',
        'df_scf_guess': True,
        'maxiter': 200,
        'e_convergence': 1e-6,
        'd_convergence': 1e-6,
        'fail_on_maxiter': False,
        'soscf': False,
        'diis': True,
        'guess': 'sad',
    })

def read_xyz_geometry(xyz_file_path, name):
    """
    Lit un fichier .xyz et retourne la géométrie pour Psi4
    """
    try:
        if not os.path.exists(xyz_file_path):
            raise FileNotFoundError(f"Fichier .xyz non trouve: {xyz_file_path}")
        
        with open(xyz_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Première ligne = nombre d'atomes
        num_atoms = int(lines[0].strip())
        # Deuxième ligne = commentaire (ignorée)
        # Lignes suivantes = atomes et coordonnées
        
        atoms = []
        coords = []
        geometry_lines = []
        
        for i in range(2, 2 + num_atoms):
            line = lines[i].strip()
            if not line:  # Ignorer les lignes vides
                continue
                
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0].strip()
                # Remplacer les virgules par des points pour les coordonnées (format français)
                x = float(parts[1].replace(',', '.'))
                y = float(parts[2].replace(',', '.'))
                z = float(parts[3].replace(',', '.'))
                
                atoms.append(symbol)
                coords.append([x, y, z])
                geometry_lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
        
        if len(atoms) != num_atoms:
            print(f"Attention: {len(atoms)} atomes lus au lieu de {num_atoms} attendus pour {name}")
        
        geometry_string = "\n".join(geometry_lines)
        
        # Pour les molécules organiques neutres, charge = 0, multiplicité = 1
        charge = 0
        multiplicity = 1
        
        print(f"Geometrie lue pour {name}: {len(atoms)} atomes")
        print(f"Exemple d'atome: {atoms[0]} {coords[0][0]:.3f} {coords[0][1]:.3f} {coords[0][2]:.3f}")
        
        return geometry_string, charge, multiplicity, atoms, coords
        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier .xyz pour {name}: {e}")
        print(f"Fichier: {xyz_file_path}")
        return None, None, None, None, None

def find_xyz_file(molecule_name, xyz_directory):
    """
    Trouve le fichier .xyz correspondant à une molécule
    """
    # Mapping des noms de molécules vers les fichiers .xyz
    name_mapping = {
  
        "3,4,5-triMeOCDCAm": "3,4,5-triMeOCDCAm",
        "3,4,5-triHOCDCAm": "3,4,5-triHOCDCAm",
        
    }
    
    # Utiliser le mapping ou le nom original
    file_base_name = name_mapping.get(molecule_name, molecule_name)
    
    # Construire le nom du fichier hplc
    xyz_filename = f"{file_base_name}_hplc_hplc_mobile_optimized.xyz"
    xyz_path = os.path.join(xyz_directory, xyz_filename)
    
    if os.path.exists(xyz_path):
        return xyz_path
    else:
        print(f"Fichier .xyz non trouve: {xyz_filename}")
        # Essayer d'autres variantes
        alternative_names = [
            f"{file_base_name}_hplc_optimized.xyz",
            f"{file_base_name}_optimized.xyz",
            f"{file_base_name}.xyz"
        ]
        
        for alt_name in alternative_names:
            alt_path = os.path.join(xyz_directory, alt_name)
            if os.path.exists(alt_path):
                print(f"Fichier alternatif trouve: {alt_name}")
                return alt_path
        
        return None

def calculate_electron_density_robust(geometry_string, charge, multiplicity, name,
                                      functional=None, basis=None):
    """
    Calcule la densité électronique avec gestion robuste.

    Parameters
    ----------
    functional : str, optional
        Fonctionnelle DFT. Défaut 'B3LYP'.
    basis : str, optional
        Base. Défaut 'def2-SVP'.
    """
    basis_to_use = basis or 'def2-SVP'
    func_to_use  = functional or 'B3LYP'

    try:
        print(f"[SCF] Configuration pour {name}: charge={charge}, multiplicité={multiplicity}")
        print(f"[SCF] Fonctionnelle: {func_to_use}, Base: {basis_to_use}")

        charge = int(charge)
        multiplicity = int(multiplicity)
        mol_string = (
            f"{charge} {multiplicity}\n"
            f"{geometry_string}\n"
            f"units angstrom\n"
            f"symmetry c1\n"
            f"no_reorient\n"
            f"no_com\n"
        )

        molecule = psi4.geometry(mol_string)

        scf_options = {
            'basis': basis_to_use,
            'scf_type': 'df',
            'maxiter': 200,
            'e_convergence': 1e-6,
            'd_convergence': 1e-6,
            'fail_on_maxiter': False,
            'diis': True,
            'guess': 'sad'
        }

        # Adapter la référence selon charge et multiplicité
        if charge != 0:
            if multiplicity == 1:
                scf_options['reference'] = 'rhf'
                print(f"[SCF] Référence RHF pour singulet chargé")
            else:
                scf_options['reference'] = 'uhf'
                print(f"[SCF] Référence UHF pour multiplicité {multiplicity}")
        else:
            scf_options['reference'] = 'rhf'
            print(f"[SCF] Référence RHF pour molécule neutre")

        psi4.set_options(scf_options)

        # Calcul SCF avec wavefunction
        print(f"[SCF] Calcul {func_to_use}/{basis_to_use} pour {name}...")
        energy, wfn = psi4.energy('scf', molecule=molecule, return_wfn=True)

        if energy != energy or wfn is None:
            raise ValueError(f"Résultats SCF invalides pour {name}")

        print(f"[OK] SCF convergé pour {name}: {energy:.8f} Hartree")

        return wfn, energy

    except Exception as e:
        print(f"[ERREUR] Calcul SCF échoué pour {name}: {e}")
        try:
            psi4.core.clean()
        except:
            pass
        return None, None

def calculate_optimal_grid_size(atoms, coords, name):
    """
    Calcule la taille de grille optimale basée sur les dimensions moléculaires.
    Retourne deux jeux de paramètres :
      - Grille compacte (densité, orbitales, dual descriptor) : 3 Å / 0.20 Å
      - Grille large (ESP) : 35 Å / 0.50 Å
    """
    # Grille compacte pour densité / orbitales (courte portée)
    overage_compact = 5    # Å — la densité décroît exponentiellement
    spacing_compact = 0.20 # Å — résolution fine pour orbitales

    # Grille large pour ESP (longue portée)
    overage_esp = 35       # Å — potentiel coulombien décroît lentement
    spacing_esp = 0.50     # Å — compromis taille/précision pour ESP

    # Rétro-compatibilité : retourne les paramètres compacts par défaut
    overage = overage_compact
    spacing = spacing_compact
    
    try:
        if not coords or not atoms:
            print(f"[GRID] Coordonnées manquantes pour {name}, grille par défaut")
            return overage, spacing
        
        # Calculer les dimensions moléculaires (pour info)
        coords_array = np.array(coords)
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        dimensions = max_coords - min_coords
        max_dimension = np.max(dimensions)
        
        # Infos pour les deux grilles
        box_compact = dimensions + 2 * overage_compact
        box_esp     = dimensions + 2 * overage_esp
        n_compact = int(np.prod(box_compact / spacing_compact))
        n_esp     = int(np.prod(box_esp / spacing_esp))
        
        print(f"[GRID] {name} - Dimensions moléculaires:")
        print(f"  X: {dimensions[0]:.2f} Å, Y: {dimensions[1]:.2f} Å, Z: {dimensions[2]:.2f} Å")
        print(f"  Dimension maximale: {max_dimension:.2f} Å")
        print(f"  Grille densité/orbitales : +{overage_compact} Å, pas {spacing_compact} Å → ~{n_compact:,} pts")
        print(f"  Grille ESP              : +{overage_esp} Å, pas {spacing_esp} Å → ~{n_esp:,} pts")
        
        return overage, spacing
        
    except Exception as e:
        print(f"[GRID] Erreur calcul grille pour {name}: {e}")
        return overage, spacing

def generate_electron_density_cube(wfn, name, output_dir, atoms=None, coords=None):
    """
    Génère des fichiers .cube pour densité électronique, potentiel électrostatique et réactivité
    """
    try:
        # Sauvegarder le répertoire de travail actuel
        original_dir = os.getcwd()
        
        # Changer vers le répertoire de sortie
        os.chdir(output_dir)
        
        # Nom final des fichiers cube
        density_cube_file = f"{name}_electron_density.cube"
        esp_cube_file = f"{name}_electrostatic_potential.cube"
        dual_cube_file = f"{name}_dual_descriptor.cube"
        
        print(f"Generation des fichiers cube pour {name}...")
        print(f"- Densite electronique (toujours positive)")
        print(f"- Potentiel electrostatique (positif/negatif selon charges)")
        print(f"- Dual descriptor (reactivite nucleophile/electrophile)")
        print(f"Repertoire de travail: {output_dir}")
        
        # Calcul de la taille de grille optimale
        if atoms and coords:
            _, _ = calculate_optimal_grid_size(atoms, coords, name)
        
        # ── Passe 1 : Grille COMPACTE pour densité, orbitales, dual descriptor ──
        overage_compact, spacing_compact = 5, 0.20
        print(f"[PASSE 1] Densité + orbitales + dual : +{overage_compact} Å, pas {spacing_compact} Å")
        # Forcer locale C juste avant cubeprop (sécurité contre reset par sous-processus)
        try:
            locale.setlocale(locale.LC_ALL, 'C')
        except Exception:
            pass
        psi4.set_options({
            'cubic_grid_spacing': [spacing_compact] * 5,
            'cubic_grid_overage': [overage_compact] * 5,
            'cubeprop_tasks': ['density', 'dual_descriptor'],
            'cubeprop_orbitals': [-1, 1],
        })
        psi4.cubeprop(wfn, density=True, dual_descriptor=True, orbitals=True, filepath=name)
        time.sleep(1)
        
        # ── Sauvegarder les fichiers de la passe 1 AVANT la passe 2 ──
        # (Psi4 peut régénérer Dt.cube lors de l'ESP et écraser la densité compacte)
        pass1_renames = [
            (f"{name}_Dt.cube", f"{name}_electron_density.cube"),
            (f"{name}_Da.cube", f"{name}_alpha_density.cube"),
            ("Dt.cube", f"{name}_electron_density.cube"),
            ("Da.cube", f"{name}_alpha_density.cube"),
            (f"{name}_DUAL.cube", f"{name}_dual_descriptor.cube"),
            (f"{name}_dual.cube", f"{name}_dual_descriptor.cube"),
            ("DUAL.cube", f"{name}_dual_descriptor.cube"),
            ("dual.cube", f"{name}_dual_descriptor.cube"),
        ]
        # Inclure aussi les fichiers DUAL avec patterns d'orbitales
        for f in os.listdir('.'):
            if f.startswith('DUAL_') and f.endswith('.cube'):
                pass1_renames.append((f, f"{name}_dual_descriptor.cube"))
        
        pass1_saved = []
        for src, dst in pass1_renames:
            if os.path.exists(src) and dst not in pass1_saved:
                shutil.move(src, dst)
                size_mb = os.path.getsize(dst) / (1024 * 1024)
                pass1_saved.append(dst)
                print(f"[PASSE 1 OK] {src} → {dst} ({size_mb:.1f} MB)")
        
        # ── Passe 2 : Grille LARGE pour ESP (potentiel longue portée) ──
        overage_esp, spacing_esp = 35, 0.50
        print(f"[PASSE 2] ESP : +{overage_esp} Å, pas {spacing_esp} Å")
        try:
            locale.setlocale(locale.LC_ALL, 'C')
        except Exception:
            pass
        psi4.set_options({
            'cubic_grid_spacing': [spacing_esp] * 3,
            'cubic_grid_overage': [overage_esp] * 3,
            'cubeprop_tasks': ['esp'],
            'cubeprop_orbitals': [],
        })
        psi4.cubeprop(wfn, esp=True, filepath=f"{name}_esp")
        time.sleep(1)
        
        # ── Renommer les fichiers ESP de la passe 2 ──
        # (les fichiers densité/dual ont déjà été sauvés après la passe 1)
        files_found = list(pass1_saved)  # Commencer avec les fichiers déjà sauvés
        
        esp_renames = [
            (f"{name}_esp_ESP.cube", f"{name}_electrostatic_potential.cube"),
            (f"{name}_esp_esp.cube", f"{name}_electrostatic_potential.cube"),
            (f"{name}_ESP.cube", f"{name}_electrostatic_potential.cube"),
            (f"{name}_esp.cube", f"{name}_electrostatic_potential.cube"),
            ("ESP.cube", f"{name}_electrostatic_potential.cube"),
            ("esp.cube", f"{name}_electrostatic_potential.cube"),
        ]
        
        for src, dst in esp_renames:
            if os.path.exists(src) and dst not in files_found:
                shutil.move(src, dst)
                size_mb = os.path.getsize(dst) / (1024 * 1024)
                files_found.append(dst)
                print(f"[PASSE 2 OK] {src} → {dst} ({size_mb:.1f} MB)")
        
        # Nettoyer les éventuels fichiers Dt/Da parasites de la passe 2
        for stale in [f"{name}_esp_Dt.cube", "Dt.cube", f"{name}_esp_Da.cube", "Da.cube"]:
            if os.path.exists(stale):
                os.remove(stale)
                print(f"[CLEAN] Supprimé fichier parasite passe 2: {stale}")
        
        if files_found:
            total_size_mb = sum(os.path.getsize(f) for f in files_found if os.path.exists(f)) / (1024 * 1024)
            print(f"[OK] {len(files_found)} fichiers cube générés pour {name} (Total: {total_size_mb:.1f} MB)")
            
            # Retourner le fichier de densité principal ou le premier trouvé
            density_file = next((f for f in files_found if "electron_density" in f), files_found[0])
            return os.path.join(output_dir, density_file)
        else:
            print(f"[ERREUR] Aucun fichier cube trouvé pour {name}")
            print(f"[DEBUG] Fichiers présents: {os.listdir('.')}")
            return None
        
    except Exception as e:
        print(f"[ERREUR] Erreur lors de la generation du fichier cube pour {name}: {e}")
        return None
    finally:
        # Toujours revenir au répertoire original
        os.chdir(original_dir)

def organize_orbital_cubes(name, output_dir):
    """
    Organise et renomme les fichiers d'orbitales générés par Psi4
    Les fichiers sont rangés dans le même dossier que les autres cubes de la molécule
    Nommage: {name}_alpha_density.cube, {name}_beta_density.cube, {name}_spin_density.cube
    """
    try:
        search_dirs = [output_dir, os.getcwd()]
        files_moved = 0
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            
            files_in_dir = os.listdir(search_dir)
            # Chercher les fichiers cube d'orbitales
            orbital_files = [f for f in files_in_dir if f.endswith('.cube') and 
                           ('Psi_' in f or 'psi_' in f or 
                            f.startswith('Da') or f.startswith('Db') or 
                            f.startswith('Ds') or f.startswith('Dt'))]
            
            # Exclure les fichiers déjà traités
            already_done = {'electron_density', 'electrostatic_potential', 'dual_descriptor',
                           'alpha_density', 'beta_density', 'spin_density'}
            orbital_files = [f for f in orbital_files if not any(d in f for d in already_done)]
            
            for file in orbital_files:
                source_file = os.path.join(search_dir, file)
                
                # Déterminer le nom de destination
                if file.startswith('Da'):
                    new_name = f"{name}_alpha_density.cube"
                elif file.startswith('Db'):
                    new_name = f"{name}_beta_density.cube"
                elif file.startswith('Ds'):
                    new_name = f"{name}_spin_density.cube"
                elif file.startswith('Dt'):
                    # Densité totale - normalement déjà capturée par generate_electron_density_cube
                    new_name = f"{name}_electron_density.cube"
                elif 'Psi_a_' in file:
                    parts = file.split('Psi_a_')[1].split('-')[0].split('_')
                    orbital_num = parts[0]
                    new_name = f"{name}_orbital_alpha_{orbital_num}.cube"
                elif 'Psi_b_' in file:
                    parts = file.split('Psi_b_')[1].split('-')[0].split('_')
                    orbital_num = parts[0]
                    new_name = f"{name}_orbital_beta_{orbital_num}.cube"
                else:
                    new_name = f"{name}_{file}"
                
                dest_file = os.path.join(output_dir, new_name)
                
                # Ne pas écraser un fichier existant
                if os.path.exists(dest_file):
                    print(f"[SKIP] {new_name} existe déjà")
                    # Supprimer le doublon source
                    try:
                        os.remove(source_file)
                    except:
                        pass
                    continue
                
                try:
                    shutil.move(source_file, dest_file)
                    files_moved += 1
                    print(f"[OK] {file} -> {new_name}")
                except Exception as move_error:
                    print(f"[ATTENTION] Erreur deplacement {file}: {move_error}")
        
        if files_moved > 0:
            print(f"[OK] {files_moved} fichiers d'orbitales organises dans: {output_dir}")
            return output_dir
        else:
            print(f"[INFO] Aucun fichier d'orbitale supplementaire pour {name}")
            return None
        
    except Exception as e:
        print(f"[ERREUR] Organisation orbitales pour {name}: {e}")
        return None

def extract_electronic_properties(wfn, name):
    """
    Extrait les propriétés électroniques clés de la wavefunction Psi4 :
    - Énergies HOMO / LUMO et gap
    - Moment dipolaire
    - Dureté / mollesse chimique
    - Potentiel chimique électronique
    - Nombre d'électrons et d'orbitales
    """
    props = {}
    HARTREE_TO_EV = 27.2114

    try:
        # ── Orbitales ──
        eps = wfn.epsilon_a().np  # énergies orbitales alpha (Hartree)
        n_occ = wfn.nalpha()      # nombre d'électrons alpha occupés
        n_mo  = len(eps)
        homo_idx = n_occ - 1
        lumo_idx = n_occ

        if homo_idx >= 0:
            homo_ha = float(eps[homo_idx])
            props["homo_eV"] = round(homo_ha * HARTREE_TO_EV, 4)
        if lumo_idx < n_mo:
            lumo_ha = float(eps[lumo_idx])
            props["lumo_eV"] = round(lumo_ha * HARTREE_TO_EV, 4)
        if "homo_eV" in props and "lumo_eV" in props:
            gap = props["lumo_eV"] - props["homo_eV"]
            props["homo_lumo_gap_eV"] = round(gap, 4)
            # Dureté chimique η = (LUMO - HOMO) / 2
            props["hardness_eV"] = round(gap / 2, 4)
            # Potentiel chimique électronique μ = (HOMO + LUMO) / 2
            props["chemical_potential_eV"] = round((props["homo_eV"] + props["lumo_eV"]) / 2, 4)
            # Électrophilicité ω = μ² / (2η)
            if props["hardness_eV"] > 0:
                props["electrophilicity_eV"] = round(
                    props["chemical_potential_eV"] ** 2 / (2 * props["hardness_eV"]), 4
                )

        props["n_electrons"] = wfn.nalpha() + wfn.nbeta()
        props["n_orbitals"]  = n_mo
        print(f"[PROP] {name}: HOMO={props.get('homo_eV','?')} eV, "
              f"LUMO={props.get('lumo_eV','?')} eV, gap={props.get('homo_lumo_gap_eV','?')} eV")
    except Exception as e:
        print(f"[PROP] Erreur extraction HOMO/LUMO pour {name}: {e}")

    # ── Moment dipolaire ──
    try:
        psi4.oeprop(wfn, 'DIPOLE')
        dx = wfn.variable('SCF DIPOLE X')
        dy = wfn.variable('SCF DIPOLE Y')
        dz = wfn.variable('SCF DIPOLE Z')
        dipole_debye = (dx**2 + dy**2 + dz**2) ** 0.5
        props["dipole_debye"] = round(dipole_debye, 4)
        props["dipole_x"] = round(dx, 4)
        props["dipole_y"] = round(dy, 4)
        props["dipole_z"] = round(dz, 4)
        print(f"[PROP] {name}: Dipôle = {dipole_debye:.4f} Debye")
    except Exception as e:
        print(f"[PROP] Erreur extraction dipôle pour {name}: {e}")

    return props


def calculate_atomic_charges(wfn, atoms, name):
    """
    Calcule les charges atomiques en utilisant l'analyse de population de Mulliken
    """
    try:
        # Calcul des charges de Mulliken
        psi4.oeprop(wfn, 'MULLIKEN_CHARGES')
        
        # Accéder aux charges de Mulliken via les variables Psi4 (version corrigée)
        try:
            # Nouvelle syntaxe recommandée
            mulliken_charges = wfn.variable('MULLIKEN CHARGES')
        except:
            # Fallback vers l'ancienne syntaxe si nécessaire
            mulliken_charges = wfn.variable('MULLIKEN_CHARGES')
        
        # Convertir en liste Python
        atomic_charges = []
        for i in range(len(atoms)):
            atomic_charges.append(float(mulliken_charges[i]))
        
        return atomic_charges
        
    except Exception as e:
        print(f"Erreur lors du calcul des charges pour {name}: {e}")
        # Essayer une méthode alternative
        try:
            # Méthode alternative : utiliser les arrays numpy
            charges_array = wfn.atomic_point_charges().np
            atomic_charges = charges_array.tolist()
            return atomic_charges
        except:
            print(f"Methode alternative echouee pour {name}, utilisation de charges nulles")
            return [0.0] * len(atoms)  # Retourner des charges nulles en cas d'erreur

def save_coordinates_and_charges(atoms, coords, charges, energy, name, output_dir):
    """
    Sauvegarde les coordonnées et charges dans un fichier .txt
    """
    try:
        txt_filename = os.path.join(output_dir, f"{name}_coordinates_and_charges.txt")
        
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"ANALYSE QUANTIQUE - MOLECULE: {name}\n")
            f.write(f"Energie SCF: {energy:.8f} Hartree\n")
            f.write(f"Energie SCF: {energy * 27.2114:.4f} eV\n")  # Conversion en eV
            f.write("="*60 + "\n")
            f.write("COORDONNEES ATOMIQUES ET CHARGES DE MULLIKEN\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Atome':<6} {'X (A)':<12} {'Y (A)':<12} {'Z (A)':<12} {'Charge':<10}\n")
            f.write("-"*60 + "\n")
            
            for i, (atom, coord, charge) in enumerate(zip(atoms, coords, charges)):
                f.write(f"{atom:<6} {coord[0]:>11.6f} {coord[1]:>11.6f} {coord[2]:>11.6f} {charge:>9.6f}\n")
            
            f.write("-"*60 + "\n")
            f.write(f"Nombre total d'atomes: {len(atoms)}\n")
            f.write(f"Charge totale calculee: {sum(charges):.6f}\n")
            
            # Statistiques sur les charges
            f.write("\n" + "="*60 + "\n")
            f.write("STATISTIQUES DES CHARGES\n")
            f.write("-"*60 + "\n")
            f.write(f"Charge maximale: {max(charges):>8.6f} (atome {atoms[charges.index(max(charges))]})\n")
            f.write(f"Charge minimale: {min(charges):>8.6f} (atome {atoms[charges.index(min(charges))]})\n")
            f.write(f"Charge moyenne: {sum(charges)/len(charges):>10.6f}\n")
        
        print(f"[OK] Fichier de coordonnees sauvegarde: {txt_filename}")
        return txt_filename
        
    except Exception as e:
        print(f"[ERREUR] Erreur lors de la sauvegarde pour {name}: {e}")
        return None

def save_optimized_geometry(atoms, coords, name, fallback_dir):
    """
    Sauvegarde une géométrie optimisée au format .xyz
    Utilisé pour les cas de fallback (optimisation B3LYP/6-31G)
    """
    try:
        # Créer le répertoire de sauvegarde s'il n'existe pas
        os.makedirs(fallback_dir, exist_ok=True)
        
        # Nom du fichier de sortie
        xyz_filename = os.path.join(fallback_dir, f"{name}_hplc_hplc_mobile_optimized.xyz")
        
        with open(xyz_filename, 'w', encoding='utf-8') as f:
            # Header .xyz standard
            f.write(f"{len(atoms)}\n")
            f.write(f"Geometrie optimisee B3LYP/6-31G pour {name} (conditions HPLC)\n")
            
            # Coordonnées atomiques
            for atom, coord in zip(atoms, coords):
                f.write(f"{atom:<3} {coord[0]:>12.8f} {coord[1]:>12.8f} {coord[2]:>12.8f}\n")
        
        print(f"[SAUVEGARDE] Géométrie optimisée sauvée: {xyz_filename}")
        return xyz_filename
        
    except Exception as e:
        print(f"[ERREUR] Échec sauvegarde géométrie pour {name}: {e}")
        return None

def check_results_exist(molecule_name, output_dir):
    """
    Vérifie si les résultats de calcul existent déjà pour une molécule.
    
    Cherche dans output_dir/molecule_name/ les fichiers essentiels :
      - {molecule_name}_electron_density.cube
      - {molecule_name}_coordinates_and_charges.txt
    
    Returns:
        bool: True si les résultats existent déjà, False sinon
    """
    mol_dir = os.path.join(output_dir, molecule_name)
    if not os.path.isdir(mol_dir):
        return False

    required_files = [
        f"{molecule_name}_electron_density.cube",
        f"{molecule_name}_coordinates_and_charges.txt",
    ]

    for fname in required_files:
        fpath = os.path.join(mol_dir, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            return False

    return True


def scan_xyz_files_in_subdirectories(base_directory, target_subdirs=None):
    """
    Scanne récursivement les fichiers XYZ dans les sous-dossiers spécifiés
    
    Args:
        base_directory: Répertoire de base à scanner
        target_subdirs: Liste des sous-dossiers à scanner (ex: ['CCAm', 'CDCAm'])
    
    Returns:
        List[Dict]: Liste des fichiers trouvés avec leurs informations
    """
    if target_subdirs is None:
        target_subdirs = ['CCAm', 'CDCAm']
    
    found_files = []
    
    print(f"🔍 SCAN DES FICHIERS XYZ")
    print(f"Répertoire base: {base_directory}")
    print(f"Sous-dossiers cibles: {target_subdirs}")
    print("="*60)
    
    if not os.path.exists(base_directory):
        print(f"❌ Erreur: Répertoire base introuvable: {base_directory}")
        return found_files
    
    for subdir in target_subdirs:
        subdir_path = os.path.join(base_directory, subdir)
        
        if not os.path.exists(subdir_path):
            print(f"⚠️ Sous-dossier introuvable: {subdir_path}")
            continue
        
        print(f"\n📁 Scan du dossier: {subdir}")
        files_in_subdir = 0
        
        # Scanner tous les fichiers XYZ dans le sous-dossier
        for file in os.listdir(subdir_path):
            if file.endswith('.xyz') and '_optimized.xyz' in file and 'backup' not in file:
                full_path = os.path.join(subdir_path, file)
                
                # Extraire le nom de la molécule du nom de fichier
                # Format attendu: MoleculeName_hplc_mobile_optimized.xyz
                base_name = file.replace('_hplc_mobile_optimized.xyz', '')
                base_name = base_name.replace('_optimized.xyz', '')
                
                file_info = {
                    'molecule_name': base_name,
                    'file_path': full_path,
                    'subdirectory': subdir,
                    'file_name': file
                }
                
                found_files.append(file_info)
                files_in_subdir += 1
                print(f"   ✅ {base_name} → {file}")
        
        print(f"   📊 Total {subdir}: {files_in_subdir} fichiers")
    
    print(f"\n📈 RÉSUMÉ SCAN:")
    print(f"   📁 Sous-dossiers scannés: {len(target_subdirs)}")
    print(f"   📄 Fichiers XYZ trouvés: {len(found_files)}")
    
    return found_files

def extract_smiles_from_json(xyz_file_path):
    """
    Extrait le SMILES depuis le fichier JSON associé au fichier XYZ
    
    Args:
        xyz_file_path: Chemin vers le fichier XYZ
        
    Returns:
        str: SMILES de la molécule ou None si non trouvé
    """
    # Remplacer l'extension .xyz par .json
    json_file_path = xyz_file_path.replace('_optimized.xyz', '_data.json')
    
    if os.path.exists(json_file_path):
        try:
            import json
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('smiles', None)
        except Exception as e:
            print(f"⚠️ Erreur lecture JSON {json_file_path}: {e}")
    
    return None

def main():
    """Fonction principale"""
    
    # Répertoire contenant les fichiers .xyz avec sous-dossiers
    xyz_directory = r"C:\Users\ugo.pasco\Documents\2A\Data finale MD\molecules_data\lambda max"
    
    # Vérifier que le répertoire existe
    if not os.path.exists(xyz_directory):
        print(f"ERREUR: Repertoire non trouvé: {xyz_directory}")
        return
    
    # Scanner automatiquement les fichiers XYZ dans les sous-dossiers
    print("DEMARRAGE ANALYSE DENSITE ELECTRONIQUE")
    print("="*70)
    
    found_files = scan_xyz_files_in_subdirectories(
        xyz_directory, 
        target_subdirs=['Cam', 'CCA', 'CCAm', 'CCE', 'CDCAm']
    )
    
    if not found_files:
        print("Aucun fichier XYZ trouvé dans les sous-dossiers")
        return
    
    molecule_names = [f['molecule_name'] for f in found_files]
    
    print(f"\nMOLECULES A TRAITER: {len(molecule_names)}")
    
    # Créer les répertoires de sortie
    output_dir = "electron_density_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer un dictionnaire de recherche rapide par nom de molécule
    file_lookup = {file_info['molecule_name']: file_info for file_info in found_files}
    
    # Configuration Psi4
    setup_psi4()
    
    # Fichier de résumé
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    
    successful_calculations = 0
    failed_calculations = 0
    skipped_calculations = 0
    
    with open(summary_file, 'w', encoding='utf-8') as summary:
        summary.write("ANALYSE DE DENSITE ELECTRONIQUE - RESUME\n")
        summary.write(f"Source: {xyz_directory}\n")
        summary.write("="*60 + "\n\n")
        
        for name in molecule_names:
            print(f"\nTRAITEMENT: {name}")
            print("-" * 50)
            
            # Vérifier si les résultats existent déjà
            if check_results_exist(name, output_dir):
                print(f"Résultats déjà existants pour {name}, calcul ignoré.")
                skipped_calculations += 1
                summary.write(f"{name}: IGNORE - Résultats déjà présents\n")
                continue
            
            try:
                # Rechercher le fichier XYZ
                xyz_file = None
                if name in file_lookup:
                    xyz_file = file_lookup[name]['file_path']
                    subdir = file_lookup[name]['subdirectory']
                    print(f"Source: {subdir}/{file_lookup[name]['file_name']}")
                
                if xyz_file is None or not os.path.exists(xyz_file):
                    print(f"Fichier .xyz non trouvé pour {name}")
                    failed_calculations += 1
                    summary.write(f"{name}: ECHEC - Fichier .xyz non trouvé\n")
                    continue
                
                print(f"Utilisation du fichier: {os.path.basename(xyz_file)}")
                
                # Lire la géométrie depuis le fichier .xyz
                geometry, charge, multiplicity, atoms, coords = read_xyz_geometry(xyz_file, name)
                
                if geometry is None:
                    print(f"Echec de la lecture du fichier .xyz pour {name}")
                    failed_calculations += 1
                    summary.write(f"{name}: ECHEC - Lecture fichier .xyz\n")
                    continue
                
                # Calcul de la densité électronique avec gestion robuste
                wfn, energy = calculate_electron_density_robust(geometry, charge, multiplicity, name)
                
                if wfn is None:
                    print(f"Echec du calcul SCF pour {name}")
                    failed_calculations += 1
                    summary.write(f"{name}: ECHEC - Calcul SCF\n")
                    continue
                
                # Créer le dossier de la molécule
                mol_output_dir = os.path.join(output_dir, name)
                os.makedirs(mol_output_dir, exist_ok=True)
                
                # Génération des fichiers cube
                print(f"\n=== Generation des fichiers cube pour {name} ===")
                print(f"    Dossier: {mol_output_dir}")
                
                # 1. Fichier cube de densité électronique totale
                density_cube_file = generate_electron_density_cube(wfn, name, mol_output_dir, atoms, coords)
                
                # 2. Organiser les fichiers d'orbitales
                orbitals_dir = organize_orbital_cubes(name, mol_output_dir)
                
                # Calcul des charges atomiques
                atomic_charges = calculate_atomic_charges(wfn, atoms, name)
                
                # Sauvegarde des coordonnées et charges
                coord_file = save_coordinates_and_charges(atoms, coords, atomic_charges, energy, name, mol_output_dir)
                
                successful_calculations += 1
                summary.write(f"{name}: SUCCES - Energie: {energy:.8f} Hartree - Fichier: {os.path.basename(xyz_file)}\n")
                summary.write(f"    - Densite electronique: {'OK' if density_cube_file else 'ECHEC'}\n")
                summary.write(f"    - Orbitales moleculaires: {'OK' if orbitals_dir else 'ECHEC'}\n")
                summary.write(f"    - Coordonnees + charges Mulliken: {'OK' if coord_file else 'ECHEC'}\n")
                
                print(f"[OK] {name} traite avec succes")
                
            except Exception as e:
                print(f"Erreur generale pour {name}: {e}")
                failed_calculations += 1
                summary.write(f"{name}: ECHEC - {str(e)}\n")
        
        summary.write("\n" + "="*60 + "\n")
        summary.write(f"STATISTIQUES:\n")
        summary.write(f"Calculs reussis: {successful_calculations}\n")
        summary.write(f"Calculs ignores (deja existants): {skipped_calculations}\n")
        summary.write(f"Calculs echoues: {failed_calculations}\n")
        summary.write(f"Total: {len(molecule_names)}\n")
        summary.write(f"Source: {xyz_directory}\n")
    
    print(f"\n{'='*60}")
    print(f"ANALYSE TERMINEE")
    print(f"{'='*60}")
    print(f"Calculs reussis: {successful_calculations}")
    print(f"Calculs ignores (deja existants): {skipped_calculations}")
    print(f"Calculs echoues: {failed_calculations}")
    print(f"Resultats sauvegardes dans: {output_dir}")
    print(f"Resume disponible dans: {summary_file}")

if __name__ == "__main__":
    main()