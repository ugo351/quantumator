"""Run xTB/CREST geometry optimization and analyze molecular dihedrals."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def calculate_dihedral(coords: Sequence[Sequence[float]], atom_indices: Sequence[int]) -> float:
    """Return a signed dihedral in degrees; atom indices are zero-based."""
    if len(atom_indices) != 4:
        raise ValueError("A dihedral requires exactly four atom indices.")
    if any(not isinstance(index, int) or index < 0 for index in atom_indices):
        raise ValueError("Dihedral atom indices must be non-negative integers.")
    if len(set(atom_indices)) != 4:
        raise ValueError("A dihedral must reference four distinct atoms.")
    try:
        p0, p1, p2, p3 = ([float(value) for value in coords[index]]
                          for index in atom_indices)
    except (IndexError, TypeError) as exc:
        raise ValueError("Dihedral atom index is outside the coordinate list.") from exc
    if any(len(point) != 3 for point in (p0, p1, p2, p3)):
        raise ValueError("Each coordinate must contain x, y, and z.")

    def subtract(left, right):
        return [a - b for a, b in zip(left, right)]

    def dot(left, right):
        return sum(a * b for a, b in zip(left, right))

    def cross(left, right):
        return [left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0]]

    b0 = subtract(p0, p1)
    b1 = subtract(p2, p1)
    b2 = subtract(p3, p2)
    norm = math.sqrt(dot(b1, b1))
    if norm <= 1e-12:
        raise ValueError("Dihedral is undefined for coincident central atoms.")
    b1 = [value / norm for value in b1]
    v = [a - dot(b0, b1) * b for a, b in zip(b0, b1)]
    w = [a - dot(b2, b1) * b for a, b in zip(b2, b1)]
    if math.sqrt(dot(v, v)) <= 1e-12 or math.sqrt(dot(w, w)) <= 1e-12:
        raise ValueError("Dihedral is undefined for collinear atoms.")
    return math.degrees(math.atan2(dot(cross(b1, v), w), dot(v, w)))


def analyze_planarity(
    coords: Sequence[Sequence[float]],
    atom_indices: Sequence[int],
    threshold_deg: float = 5.0,
) -> dict[str, Any]:
    angle = calculate_dihedral(coords, atom_indices)
    absolute_angle = abs(angle)
    deviation = min(absolute_angle, abs(180.0 - absolute_angle))
    return {
        "dihedral_deg": angle,
        "absolute_dihedral_deg": absolute_angle,
        "planarity_deviation_deg": deviation,
        "planar": deviation <= threshold_deg,
        "threshold_deg": float(threshold_deg),
    }


def analyze_dihedrals(
    coords: Sequence[Sequence[float]],
    dihedrals: Iterable[Sequence[int]],
    threshold_deg: float = 5.0,
) -> dict[str, Any]:
    results = {}
    for index, atom_indices in enumerate(dihedrals, 1):
        results[f"dihedral_{index}"] = {
            "atom_indices": list(atom_indices),
            **analyze_planarity(coords, atom_indices, threshold_deg),
        }
    results["all_planar"] = all(item["planar"] for item in results.values())
    return results


def read_xyz_frames(path: str | Path) -> list[str]:
    """Read a single- or multi-frame XYZ file and return complete XYZ frames."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom count at line {cursor + 1}.") from exc
        end = cursor + atom_count + 2
        if atom_count < 1 or end > len(lines):
            raise ValueError("Incomplete XYZ frame.")
        frame_lines = lines[cursor:end]
        for atom_line in frame_lines[2:]:
            fields = atom_line.split()
            if len(fields) < 4:
                raise ValueError("Invalid atom line in XYZ frame.")
            [float(value) for value in fields[1:4]]
        frames.append("\n".join(frame_lines) + "\n")
        cursor = end
    if not frames:
        raise ValueError("XYZ file contains no molecular frames.")
    return frames


class XTBOptimizer:
    OPT_LEVELS = {"normal", "tight", "verytight"}

    def __init__(
        self,
        xtb_exe: str = "xtb",
        crest_exe: str = "crest",
        crest_use_wsl: bool = False,
        crest_wsl_exe: str = "crest",
        crest_wsl_xtb_exe: str = "/home/ugopasco/miniforge3/envs/crest_xtb/bin/xtb",
        wsl_exe: str = "wsl.exe",
    ):
        self.xtb_exe = xtb_exe
        self.crest_exe = crest_exe
        self.crest_use_wsl = crest_use_wsl
        self.crest_wsl_exe = crest_wsl_exe
        self.crest_wsl_xtb_exe = crest_wsl_xtb_exe
        self.wsl_exe = wsl_exe

    @staticmethod
    def _create_wsl_crest_dir(wsl_executable: str) -> str:
        completed = subprocess.run(
            [
                wsl_executable, "--cd", "/home/ugopasco", "--exec", "mktemp",
                "-d", "-p", "/home/ugopasco", "quantumator_crest_XXXXXX",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, shell=False,
        )
        linux_dir = completed.stdout.strip()
        if completed.returncode != 0 or not linux_dir.startswith(
            "/home/ugopasco/quantumator_crest_"
        ):
            detail = completed.stderr.strip() or "mktemp did not return a valid directory."
            raise RuntimeError(f"Cannot create a Linux CREST work directory: {detail}")
        return linux_dir

    @staticmethod
    def _write_wsl_file(wsl_executable: str, linux_dir: str, name: str, data: bytes,
                        timeout: Optional[float] = None) -> None:
        completed = subprocess.run(
            [wsl_executable, "--cd", linux_dir, "--exec", "tee", name],
            input=data, capture_output=True, check=False, shell=False,
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Cannot stage {name} in WSL: {detail}")

    @staticmethod
    def _read_wsl_file(wsl_executable: str, linux_dir: str, name: str,
                       timeout: Optional[float] = None) -> bytes:
        completed = subprocess.run(
            [wsl_executable, "--cd", linux_dir, "--exec", "cat", name],
            capture_output=True, check=False, shell=False, timeout=timeout,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Cannot retrieve {name} from WSL: {detail}")
        return completed.stdout

    @staticmethod
    def _list_wsl_crest_rotamer_files(
        wsl_executable: str, linux_dir: str,
        timeout: Optional[float] = None,
    ) -> list[str]:
        completed = subprocess.run(
            [
                wsl_executable, "--cd", linux_dir, "--exec", "find", ".",
                "-maxdepth", "1", "-type", "f", "-name",
                "crest_rotamers_*.xyz", "-printf", "%f\\n",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, shell=False, timeout=timeout,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "find failed in the CREST work directory."
            raise RuntimeError(f"Cannot list CREST rotamer files in WSL: {detail}")

        def file_number(name: str) -> int:
            match = re.search(r"crest_rotamers_(\d+)\.xyz$", name)
            return int(match.group(1)) if match else -1

        return sorted(
            (name for name in completed.stdout.splitlines() if name),
            key=file_number,
        )

    def _capture_wsl_crest_rotamers(
        self, wsl_executable: str, linux_dir: str, destination: Path,
        timeout: Optional[float] = None,
    ) -> None:
        try:
            names = self._list_wsl_crest_rotamer_files(
                wsl_executable, linux_dir, timeout,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return
        for name in names:
            try:
                data = self._read_wsl_file(
                    wsl_executable, linux_dir, name, timeout,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError):
                continue
            (destination / name).write_bytes(data)

    @staticmethod
    def _remove_wsl_crest_dir(wsl_executable: str, linux_dir: str) -> None:
        if not linux_dir.startswith("/home/ugopasco/quantumator_crest_"):
            return
        subprocess.run(
            [wsl_executable, "--cd", "/home/ugopasco", "--exec", "rm", "-rf", "--", linux_dir],
            capture_output=True, check=False, shell=False,
        )

    @staticmethod
    def _energy_from_xyz_frame(frame: str) -> Optional[float]:
        lines = frame.splitlines()
        if len(lines) < 2:
            return None
        comment = lines[1]
        number = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"
        match = re.search(rf"\bEtot\s*=\s*{number}", comment, re.I)
        if match is None:
            match = re.match(rf"\s*{number}(?=\s|$)", comment)
        return float(match.group(1)) if match else None

    @staticmethod
    def _write_crest_energy_landscape(path: Path, conformers: list[dict]) -> None:
        kcal_per_hartree = 627.509474
        crest_energies = [
            (entry["conformer_index"], entry["crest_energy_hartree"])
            for entry in conformers if entry["crest_energy_hartree"] is not None
        ]
        xtb_energies = [
            (entry["conformer_index"], entry["xtb_refined_energy_hartree"])
            for entry in conformers if entry["xtb_refined_energy_hartree"] is not None
        ]
        crest_rank = {
            index: rank for rank, (index, _energy) in enumerate(
                sorted(crest_energies, key=lambda item: item[1]), 1
            )
        }
        xtb_rank = {
            index: rank for rank, (index, _energy) in enumerate(
                sorted(xtb_energies, key=lambda item: item[1]), 1
            )
        }
        crest_min = min((energy for _index, energy in crest_energies), default=None)
        xtb_min = min((energy for _index, energy in xtb_energies), default=None)
        fields = [
            "conformer_index", "crest_rank", "crest_energy_hartree",
            "crest_relative_energy_kcal_mol", "xtb_refined_rank",
            "xtb_refined_energy_hartree", "xtb_relative_energy_kcal_mol",
            "refinement_status", "crest_xyz", "refined_xyz", "error",
        ]
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for entry in conformers:
                crest_energy = entry["crest_energy_hartree"]
                xtb_energy = entry["xtb_refined_energy_hartree"]
                writer.writerow({
                    **entry,
                    "crest_rank": crest_rank.get(entry["conformer_index"], ""),
                    "crest_relative_energy_kcal_mol": (
                        (crest_energy - crest_min) * kcal_per_hartree
                        if crest_energy is not None and crest_min is not None else ""
                    ),
                    "xtb_refined_rank": xtb_rank.get(entry["conformer_index"], ""),
                    "xtb_relative_energy_kcal_mol": (
                        (xtb_energy - xtb_min) * kcal_per_hartree
                        if xtb_energy is not None and xtb_min is not None else ""
                    ),
                })

    @staticmethod
    def _parse_crest_energy_scan(stdout: str) -> list[dict]:
        markers = list(re.finditer(
            r"total number unique points considered further\s*:\s*\d+",
            stdout, re.I,
        ))
        if not markers:
            return []
        number = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"
        row_pattern = re.compile(
            rf"^\s*(\d+)\s+{number}\s+{number}\s+{number}(?:\s+{number})?"
            r"(?:\s+(\d+)\s+(\d+))?\s*$"
        )
        records = []
        conformer_index = None
        degeneracy = None
        for line in stdout[markers[-1].end():].splitlines():
            match = row_pattern.match(line)
            if match:
                if match.group(6) is not None:
                    conformer_index = int(match.group(6))
                    degeneracy = int(match.group(7))
                if conformer_index is None:
                    continue
                records.append({
                    "state_index": int(match.group(1)),
                    "relative_energy_kcal_mol": float(match.group(2)),
                    "crest_energy_hartree": float(match.group(3)),
                    "boltzmann_weight": float(match.group(4)),
                    "conformer_population": float(match.group(5)) if match.group(6) else "",
                    "conformer_index": conformer_index,
                    "degeneracy": degeneracy,
                    "xtb_refined_energy_hartree": "",
                    "xtb_relative_energy_kcal_mol": "",
                })
            elif records and line.strip():
                break
        return records

    @staticmethod
    def _write_crest_scan_csv(path: Path, records: list[dict]) -> None:
        fields = [
            "state_index", "relative_energy_kcal_mol", "crest_energy_hartree",
            "boltzmann_weight", "conformer_population", "conformer_index",
            "degeneracy", "xtb_refined_energy_hartree",
            "xtb_relative_energy_kcal_mol",
        ]
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _parse_crest_search_states(frames: list[str]) -> list[dict]:
        energies = [XTBOptimizer._energy_from_xyz_frame(frame) for frame in frames]
        valid_energies = [energy for energy in energies if energy is not None]
        if not valid_energies:
            return []
        lowest_energy = min(valid_energies)
        return [
            {
                "state_index": index,
                "crest_energy_hartree": energy if energy is not None else "",
                "relative_energy_kcal_mol": (
                    (energy - lowest_energy) * 627.509474
                    if energy is not None else ""
                ),
            }
            for index, energy in enumerate(energies, 1)
        ]

    @staticmethod
    def _crest_plot_inputs(
        search_frames: list[str], conformer_frames: list[str],
        energy_scan_records: Optional[list[dict]] = None,
    ) -> tuple[list[str], list[dict], str]:
        search_records = XTBOptimizer._parse_crest_search_states(search_frames)
        if search_records:
            return search_frames, search_records, "search_states"
        conformer_by_index = {
            index: frame for index, frame in enumerate(conformer_frames, 1)
        }
        mapped_frames = []
        mapped_records = []
        for record in energy_scan_records or []:
            try:
                conformer_index = int(record["conformer_index"])
                state_index = int(record["state_index"])
                energy = float(record["crest_energy_hartree"])
            except (KeyError, TypeError, ValueError):
                continue
            frame = conformer_by_index.get(conformer_index)
            if frame is None:
                continue
            mapped_frames.append(frame)
            mapped_records.append({
                **record,
                "state_index": state_index,
                "crest_energy_hartree": energy,
            })
        if mapped_records:
            return mapped_frames, mapped_records, "energy_scan"
        conformer_records = XTBOptimizer._parse_crest_search_states(conformer_frames)
        if conformer_records:
            return conformer_frames, conformer_records, "final_conformers"
        return [], [], "unavailable"

    @staticmethod
    def _write_crest_search_scan_csv(path: Path, records: list[dict]) -> None:
        fields = ["state_index", "crest_energy_hartree", "relative_energy_kcal_mol"]
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _conformer_reference_energy(entry: dict) -> Optional[float]:
        for key in ("xtb_refined_energy_hartree", "crest_energy_hartree"):
            energy = entry.get(key)
            if isinstance(energy, (int, float)):
                return float(energy)
        return None

    @staticmethod
    def _crest_energy_distribution_records(
        search_records: list[dict], conformers: list[dict],
    ) -> list[dict]:
        conformer_energies = [
            XTBOptimizer._conformer_reference_energy(entry) for entry in conformers
            if XTBOptimizer._conformer_reference_energy(entry) is not None
        ]
        if not conformer_energies:
            return []
        reference_energy = min(conformer_energies)
        return [
            {
                "candidate_index": entry["state_index"],
                "crest_energy_hartree": energy,
                "relative_to_min_optimized_xtb_kcal_mol": (
                    (energy - reference_energy) * 627.509474
                ),
            }
            for entry in search_records
            if isinstance((energy := entry.get("crest_energy_hartree")), (int, float))
        ]

    @staticmethod
    def _write_crest_energy_distribution_csv(path: Path, records: list[dict]) -> None:
        fields = [
            "candidate_index", "crest_energy_hartree",
            "relative_to_min_optimized_xtb_kcal_mol",
        ]
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _crest_conformer_kbt_pairs(
        conformers: list[dict], temperature_k: float = 298.15,
    ) -> tuple[float, list[dict]]:
        energies = [
            (entry["conformer_index"], XTBOptimizer._conformer_reference_energy(entry))
            for entry in conformers
            if XTBOptimizer._conformer_reference_energy(entry) is not None
        ]
        kbt_kcal_mol = 0.00198720425864083 * temperature_k
        pairs = []
        for left_index, (left_id, left_energy) in enumerate(energies):
            for right_id, right_energy in energies[left_index + 1:]:
                difference = abs(left_energy - right_energy) * 627.509474
                pairs.append({
                    "conformer_indices": (left_id, right_id),
                    "difference_kcal_mol": difference,
                    "within_kbt": difference < kbt_kcal_mol,
                })
        return kbt_kcal_mol, pairs

    @staticmethod
    def _plot_crest_energy_distribution(
        path: Path, records: list[dict], conformers: list[dict],
    ) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            return False

        candidates = [
            entry["relative_to_min_optimized_xtb_kcal_mol"] for entry in records
        ]
        if not candidates:
            return False
        conformer_energies = [
            XTBOptimizer._conformer_reference_energy(entry) for entry in conformers
            if XTBOptimizer._conformer_reference_energy(entry) is not None
        ]
        if not conformer_energies:
            return False
        reference_energy = min(conformer_energies)
        minima = [
            (
                entry["conformer_index"],
                (energy - reference_energy) * 627.509474,
            )
            for entry in conformers
            if (energy := XTBOptimizer._conformer_reference_energy(entry)) is not None
        ]
        edges_min = min(candidates)
        edges_max = max(candidates)
        if edges_min == edges_max:
            edges_min -= 0.5
            edges_max += 0.5
        bin_edges = [edges_min + (edges_max - edges_min) * i / 30 for i in range(31)]
        bin_width = bin_edges[1] - bin_edges[0]
        figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        counts, _, _ = axis.hist(
            candidates, bins=bin_edges, color="#287C8E", alpha=0.42,
            edgecolor="white", linewidth=0.65,
            label=f"Candidats (tranches de {bin_width:.2f} kcal/mol)",
        )
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            gaussian_kde = None
        if gaussian_kde is not None and len(set(candidates)) > 1:
            kde = gaussian_kde(candidates, bw_method="silverman")
            grid_min = min(-0.5, edges_min)
            grid_max = max(edges_max, *(energy for _index, energy in minima))
            grid = [grid_min + (grid_max - grid_min) * i / 699 for i in range(700)]
            axis.plot(
                grid, [value * len(candidates) * bin_width for value in kde(grid)],
                color="#D17A22", linewidth=2.7, label="Tendance KDE",
            )

        optimized_color = "#B23A48"
        marker_y = max(counts, default=0) + 7
        for index, energy in minima:
            axis.vlines(
                energy, marker_y - 4, marker_y + 4,
                color=optimized_color, linewidth=2.2,
            )
            axis.annotate(
                f"C{index}", (energy, marker_y + 4), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=9, color=optimized_color,
            )
        handles, _labels = axis.get_legend_handles_labels()
        handles.append(Line2D(
            [0], [0], color=optimized_color, marker="|", linestyle="None",
            markersize=12, markeredgewidth=2, label="Structures optimisées xTB",
        ))
        kbt_kcal_mol, pairs = XTBOptimizer._crest_conformer_kbt_pairs(conformers)
        axis.axvline(
            kbt_kcal_mol, color="#3A7D44", linestyle="-.", linewidth=1.8,
        )
        handles.append(Line2D(
            [0], [0], color="#3A7D44", linestyle="-.", linewidth=1.8,
            label=f"kBT ≈ RT = {kbt_kcal_mol:.3f} kcal/mol (298 K)",
        ))
        within_kbt = [pair for pair in pairs if pair["within_kbt"]]
        if len(within_kbt) <= 4:
            within_text = ", ".join(
                f"C{left}-C{right} ({pair['difference_kcal_mol']:.3f})"
                for pair in within_kbt
                for left, right in [pair["conformer_indices"]]
            ) or "aucune"
        else:
            within_text = f"{len(within_kbt)} paires"
        comparison_text = (
            f"298 K, kBT = {kbt_kcal_mol:.3f} kcal/mol\n"
            f"ΔE < kBT : {within_text}\n"
            f"ΔE ≥ kBT : {len(pairs) - len(within_kbt)} autres paires"
        )
        axis.legend(handles=handles, frameon=False, loc="upper left")
        axis.text(
            0.98, 0.65, comparison_text, transform=axis.transAxes,
            ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#cccccc", "alpha": 0.92},
        )
        axis.set_title("Candidats CREST et structures optimisées xTB")
        axis.set_xlabel("Énergie relative (kcal/mol; zéro = minimum xTB optimisé)")
        axis.set_ylabel("Nombre de candidats par tranche")
        axis.set_xlim(min(-0.5, edges_min), max(edges_max, *(e for _i, e in minima)))
        axis.set_ylim(0, marker_y + 28)
        axis.set_yticks(range(0, int(max(counts, default=0) // 20 + 1) * 20 + 1, 20))
        axis.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        figure.savefig(path, dpi=190)
        plt.close(figure)
        return True

    @staticmethod
    def _crest_plot_torsions(topology_smiles: Optional[str]):
        if not topology_smiles:
            return None
        try:
            from rdkit import Chem
        except ImportError:
            return None
        molecule = Chem.MolFromSmiles(topology_smiles)
        pattern = Chem.MolFromSmarts("[O:1]=[C:2]-[C:3]=[C:4]")
        if molecule is None or pattern is None:
            return None

        def conjugated_tail(current, previous, expected_bond, visited):
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index == previous or neighbor_index in visited:
                    continue
                bond = molecule.GetBondBetweenAtoms(current, neighbor_index)
                if neighbor.GetIsAromatic():
                    if (expected_bond == Chem.BondType.SINGLE
                            and bond.GetBondType() == Chem.BondType.SINGLE):
                        return [neighbor_index]
                    continue
                if (neighbor.GetSymbol() != "C"
                        or bond.GetBondType() != expected_bond):
                    continue
                next_bond = (
                    Chem.BondType.DOUBLE
                    if expected_bond == Chem.BondType.SINGLE
                    else Chem.BondType.SINGLE
                )
                tail = conjugated_tail(
                    neighbor_index, current, next_bond,
                    visited | {neighbor_index},
                )
                if tail:
                    return [neighbor_index] + tail
            return None

        for match in molecule.GetSubstructMatches(pattern):
            oxygen, carbonyl, alpha, beta = match
            tail = conjugated_tail(
                beta, alpha, Chem.BondType.SINGLE, {alpha, beta},
            )
            if not tail:
                continue
            aryl = tail[-1]
            conjugated_chain = [alpha, beta] + tail[:-1]
            aryl_neighbors = sorted(
                atom.GetIdx()
                for atom in molecule.GetAtomWithIdx(aryl).GetNeighbors()
                if atom.GetIsAromatic()
            )
            if not aryl_neighbors:
                continue

            tau2 = None
            for index in range(1, len(conjugated_chain) - 2):
                before, left, right, after = conjugated_chain[index - 1:index + 3]
                if (
                    molecule.GetBondBetweenAtoms(before, left).GetBondType()
                    == Chem.BondType.DOUBLE
                    and molecule.GetBondBetweenAtoms(left, right).GetBondType()
                    == Chem.BondType.SINGLE
                    and molecule.GetBondBetweenAtoms(right, after).GetBondType()
                    == Chem.BondType.DOUBLE
                ):
                    tau2 = (before, left, right, after)
                    tau2_label = (
                        "τ₂ = torsion entre doubles conjuguées (°) : s-cis / s-trans"
                    )
                    break
            if tau2 is None:
                tau2 = (aryl, beta, alpha, carbonyl)
                tau2_label = "τ₂ = Ar–C=C–C(=O) (°) : planéité"

            return {
                "tau1": (oxygen, carbonyl, alpha, beta),
                "tau2": tau2,
                "tau2_label": tau2_label,
                "aryl_alkene": (
                    conjugated_chain[-2], conjugated_chain[-1],
                    aryl, aryl_neighbors[0],
                ),
            }
        return None

    @staticmethod
    def _plot_crest_interactive(
        landscape_3d_path: Path, absolute_dihedral_path: Path,
        search_frames: list[str], records: list[dict], conformers: list[dict],
        topology_smiles: Optional[str], candidate_source: str = "search_states",
    ) -> tuple[bool, bool]:
        try:
            import plotly.graph_objects as go
            from rdkit import Chem
        except ImportError:
            return False, False

        torsions = XTBOptimizer._crest_plot_torsions(topology_smiles)
        if not torsions or len(search_frames) != len(records):
            return False, False

        optimized = [
            entry for entry in conformers
            if isinstance(entry.get("xtb_refined_energy_hartree"), (int, float))
            and entry.get("refined_xyz")
        ]
        if not optimized:
            return False, False
        reference_energy = min(
            entry["xtb_refined_energy_hartree"] for entry in optimized
        )

        def frame_coordinates(frame: str) -> list[list[float]]:
            return [
                [float(value) for value in line.split()[1:4]]
                for line in frame.splitlines()[2:] if line.split()
            ]

        def molecule_coordinates(molecule) -> list[list[float]]:
            conformer = molecule.GetConformer()
            return [
                list(conformer.GetAtomPosition(index))
                for index in range(molecule.GetNumAtoms())
            ]

        def point_data(coords, energy, state_index):
            tau1 = calculate_dihedral(coords, torsions["tau1"])
            tau2 = calculate_dihedral(coords, torsions["tau2"])
            aryl_angle = abs(calculate_dihedral(coords, torsions["aryl_alkene"]))
            return {
                "index": state_index,
                "tau1": tau1,
                "tau2": tau2,
                "aryl_angle": min(180.0, aryl_angle),
                "energy": (energy - reference_energy) * 627.509474,
            }

        candidates = []
        candidate_mol_blocks = {}
        for frame, record in zip(search_frames, records):
            energy = record.get("crest_energy_hartree")
            if not isinstance(energy, (int, float)):
                continue
            candidate_molecule = XTBOptimizer._molecule_from_xyz_frame(
                frame, topology_smiles=topology_smiles,
            )
            candidates.append(point_data(
                molecule_coordinates(candidate_molecule), energy,
                record["state_index"],
            ))
            candidate_mol_blocks[str(record["state_index"])] = Chem.MolToMolBlock(
                candidate_molecule,
                forceV3000=candidate_molecule.GetNumAtoms() > 999,
            )
        refined = []
        for entry in optimized:
            xyz_path = Path(entry["refined_xyz"])
            if not xyz_path.is_file():
                continue
            frame = read_xyz_frames(xyz_path)[0]
            refined_molecule = XTBOptimizer._molecule_from_xyz_frame(
                frame, topology_smiles=topology_smiles,
            )
            refined.append(point_data(
                molecule_coordinates(refined_molecule),
                entry["xtb_refined_energy_hartree"],
                f"C{entry['conformer_index']} xTB",
            ))
        if not candidates or not refined:
            return False, False

        candidate_label = (
            "Conformères finaux CREST"
            if candidate_source == "final_conformers" else
            "États CREST (géométrie du conformère associé)"
            if candidate_source == "energy_scan" else "Candidats CREST"
        )

        molecule = Chem.MolFromSmiles(topology_smiles)
        if molecule is None:
            return False, False
        molecule = Chem.AddHs(molecule)
        atom_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        bond_pairs = [
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
             4 if bond.GetIsAromatic() else int(bond.GetBondTypeAsDouble())]
            for bond in molecule.GetBonds()
        ]
        if len(frame_coordinates(search_frames[0])) != len(atom_elements):
            return False, False
        plotly_mol_blocks = json.dumps(
            candidate_mol_blocks, ensure_ascii=True,
        ).replace("</", "<\\/")
        file_prefix = landscape_3d_path.stem.replace("_crest_landscape_3d", "")
        post_script = (
            "var plot = document.getElementById('{plot_id}');\n"
            f"var molBlocks = {plotly_mol_blocks};\n"
            f"var filePrefix = {json.dumps(file_prefix)};\n"
            "function downloadCandidate(candidateId) {\n"
            "  var molBlock = molBlocks[candidateId];\n"
            "  if (!molBlock) return;\n"
            "  var url = URL.createObjectURL(new Blob([molBlock], {type: 'chemical/x-mdl-molfile'}));\n"
            "  var link = document.createElement('a');\n"
            "  link.href = url;\n"
            "  link.download = filePrefix + '_crest_candidate_' + String(candidateId).padStart(3, '0') + '.mol';\n"
            "  document.body.appendChild(link); link.click(); link.remove();\n"
            "  setTimeout(function() { URL.revokeObjectURL(url); }, 1000);\n"
            "}\n"
            "plot.on('plotly_click', function(event) {\n"
            "  var selected = event.points.find(function(point) { return point.curveNumber === 0; });\n"
            "  if (selected) downloadCandidate(String(selected.customdata[0]));\n"
            "});"
        )
        landscape_figure = go.Figure()
        landscape_figure.add_trace(go.Scatter3d(
            x=[point["tau1"] for point in candidates],
            y=[point["tau2"] for point in candidates],
            z=[point["energy"] for point in candidates],
            mode="markers",
            marker={
                "size": 4, "opacity": 0.68,
                "color": [point["energy"] for point in candidates],
                "colorscale": "Viridis",
                "colorbar": {"title": "ΔE<br>(kcal/mol)"},
            },
            customdata=[
                [point["index"], point["aryl_angle"]]
                for point in candidates
            ],
            hovertemplate=(
                "Candidat %{customdata[0]}<br>"
                "τ₁ = %{x:.1f}°<br>τ₂ = %{y:.1f}°<br>"
                "ΔE = %{z:.3f} kcal/mol<br>"
                "|dièdre aryle–alcène| = %{customdata[1]:.1f}°<extra></extra>"
            ),
            name=f"{candidate_label} (n={len(candidates)})",
        ))
        landscape_figure.add_trace(go.Scatter3d(
            x=[point["tau1"] for point in refined],
            y=[point["tau2"] for point in refined],
            z=[point["energy"] for point in refined],
            mode="markers+text",
            text=[point["index"] for point in refined],
            textposition="top center",
            marker={"size": 8, "symbol": "diamond", "color": "#C83E4D",
                    "line": {"color": "white", "width": 1}},
            name="Minima xTB raffinés",
            hovertemplate=(
                "%{text}<br>τ₁ = %{x:.1f}°<br>τ₂ = %{y:.1f}°<br>"
                "ΔE = %{z:.3f} kcal/mol<extra></extra>"
            ),
        ))
        landscape_figure.update_layout(
            title="Paysage CREST selon τ₁, τ₂ et ΔE",
            scene={
                "xaxis_title": "τ₁ = O=C–C=C (°) : s-cis / s-trans",
                "yaxis_title": torsions["tau2_label"],
                "zaxis_title": "ΔE (kcal/mol; référence = minimum xTB)",
                "xaxis": {"range": [-180, 180]},
                "yaxis": {"range": [-180, 180]},
                "aspectmode": "auto",
            },
            annotations=[{
                "text": "Cliquer un candidat pour télécharger son fichier .mol",
                "x": 0.5, "y": 1.02, "xref": "paper", "yref": "paper",
                "showarrow": False,
            }],
            legend={"orientation": "h", "y": 1.04, "x": 0},
            height=700, margin={"l": 0, "r": 0, "t": 80, "b": 0},
            clickmode="event",
        )
        landscape_figure.write_html(
            landscape_3d_path, include_plotlyjs="directory", full_html=True,
            auto_open=False, post_script=post_script,
        )

        angle_figure = go.Figure()
        sparse_bins = set(range(18))
        bins: dict[int, list[float]] = {}
        for point in candidates:
            bin_index = min(17, int(point["aryl_angle"] // 10))
            bins.setdefault(bin_index, []).append(point["energy"])
        medians = {
            index: statistics.median(energies)
            for index, energies in bins.items() if len(energies) >= 10
        }
        sparse_bins.difference_update(medians)
        if sparse_bins:
            start = previous = min(sparse_bins)
            for index in sorted(sparse_bins)[1:] + [None]:
                if index is not None and index == previous + 1:
                    previous = index
                    continue
                angle_figure.add_vrect(
                    x0=start * 10, x1=(previous + 1) * 10,
                    fillcolor="#E7E7E7", opacity=0.5, line_width=0,
                    layer="below", showlegend=False,
                )
                if index is not None:
                    start = previous = index
        angle_figure.add_trace(go.Scatter(
            x=[point["aryl_angle"] for point in candidates],
            y=[point["energy"] for point in candidates],
            mode="markers",
            marker={
                "size": 7, "opacity": 0.58,
                "color": [point["tau1"] for point in candidates],
                "colorscale": "RdBu", "cmin": -180, "cmax": 180,
                "colorbar": {"title": "τ₁ (°)"},
            },
            customdata=[
                [point["index"], point["tau1"], point["tau2"]]
                for point in candidates
            ],
            hovertemplate=(
                "Candidat %{customdata[0]}<br>"
                "|dièdre aryle–alcène| = %{x:.1f}°<br>"
                "ΔE = %{y:.3f} kcal/mol<br>"
                "τ₁ = %{customdata[1]:.1f}°<br>"
                "τ₂ = %{customdata[2]:.1f}°<extra></extra>"
            ),
            name=f"{candidate_label} (n={len(candidates)})",
        ))
        for index in sorted(medians):
            if index - 1 in medians:
                angle_figure.add_trace(go.Scatter(
                    x=[(index - 1) * 10 + 5, index * 10 + 5],
                    y=[medians[index - 1], medians[index]],
                    mode="lines", line={"color": "#D8791D", "width": 3},
                    showlegend=False, hoverinfo="skip",
                ))
        angle_figure.add_trace(go.Scatter(
            x=[index * 10 + 5 for index in sorted(medians)],
            y=[medians[index] for index in sorted(medians)],
            mode="markers", marker={"size": 10, "color": "#D8791D"},
            name="Médiane par tranche de 10° (n ≥ 10)",
            hovertemplate="|dièdre| = %{x:.0f}°<br>Médiane ΔE = %{y:.3f}<extra></extra>",
        ))
        angle_figure.add_trace(go.Scatter(
            x=[point["aryl_angle"] for point in refined],
            y=[point["energy"] for point in refined],
            mode="markers+text",
            text=[point["index"] for point in refined],
            textposition="top center",
            marker={"size": 12, "symbol": "diamond", "color": "#C83E4D",
                    "line": {"color": "white", "width": 1}},
            name="Minima xTB raffinés",
            hovertemplate=(
                "%{text}<br>|dièdre| = %{x:.1f}°<br>"
                "ΔE = %{y:.3f} kcal/mol<extra></extra>"
            ),
        ))
        angle_figure.update_layout(
            title="Énergie CREST selon le dièdre aryle–alcène",
            xaxis={"title": "|Dièdre aryle–alcène| (°)", "range": [0, 180],
                   "dtick": 30},
            yaxis={"title": "ΔE (kcal/mol; référence = minimum xTB)"},
            legend={"orientation": "h", "y": 1.04, "x": 0},
            margin={"l": 70, "r": 35, "t": 75, "b": 60},
        )
        angle_figure.write_html(
            absolute_dihedral_path, include_plotlyjs="directory", full_html=True,
            auto_open=False,
        )
        return True, True

    @staticmethod
    def _plot_crest_search_landscape(
        path: Path, records: list[dict], conformers: list[dict],
    ) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            return False

        optimized = [
            entry for entry in conformers
            if isinstance(entry.get("xtb_refined_energy_hartree"), (int, float))
        ]
        if not optimized:
            return False
        reference_energy = min(
            entry["xtb_refined_energy_hartree"] for entry in optimized
        )
        points = [
            (
                record["state_index"],
                (record["crest_energy_hartree"] - reference_energy) * 627.509474,
            )
            for record in records
            if isinstance(record.get("crest_energy_hartree"), (int, float))
        ]
        if not points:
            return False
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        axis.scatter(
            [point[0] for point in points], [point[1] for point in points],
            s=17, color="#287C8E", alpha=0.65, edgecolors="none",
            label=f"Candidats CREST (n={len(points)})",
        )
        optimized_levels = [
            (
                entry["conformer_index"],
                (entry["xtb_refined_energy_hartree"] - reference_energy) * 627.509474,
            )
            for entry in optimized
        ]
        for conformer_index, energy in optimized_levels:
            axis.axhline(energy, color="#B23A48", linestyle="--", linewidth=1.4)
            axis.annotate(
                f"C{conformer_index} xTB", (1, energy), xytext=(7, 0),
                textcoords="offset points", ha="left", va="center",
                color="#B23A48", fontsize=8,
            )
        kbt_kcal_mol, pairs = XTBOptimizer._crest_conformer_kbt_pairs(optimized)
        axis.axhline(
            kbt_kcal_mol, color="#3A7D44", linestyle="-.", linewidth=1.8,
        )
        within_kbt = [pair for pair in pairs if pair["within_kbt"]]
        within_text = ", ".join(
            f"C{left}-C{right}"
            for pair in within_kbt
            for left, right in [pair["conformer_indices"]]
        ) or "aucune"
        axis.text(
            0.02, 0.98,
            f"298 K, kBT = {kbt_kcal_mol:.3f} kcal/mol\nΔE < kBT : {within_text}",
            transform=axis.transAxes, ha="left", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#cccccc", "alpha": 0.92},
        )
        handles, _labels = axis.get_legend_handles_labels()
        handles.extend([
            Line2D([0], [0], color="#B23A48", linestyle="--", linewidth=1.4,
                   label="Structures optimisées xTB"),
            Line2D([0], [0], color="#3A7D44", linestyle="-.", linewidth=1.8,
                   label=f"kBT = {kbt_kcal_mol:.3f} kcal/mol"),
        ])
        axis.legend(handles=handles, frameon=False, loc="upper right")
        axis.set_title(f"Paysage de recherche CREST ({len(points)} candidats)")
        axis.set_xlabel("Ordre des candidats dans la sortie CREST")
        axis.set_ylabel("Énergie relative (kcal/mol; zéro = minimum xTB optimisé)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlim(0, max(point[0] for point in points) * 1.06)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return True

    @staticmethod
    def _plot_crest_energy_landscape(
        path: Path, scan_records: list[dict], conformers: list[dict],
    ) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False

        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        scan_x = [entry["state_index"] for entry in scan_records]
        scan_y = [entry["relative_energy_kcal_mol"] for entry in scan_records]
        scan_weights = [entry.get("boltzmann_weight") or 0 for entry in scan_records]
        sizes = [30 + 100 * weight for weight in scan_weights]
        axis.plot(scan_x, scan_y, color="#287C8E", alpha=0.45, linewidth=1.5)
        axis.scatter(
            scan_x, scan_y, s=sizes, color="#287C8E", edgecolor="white",
            linewidth=0.7, label="Etats CREST", zorder=3,
        )
        refined = [
            entry for entry in conformers
            if entry.get("xtb_refined_energy_hartree") is not None
        ]
        refined_x = []
        refined_y = []
        refined_labels = []
        scan_state_for_conformer = {}
        for entry in scan_records:
            scan_state_for_conformer.setdefault(entry["conformer_index"], entry["state_index"])
        for entry in refined:
            conformer_index = entry["conformer_index"]
            if conformer_index not in scan_state_for_conformer:
                continue
            refined_x.append(scan_state_for_conformer[conformer_index])
            refined_y.append(entry["xtb_relative_energy_kcal_mol"])
            refined_labels.append(conformer_index)
        if refined_x:
            axis.plot(
                refined_x, refined_y, color="#D17A22", linestyle="--",
                linewidth=1.2, label="Raffinement xTB",
            )
            axis.scatter(
                refined_x, refined_y, marker="D", s=62, color="#D17A22",
                edgecolor="white", linewidth=0.7, label="Raffinement xTB", zorder=4,
            )
            for x_value, y_value, label in zip(refined_x, refined_y, refined_labels):
                axis.annotate(f"C{label}", (x_value, y_value), xytext=(5, -13),
                              textcoords="offset points", fontsize=8, color="#8A4B10")

        axis.set_title("CREST energy landscape")
        axis.set_xlabel("Rang de l'etat CREST")
        axis.set_ylabel("Energie relative (kcal/mol)")
        axis.set_xticks(scan_x)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return True

    @staticmethod
    def _is_major_crest_line(line: str, is_stderr: bool = False) -> bool:
        if re.search(r"\b(?:warning|error|fatal|failed|exception)\b", line, re.I):
            return True
        if is_stderr:
            return False
        return bool(re.search(
            r"(?:\bCREST\b.*\b(?:version|started|finished|terminated|completed)\b|"
            r"\b(?:started|starting|finished|completed|converged|generated|found|selected|retained)\b.*"
            r"\b(?:search|sampling|conformer|ensemble|structure)\b|"
            r"\b(?:search|sampling|conformer|ensemble|structure)\b.*"
            r"\b(?:started|starting|finished|completed|converged|generated|found|selected|retained)\b|"
            r"\bnormal termination\b|\btotal\b.*\bconformers?\b)",
            line, re.I,
        ))

    @staticmethod
    def _run_streamed(
        command, cwd, timeout, environment, line_callback=None, log_dir=None,
    ):
        log_handles = []
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_handles = [
                (log_dir / "stdout.log").open("w", encoding="utf-8", newline=""),
                (log_dir / "stderr.log").open("w", encoding="utf-8", newline=""),
            ]
        try:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                shell=False, env=environment,
            )
        except Exception:
            for handle in log_handles:
                handle.close()
            raise
        stdout_lines = []
        stderr_lines = []

        def forward_lines(stream, lines, sink, label, log_handle=None):
            for line in iter(stream.readline, ""):
                lines.append(line)
                if log_handle is not None:
                    log_handle.write(line)
                    log_handle.flush()
                if line_callback is not None:
                    try:
                        line_callback(line)
                    except Exception:
                        pass
                if XTBOptimizer._is_major_crest_line(
                    line, is_stderr=label == "CREST stderr"
                ):
                    sink.write(f"[{label}] {line}")
                    sink.flush()
            stream.close()

        readers = [
            threading.Thread(
                target=forward_lines,
                args=(
                    process.stdout, stdout_lines, sys.stdout, "CREST",
                    log_handles[0] if log_handles else None,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=forward_lines,
                args=(
                    process.stderr, stderr_lines, sys.stderr, "CREST stderr",
                    log_handles[1] if log_handles else None,
                ),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if process.poll() is None:
                process.kill()
            process.wait()
        finally:
            for reader in readers:
                reader.join()
            for handle in log_handles:
                handle.close()

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        if timed_out:
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    @staticmethod
    def resolve_executable(executable: str) -> str:
        candidate = Path(executable).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            if not candidate.is_file():
                raise FileNotFoundError(f"Executable not found: {candidate}")
            return str(candidate.resolve())
        resolved = shutil.which(executable)
        if not resolved:
            raise FileNotFoundError(
                f"Executable '{executable}' was not found in PATH. "
                "Set xtb_exe/crest_exe to its full path."
            )
        return resolved

    def is_available(self, executable: Optional[str] = None) -> bool:
        try:
            self.resolve_executable(executable or self.xtb_exe)
            return True
        except (FileNotFoundError, OSError):
            return False

    @staticmethod
    def generate_xyz_from_smiles(smiles: str, output_path: str | Path) -> Path:
        """Generate an unconstrained 3D starting structure using RDKit ETKDG/MMFF."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as exc:
            raise RuntimeError("RDKit is required to generate XYZ from SMILES.") from exc

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        molecule = Chem.AddHs(molecule)
        embedding_status = AllChem.EmbedMolecule(molecule, randomSeed=42)
        if embedding_status != 0:
            for use_random_coords in (False, True):
                molecule.RemoveAllConformers()
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                params.maxIterations = 500
                params.useMacrocycleTorsions = True
                params.useMacrocycle14config = True
                params.useRandomCoords = use_random_coords
                embedding_status = AllChem.EmbedMolecule(molecule, params)
                if embedding_status == 0:
                    break
        if embedding_status != 0:
            raise RuntimeError(
                "RDKit could not generate a 3D starting geometry after standard, "
                "macrocycle, and random-coordinate ETKDG attempts."
            )
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000)
        conformer = molecule.GetConformer()
        lines = [str(molecule.GetNumAtoms()), "RDKit ETKDG starting geometry"]
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol():<2} {point.x:.8f} {point.y:.8f} {point.z:.8f}")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    @staticmethod
    def _parse_output(stdout: str, stderr: str, returncode: Optional[int], optimized_xyz: Optional[Path]) -> dict:
        combined = f"{stdout}\n{stderr}"
        energy_matches = re.findall(
            r"TOTAL\s+ENERGY\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s*(?:Eh|a\.u\.)",
            combined,
            re.IGNORECASE,
        )
        iteration_matches = re.findall(
            r"(?:CONVERGED\s+AFTER|CYCLE)\s+(\d+)\s+"
            r"(?:CYCLES?|ITERATIONS?|OPTIMIZATION)",
            combined,
            re.IGNORECASE,
        )
        converged = bool(re.search(r"GEOMETRY\s+OPTIMIZATION\s+CONVERGED", combined, re.I))
        energy = float(energy_matches[-1]) if energy_matches else None
        iterations = int(iteration_matches[-1]) if iteration_matches else None
        return {
            "success": returncode == 0 and converged and optimized_xyz is not None,
            "energy_hartree": energy,
            "converged": converged,
            "optimized_xyz": str(optimized_xyz) if optimized_xyz else None,
            "n_iterations": iterations,
            "returncode": returncode,
            "stdout_tail": "\n".join(stdout.splitlines()[-30:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
        }

    def _run_single(
        self,
        xyz_path: str | Path,
        method: str,
        opt_level: str,
        charge: int,
        multiplicity: int,
        threads: int,
        work_dir: Path,
        timeout: Optional[float] = None,
        constrained_dihedrals: Optional[Sequence[Sequence[Any]]] = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        result = {
            "success": False, "energy_hartree": None, "converged": False,
            "optimized_xyz": None, "n_iterations": None, "runtime_s": None,
            "error": None,
        }
        try:
            executable = self.resolve_executable(self.xtb_exe)
            method_number = {"GFN1": "1", "GFN2": "2"}.get(method.upper())
            if method_number is None:
                raise ValueError("xtb_method must be GFN1 or GFN2.")
            if opt_level.lower() not in self.OPT_LEVELS:
                raise ValueError("opt_level must be normal, tight, or verytight.")
            if threads < 1 or multiplicity < 1:
                raise ValueError("threads and multiplicity must be positive integers.")

            work_dir.mkdir(parents=True, exist_ok=False)
            input_path = work_dir / "input.xyz"
            shutil.copy2(str(xyz_path), input_path)
            command = [executable, str(input_path), "--gfn", method_number,
                       "--opt", opt_level.lower(), "--chrg", str(charge),
                       "--uhf", str(multiplicity - 1), "-P", str(threads)]
            if constrained_dihedrals is not None:
                if not constrained_dihedrals:
                    raise ValueError("At least one constrained dihedral is required.")
                force_constants = {float(item[2]) for item in constrained_dihedrals}
                if len(force_constants) != 1:
                    raise ValueError("All constrained dihedrals must share one force constant.")
                xcontrol = work_dir / "xcontrol.inp"
                xcontrol_lines = ["$constrain", f"  force constant={force_constants.pop()}"]
                for atom_indices, target_deg, _force_constant in constrained_dihedrals:
                    if len(atom_indices) != 4:
                        raise ValueError("Each constrained dihedral requires four indices.")
                    if any(not isinstance(index, int) or index < 0 for index in atom_indices):
                        raise ValueError("Constrained dihedral indices must be non-negative integers.")
                    one_based = [index + 1 for index in atom_indices]
                    xcontrol_lines.append(
                        f"  dihedral: {', '.join(map(str, one_based))}, {float(target_deg)}"
                    )
                xcontrol_lines.append("$end")
                xcontrol.write_text("\n".join(xcontrol_lines) + "\n", encoding="utf-8")
                command.extend(["--input", str(xcontrol)])
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = str(threads)
            completed = subprocess.run(
                command, cwd=work_dir, capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False, shell=False,
                timeout=timeout, env=environment,
            )
            (work_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (work_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            optimized = work_dir / "xtbopt.xyz"
            if not optimized.is_file():
                optimized = None
            result.update(self._parse_output(
                completed.stdout, completed.stderr, completed.returncode, optimized
            ))
            if completed.returncode != 0:
                result["error"] = f"xTB exited with status {completed.returncode}."
            elif not result["converged"]:
                result["error"] = "xTB did not report geometry optimization convergence."
            elif optimized is None:
                result["error"] = "xTB converged but did not create xtbopt.xyz."
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
            result["error"] = str(exc)
        except Exception as exc:
            result["error"] = f"Unexpected xTB error: {exc}"
        result["runtime_s"] = round(time.perf_counter() - start, 3)
        return result

    def optimize(
        self,
        xyz_path: str | Path,
        output_dir: str | Path,
        method: str = "GFN2",
        opt_level: str = "tight",
        charge: int = 0,
        multiplicity: int = 1,
        threads: int = 1,
        run_crest: bool = False,
        crest_n_conformers: int = 10,
        timeout: Optional[float] = None,
        scratch_dir: Optional[str | Path] = None,
        output_name: str = "xtbopt.xyz",
        receptor_name: str = "xtbopt_receptor.mol",
        topology_smiles: Optional[str] = None,
        log_dir: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        source = Path(xyz_path).resolve()
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        scratch = Path(scratch_dir).resolve() if scratch_dir else destination
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="xtb_", dir=scratch))
        run_log_dir = Path(log_dir).resolve() if log_dir else destination / "logs"
        run_log_dir = run_log_dir / root.name
        if not run_crest:
            result = self._run_single(source, method, opt_level, charge, multiplicity,
                                      threads, root / "run", timeout)
            self._archive_process_logs(root / "run", run_log_dir / "xtb")
            result["process_logs_dir"] = str(run_log_dir)
            return self._persist_result(
                result, destination, charge, output_name, receptor_name,
                topology_smiles,
            )
        crest_threads = threads
        return self._optimize_crest(source, destination, root, method, opt_level,
                                    charge, multiplicity, threads, crest_threads,
                                    crest_n_conformers, timeout,
                                    output_name, receptor_name, topology_smiles,
                                    run_log_dir)

    def _optimize_crest(self, source, destination, root, method, opt_level,
                        charge, multiplicity, threads, crest_threads,
                        n_conformers, timeout,
                        output_name, receptor_name, topology_smiles, run_log_dir):
        started = time.perf_counter()
        try:
            if n_conformers < 1:
                raise ValueError("crest_n_conformers must be at least 1.")
            method_flag = {"GFN1": "--gfn1", "GFN2": "--gfn2"}.get(method.upper())
            if method_flag is None:
                raise ValueError("xtb_method must be GFN1 or GFN2.")
            crest_dir = root / "crest"
            crest_dir.mkdir()
            seed = crest_dir / "input.xyz"
            shutil.copy2(source, seed)
            linux_crest_dir = None
            wsl = None
            if self.crest_use_wsl:
                wsl = self.resolve_executable(self.wsl_exe)
                linux_crest_dir = self._create_wsl_crest_dir(wsl)
                self._write_wsl_file(wsl, linux_crest_dir, "input.xyz", seed.read_bytes(), timeout)
                command = [
                    wsl, "--cd", linux_crest_dir, "--exec", self.crest_wsl_exe,
                    "input.xyz", "-xnam", self.crest_wsl_xtb_exe, method_flag,
                    "--chrg", str(charge), "--uhf", str(multiplicity - 1),
                    "--T", str(crest_threads),
                ]
                command_cwd = None
            else:
                crest = self.resolve_executable(self.crest_exe)
                xtb = self.resolve_executable(self.xtb_exe)
                command = [crest, str(seed), "-xnam", xtb, method_flag, "--chrg",
                           str(charge), "--uhf", str(multiplicity - 1),
                           "--T", str(threads)]
                command_cwd = crest_dir
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = str(crest_threads)
            crest_log_dir = run_log_dir / "crest"
            crest_log_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "command": command,
                "working_directory": linux_crest_dir or str(crest_dir),
                "local_scratch_directory": str(crest_dir),
                "wsl_scratch_directory": linux_crest_dir,
                "crest_threads": crest_threads,
                "xtb_refinement_threads": threads,
                "crest_executable": self.crest_wsl_exe if self.crest_use_wsl else self.crest_exe,
                "xtb_executable": self.crest_wsl_xtb_exe if self.crest_use_wsl else self.xtb_exe,
                "environment": {
                    key: environment[key]
                    for key in (
                        "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_NESTED",
                        "OMP_MAX_ACTIVE_LEVELS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                    if key in environment
                },
            }
            (crest_log_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("[CREST] Recherche de conformeres demarree.", flush=True)
            line_callback = None
            if linux_crest_dir and wsl:
                line_callback = lambda line: (
                    self._capture_wsl_crest_rotamers(
                        wsl, linux_crest_dir, crest_dir, timeout,
                    )
                    if "crest_rotamers_" in line else None
                )
            completed = self._run_streamed(
                command, cwd=command_cwd, timeout=timeout, environment=environment,
                line_callback=line_callback, log_dir=crest_log_dir,
            )
            (crest_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (crest_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            self._archive_process_logs(crest_dir, run_log_dir / "crest")
            scan_records = self._parse_crest_energy_scan(completed.stdout)
            conformer_file = crest_dir / "crest_conformers.xyz"
            search_state_files = []
            if completed.returncode != 0:
                diagnostics = "\n".join(
                    f"{stream_name}:\n{stream.strip()[-1200:]}"
                    for stream_name, stream in (
                        ("stderr", completed.stderr), ("stdout", completed.stdout)
                    )
                    if stream.strip()
                )
                scratch_note = (
                    f" WSL scratch retained at {linux_crest_dir}."
                    if linux_crest_dir else ""
                )
                raise RuntimeError(
                    f"CREST exited with status {completed.returncode}: "
                    f"{diagnostics}{scratch_note}"
                )
            if linux_crest_dir and wsl:
                conformer_file.write_bytes(self._read_wsl_file(
                    wsl, linux_crest_dir, "crest_conformers.xyz", timeout
                ))
                try:
                    rotamer_names = self._list_wsl_crest_rotamer_files(
                        wsl, linux_crest_dir, timeout,
                    )
                    for name in rotamer_names:
                        local_file = crest_dir / name
                        local_file.write_bytes(self._read_wsl_file(
                            wsl, linux_crest_dir, name, timeout,
                        ))
                        search_state_files.append(local_file)
                    search_state_files.extend(
                        path for path in sorted(crest_dir.glob("crest_rotamers_*.xyz"))
                        if path not in search_state_files
                    )
                except RuntimeError as exc:
                    print(f"[CREST] Etats de recherche detailles indisponibles: {exc}", flush=True)
            else:
                search_state_files = sorted(
                    crest_dir.glob("crest_rotamers_*.xyz"),
                    key=lambda path: int(re.search(r"_(\d+)\.xyz$", path.name).group(1)),
                )
            if not conformer_file.is_file():
                raise RuntimeError("CREST did not create crest_conformers.xyz.")
            all_conformers = read_xyz_frames(conformer_file)
            search_state_frames = [
                frame for path in search_state_files for frame in read_xyz_frames(path)
            ]
            if not search_state_frames:
                print(
                    "[CREST] Aucun crest_rotamers_*.xyz trouve; "
                    "les angles seront associes aux energies du scan quand disponibles.",
                    flush=True,
                )
            if linux_crest_dir and wsl:
                self._remove_wsl_crest_dir(wsl, linux_crest_dir)
                linux_crest_dir = None
            artifact_stem = Path(output_name).stem
            if artifact_stem.endswith("_geomxTB_opti"):
                artifact_stem = artifact_stem[:-len("_geomxTB_opti")]
            ensemble_path = destination / f"{artifact_stem}_crest_conformers.xyz"
            ensemble_path.write_text("".join(all_conformers), encoding="utf-8")
            landscape_path = destination / f"{artifact_stem}_crest_energy_landscape.csv"
            landscape = []
            for index, frame in enumerate(all_conformers, 1):
                crest_xyz_path = destination / f"{artifact_stem}_crest_conformer_{index:03d}.xyz"
                crest_xyz_path.write_text(frame, encoding="utf-8")
                landscape.append({
                    "conformer_index": index,
                    "crest_energy_hartree": self._energy_from_xyz_frame(frame),
                    "xtb_refined_energy_hartree": None,
                    "refinement_status": "pending" if index <= n_conformers else "not_selected",
                    "crest_xyz": str(crest_xyz_path),
                    "refined_xyz": "",
                    "error": "",
                })
            search_candidates_path = None
            search_scan_path = None
            landscape_3d_path = None
            absolute_dihedral_path = None
            energy_distribution_csv_path = None
            search_records = self._parse_crest_search_states(search_state_frames)
            plot_frames, plot_records, plot_source = self._crest_plot_inputs(
                search_state_frames, all_conformers, scan_records,
            )
            if search_records:
                search_candidates_path = destination / (
                    f"{artifact_stem}_crest_search_candidates.xyz"
                )
                search_candidates_path.write_text(
                    "".join(search_state_frames), encoding="utf-8"
                )
                search_scan_path = destination / f"{artifact_stem}_crest_search_scan.csv"
                self._write_crest_search_scan_csv(search_scan_path, search_records)
                distribution_records = self._crest_energy_distribution_records(
                    search_records, landscape,
                )
                if distribution_records:
                    energy_distribution_csv_path = destination / (
                        f"{artifact_stem}_crest_energy_distribution.csv"
                    )
                    self._write_crest_energy_distribution_csv(
                        energy_distribution_csv_path, distribution_records,
                    )
            if plot_records:
                landscape_3d_path = destination / (
                    f"{artifact_stem}_crest_landscape_3d.html"
                )
                absolute_dihedral_path = destination / (
                    f"{artifact_stem}_crest_dihedral_energy_absolute.html"
                )
                if plot_source == "energy_scan":
                    print(
                        "[CREST] Graphiques bases sur les "
                        f"{len(plot_records)} etats energetiques du scan; "
                        "les coordonnees proviennent de leurs conformeres associes.",
                        flush=True,
                    )
                elif plot_source == "final_conformers":
                    print(
                        "[CREST] Etats de recherche detailles absents; "
                        f"graphiques bases sur {len(plot_records)} conformeres finaux.",
                        flush=True,
                    )
            else:
                print(
                    "[CREST] Graphiques non generes: aucun etat avec energie exploitable.",
                    flush=True,
                )
            self._write_crest_energy_landscape(landscape_path, landscape)
            conformers = all_conformers[:n_conformers]
            print(
                f"[CREST] {len(all_conformers)} conformeres generes; "
                f"raffinement de {len(conformers)} conformere(s).",
                flush=True,
            )
            ranked = []
            for index, frame in enumerate(conformers, 1):
                print(
                    f"[CREST] Raffinement xTB du conformere {index}/{len(conformers)}.",
                    flush=True,
                )
                frame_path = root / f"conformer_{index:04d}.xyz"
                frame_path.write_text(frame, encoding="utf-8")
                optimized = self._run_single(
                    frame_path, method, opt_level, charge, multiplicity, threads,
                    root / f"refine_{index:04d}", timeout,
                )
                self._archive_process_logs(
                    root / f"refine_{index:04d}",
                    run_log_dir / f"xtb_conformer_{index:03d}",
                )
                if optimized["success"]:
                    refined_xyz_path = destination / (
                        f"{artifact_stem}_crest_conformer_{index:03d}_refined.xyz"
                    )
                    shutil.copy2(optimized["optimized_xyz"], refined_xyz_path)
                    landscape[index - 1].update({
                        "xtb_refined_energy_hartree": optimized.get("energy_hartree"),
                        "refinement_status": "refined",
                        "refined_xyz": str(refined_xyz_path),
                    })
                    optimized["conformer_index"] = index
                    optimized["crest_energy_hartree"] = landscape[index - 1]["crest_energy_hartree"]
                    optimized["refined_xyz"] = str(refined_xyz_path)
                    ranked.append(optimized)
                else:
                    landscape[index - 1].update({
                        "refinement_status": "failed",
                        "error": optimized.get("error") or "xTB refinement failed.",
                    })
                self._write_crest_energy_landscape(landscape_path, landscape)
            crest_energies = [
                entry["crest_energy_hartree"]
                for entry in landscape if entry["crest_energy_hartree"] is not None
            ]
            lowest_crest_energy = min(crest_energies, default=None)
            for entry in landscape:
                crest_energy = entry["crest_energy_hartree"]
                entry["crest_relative_energy_kcal_mol"] = (
                    (crest_energy - lowest_crest_energy) * 627.509474
                    if crest_energy is not None and lowest_crest_energy is not None
                    else ""
                )
            refined_energies = [
                entry["xtb_refined_energy_hartree"]
                for entry in landscape if entry["xtb_refined_energy_hartree"] is not None
            ]
            lowest_refined_energy = min(refined_energies, default=None)
            for entry in landscape:
                refined_energy = entry["xtb_refined_energy_hartree"]
                entry["xtb_relative_energy_kcal_mol"] = (
                    (refined_energy - lowest_refined_energy) * 627.509474
                    if refined_energy is not None and lowest_refined_energy is not None
                    else ""
                )
            if (plot_records and landscape_3d_path is not None
                    and absolute_dihedral_path is not None):
                landscape_created, dihedral_created = self._plot_crest_interactive(
                    landscape_3d_path, absolute_dihedral_path,
                    plot_frames, plot_records, landscape, topology_smiles,
                    candidate_source=plot_source,
                )
                if not landscape_created:
                    landscape_3d_path = None
                if not dihedral_created:
                    absolute_dihedral_path = None
            if not scan_records:
                scan_records = [
                    {
                        "state_index": entry["conformer_index"],
                        "relative_energy_kcal_mol": entry["crest_relative_energy_kcal_mol"],
                        "crest_energy_hartree": entry["crest_energy_hartree"],
                        "boltzmann_weight": "",
                        "conformer_population": "",
                        "conformer_index": entry["conformer_index"],
                        "degeneracy": 1,
                        "xtb_refined_energy_hartree": entry["xtb_refined_energy_hartree"] or "",
                        "xtb_relative_energy_kcal_mol": entry.get("xtb_relative_energy_kcal_mol", ""),
                    }
                    for entry in landscape
                ]
            else:
                refined_by_index = {entry["conformer_index"]: entry for entry in landscape}
                refined_energies = [
                    entry["xtb_refined_energy_hartree"]
                    for entry in landscape
                    if entry["xtb_refined_energy_hartree"] is not None
                ]
                lowest_refined_energy = min(refined_energies, default=None)
                for entry in scan_records:
                    refined_entry = refined_by_index.get(entry["conformer_index"], {})
                    refined_energy = refined_entry.get("xtb_refined_energy_hartree")
                    entry["xtb_refined_energy_hartree"] = refined_energy or ""
                    entry["xtb_relative_energy_kcal_mol"] = (
                        (refined_energy - lowest_refined_energy) * 627.509474
                        if refined_energy is not None and lowest_refined_energy is not None
                        else ""
                    )
            scan_path = destination / f"{artifact_stem}_crest_energy_scan.csv"
            self._write_crest_scan_csv(scan_path, scan_records)
            if not ranked:
                raise RuntimeError("xTB refinement failed for every CREST conformer.")
            ranked_with_energy = [item for item in ranked
                                  if item.get("energy_hartree") is not None]
            if not ranked_with_energy:
                raise RuntimeError("xTB did not report energies for the CREST conformers.")
            best = min(ranked_with_energy, key=lambda item: item["energy_hartree"])
            best["conformers_refined"] = len(ranked)
            best["crest_conformers_xyz"] = str(ensemble_path)
            best["crest_energy_landscape_csv"] = str(landscape_path)
            best["crest_energy_scan_csv"] = str(scan_path)
            if search_candidates_path is not None and search_scan_path is not None:
                best["crest_search_candidates_xyz"] = str(search_candidates_path)
                best["crest_search_scan_csv"] = str(search_scan_path)
            if plot_records:
                best["crest_plot_source"] = plot_source
            if landscape_3d_path is not None:
                best["crest_conformer_landscape_3d_html"] = str(landscape_3d_path)
            if absolute_dihedral_path is not None:
                best["crest_dihedral_energy_absolute_html"] = str(
                    absolute_dihedral_path
                )
            if energy_distribution_csv_path is not None:
                best["crest_search_energy_distribution_csv"] = str(
                    energy_distribution_csv_path
                )
            best["crest_conformer_xyz_files"] = [entry["crest_xyz"] for entry in landscape]
            best["crest_refined_xyz_files"] = [
                entry["refined_xyz"] for entry in landscape if entry["refined_xyz"]
            ]
            best["process_logs_dir"] = str(run_log_dir)
            best["crest_search_threads"] = crest_threads
            best["crest_stdout_tail"] = "\n".join(completed.stdout.splitlines()[-30:])
            best["runtime_s"] = round(time.perf_counter() - started, 3)
            return self._persist_result(
                best, destination, charge, output_name, receptor_name,
                topology_smiles,
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return {
                "success": False, "energy_hartree": None, "converged": False,
                "optimized_xyz": None, "n_iterations": None,
                "runtime_s": round(time.perf_counter() - started, 3),
                "error": str(exc), "method": method, "opt_level": opt_level,
                "process_logs_dir": str(run_log_dir),
            }

    @staticmethod
    def _archive_process_logs(source_dir: Path, log_dir: Path) -> None:
        log_files = [source_dir / "stdout.log", source_dir / "stderr.log"]
        existing_logs = [path for path in log_files if path.is_file()]
        if not existing_logs:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        for source_path in existing_logs:
            shutil.copy2(source_path, log_dir / source_path.name)

    @staticmethod
    def _persist_result(
        result: dict[str, Any], output_dir: Path, charge: int = 0,
        output_name: str = "xtbopt.xyz",
        receptor_name: str = "xtbopt_receptor.mol",
        topology_smiles: Optional[str] = None,
    ) -> dict[str, Any]:
        result = dict(result)
        source = result.get("optimized_xyz")
        if result.get("success") and source:
            saved = output_dir / output_name
            shutil.copy2(source, saved)
            result["optimized_xyz"] = str(saved)
            try:
                receptor_mol = output_dir / receptor_name
                XTBOptimizer.export_receptor_mol(
                    saved, receptor_mol, charge=charge, topology_smiles=topology_smiles
                )
                result["receptor_mol"] = str(receptor_mol)
            except Exception as exc:
                result["receptor_mol"] = None
                result["receptor_export_error"] = str(exc)
        return result

    @staticmethod
    def _molecule_from_xyz_frame(
        xyz_frame: str, charge: int = 0,
        topology_smiles: Optional[str] = None,
    ):
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        coordinate_molecule = Chem.MolFromXYZBlock(xyz_frame)
        if coordinate_molecule is None:
            raise ValueError("Cannot read candidate XYZ frame.")

        if topology_smiles:
            molecule = Chem.MolFromSmiles(topology_smiles)
            if molecule is None:
                raise ValueError("Cannot parse source SMILES for candidate topology.")
            molecule = Chem.AddHs(molecule)
            if molecule.GetNumAtoms() != coordinate_molecule.GetNumAtoms():
                raise ValueError(
                    "Source SMILES atom count does not match candidate XYZ."
                )
            rdDetermineBonds.DetermineBonds(coordinate_molecule, charge=int(charge))
            atom_mapping = coordinate_molecule.GetSubstructMatch(molecule)
            if len(atom_mapping) != molecule.GetNumAtoms():
                raise ValueError(
                    "Candidate XYZ connectivity does not match source SMILES."
                )
            source_conformer = coordinate_molecule.GetConformer()
            conformer = Chem.Conformer(molecule.GetNumAtoms())
            for index in range(molecule.GetNumAtoms()):
                point = source_conformer.GetAtomPosition(atom_mapping[index])
                conformer.SetAtomPosition(index, Point3D(point.x, point.y, point.z))
            conformer.Set3D(True)
            molecule.AddConformer(conformer, assignId=True)
        else:
            molecule = coordinate_molecule
            rdDetermineBonds.DetermineBonds(molecule, charge=int(charge))
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("Candidate XYZ produced multiple disconnected fragments.")
        try:
            Chem.AssignStereochemistryFrom3D(molecule)
        except Exception:
            pass
        return molecule

    @staticmethod
    def export_receptor_mol(
        xyz_path: str | Path, output_path: str | Path, charge: int = 0,
        topology_smiles: Optional[str] = None,
    ) -> Path:
        """Write xTB coordinates as a 3D MOL, preferring known SMILES topology."""
        from rdkit import Chem
        xyz_frame = Path(xyz_path).read_text(encoding="utf-8")
        molecule = XTBOptimizer._molecule_from_xyz_frame(
            xyz_frame, charge=charge, topology_smiles=topology_smiles,
        )

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Chem.MolToMolFile(molecule, str(destination), forceV3000=molecule.GetNumAtoms() > 999)
        check = Chem.MolFromMolFile(str(destination), removeHs=False)
        if check is None or check.GetNumAtoms() != molecule.GetNumAtoms():
            raise ValueError("The exported receptor MOL could not be read back intact.")
        return destination

    def optimize_constrained_dihedral(
        self, xyz_path: str | Path, output_dir: str | Path,
        atom_indices: Sequence[int], target_deg: float, method: str = "GFN2",
        opt_level: str = "tight", charge: int = 0, multiplicity: int = 1,
        threads: int = 1, force_constant: float = 0.5,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        result = self.optimize_constrained_dihedrals(
            xyz_path, output_dir, [atom_indices], target_deg,
            method=method, opt_level=opt_level, charge=charge,
            multiplicity=multiplicity, threads=threads,
            force_constant=force_constant, timeout=timeout,
        )
        first_dihedral = result.get("dihedrals", {}).get("dihedral_1")
        if first_dihedral:
            result["dihedral_deg"] = first_dihedral["dihedral_deg"]
        return result

    def optimize_constrained_dihedrals(
        self, xyz_path: str | Path, output_dir: str | Path,
        atom_indices: Sequence[Sequence[int]], target_deg: float = 0.0,
        method: str = "GFN2", opt_level: str = "tight", charge: int = 0,
        multiplicity: int = 1, threads: int = 1,
        force_constant: float = 0.5, timeout: Optional[float] = None,
        scratch_dir: Optional[str | Path] = None,
        output_name: str = "xtbopt_constrained.xyz",
        receptor_name: str = "xtbopt_constrained_receptor.mol",
        topology_smiles: Optional[str] = None,
    ) -> dict[str, Any]:
        if not atom_indices:
            raise ValueError("At least one constrained dihedral is required.")
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        scratch = Path(scratch_dir).resolve() if scratch_dir else destination
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="xtb_constrained_", dir=scratch))
        result = self._run_single(
            Path(xyz_path).resolve(), method, opt_level, charge, multiplicity,
            threads, root / "run", timeout,
            constrained_dihedrals=[
                (indices, target_deg, force_constant) for indices in atom_indices
            ],
        )
        if result.get("success"):
            result = self._persist_result(
                result, destination, charge, output_name, receptor_name,
                topology_smiles,
            )
            coords = self._read_coordinates(result["optimized_xyz"])
            result["dihedrals"] = {
                f"dihedral_{index}": {
                    "atom_indices": list(indices),
                    "target_deg": float(target_deg),
                    "dihedral_deg": calculate_dihedral(coords, indices),
                }
                for index, indices in enumerate(atom_indices, 1)
            }
        return result

    @staticmethod
    def _read_coordinates(path: str | Path) -> list[tuple[float, float, float]]:
        frame = read_xyz_frames(path)[0].splitlines()
        return [tuple(map(float, line.split()[1:4])) for line in frame[2:]]

    def compare_constrained_dihedral(self, xyz_path: str | Path, output_dir: str | Path,
                                    atom_indices: Sequence[int], target_deg: float,
                                    **kwargs) -> dict[str, Any]:
        destination = Path(output_dir).resolve()
        free = self.optimize(xyz_path, destination / "free", **kwargs)
        constrained = self.optimize_constrained_dihedral(
            xyz_path, destination / "constrained", atom_indices, target_deg,
            **{key: value for key, value in kwargs.items()
               if key in {"method", "opt_level", "charge", "multiplicity", "threads", "timeout"}},
        )
        if free.get("success"):
            free["dihedral_deg"] = calculate_dihedral(
                self._read_coordinates(free["optimized_xyz"]), atom_indices
            )
        difference = None
        if free.get("energy_hartree") is not None and constrained.get("energy_hartree") is not None:
            difference = constrained["energy_hartree"] - free["energy_hartree"]
        return {"free": free, "constrained": constrained,
                "energy_difference_hartree": difference}