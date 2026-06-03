"""
utils_paths.py
==============
Utilitaires partagés de validation et d'assainissement de chemins.
Importé par pipeline_orchestrator.py et pipeline_gui.py pour garantir
un comportement identique entre le CLI et la GUI.
"""

import re


def _sanitize_mol_name(name: str) -> str:
    """Élimine les caractères qui permettraient une traversée de chemin."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00]', '_', name)
    sanitized = re.sub(r'\.{2,}', '.', sanitized)
    sanitized = sanitized.strip('. ')
    if not sanitized:
        raise ValueError(f"Nom de molécule invalide après assainissement : '{name}'")
    return sanitized
