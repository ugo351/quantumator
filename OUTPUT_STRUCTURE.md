# Structure des sorties

Chaque calcul lancé depuis `pipeline_gui.py` crée un dossier racine par molécule. `PipelineConfig.output_dir` reste cette racine pour préserver les appels existants.

```text
<output_dir>/<NomMolécule>/
├── fonctionnel/
│   ├── logs/
│   │   └── <NomMolécule>_pipeline.log
│   ├── temp/
│   │   ├── xtb/                         # dossiers uniques xTB/CREST, input, stdout/stderr
│   │   ├── xtb_constrained/             # calcul xTB sous contrainte
│   │   ├── psi4/                        # <NomMolécule>_psi4_*.psi4out
│   │   └── binding/                     # PDBQT de travail et conversions SDF temporaires
│   └── checkpoints/
│       └── binding/<NomMolécule>_docking_checkpoint.csv
└── utilisable/
    ├── geometries/
    │   ├── <NomMolécule>_geomRDKit_initiale.xyz             # si générée depuis SMILES
    │   ├── <NomMolécule>_geomUtilisateur_initiale.xyz        # si fournie par l'utilisateur
    │   ├── <NomMolécule>_geomxTB_opti.xyz
    │   ├── <NomMolécule>_receptor_xTB.mol
    │   ├── <NomMolécule>_geomxTB_contrainte.xyz          # optionnel
    │   ├── <NomMolécule>_receptor_xTB_contrainte.mol     # optionnel
    │   └── <NomMolécule>_geomDFT_opti.xyz                # si DFT exécutée
    ├── tddft/
    │   ├── <NomMolécule>_tddft_transitions.json
    │   └── <NomMolécule>_*                              # exports détaillés Psi4
    ├── density/
    │   ├── <NomMolécule>_electron_density.cube
    │   ├── <NomMolécule>_electrostatic_potential.cube
    │   ├── <NomMolécule>_dual_descriptor.cube
    │   └── <NomMolécule>_coordinates_and_charges.txt
    ├── binding/
    │   ├── <NomMolécule>_receptor.xyz
    │   ├── <NomMolécule>.xyz
    │   ├── <NomMolécule>_docking_results.csv
    │   ├── <NomMolécule>_docking_results.png
    │   ├── <NomMolécule>_pi_contact_geometry.csv
    │   └── poses/<NomMolécule>_runXX/<NomMolécule>_runXX_poseN.sdf
    └── rapports/
        ├── <NomMolécule>_report.json
        └── <NomMolécule>_properties.txt
```

Les géométries passent entre étapes par le chemin effectif enregistré dans `results`; leur emplacement est désormais `utilisable/geometries`. Les dossiers `fonctionnel/temp` peuvent contenir des fichiers volumineux et reproductibles; ils ne doivent pas être utilisés comme livrables. Aucun nettoyage automatique n'est fait.

## Batch

Les résumés batch et multi-récepteur sont écrits dans `utilisable/rapports/` de leur dossier de résultats. Le fichier source CSV/Excel y est copié.

## Scripts standalone historiques

Le rangement ci-dessus s'applique à l'orchestrateur chargé par la GUI. Les blocs `main()` historiques de `psi4_calculator_fixed_y.py`, `electron_density_analysis_fixed.py` et `docking_kd_pipeline.py` conservent leurs répertoires/configurations standalone afin de ne pas casser leurs usages directs. Ils ne sont pas appelés comme points d'entrée lors d'un run GUI normal.
