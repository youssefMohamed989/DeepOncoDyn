A complete computational drug discovery pipeline for oncology research, implementing six sequential stages from target identification through multi-objective lead optimisation, culminating in a publication-quality dashboard figure.

Overview
insilico_drug_discovery.py is a self-contained simulation pipeline that models the full preclinical drug discovery workflow:
Target Identification → Virtual Screening → Generative Chemistry → ADMET Profiling → Lead Optimisation → Figure Generation
All molecular data is synthetically generated using chemically realistic distributions, making the pipeline fully reproducible without external databases or licensed compound libraries.

Pipeline Stages
Stage 1 — Target Identification & Network Analysis (TargetIdentification)

Constructs a simulated protein–protein interaction (PPI) network of 120 proteins using a Barabási–Albert preferential attachment model seeded by 15 canonical cancer driver genes (TP53, EGFR, KRAS, PIK3CA, etc.)
Computes network centrality metrics: betweenness, degree centrality, PageRank, clustering coefficient
Scores each protein on a composite druggability index combining network topology, simulated pocket volume, hydrophobicity, and disease association (OMIM/COSMIC proxy)
Output: target_druggability_scores.csv

Stage 2 — Virtual Screening (VirtualScreening)

Generates a 5,000-compound library with realistic physicochemical descriptors (MW, LogP, HBA, HBD, TPSA, RotBonds) and 2048-bit Morgan fingerprint surrogates
Applies a pharmacophore filter based on configurable feature constraints
Ranks hits using a composite virtual screening score weighted across QSAR-predicted docking ΔG, shape similarity to a reference ligand, and synthetic accessibility
Output: virtual_library.csv, vs_top500_hits.csv, morgan_fingerprints.npy

Stage 3 — Generative Chemistry (GenerativeChemistry)

Simulates a fragment-based Variational Autoencoder (VAE) encoding top screening hits into a 64-dimensional latent space via PCA projection
Runs a REINFORCE policy gradient optimisation (80 steps, 25 samples/step) to navigate towards high-reward (low ΔG) chemical regions
Decodes 2,000 novel molecules via fragment assembly from a 15-fragment library (benzene, pyridine, indole, quinoline, etc.)
Output: generated_compounds.csv

Stage 4 — ADMET Profiling (ADMETProfiler)

Predicts 12 ADMET endpoints using physicochemically calibrated Random Forest models:

Absorption: Caco-2 permeability, P-gp substrate probability
Distribution: plasma protein binding, BBB penetration
Metabolism: CYP3A4 & CYP2D6 inhibition, hepatic intrinsic clearance
Excretion/Toxicity: hERG pIC₅₀, Ames mutagenicity, hepatotoxicity, LD50


Computes a composite ADMET score per compound
Output: admet_profiles.csv

Stage 5 — Multi-Objective Lead Optimisation (LeadOptimiser)

Builds a four-objective scoring frame: potency, selectivity, ADMET, and synthetic accessibility
Identifies Pareto-optimal fronts (non-dominated sorting) over 500 sampled candidates
Shortlists the top 30 Pareto front 1 leads
Output: final_lead_candidates.csv

Stage 6 — Figure 7 Generation (generate_figure7)
Produces a 10-panel publication-quality dashboard at 300 DPI covering:
PanelContentAPPI network with druggability heatmapBTarget druggability ranked bar chartCVirtual screening score distributionDChemical space PCA (generated vs screened)EREINFORCE reward trajectoryFADMET radar chart (top-5 candidates)GPareto front scatter (potency vs ADMET)HSafety–potency window (hERG vs docking ΔG)ILipinski property space (MW vs LogP)JADMET heatmap (top-20 candidates × 6 endpoints)

Output: Figure7_InSilicoDrugDiscovery.png


Requirements
python >= 3.8
numpy
pandas
matplotlib
networkx
scipy
scikit-learn
Install all dependencies:
bashpip install numpy pandas matplotlib networkx scipy scikit-learn

Directory Structure
The pipeline expects (and creates) the following layout:
oncology_paper/
├── data/                          # All CSV and .npy outputs
│   ├── target_druggability_scores.csv
│   ├── virtual_library.csv
│   ├── vs_top500_hits.csv
│   ├── morgan_fingerprints.npy
│   ├── generated_compounds.csv
│   ├── admet_profiles.csv
│   └── final_lead_candidates.csv
├── figures/
│   └── Figure7_InSilicoDrugDiscovery.png
└── code/
    └── insilico_drug_discovery.py
Create the directories before running:
bashmkdir -p /home/claude/oncology_paper/{data,figures,code}

Usage
bashpython insilico_drug_discovery.py
The pipeline runs all six stages sequentially and prints progress to stdout:
=================================================================
  DeepOncoDyn  |  In-Silico Drug Discovery Pipeline
=================================================================

[STAGE 1]  Target Identification & Network Analysis
[STAGE 2]  Virtual Screening (5,000 compound library)
[STAGE 3]  Generative Chemistry (VAE + REINFORCE)
[STAGE 4]  ADMET Profiling
[STAGE 5]  Multi-Objective Lead Optimisation (Pareto)
[STAGE 6]  Generating Figure 7 (In-Silico Drug Discovery Dashboard)
Programmatic use
pythonfrom insilico_drug_discovery import run_pipeline

results = run_pipeline()
# results keys: target_df, vs_hits, gen_df, admet_df, pareto_df
Reproducibility
The random seed is fixed globally at the top of the script:
pythonnp.random.seed(2024)
Change this value to generate alternative realisations.

Key Parameters
ParameterLocationDefaultDescriptionn_proteinsTargetIdentification.__init__120PPI network sizen_compoundsVirtualScreening.__init__5000Screening library sizelatent_dimGenerativeChemistry.__init__64VAE latent dimensionsn_generateGenerativeChemistry.__init__2000Molecules to generaten_stepsreinforce_sample80RL optimisation stepssample_sizerun_pipeline500Pareto subset size

Outputs Summary
FileStageDescriptiontarget_druggability_scores.csv1120 proteins with composite druggability scoresvirtual_library.csv2Full 5,000-compound library with descriptorsvs_top500_hits.csv2Top 500 virtual screening hitsmorgan_fingerprints.npy25000 × 2048 binary fingerprint matrixgenerated_compounds.csv32,000 generatively designed moleculesadmet_profiles.csv412-endpoint ADMET predictionsfinal_lead_candidates.csv5Top 30 Pareto-optimal leadsFigure7_InSilicoDrugDiscovery.png6300 DPI 10-panel dashboard

Notes

All molecular data is simulated. No real compound structures (SMILES) are generated or used; scaffolds are represented by fragment name strings.
The pipeline is designed for figure generation and workflow illustration in a research manuscript context, not for production drug discovery.
Pareto front computation uses an O(n²) non-dominated sort; the pipeline subsamples to 500 compounds to keep runtime reasonable. Increase sample_size in run_pipeline() for higher coverage.
Matplotlib's Agg backend is used so the pipeline runs in headless/server environments with no display required.
