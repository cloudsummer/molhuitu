# 🧬 MolHuiTu — Molecular HyperGraph V8.1
### Intelligent Drug–Target Interaction (DTI) Prediction Platform

> *A next-generation AI platform for interpretable drug–target interaction prediction based on hypergraph masked autoencoders.*


<img width="3548" height="1652" alt="MolHuiTu Overview" src="https://github.com/user-attachments/assets/0bf60f5b-a63f-4708-9910-d043bc655497" />


## 🗂️ Table of Contents
- [1. Overview](#1-overview)
- [2. Highlights](#2-highlights)
- [3. Model Overview](#3-model-overview)
- [4. Capabilities](#4-capabilities)
- [5. Quick Start](#5-quick-start)
- [6. Use Cases](#6-use-cases)
- [7. Technical Stack](#7-technical-stack)
- [8. Latest Update (V8.1)](#8-latest-update-v81)
- [9. Citation & Acknowledgements](#9-citation--acknowledgements)

---

## 1. Overview

**MolHuiTu (Molecular HyperGraph)** is an **intelligent drug–target interaction (DTI) prediction platform** that integrates **hypergraph self-supervised learning** with **protein language modeling**.  
By simply inputting a **SMILES** string and a **protein FASTA** sequence, users can obtain:
- Predicted **DTI probability or regression score**  
- **Explainable key atoms/residues** contributing to molecular binding  
- Interactive **3D visualization** for structural interpretation  

The platform supports both **single-sample prediction** and **high-throughput screening**, serving as an end-to-end AI system for **drug discovery**, **target validation**, and **repurposing studies**.

---

## 2. Highlights

1. **Hypergraph Representation × Self-Supervised Learning**  
   - Encodes complex molecular relations (bonds, rings, functional groups, hydrogen bonds) using **hypergraphs**, overcoming the binary-edge limitation of traditional molecular graphs.  
   - Employs an **enhanced masked autoencoder (HyperGraph-MAE)** with attention aggregation to capture high-order interactions.

2. **Protein Semantic Modeling**  
   - Utilizes **ProtBert** for deep contextual protein embeddings.  
   - Jointly optimizes drug and protein representations for robust DTI scoring.

3. **End-to-End Inference Pipeline**  
   - Supports single prediction and **CSV batch processing**.  
   - Includes thresholding, calibration (Platt/Isotonic), and metric outputs for large-scale virtual screening.

4. **Interpretability**  
   - Dual-level explanations:  
     - **Atom-level:** Hypergraph node SHAP weights  
     - **Residue-level:** Leave-one-out masking  
   - Outputs **Top-K critical sites** with reliability consistency metrics.

5. **Knowledge Integration & Visualization**  
   - Cross-links with **PubChem**, **UniProt**, **AlphaFold**, and **RCSB PDB** databases.  
   - Provides high-fidelity 3D rendering via **3Dmol.js** with interactive highlighting of key atoms/residues.

---

## 3. Model Overview

| Component | Description |
|:--|:--|
| **Drug Encoder** | HyperGraph-MAE learns molecular embeddings from hypergraph topologies |
| **Protein Encoder** | ProtBert generates amino-acid embeddings (mean pooling or CLS token) |
| **Fusion & Predictor** | Concatenated `[drug | protein]` vectors fed into **XGBoost** for classification or regression outputs |

---

## 4. Capabilities

- **DTI Prediction:** Binary or regression inference with optional calibration.  
- **Explainability:** Atom- and residue-level contributions with SHAP and masking.  
- **High-Throughput Screening:** Batch-mode CSV inference with structured outputs.  
- **3D Visualization:** Automatic rendering of available structures; fallback alerts for missing conformations.  
- **Applications:** Drug discovery, repositioning, target validation, and off-target analysis.

---

## 5. Quick Start

1. Navigate to the **Prediction** page.  
2. Enter a **SMILES** string and a **protein FASTA** sequence.  
3. Enable “Atom Contribution” or “Residue Contribution” (optional).  
4. Adjust **Top-K** in *Advanced Settings* for interpretability depth.  
5. Submit the job and explore interactive 3D visualization with heatmap overlays.  
6. Export structured reports and CSV result tables.

---

## 6. Use Cases

- **Lead discovery & optimization**  
- **Drug repurposing** for new indications  
- **Target validation** and selectivity profiling  
- **Safety assessment** and off-target interaction analysis  

---

## 7. Technical Stack

- **Backend:** PyTorch · Torch-Geometric · RDKit · XGBoost  
- **Frontend:** Vue.js · 3Dmol.js  
- **Core Models:** HyperGraph-MAE · ProtBert · SHAP · Attention Aggregator  
- **Deployment:** Flask / FastAPI · Docker  

---

## 8. Latest Update (V8.1)

- Refined **frontend interactivity** and **3D rendering stability**  
- Added **Top-K key-site visualization** and **form validation prompts**  
- Optimized **batch prediction interface** and result exporting  

---

## 9. Citation & Acknowledgements

This platform acknowledges the contributions of the open-source community, including:  
`RDKit`, `PyTorch`, `torch-geometric`, `Transformers`, `XGBoost`, `3Dmol.js`, and `SHAP`.

> © 2025 MolHuiTu (Molecular HyperGraph) Team ·  
> Software Copyright Registration No. **2025SR1938362**  
> Version **V8.1.2025** · Developed by **XY-Lab**

---


📌 *“Bridging AI and molecular science — towards interpretable and trustworthy drug discovery.”*
