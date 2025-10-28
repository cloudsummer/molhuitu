# molhuitu

[Update 25.10.28]

Molecular HyperGraph: A Drug–Target Interaction Prediction Platform Based on a Hypergraph Masked Autoencoder

<img width="3548" height="1652" alt="fig1" src="https://github.com/user-attachments/assets/0bf60f5b-a63f-4708-9910-d043bc655497" />


# 🧬 Molecular HyperGraph V8.1 — Intelligent Drug–Target Interaction (DTI) Prediction Platform

**Molecular HyperGraph** is an intelligent **drug–target interaction (DTI) prediction platform** powered by **hypergraph masked autoencoders** and **protein language models**.  
By simply inputting a **SMILES string** and a **protein FASTA sequence**, users can instantly obtain:
- Predicted **DTI probability or regression score**
- **Explainable key atoms/residues** contributing to binding
- Interactive **3D visualization** of molecular and protein structures  
This platform supports both **single-sample inference** and **high-throughput virtual screening**, accelerating **drug discovery** and **target validation**.

---
## 🚀 Highlights

1. **Hypergraph Representation × Self-Supervised Learning**  
   - Represents molecular higher-order relations (bonds, rings, functional groups, hydrogen bonds) via **hypergraphs**, surpassing binary edge limitations of standard molecular graphs.  
   - Enhanced **masked autoencoder (MAE)** and **attention aggregation** modules capture multi-body molecular interactions with high precision.

2. **Protein Semantic Representation**  
   - Leverages **ProtBert** for amino-acid sequence embedding.  
   - Joint drug–protein representation enables accurate DTI scoring.

3. **End-to-End Inference**  
   - Instant single-sample prediction.  
   - Supports **CSV batch mode**, configurable thresholds, and evaluation metrics — ideal for large-scale screening pipelines.

4. **Explainability**  
   - Two-level interpretability:  
     - **Atom-level** (Hypergraph node SHAP weights)  
     - **Residue-level** (Leave-one-out masking)  
   - Outputs **Top-K key sites** with consistency metrics for reliability assessment.

5. **Integrated Knowledge & Visualization**  
   - Built-in data linking and visualization with **PubChem**, **UniProt**, **AlphaFold**, and **RCSB PDB** via interactive **3Dmol.js** rendering.

---
## 🧠 Model Overview

| Component | Description |
|:--|:--|
| **Drug Encoder** | HyperGraph-MAE generates molecular embeddings from hypergraph topology |
| **Protein Encoder** | ProtBert produces sequence embeddings (mean pooling or CLS token) |
| **Fusion & Predictor** | Concatenated [drug \| protein] vector fed into **XGBoost** for DTI score or probability output |

---

## ⚡ Capabilities

- **DTI Prediction:** Binary or regression scoring with optional calibration (Platt / Isotonic).  
- **Explainability:** Atom- and residue-level importance via KernelSHAP and leave-one masking.  
- **High-Throughput Screening:** Batch CSV inference, exporting ranked results and statistics.  
- **3D Visualization:** Automatic rendering of molecular and protein structures (or fallback messages if unavailable).  
- **Applications:** Drug discovery, repositioning, target validation, and off-target toxicity assessment.

---

## 🧩 Quick Start

1. Navigate to the **“Prediction”** page.  
2. Input a **SMILES** string and a **protein FASTA** sequence.  
3. Optionally enable “Atom Contribution” or “Residue Contribution” and adjust **Top-K** settings.  
4. Submit and explore the interactive 3D visualization and heat-mapped contributions.  
5. Export detailed reports or CSV result tables.

---

## 🧭 Use Cases

- **Lead discovery & optimization**  
- **Drug repositioning** (new indications for known compounds)  
- **Target validation** (binding selectivity & druggability)  
- **Safety profiling** (off-target interaction risk)

---

## 🧱 Technical Stack

- **Backend:** PyTorch, Torch-Geometric, RDKit, XGBoost  
- **Frontend:** Vue.js + 3Dmol.js  
- **Model Components:** HyperGraph-MAE, ProtBert, SHAP, Attention Aggregator  
- **Deployment:** Flask / FastAPI + Docker

---

## 📅 Latest Update (V8.1, 2025)

- Enhanced **frontend interactivity** and **3D rendering robustness**  
- Added **Top-K key-site visualization** and **form validation prompts**  

---

## 🧾 Citation & Acknowledgements

This platform builds upon and acknowledges contributions from the open-source community, including but not limited to:  
`RDKit`, `PyTorch`, `torch-geometric`, `Transformers`, `XGBoost`, `3Dmol.js`, and `SHAP`.

> © 2025 Molecular HyperGraph Team · Software Copyright Registration No. **2025SR1938362** · Version: V8.1.2025 by: XY-lab
