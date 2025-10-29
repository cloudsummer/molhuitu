# 🧬 MolHuiTu — Molecular HyperGraph V8.1
### Intelligent Drug–Target Interaction (DTI) Prediction Platform

> *A next-generation AI platform for interpretable drug–target interaction prediction based on hypergraph masked autoencoders.*
原理流程示意图：
<img width="3548" height="1652" alt="MolHuiTu Overview" src="https://github.com/user-attachments/assets/0bf60f5b-a63f-4708-9910-d043bc655497" />
分子绘图前端界面主页：
<img width="3172" height="1582" alt="image" src="https://github.com/user-attachments/assets/185000bc-4b54-4178-81e6-f7050db1f3cf" />

药物靶点相互作用预测：
<img width="2814" height="1492" alt="image" src="https://github.com/user-attachments/assets/b400ecd4-d50b-4c79-9718-c95243c61ac3" />

批量任务提交示例：
<img width="1416" height="394" alt="image" src="https://github.com/user-attachments/assets/1a96e41f-dd1a-4231-8f16-be7045243fd4" />

单次预测界面：
<img width="2646" height="1404" alt="image" src="https://github.com/user-attachments/assets/45176290-8fe9-4349-95f0-428bec62b5da" />

单次预测等待：
<img width="2248" height="868" alt="image" src="https://github.com/user-attachments/assets/f356af59-b01f-4e61-84ba-44877d8b384f" />

批量任务运行完成示例：
<img width="2920" height="752" alt="image" src="https://github.com/user-attachments/assets/371e45a6-ef83-43d2-a9b2-6675680ccb30" />

药物靶点相互作用预测报告1：
<img width="2870" height="1250" alt="image" src="https://github.com/user-attachments/assets/73de69af-97b0-49d7-a709-ba364b5899c9" />


药物靶点相互作用预测报告2：
<img width="1064" height="568" alt="image" src="https://github.com/user-attachments/assets/914e7efe-a150-4e7d-8411-cf0d78e0cb7e" />

运行的时候可以关注cpu和gpu的占用：
<img width="2026" height="600" alt="image" src="https://github.com/user-attachments/assets/184e0fc4-4d8a-498f-a039-9d8e0f3e7b99" />
<img width="594" height="288" alt="image" src="https://github.com/user-attachments/assets/af0e8d3c-aad1-43c9-951c-e161d0fac141" />


---

## 🗂️ Table of Contents
- [1. Overview](#1-overview)
- [2. Highlights](#2-highlights)
- [3. Model Overview](#3-model-overview)
- [4. Capabilities](#4-capabilities)
- [5. Quick Start](#5-quick-start)
- [6. Use Cases](#6-use-cases)
- [7. Technical Stack](#7-technical-stack)
- [8. Latest Update (V8.1)](#8-latest-update-v81)
- [9. Architecture](#9-architecture)
- [10. Repository Layout](#10-repository-layout)
- [11. Installation & Environment](#11-installation--environment)
- [12. Model Assets](#12-model-assets)
- [13. Configuration](#13-configuration)
- [14. Running the Web App](#14-running-the-web-app)
- [15. API Reference](#15-api-reference)
- [16. CLI Usage (Single & Batch)](#16-cli-usage-single--batch)
- [17. Explainability Outputs](#17-explainability-outputs)
- [18. 3Dmol (Offline-Friendly)](#18-3dmol-offline-friendly)
- [19. Reverse Proxy / FRP / Same-Origin](#19-reverse-proxy--frp--same-origin)
- [20. Performance Tuning](#20-performance-tuning)
- [21. Troubleshooting](#21-troubleshooting)
- [22. Hardware & Platform Validation](#22-hardware--platform-validation)
- [23. Security Notes](#23-security-notes)
- [24. License](#24-license)
- [25. Citation & Acknowledgements](#25-citation--acknowledgements)
- [Appendix — systemd Service (Linux)](#appendix--systemd-service-linux)

---

## 1. Overview

**MolHuiTu (Molecular HyperGraph)** is an **intelligent DTI platform** unifying **hypergraph self-supervised learning** (for small molecules) and **protein language modeling** (ProtBert).

Provide a **SMILES** string and a **protein FASTA** sequence to obtain:

- Predicted **DTI probability** or **regression score**
- **Explainable** key atoms/residues that drive binding
- Interactive **3D visualization** for structural interpretation

Supports **single-sample prediction** and **high-throughput screening** for **drug discovery**, **target validation**, and **repurposing**.

---

## 2. Highlights

1. **Hypergraph Representation × Self-Supervised Learning**  
   Encodes rings, functional groups, and higher-order relations via **hypergraphs**, overcoming binary-edge limitations. **HyperGraph-MAE** captures long-range, high-order interactions.

2. **Protein Semantic Modeling**  
   **ProtBert** produces deep contextual embeddings (mean/CLS pooling).

3. **End-to-End Inference**  
   Single prediction + **CSV batch** processing, optional calibration (Platt/Isotonic).

4. **Interpretability**  
   **Atom-level** SHAP (hypergraph nodes, RDKit heatmaps) and **residue-level** occlusion/KernelSHAP with Top-K reporting and reliability checks.

5. **Knowledge Integration & Visualization**  
   Integrates **PubChem**, **UniProt**, **AlphaFold**, **RCSB PDB** (best-effort).  
   **3Dmol.js** for high-fidelity, interactive structure rendering.

---

## 3. Model Overview

| Component | Description |
|:--|:--|
| **Drug Encoder** | HyperGraph-MAE learns embeddings from molecular **hypergraphs** |
| **Protein Encoder** | **ProtBert** generates sequence embeddings (mean or CLS pooling) |
| **Fusion & Predictor** | Concatenate `[drug \| protein]` → **XGBoost** (binary/regression) |

---

## 4. Capabilities

- **DTI Prediction**: binary or regression, optional calibration  
- **Explainability**: atom & residue contributions (SHAP / occlusion)  
- **High-Throughput**: batch CSV inference with JSON/CSV outputs  
- **3D Visualization**: automatic molecule/protein viewers with highlights  
- **Applications**: lead discovery, repurposing, target validation, off-target analysis

---

## 5. Quick Start

**Web UI**
1. Open the **Prediction** page.  
2. Paste **SMILES** and **FASTA** (or upload CSV in batch mode).  
3. Toggle **Atom**/**Residue** explanations (optional).  
4. Adjust **Top-K** and SHAP parameters in *Advanced Settings*.  
5. Submit and view the interactive report (3D viewers + heatmaps).  
6. Export structured JSON/CSV; view explainability artifacts.

**CLI**  
Run a batch CSV or a single pair from the terminal (see [16. CLI Usage](#16-cli-usage-single--batch)).

---

## 6. Use Cases

- **Lead discovery & optimization**  
- **Drug repurposing**  
- **Target validation** & selectivity profiling  
- **Safety assessment** / off-target interaction analysis

---

## 7. Technical Stack

- **Backend**: **FastAPI** (Uvicorn), Python 3.9–3.11  
- **Core ML**: PyTorch, RDKit, XGBoost, Transformers (ProtBert), SHAP  
- **Frontend**: Static HTML/JS (Tailwind CSS), **3Dmol.js**  
- **Packaging/Deploy**: Conda, optional Docker/FRP/Nginx

> The web UI is **static HTML + JS** served by the same FastAPI process (no SPA framework required).

---

## 8. Latest Update (V8.1)

- Stabilized **3D rendering** and local-first 3Dmol loading with CDN fallback  
- **Top-K** key-site UX & form validation improvements  
- Optimized **batch prediction** interface and result exporting  
- Hardened **same-origin** defaults to avoid cross-cluster API calls

---

## 9. Architecture
