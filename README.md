# 🧬 MolHuiTu — Molecular HyperGraph V8.1
**Intelligent Drug–Target Interaction (DTI) Prediction Platform**

> _A next-generation, GPU-accelerated DTI system that fuses **hypergraph** molecular encoders, **protein language models** (ProtBert), rich **explainability**, and a sleek web UI._

<p align="center">
  <img width="100%" alt="MolHuiTu Overview" src="https://github.com/user-attachments/assets/0bf60f5b-a63f-4708-9910-d043bc655497" />
</p>

---

## 🗂️ Table of Contents
- [1. Overview](#1-overview)
- [2. Feature Highlights](#2-feature-highlights)
- [3. Guided Tour (with Screenshots)](#3-guided-tour-with-screenshots)
- [4. Architecture](#4-architecture)
- [5. Prerequisites](#5-prerequisites)
- [6. Installation (Ubuntu 24.04 + Conda + RTX 4090)](#6-installation-ubuntu-2404--conda--rtx-4090)
- [7. Repository Layout](#7-repository-layout)
- [8. Quick Start](#8-quick-start)
- [9. CLI — Single & Batch Prediction](#9-cli--single--batch-prediction)
- [10. Explainability (Technical)](#10-explainability-technical)
- [11. 3D Viewer & Offline/CDN Fallback](#11-3d-viewer--offlinecdn-fallback)
- [12. Performance & Monitoring](#12-performance--monitoring)
- [13. Run as a Service (systemd)](#13-run-as-a-service-systemd)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Security & Production Notes](#15-security--production-notes)
- [16. License](#16-license)
- [17. Acknowledgements](#17-acknowledgements)
- [Appendix — Full CLI Arguments](#appendix--full-cli-arguments)

---

## 1. Overview
**MolHuiTu** (Molecular Intelligence Graph) predicts drug–target interactions from a **SMILES** (ligand) and a **FASTA** (protein). It returns a calibrated score and **explains** the prediction by highlighting key **atoms** and **residues**. The web UI includes **interactive 3D visualization**, batch job management, and downloadable reports.

---

## 2. Feature Highlights
- **Hypergraph Molecular Encoder** — Captures multi-body patterns (rings, functional groups, H-bonds) beyond pairwise bonds via **hyperedges** and a masked-autoencoder pretrain; improves modeling of complex chemistry.
- **Protein Language Model (ProtBert)** — Transformer embeddings of amino-acid sequences (mean/CLS pooling), fused with ligand embeddings for robust DTI scoring.
- **End-to-End Inference** — Single query and high-throughput **batch CSV** screening; optional probability calibration for deployment realism.
- **Integrated Explainability** — **Atom-level SHAP** and **residue-level occlusion** with **Top-K** contributors and a consistency check.
- **One-Stop Context** — Hooks for PubChem / UniProt / AlphaFold / RCSB PDB to enrich reports and drive **3Dmol.js** visualization.
- **Practical UX** — Clean web UI, job history, CSV export, and report pages; GPU-optimized backend validated on **NVIDIA RTX 4090**.

> _Traditional graph vs hypergraph_: a simple graph restricts bonds to pairs; **MolHuiTu** uses **hyperedges** to connect any number of atoms so functional motifs are represented natively.

---

## 3. Guided Tour (with Screenshots)

## Usage

With the environment set up, you can now run the MolHuiTu application and perform predictions.

### Starting the Backend Server

Start the MolHuiTu backend server which handles the predictions. Depending on how the project is structured, this could be done via a provided launch script or a direct Python command. For example:
```bash
# Example: if there's a script or module to run the web server
python src/app.py
```
Or, if using a framework like FastAPI with Uvicorn:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
After running the above, you should see the server starting up, loading models into memory, and listening on a port (e.g. 8000). Ensure that this port is accessible if you are using a remote server.

### Accessing the Web Interface

Once the backend is running, open a web browser and navigate to the MolHuiTu interface. If you’re running locally, use:  
```
http://localhost:8000
```  
(Adjust the port if needed, and use the server’s IP or domain if accessing remotely.)

You should see the MolHuiTu home page with options for different prediction modes.

**MolHuiTu Drug-Target Prediction Interface:**  
<img width="2814" height="1492" alt="image" src="https://github.com/user-attachments/assets/b400ecd4-d50b-4c79-9718-c95243c61ac3" />  
*_(Screenshot: The interface for Drug-Target Interaction prediction, where users can input a molecule (SMILES or structure file) and a protein sequence or identifier.)_*

#### Single Prediction

For a single DTI prediction:

1. Navigate to the **Single Prediction** section of the interface.
2. Input the required data:  
   - **Drug Molecule:** Provide the molecule, typically as a SMILES string, a MOL file, or an identifier of the compound. The interface may also allow drawing the structure or uploading a file.  
   - **Target Protein:** Provide the protein information, either as an amino acid sequence (FASTA format) or a known identifier (such as a UniProt ID).
3. Click the **Submit** button to start the prediction.

After submission, the task will be sent to the backend for processing. You will see a status indicator for the job.

**Single Prediction Submission Example:**  
<img width="2646" height="1404" alt="image" src="https://github.com/user-attachments/assets/45176290-8fe9-4349-95f0-428bec62b5da" />  
*_(Screenshot: A single prediction entry form filled with a sample drug and target, ready to be submitted.)_*

While the prediction is running, MolHuiTu will indicate that the task is in progress.

**Single Prediction In Progress:**  
<img width="2248" height="868" alt="image" src="https://github.com/user-attachments/assets/f356af59-b01f-4e61-84ba-44877d8b384f" />  
*_(Screenshot: The interface showing a single prediction task in progress. Users are advised to wait as the model computes the results.)_*

Once the prediction is complete, a result report will be available for viewing and download.

#### Batch Prediction

MolHuiTu also supports batch processing, allowing you to run multiple predictions in one go:

1. Go to the **Batch Prediction** section.
2. Prepare an input file (CSV format) with each row representing a drug-target pair. For example, the repository provides a `batch_template.csv` as a template:  
   *Each row might contain columns such as Drug_SMILES (or compound ID) and Target_Sequence (or target ID).*
3. Upload the CSV file through the interface (or as instructed on the page).
4. Submit the batch job. The interface will queue all tasks and start processing them one by one on the backend.

After submitting, you'll see a list of tasks with their statuses (e.g. queued, running, completed).

**Batch Task Submission Example:**  
<img width="1416" height="394" alt="image" src="https://github.com/user-attachments/assets/1a96e41f-dd1a-4231-8f16-be7045243fd4" />  

*_(Screenshot: A batch submission form where a CSV file has been selected for upload.)_*

While the batch is running, you can monitor the progress of each task in real-time. Each job will update its status from "pending" to "running" to "completed" (or "failed" if an issue occurs).

**Batch Tasks Status Dashboard:**  
<img width="2920" height="752" alt="image" src="https://github.com/user-attachments/assets/371e45a6-ef83-43d2-a9b2-6675680ccb30" />  
*_(Screenshot: Batch task list showing multiple tasks and their current status. Completed tasks have results available and links to view reports.)_*

When all tasks are finished, you can review the results for each pair. You may also download a consolidated results file (e.g. a CSV similar to `batch_template.pred.csv` with added prediction outcomes for all input pairs).

### Viewing Results and Reports

For each completed prediction (single or batch), MolHuiTu provides a detailed report. A report typically includes:

- The input details (drug and target, with identifiers or sequence).
- The predicted interaction score or probability (indicating how likely the drug is to interact with the target).
- Visualizations of the molecular structure of the drug (and possibly the target, if structural data is available or relevant).
- Additional information such as confidence metrics, similarity to known compounds, target annotations, etc.

**Example DTI Prediction Report:**  
<img width="2870" height="1250" alt="image" src="https://github.com/user-attachments/assets/73de69af-97b0-49d7-a709-ba364b5899c9" />  
*_(Screenshot: Part of a DTI prediction report, listing the input details and the predicted interaction score among other details.)_*

Large reports may contain multiple sections, possibly including tables of results or interactive components.

**Continuation of DTI Report:**  
<img width="1064" height="568" alt="image" src="https://github.com/user-attachments/assets/914e7efe-a150-4e7d-8411-cf0d78e0cb7e" />  
*_(Screenshot: Another section of the report, possibly showing additional metrics or a summary of results.)_*

Reports can be viewed in the web interface and are also saved to the server (e.g., in the `outputs/` directory) for future reference or downloading.

## 4. Architecture
- **Drug Encoder — HyperGraph-MAE**  
  Represent molecules as **hypergraphs** (nodes = atoms; hyperedges = rings/groups/relations). Pretrain with degree-aware masking and reconstruction; aggregate via multi-head attention → fixed-size ligand embedding.
- **Protein Encoder — ProtBert**  
  Transformer embeddings from **ProtBert** (HuggingFace); mean/CLS pooling configurable → protein embedding.
- **Fusion & Prediction — XGBoost Head**  
  Concatenate (or bilinear fuse) ligand/protein embeddings → **XGBoost** for classification (probability) or regression (affinity). Optional **Platt / Isotonic** calibration improves reliability.

_Backend stack_: PyTorch (+ CUDA), PyTorch Geometric, RDKit, FastAPI/Uvicorn, XGBoost, SHAP, 3Dmol.js (frontend).

---

## 5. Prerequisites
- **OS**: Ubuntu 24.04 LTS (assumed below).
- **GPU**: NVIDIA (tested on **RTX 4090**, ≥24 GB VRAM recommended for SHAP/occlusion).
- **Driver/CUDA**: Recent NVIDIA driver; CUDA 11.8+ or CUDA 12.x supported by your PyTorch build.
- **Conda**: Miniconda/Anaconda for clean, reproducible environments.

---

## 6. Installation (Ubuntu 24.04 + Conda + RTX 4090)

```bash
# 0) Essentials
sudo apt update && sudo apt upgrade -y
sudo apt install -y git

# 1) Clone
git clone https://github.com/yourusername/molhuitu.git
cd molhuitu

# 2) Conda env (Python 3.10)
conda create -n molhuitu python=3.10 -y
conda activate molhuitu

# 3) Channels & core deps
conda config --add channels conda-forge

# RDKit for cheminformatics
conda install -y rdkit

# PyTorch (CUDA 11.8 shown; pick the variant that matches your driver)
conda install -y pytorch torchvision torchtext pytorch-cuda=11.8 -c pytorch -c nvidia

# PyG ops (use pip wheels compatible with your torch/cuda)
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Misc libs
pip install "transformers==4.*" xgboost shap fastapi "uvicorn[standard]" 3dmol

# 4) Project install (editable)
pip install -e .

# 5) Sanity checks
python - <<'PY'
import torch, rdkit
print("CUDA available:", torch.cuda.is_available())
PY
nvidia-smi
```

> **Models**: place ProtBert under `./protbert_model/` (or allow first-run auto-download). Keep HyperGraph-MAE checkpoints in `./hydra/.../checkpoints/` and XGBoost models in `./xgbout/`.

---

## 7. Repository Layout
```
molhuitu/
├─ hydra/                         # configs & training/infer outputs
├─ protbert_model/                # local ProtBert (optional; else auto-download)
├─ scripts/
│   └─ dti_e2e_predict.py         # end-to-end CLI entry
├─ src/                           # Python sources (backend, models, encoders, API)
├─ web_frontend/                  # static web app (HTML/CSS/JS, 3Dmol.js)
├─ xgbout/                        # xgboost heads (.json)
├─ outputs/                       # predictions, reports, SHAP, assets
├─ batch_template.csv             # batch input template
├─ batch_template.pred.csv        # batch output example
├─ transferconda.yml              # optional env recipe
└─ requirements.txt               # optional pip requirements
```

---

## 8. Quick Start

### 8.1 Run the web API (FastAPI + Uvicorn)
```bash
# from project root
uvicorn src.app:app --host 0.0.0.0 --port 8000
# Open: http://<server-ip>:8000
```

> **Same-origin tip**: Serve the static UI from the **same** origin as the API to avoid CORS hassle.

### 8.2 Frontend
Point your static site (e.g., `web_frontend/`) to the API origin above. 3Dmol.js is bundled locally with CDN fallback (see §11).

---

## 9. CLI — Single & Batch Prediction

### 9.1 **Single sample** (reference command you already ran)
```bash
python scripts/dti_e2e_predict.py \
  --smiles 'CCO' \
  --sequence 'ACDEFGHIKLMNPQRSTVWY' \
  --xgb_model xgbout/davisreg_xgb.json \
  --hg_ckpt hydra/version2/outputs/max_full_baseline/pretrain_with_delta_20250919_175430/checkpoints/checkpoint_step_1500.pth \
  --hg_config hydra/version2/outputs/max_full_baseline/pretrain_with_delta_20250919_175430/config.json \
  --protbert_model ./protbert_model \
  --device cuda \
  --task regression \
  --output outputs/pred/pred.json \
  --explain_atoms \
  --explain_residues \
  --shap_background_strategy mix \
  --background 5 \
  --nsamples 10 \
  --shap_topk 10 \
  --shap_out outputs/dtishap/explain.json \
  --residue_explainer occlusion \
  --residue_max 512 \
  --residue_stride 1
```

### 9.2 Batch mode (CSV)
```bash
python scripts/dti_e2e_predict.py \
  --csv batch_template.csv \
  --smiles_col smiles --sequence_col sequence \
  --output_csv outputs/batch_results.pred.csv \
  --threshold 0.5 --skip_invalid
```

---

## 10. Explainability (Technical)
- **Atom-level SHAP (KernelSHAP)**: approximate Shapley values by masking ligand nodes/hyperedges and observing Δscore. High positive SHAP → atom critical for binding. Output includes **Top-K atoms** with contributions; visual overlays (2D/3D) reflect magnitude.
- **Residue-level Occlusion**: leave-one-out masking of residues (or windows with `--residue_stride`) to estimate each position’s importance. Reports **Top-K residues**, typically aligning with pocket residues in 3D.
- **Consistency Check**: optional metric correlating atom hotspots and nearby residue hotspots in 3D; high score suggests stable, geometry-consistent rationale.

---

## 11. 3D Viewer & Offline/CDN Fallback
Frontend uses **3Dmol.js** (protein from AlphaFold/PDB; ligand from MOL/SDF or generated conformers).

```html
<script src="./3Dmol-min.js"
  onerror="(function(){
    var s=document.createElement('script');
    s.src='https://3Dmol.org/build/3Dmol-min.js';
    document.head.appendChild(s);
  })();">
</script>
<!-- your app scripts -->
<script src="./results_ligand_fix.js"></script>
<div id="ligand3dViewer" style="width:100%;height:360px;border:1px solid #eee;border-radius:8px;"></div>
```

> Keep a **local** copy under your static root for intranet use; the `onerror` hook pulls CDN if reachable.

---

## 12. Performance & Monitoring

During heavy runs (esp. SHAP/occlusion), watch CPU/GPU:

<img width="1013" height="300" alt="CPU htop" src="https://github.com/user-attachments/assets/184e0fc4-4d8a-498f-a039-9d8e0f3e7b99" />
<img width="297" height="144" alt="GPU nvidia-smi" src="https://github.com/user-attachments/assets/af0e8d3c-aad1-43c9-951c-e161d0fac141" />

*_(Screenshots: Terminal output of `htop` (top image) showing CPU usage across cores, and `nvidia-smi` (bottom image) showing the GPU (RTX 4090) memory and utilization during a batch inference.)_*

Monitoring these resources can help in understanding performance. For instance, you can verify that the GPU is fully utilized during heavy computations. If the GPU is underutilized, you might consider increasing batch sizes or running multiple tasks in parallel (if supported) to better leverage the hardware.

**Tips**
- Prefer ≥24 GB VRAM for explainability.
- Omit `--explain_*` for fast screening; add only to shortlisted candidates.
- Mixed precision (FP16) can help; validate SHAP stability.
- Keep the API warm to avoid repeated model loads.
- Parallelize cautiously; respect VRAM headroom.
- Use NGINX to serve static assets and reverse-proxy API; enable HTTPS.

---

## 13. Run as a Service (systemd)
Create `/etc/systemd/system/molhuitu.service`:
```ini
[Unit]
Description=MolHuiTu DTI Prediction Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/path/to/molhuitu
ExecStart=/bin/bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate molhuitu && uvicorn src.app:app --host 0.0.0.0 --port 8000'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Enable & start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable molhuitu
sudo systemctl start molhuitu
sudo systemctl status molhuitu --no-pager
```

---

## 14. Troubleshooting
- **Conda resolution issues** → install deps in smaller groups or use `mamba`.
- **RDKit ImportError** → ensure **conda-forge** RDKit (not pip) is used.
- **CUDA unavailable** → check driver; `python -c "import torch; print(torch.cuda.is_available())"`.
- **OOM during explainability** → reduce `--nsamples`, `--background`, `--shap_topk`, or `--residue_max`; disable one explainer.
- **UI can’t fetch 3D** → verify internet/DB access; local ligand fallback; “No available 3D structure” is normal if none exists.
- **Long SHAP runtimes** → fewer samples or approximate settings; precompute explanations offline and cache.

---

## 15. Security & Production Notes
- **Auth**: Demo setups may use static credentials; for production integrate proper auth (tokens/OIDC).
- **CORS**: Prefer serving UI and API on the **same origin**.
- **Rate-limiting & Timeouts**: Configure NGINX/Gunicorn/Uvicorn appropriately for long jobs.
- **Data Privacy**: Avoid uploading proprietary sequences/ligands to external services when fetching structures.

---

## 16. License
This repository is released for **research and evaluation**.  
> **Default stance**: _All rights reserved_ unless a `LICENSE` file is provided.  
If you wish to allow limited reuse while protecting commercial rights, consider a **Non-Commercial** license (e.g., CC BY-NC-SA) or a **copyleft network license** (e.g., AGPL-3.0). Set your final choice in `LICENSE`.

---

## 17. Acknowledgements
Built on **RDKit**, **PyTorch**, **PyG**, **HuggingFace (ProtBert)**, **XGBoost**, **SHAP**, and **3Dmol.js**.  
Software registration: _China National Copyright
Administration_ **2025SR1938362** (MolHuiTu V8.1.2025).  
Thanks to all contributors and the community. **Happy researching!** 🎉Developed by **XY-Lab**

---

## Appendix — Full CLI Arguments

**Basics**
- `--task {binary|regression}`: task type (default `binary`)
- `--device {cuda|cpu}`: compute device (auto if omitted)
- `--output PATH`: JSON output for single-sample prediction

**Model & required assets**
- `--hg_ckpt FILE.pth` **(required)**: HyperGraph-MAE weights
- `--hg_config FILE.(json|yaml)`: HG-MAE config (default project config)
- `--protbert_model NAME|DIR`: ProtBert model ID or local path (default `Rostlab/prot_bert_bfd`)
- `--xgb_model FILE.json`: trained XGBoost head (**required** in predict mode)
- `--timeout_seconds N`: hypergraph construction timeout (seconds)

**Single-sample prediction**
- `--smiles STR`: ligand SMILES (required with `--sequence`)
- `--sequence STR`: protein sequence in FASTA/plain (required with `--smiles`)
- `--pool {mean|max|sum}`: ligand pooling (default `mean`)
- `--prot_pool {mean|cls}`: protein pooling (default `mean`)
- `--no_norm`: disable L2 normalization for ligand embeddings

**Batch CSV prediction**
- `--csv FILE.csv`: input CSV
- `--smiles_col NAME` (default `smiles`)
- `--sequence_col NAME` (default `sequence`)
- `--id_col NAME`: optional identifier column
- `--output_csv OUT.csv` (default `input.pred.csv`)
- `--skip_invalid`: skip malformed rows
- `--label_col NAME`: optional ground-truth labels (0/1) for metrics
- `--threshold FLOAT` (default `0.5`): classification threshold

**Preprocessing (standardization & stats)**
- `--use_preprocess` / `--no_preprocess` (default off)
- `--no_standardize`: disable standardization (only if preprocess on)
- `--keep_metals`: keep metal-containing molecules
- `--max_atoms N` (default `200`): max atoms during standardization
- `--stats_sample_size N` (default `10000`): global stats sampling size

**Train XGBoost (optional)**
- `--train_xgb`: switch to training mode (requires `--csv` and `--label_col`)
- `--xgb_out FILE.json`: save trained XGB head (default `input.csv.xgb.json`)
- `--cv5`: 5-fold CV on training set (ignores `--test_csv`)
- `--drug_emb_parquet FILE.parquet`: precomputed drug embeddings (columns: `smiles`, `emb_*`)
- `--prot_emb_parquet FILE.parquet`: precomputed protein embeddings (columns: `protein`, `emb_*`)
- `--val_ratio FLOAT` (default `0.1`)
- `--test_ratio FLOAT` (default `0.1`; set `0` to disable)
- `--test_csv FILE.csv`: separate test CSV
- `--seed INT` (default `42`)

**XGBoost hyper-parameters**
- `--xgb_lr FLOAT` (default `0.1`)
- `--xgb_n_round INT` (default `1000`)
- `--xgb_early_stopping INT` (default `100`)
- `--xgb_max_depth INT` (default `0`)
- `--xgb_max_leaves INT` (default `1024`; >0 uses `lossguide` and sets depth=0)
- `--xgb_subsample FLOAT` (default `0.8`)
- `--xgb_colsample FLOAT` (default `0.8`)
- `--xgb_max_bin INT` (default `1024`)
- `--xgb_reg_lambda FLOAT` (default `9.0`)
- `--xgb_reg_alpha FLOAT` (default `0.0`)
- `--xgb_min_child_weight FLOAT` (default `7.0`)
- `--xgb_gamma FLOAT` (default `1.0`)
- `--auto_scale_pos_weight`: set `scale_pos_weight=neg/pos` automatically
- `--eval_period INT` (default `1`)

**Optuna hyper-param search (optional)**
- `--optuna`: enable HPO
- `--n_trials INT` (default `30`)
- `--opt_metric {aucpr|auc|mse|rmse|mae|r2}`: optimization target (regression forces `rmse`)
- `--timeout SECONDS`: wall-time cap for HPO

**Probability calibration (optional, during training)**
- `--calibration_method {platt|isotonic}` (default `isotonic`)
- `--calibrate_in_train`: fit calibrator on validation set and apply in-memory

**Explainability / SHAP**
- `--explain_atoms`: atom-level SHAP (KernelSHAP + node masking)
- `--explain_residues`: residue-level (KernelSHAP or occlusion)
- `--prot_occlusion {drop|mask}` (default `drop`)
- `--residue_explainer {kernelshap|occlusion}` (default `occlusion`)
- `--residue_max INT` (default `512`): max residues to analyze (excl. CLS/SEP)
- `--residue_stride INT` (default `1`)
- `--shap_background_strategy {zeros|random_keep|mix}` (default `random_keep`)
- `--background INT` (default `20`): background samples for SHAP
- `--nsamples INT` (default `200`): KernelSHAP sampling budget
- `--shap_topk INT` (default `20`): Top-K contributors to keep
- `--shap_out FILE.json`: JSON output for explanations
- `--shap_batch INT` (default `64`): forward batch for SHAP
- `--viz_atoms_png FILE.png`, `--viz_atoms_svg FILE.svg`: RDKit atom heatmap outputs




















