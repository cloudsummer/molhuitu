# server.py (MODIFIED FOR ASYNC TASK MANAGEMENT, CANCELLATION, AND ARGUMENT BUG FIX)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import tempfile, json, subprocess, os, uuid, re, base64
from multiprocessing import Process, Manager
from typing import Dict, Any
import time
import psutil # 需要安装: pip install psutil
from pathlib import Path

# ---- 任务管理 ----
manager = Manager()
tasks: Dict[str, Dict[str, Any]] = manager.dict()

app = FastAPI(title="DTI Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- 静态文件挂载（用于图片等可直接通过 URL 访问）----
REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / 'outputs' / 'web' / 'static'
STATIC_VIZ_DIR = STATIC_DIR / 'viz'
STATIC_VIZ_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---- 默认配置 ----
DEFAULTS = {
    "xgb_model": "/home/cpua212/Code/zyzhu/project3/hypergraph-mae/xgbout/davisreg_xgb.json",
    "hg_ckpt": "/home/cpua212/Code/zyzhu/project3/hypergraph-mae/hydra/version2/outputs/max_full_baseline/pretrain_with_delta_20250919_175430/checkpoints/checkpoint_step_1500.pth",
    "hg_config": "/home/cpua212/Code/zyzhu/project3/hypergraph-mae/hydra/version2/outputs/max_full_baseline/pretrain_with_delta_20250919_175430/config.json",
    "protbert_model": "/home/cpua212/Code/zyzhu/project3/protbert_model",
    "device": "cuda",
    "shap_out_dir": "/home/cpua212/Code/zyzhu/project3/hypergraph-mae/outputs/dtishap/",
    "task": "regression",
}

# 固化一次 Optuna 得到的最优 XGB 超参
BEST_XGB_PARAMS = {
    "eta": 0.03401282905782875,
    "max_depth": 8,
    "subsample": 0.7746385311052707,
    "colsample_bytree": 0.7793796685010081, # BUG来源1
    "min_child_weight": 4.51374104342547,
    "lambda": 1.7397074141778888e-06,       # BUG来源2
    "alpha": 0.0008849299239384577,        # BUG来源3
    "gamma": 0.0001865112880276377,
}

# ---- API 模型 ----
class PredictReq(BaseModel):
    smiles: str
    fasta: str
    explain_atoms: bool = True
    explain_residues: bool = True
    shap_background_strategy: str = 'mix'
    background: int = 5
    nsamples: int = 10
    shap_topk: int = 10
    residue_explainer: str = 'occlusion'
    residue_max: int = 512
    residue_stride: int = 1

# ---- Helper Functions ----
def fasta_to_sequence(fasta_text: str) -> str:
    lines = [ln.strip() for ln in fasta_text.strip().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(">"): raise ValueError("FASTA 必须以 '>' 开头")
    if any(ln.startswith(">") for ln in lines[1:]): raise ValueError("不支持多 FASTA")
    seq = "".join(lines[1:])
    if not seq: raise ValueError("FASTA 序列为空")
    return seq

def kill_proc_tree(pid: int):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

# ---- 核心任务执行函数 (将在子进程中运行) ----
def run_prediction_task(task_id: str, cfg: Dict[str, Any], shared_dict: Dict):
    shared_dict[task_id] = {"status": "RUNNING", "pid": os.getpid(), "start_time": time.time(), "result": None, "error": None}
    
    try:
        out_dir = tempfile.mkdtemp(prefix="dti_pred_")
        out_path = os.path.join(out_dir, f"pred_{uuid.uuid4().hex}.json")
        shap_out_path = os.path.join(DEFAULTS["shap_out_dir"], f"explain_{uuid.uuid4().hex}.json")
        os.makedirs(DEFAULTS["shap_out_dir"], exist_ok=True)

        cmd = [
            "python", "scripts/dti_e2e_predict.py",
            "--smiles", cfg['smiles'], "--sequence", cfg['sequence'],
            "--xgb_model", DEFAULTS["xgb_model"], "--hg_ckpt", DEFAULTS["hg_ckpt"],
            "--hg_config", DEFAULTS["hg_config"], "--protbert_model", DEFAULTS["protbert_model"],
            "--device", DEFAULTS["device"], "--output", out_path, "--task", DEFAULTS["task"]
        ]

        # --- BUG FIX for XGB PARAMS ---
        # 显式映射字典键到正确的命令行参数
        cmd.extend(["--xgb_lr", str(BEST_XGB_PARAMS["eta"])])
        cmd.extend(["--xgb_max_depth", str(BEST_XGB_PARAMS["max_depth"])])
        cmd.extend(["--xgb_subsample", str(BEST_XGB_PARAMS["subsample"])])
        cmd.extend(["--xgb_colsample", str(BEST_XGB_PARAMS["colsample_bytree"])]) # Script expects --xgb_colsample
        cmd.extend(["--xgb_min_child_weight", str(BEST_XGB_PARAMS["min_child_weight"])])
        cmd.extend(["--xgb_reg_lambda", str(BEST_XGB_PARAMS["lambda"])]) # Script expects --xgb_reg_lambda
        cmd.extend(["--xgb_reg_alpha", str(BEST_XGB_PARAMS["alpha"])]) # Script expects --xgb_reg_alpha
        cmd.extend(["--xgb_gamma", str(BEST_XGB_PARAMS["gamma"])])

        is_explain_enabled = cfg.get("explain_atoms") or cfg.get("explain_residues")
        
        if is_explain_enabled:
            cmd.extend(["--shap_out", shap_out_path])
            cmd.extend(["--shap_background_strategy", cfg["shap_background_strategy"]])
            cmd.extend(["--shap_topk", str(cfg["shap_topk"])])
            
            # --- BUG FIX for SHAP PARAMS ---
            cmd.extend(["--background", str(cfg["background"])]) # Script expects --background
            cmd.extend(["--nsamples", str(cfg["nsamples"])]) # Script expects --nsamples

            if cfg.get("explain_atoms"):
                # 将原子热力图输出到静态目录，返回 URL
                atoms_png_path = STATIC_VIZ_DIR / f"{task_id}_atoms.png"
                cmd.extend(["--explain_atoms", "--viz_atoms_png", str(atoms_png_path)])
            
            if cfg.get("explain_residues"):
                cmd.append("--explain_residues")
                cmd.extend(["--residue_explainer", cfg["residue_explainer"]])
                cmd.extend(["--residue_max", str(cfg["residue_max"])])
                cmd.extend(["--residue_stride", str(cfg["residue_stride"])])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0:
            raise RuntimeError(f"脚本执行失败（returncode={proc.returncode}）。stderr 末尾：\n{proc.stderr[-2000:]}")

        if not os.path.exists(out_path):
            raise FileNotFoundError(f"未找到输出文件: {out_path}。stdout 末尾: {proc.stdout[-1000:]}")

        with open(out_path, "r", encoding="utf-8") as f: data = json.load(f)

        # 返回静态 URL（优先），保留 base64 作为兼容（若需要）
        if cfg.get("explain_atoms") and 'atoms_png_path' in locals() and atoms_png_path.exists():
            atoms_url = f"/static/viz/{atoms_png_path.name}"
            atoms_viz = data.setdefault('viz', {}).setdefault('atoms', {})
            atoms_viz['png_url'] = atoms_url
            # 如需同时返回 base64，可解除以下注释（默认KISS不返回，避免响应过大）
            # with open(atoms_png_path, "rb") as pf:
            #     b64 = base64.b64encode(pf.read()).decode("ascii")
            # atoms_viz['png_b64'] = f"data:image/png;base64,{b64}"

        explanations_json = None
        if is_explain_enabled and os.path.exists(shap_out_path):
            with open(shap_out_path, 'r', encoding='utf-8') as ef: explanations_json = json.load(ef)
        
        result = {"output_json": data, "explanations_json": explanations_json}
        shared_dict[task_id] = {"status": "SUCCESS", "result": result, "error": None}

    except Exception as e:
        shared_dict[task_id] = {"status": "FAILURE", "result": None, "error": str(e)}

# ---- API Endpoints (no changes here, they are already correct) ----
@app.post("/api/submit_task", status_code=202)
def submit_task(req: PredictReq):
    try:
        sequence = fasta_to_sequence(req.fasta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    task_id = str(uuid.uuid4())
    cfg = req.dict()
    cfg["sequence"] = sequence
    
    process = Process(target=run_prediction_task, args=(task_id, cfg, tasks))
    process.start()
    
    tasks[task_id] = {"status": "SUBMITTED", "pid": process.pid}
    return {"task_id": task_id}

@app.get("/api/task_status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return tasks.get(task_id)

@app.post("/api/cancel_task/{task_id}", status_code=200)
def cancel_task(task_id: str):
    task_info = tasks.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task ID not found")

    if task_info.get("status") in ["SUCCESS", "FAILURE", "CANCELLED"]:
        return {"message": "Task has already completed or been cancelled."}

    pid = task_info.get("pid")
    if pid:
        kill_proc_tree(pid)
        tasks[task_id] = {"status": "CANCELLED", "result": None, "error": "Task was cancelled by user."}
        return {"message": f"Task {task_id} cancellation request sent."}
    
    raise HTTPException(status_code=404, detail="Task process not found, cannot cancel.")
