#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTI 预测后端（前后端一体化单端口版）
- 修复：静态挂载顺序导致 /api/submit_task 返回 405 的问题
- 特性：主页/静态文件 + API 同时由 5050 端口提供
- 启动：conda activate molhuitu && uvicorn server:app --host 0.0.0.0 --port 5050
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from multiprocessing import Process, Manager
from typing import Dict, Any
from pathlib import Path
import tempfile, json, subprocess, os, uuid, time, psutil, logging

# -----------------------------
# 基本路径
# -----------------------------
HERE = Path(__file__).resolve().parent                           # ~/hypergraph-mae/web_frontend/dti_web_2.2
WEB_ROOT = HERE                                                  # 前端页面目录
REPO_ROOT = Path(__file__).resolve().parents[2]                  # ~/hypergraph-mae
SCRIPT_PATH = REPO_ROOT / "scripts" / "dti_e2e_predict.py"       # 绝对路径，避免找不到脚本

# -----------------------------
# 日志
# -----------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("dti-server")
log.info(f"WEB_ROOT={WEB_ROOT}")
log.info(f"REPO_ROOT={REPO_ROOT}")
log.info(f"SCRIPT_PATH={SCRIPT_PATH}")

# -----------------------------
# 任务管理
# -----------------------------
manager = Manager()
tasks: Dict[str, Dict[str, Any]] = manager.dict()

app = FastAPI(title="DTI Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------
# 静态可视化产物目录
# -----------------------------
STATIC_DIR = REPO_ROOT / "outputs" / "web" / "static"
STATIC_VIZ_DIR = STATIC_DIR / "viz"
STATIC_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 默认配置（已按你的机器路径）
# -----------------------------
DEFAULTS = {
    "xgb_model": "/home/dc_zhuzy/hypergraph-mae/xgbout/davisreg_xgb.json",
    "hg_ckpt": "/home/dc_zhuzy/hypergraph-mae/hydra/version2/outputs/max_full_baseline/pretrain_with_delta_20250919_175430/checkpoints/checkpoint_step_1500.pth",
    "hg_config": "/home/dc_zhuzy/hypergraph-mae/hydra/version2/outputs/max_full_baseline/pretrain_with_delta_20250919_175430/config.json",
    "protbert_model": "/home/dc_zhuzy/hypergraph-mae/protbert_model",
    "device": "cuda",
    "shap_out_dir": "/home/dc_zhuzy/hypergraph-mae/outputs/dtishap/",
    "task": "regression",
}

# 固化 XGB 参数
BEST_XGB_PARAMS = {
    "eta": 0.03401282905782875,
    "max_depth": 8,
    "subsample": 0.7746385311052707,
    "colsample_bytree": 0.7793796685010081,
    "min_child_weight": 4.51374104342547,
    "lambda": 1.7397074141778888e-06,
    "alpha": 0.0008849299239384577,
    "gamma": 0.0001865112880276377,
}

# -----------------------------
# 请求体
# -----------------------------
class PredictReq(BaseModel):
    smiles: str
    fasta: str
    explain_atoms: bool = True
    explain_residues: bool = True
    shap_background_strategy: str = "mix"
    background: int = 5
    nsamples: int = 10
    shap_topk: int = 10
    residue_explainer: str = "occlusion"
    residue_max: int = 512
    residue_stride: int = 1

# -----------------------------
# 辅助函数
# -----------------------------
def fasta_to_sequence(fasta_text: str) -> str:
    lines = [ln.strip() for ln in fasta_text.strip().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("FASTA 必须以 '>' 开头")
    if any(ln.startswith(">") for ln in lines[1:]):
        raise ValueError("暂不支持多 FASTA")
    seq = "".join(lines[1:])
    if not seq:
        raise ValueError("FASTA 序列为空")
    return seq

def kill_proc_tree(pid: int):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

# -----------------------------
# 子进程：执行预测
# -----------------------------
def run_prediction_task(task_id: str, cfg: Dict[str, Any], shared_dict: Dict):
    shared_dict[task_id] = {
        "status": "RUNNING", "pid": os.getpid(), "start_time": time.time(),
        "result": None, "error": None,
    }
    try:
        out_dir = tempfile.mkdtemp(prefix="dti_pred_")
        out_path = os.path.join(out_dir, f"pred_{uuid.uuid4().hex}.json")
        shap_out_path = os.path.join(DEFAULTS["shap_out_dir"], f"explain_{uuid.uuid4().hex}.json")
        os.makedirs(DEFAULTS["shap_out_dir"], exist_ok=True)

        # 绝对路径 + 在仓库根目录执行，避免路径问题
        cmd = [
            "python", str(SCRIPT_PATH),
            "--smiles", cfg["smiles"],
            "--sequence", cfg["sequence"],
            "--xgb_model", DEFAULTS["xgb_model"],
            "--hg_ckpt", DEFAULTS["hg_ckpt"],
            "--hg_config", DEFAULTS["hg_config"],
            "--protbert_model", DEFAULTS["protbert_model"],
            "--device", DEFAULTS["device"],
            "--output", out_path,
            "--task", DEFAULTS["task"],
            # XGB 参数
            "--xgb_lr", str(BEST_XGB_PARAMS["eta"]),
            "--xgb_max_depth", str(BEST_XGB_PARAMS["max_depth"]),
            "--xgb_subsample", str(BEST_XGB_PARAMS["subsample"]),
            "--xgb_colsample", str(BEST_XGB_PARAMS["colsample_bytree"]),
            "--xgb_min_child_weight", str(BEST_XGB_PARAMS["min_child_weight"]),
            "--xgb_reg_lambda", str(BEST_XGB_PARAMS["lambda"]),
            "--xgb_reg_alpha", str(BEST_XGB_PARAMS["alpha"]),
            "--xgb_gamma", str(BEST_XGB_PARAMS["gamma"]),
        ]

        is_explain = cfg.get("explain_atoms") or cfg.get("explain_residues")
        if is_explain:
            cmd += [
                "--shap_out", shap_out_path,
                "--shap_background_strategy", cfg["shap_background_strategy"],
                "--background", str(cfg["background"]),
                "--nsamples", str(cfg["nsamples"]),
                "--shap_topk", str(cfg["shap_topk"]),
            ]
            if cfg.get("explain_atoms"):
                atoms_png_path = STATIC_VIZ_DIR / f"{task_id}_atoms.png"
                cmd += ["--explain_atoms", "--viz_atoms_png", str(atoms_png_path)]
            if cfg.get("explain_residues"):
                cmd += [
                    "--explain_residues",
                    "--residue_explainer", cfg["residue_explainer"],
                    "--residue_max", str(cfg["residue_max"]),
                    "--residue_stride", str(cfg["residue_stride"]),
                ]

        log.info(f"[TASK {task_id}] CWD={REPO_ROOT}")
        log.info(f"[TASK {task_id}] CMD={' '.join(cmd)}")

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            log.error(f"[TASK {task_id}] stderr tail:\n{proc.stderr[-2000:]}")
            raise RuntimeError(f"模型执行失败：returncode={proc.returncode}")

        if not os.path.exists(out_path):
            raise FileNotFoundError(f"输出文件缺失: {out_path}")

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        explanations_json = None
        if is_explain and os.path.exists(shap_out_path):
            with open(shap_out_path, "r", encoding="utf-8") as ef:
                explanations_json = json.load(ef)

        shared_dict[task_id] = {
            "status": "SUCCESS",
            "result": {"output_json": data, "explanations_json": explanations_json},
            "error": None,
        }
    except Exception as e:
        shared_dict[task_id] = {"status": "FAILURE", "result": None, "error": str(e)}

# -----------------------------
# API 路由（注意：先注册 API，最后再挂静态！）
# -----------------------------
@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.post("/api/submit_task", status_code=202)
def submit_task(req: PredictReq):
    try:
        seq = fasta_to_sequence(req.fasta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task_id = str(uuid.uuid4())
    cfg = req.dict()
    cfg["sequence"] = seq

    process = Process(target=run_prediction_task, args=(task_id, cfg, tasks))
    process.start()
    tasks[task_id] = {"status": "SUBMITTED", "pid": process.pid}
    log.info(f"[TASK {task_id}] submitted, pid={process.pid}")
    return {"task_id": task_id}

@app.get("/api/task_status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return tasks.get(task_id)

@app.post("/api/cancel_task/{task_id}", status_code=200)
def cancel_task(task_id: str):
    info = tasks.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="Task ID not found")
    if info.get("status") in ["SUCCESS", "FAILURE", "CANCELLED"]:
        return {"message": "Task 已完成或已取消"}
    pid = info.get("pid")
    if pid:
        kill_proc_tree(pid)
        tasks[task_id] = {"status": "CANCELLED", "result": None, "error": "用户手动取消"}
        log.info(f"[TASK {task_id}] cancelled")
        return {"message": f"任务 {task_id} 已取消"}
    raise HTTPException(status_code=404, detail="未找到任务进程")

# -----------------------------
# 页面与静态资源（最后挂载，避免覆盖 /api/*）
# -----------------------------
# 首页
@app.get("/", response_class=FileResponse)
def serve_home():
    return WEB_ROOT / "homepage_index.html"

# 直接挂载输出静态资源（图片等）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 将整个前端目录作为静态目录挂载到根路径（放在最后，避免覆盖 API）
# 注意：因为 /api/* 已经在前面注册，Starlette 会优先匹配 API，再回退到静态文件
app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="frontend")

