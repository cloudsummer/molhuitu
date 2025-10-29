#!/usr/bin/env python
"""
End-to-end DTI prediction: SMILES + protein sequence -> interaction score.

Workflow:
  - Drug: build molecular hypergraph from SMILES, get HG-MAE graph embedding
  - Protein: get ProtBert embedding from sequence
  - Head: concatenate [drug_emb | prot_emb] -> XGBoost -> probability

Notes:
  - Requires: rdkit, torch, torch-geometric, transformers, xgboost
  - HG-MAE checkpoint should be trained with config consistent to hydra version2 (e.g., max_full.yaml)
  - ProtBert can be loaded from local dir via --protbert_model (recommended for offline env)
"""

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch
import pandas as pd
from rdkit import Chem

# Make project src importable
sys.path.append(str(Path(__file__).parent.parent))

from src.models.hypergraph_mae import PretrainedHyperGraphMAE, EnhancedHyperGraphMAE  # noqa: E402
from src.data.hypergraph_construction import smiles_to_hypergraph  # noqa: E402
from src.data.molecule_features import get_global_feature_statistics  # noqa: E402

# 直接复用项目内预处理逻辑（不落盘，内存中计算）
try:
    from scripts.preprocess_data import compute_statistics as preproc_compute_statistics, process_molecule as preproc_process_molecule  # noqa: E402
    from src.data.molecule_standardizer import MoleculeStandardizer  # noqa: E402
    _PREPROC_AVAILABLE = True
except Exception:
    _PREPROC_AVAILABLE = False


# ======== SHAP/解释性工具（按需导入与复用） ========
def _make_background(L: int, n: int, strategy: str, rng: np.random.Generator) -> np.ndarray:
    """生成 KernelSHAP 背景矩阵 [n, L]（复用 run_shap_downstream 的思路）。"""
    n = max(1, int(n))
    strategy = (strategy or "random_keep").lower()
    mats = []
    if strategy == "zeros":
        mats.append(np.zeros((1, L), dtype=np.float32))
        if n > 1:
            mats.extend([np.zeros((1, L), dtype=np.float32) for _ in range(n - 1)])
    elif strategy == "random_keep":
        for _ in range(n):
            keep_ratio = float(rng.uniform(0.1, 0.6))
            v = (rng.random(L) < keep_ratio).astype(np.float32)
            mats.append(v[None, :])
    else:  # mix
        n0 = max(1, int(0.3 * n))
        mats.extend([np.zeros((1, L), dtype=np.float32) for _ in range(n0)])
        for _ in range(n - n0):
            keep_ratio = float(rng.uniform(0.1, 0.6))
            v = (rng.random(L) < keep_ratio).astype(np.float32)
            mats.append(v[None, :])
    return np.concatenate(mats, axis=0).astype(np.float32)


def _xgb_predict_margin(booster, X: np.ndarray) -> np.ndarray:
    """返回 XGBoost margin（原始分数），X 形状 [N, D]。"""
    import xgboost as xgb
    if X.ndim == 1:
        X = X.reshape(1, -1)
    dm = xgb.DMatrix(X)
    # output_margin=True 返回 raw margin（log-odds/原值），更适合可加性
    y = booster.predict(dm, output_margin=True)
    return np.asarray(y, dtype=np.float64).reshape(-1)


def fuse_features(drug_vec: np.ndarray, prot_vec: np.ndarray) -> np.ndarray:
    """KISS: 跨模态融合仅保留 concat（[drug | protein]）。"""
    dv = np.asarray(drug_vec, dtype=np.float32).reshape(-1)
    pv = np.asarray(prot_vec, dtype=np.float32).reshape(-1)
    return np.concatenate([dv, pv], axis=0)


def _pool_with_mask(z: torch.Tensor, keep_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """对节点嵌入 z [N, C] 在 keep_mask=True 的节点上做池化；若全被遮蔽，返回零向量。"""
    n, c = z.size(0), z.size(1)
    if keep_mask is None or keep_mask.dtype != torch.bool:
        keep_mask = torch.ones(n, dtype=torch.bool, device=z.device)
    if int(keep_mask.sum().item()) == 0:
        return torch.zeros(c, dtype=z.dtype, device=z.device)
    z_sel = z[keep_mask]
    if z_sel.numel() == 0:
        return torch.zeros(c, dtype=z.dtype, device=z.device)
    if mode == 'mean':
        g = z_sel.mean(dim=0)
    elif mode == 'max':
        g, _ = z_sel.max(dim=0)
    elif mode == 'sum':
        g = z_sel.sum(dim=0)
    else:
        raise ValueError(f"Unknown pooling: {mode}")
    return g


def _l2_normalize_vec(g: torch.Tensor) -> torch.Tensor:
    if g.dim() == 1:
        g = g.unsqueeze(0)
    g = torch.nn.functional.normalize(g, p=2, dim=1)
    return g.squeeze(0)


def _save_rdkit_atom_heatmap(smiles: str,
                             weights: list[float] | np.ndarray,
                             out_png: str | None = None,
                             out_svg: str | None = None) -> dict:
    """保存 RDKit 原子热力图；返回包含已保存文件路径的字典。KISS：优先用 Matplotlib 路径，失败时回退 RDKit Cairo 绘制。"""
    from rdkit import Chem
    from rdkit.Chem.Draw import SimilarityMaps
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for visualization")
    n = mol.GetNumAtoms()
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size < n:
        # 右侧零填充以匹配原子数
        w = np.concatenate([w, np.zeros(n - w.size, dtype=float)], axis=0)
    elif w.size > n:
        w = w[:n]
    saved = {}
    # 尝试 Matplotlib 绘制
    tried_matplotlib = False
    try:
        import matplotlib.pyplot as _plt
        res_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, w.tolist(), colorMap=_plt.cm.coolwarm, size=(600, 450))
        fig = res_fig[0] if isinstance(res_fig, tuple) else res_fig
        if out_png:
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_png), dpi=200)
            saved['png'] = str(out_png)
        if out_svg:
            Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_svg))
            saved['svg'] = str(out_svg)
        tried_matplotlib = True
    except Exception:
        tried_matplotlib = True
    # 回退：使用 RDKit 2D 绘制（Cairo/SVG）
    if (out_png and 'png' not in saved) or (out_svg and 'svg' not in saved):
        try:
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer_png = None
            if out_png:
                drawer_png = rdMolDraw2D.MolDraw2DCairo(600, 450)
            drawer_svg = None
            if out_svg:
                drawer_svg = rdMolDraw2D.MolDraw2DSVG(600, 450)
            # 使用 SimilarityMaps 在 drawer 上绘制
            if drawer_png is not None:
                SimilarityMaps.GetSimilarityMapFromWeights(mol, w.tolist(), draw2d=drawer_png)
                drawer_png.FinishDrawing()
                Path(out_png).parent.mkdir(parents=True, exist_ok=True)
                with open(out_png, 'wb') as f:
                    f.write(drawer_png.GetDrawingText())
                saved['png'] = str(out_png)
            if drawer_svg is not None:
                SimilarityMaps.GetSimilarityMapFromWeights(mol, w.tolist(), draw2d=drawer_svg)
                drawer_svg.FinishDrawing()
                Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
                with open(out_svg, 'w', encoding='utf-8') as f:
                    f.write(drawer_svg.GetDrawingText())
                saved['svg'] = str(out_svg)
        except Exception:
            # 忽略回退失败
            pass
    return saved


def build_identity_global_stats() -> dict:
    """构造“恒等归一化”的全局统计：均值=0、方差=1，仅提供范畴枚举。
    这样连续特征 (x - mean)/std 等于恒等变换，实现“只标准化、不归一化”。"""
    stats = {
        'cont_mean': [0.0, 0.0, 0.0],
        'cont_std': [1.0, 1.0, 1.0],
        'bond_cont_means': [0.0],
        'bond_cont_stds': [1.0],
    }
    # 类别枚举（与 get_global_feature_statistics 保持一致）
    stats['degree_cat'] = list(range(7))
    stats['hybridization_cats'] = [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    stats['atom_chiral_cats'] = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    stats['bond_stereo_cats'] = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY,
    ]
    stats['bond_type_cats'] = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    return stats


def load_yaml(path: str):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _infer_in_dim_from_ckpt(ckpt_obj) -> int:
    """参考 probe3_92 的策略，从权重键或config中推断 in_dim。"""
    # 1) config.model.feature_dim
    try:
        cfg = ckpt_obj.get('config') if isinstance(ckpt_obj, dict) else None
        if isinstance(cfg, dict):
            md = cfg.get('model', {})
            if 'feature_dim' in md and md['feature_dim'] is not None:
                return int(md['feature_dim'])
    except Exception:
        pass
    # 2) checkpoint weights
    try:
        state = ckpt_obj.get('model_state_dict', ckpt_obj)
        weight_keys = [
            'encoder.conv_layers.0.convs.0.lin_node.weight',
            'encoder.input_proj.weight',
            'encoder.init_proj.weight',
        ]
        for k in weight_keys:
            if isinstance(state, dict) and k in state and hasattr(state[k], 'shape') and len(state[k].shape) == 2:
                return int(state[k].shape[1])
    except Exception:
        pass
    # 3) legacy fallback from config.features.hyperedge_dim (not ideal)
    try:
        cfg = ckpt_obj.get('config') if isinstance(ckpt_obj, dict) else None
        if isinstance(cfg, dict):
            leg = cfg.get('features', {}).get('hyperedge_dim')
            if leg is not None:
                return int(leg)
    except Exception:
        pass
    return None


def load_hg_model_robust(ckpt_path: str, cfg_path: str, device: torch.device):
    """更鲁棒的HG-MAE加载：对齐 probe3_92 逻辑，手动实例化并加载权重，避免多余kwargs问题。"""
    import yaml
    # 读取配置（支持JSON作为YAML子集）
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # 加载ckpt（weights_only=False 兼容复杂对象）
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 推断输入维度
    in_dim_cfg = None
    try:
        in_dim_cfg = (
            config.get('features', {})
                  .get('atom', {})
                  .get('dim') if isinstance(config.get('features', {}), dict) and isinstance(config['features'].get('atom', {}), dict) else None
        )
    except Exception:
        in_dim_cfg = None
    in_dim = int(in_dim_cfg) if in_dim_cfg is not None else _infer_in_dim_from_ckpt(ckpt)
    if in_dim is None:
        in_dim = 25  # 默认原子特征维度

    # 从config.model读取必要超参
    m = dict(config.get('model', {}))
    required = ['hidden_dim', 'latent_dim', 'proj_dim', 'heads', 'num_layers']
    missing = [k for k in required if k not in m]
    if missing:
        raise ValueError(f"Config.model missing keys: {missing}")
    mask_ratio = float(m.get('mask_ratio', 0.7))
    # 合法范围 (0,1)
    if not (0.0 < mask_ratio < 1.0):
        mask_ratio = 0.7

    # 实例化模型（不传多余kwargs），保留完整config供内部使用
    model = EnhancedHyperGraphMAE(
        in_dim=int(in_dim),
        hidden_dim=int(m['hidden_dim']),
        latent_dim=int(m['latent_dim']),
        proj_dim=int(m['proj_dim']),
        heads=int(m['heads']),
        num_layers=int(m['num_layers']),
        mask_ratio=mask_ratio,
        config=config,
    )

    # 加载权重（宽松匹配，剔除不兼容项）
    state = ckpt.get('model_state_dict', ckpt)
    try:
        cur = model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k.startswith('descriptor_head.'):
                continue
            if k in cur and hasattr(cur[k], 'shape') and getattr(cur[k], 'shape', None) == getattr(v, 'shape', None):
                filtered[k] = v
        model.load_state_dict(filtered, strict=False)
    except Exception:
        # 回退：直接宽松加载
        model.load_state_dict(state, strict=False)

    model.eval().to(device)
    return model, config


def build_drug_embedding(smiles: str,
                         hg_ckpt: str,
                         config_path: str,
                         device: torch.device,
                         pool: str = 'mean',
                         normalize: bool = True) -> np.ndarray:
    from rdkit import Chem

    # Load config and model
    config = load_yaml(config_path)
    model = PretrainedHyperGraphMAE.from_pretrained(hg_ckpt, config=config, strict=False)
    model.eval().to(device)

    # Prepare global stats from this molecule (KISS fallback).
    # For best parity with training, you can precompute stats on training set and pass them here.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    global_stats = get_global_feature_statistics([mol])

    # Build hypergraph data
    data = smiles_to_hypergraph(smiles=smiles, mol_id="query", config=config, global_stats=global_stats, device=device)
    if data is None:
        raise RuntimeError("Failed to construct hypergraph from SMILES")

    # Node embeddings -> graph embedding
    with torch.no_grad():
        z = model.get_embedding(data.x, data.hyperedge_index, getattr(data, 'hyperedge_attr', None))
        if pool == 'mean':
            g = z.mean(dim=0)
        elif pool == 'max':
            g, _ = z.max(dim=0)
        elif pool == 'sum':
            g = z.sum(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {pool}")
        if normalize:
            g = torch.nn.functional.normalize(g.unsqueeze(0), p=2, dim=1).squeeze(0)
    return g.detach().cpu().numpy().astype(np.float32)


def build_drug_embedding_preprocessed(smiles: str,
                                      model: PretrainedHyperGraphMAE,
                                      config: dict,
                                      device: torch.device,
                                      pool: str,
                                      normalize: bool,
                                      global_stats: dict,
                                      standardizer: MoleculeStandardizer,
                                      mol_id: str = "query") -> np.ndarray:
    """使用仓库的预处理逻辑（标准化+全局统计+超图构建）来生成药物嵌入。"""
    if not _PREPROC_AVAILABLE:
        raise RuntimeError("Preprocess module not available; cannot use integrated preprocessing.")

    # 1) 标准化 SMILES
    std_smiles = standardizer.standardize_smiles(smiles)
    if std_smiles is None:
        raise ValueError(f"Standardization failed for SMILES: {smiles}")

    # 2) 通过预处理模块构建超图（不落盘）
    # 预处理接口期望 row 含 'smiles'/'id'
    row = {'smiles': std_smiles, 'id': mol_id}
    result = preproc_process_molecule(row, config=config, global_stats=global_stats)
    if result is None or result.get('data') is None:
        err = result.get('error') if isinstance(result, dict) else 'unknown'
        raise RuntimeError(f"Hypergraph construction failed: {err}")
    data = result['data']

    # 3) 送入模型得到嵌入
    model.eval().to(device)
    with torch.no_grad():
        x = data.x.to(device)
        he_idx = data.hyperedge_index.to(device)
        he_attr = getattr(data, 'hyperedge_attr', None)
        if he_attr is not None:
            he_attr = he_attr.to(device)
        z = model.get_embedding(x, he_idx, he_attr)
        if pool == 'mean':
            g = z.mean(dim=0)
        elif pool == 'max':
            g, _ = z.max(dim=0)
        elif pool == 'sum':
            g = z.sum(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {pool}")
        if normalize:
            g = torch.nn.functional.normalize(g.unsqueeze(0), p=2, dim=1).squeeze(0)
    return g.detach().cpu().numpy().astype(np.float32)


def build_drug_embedding_with_model(smiles: str,
                                    model: PretrainedHyperGraphMAE,
                                    config: dict,
                                    device: torch.device,
                                    pool: str = 'mean',
                                    normalize: bool = True) -> np.ndarray:
    """KISS: 复用已加载的 HG-MAE 模型进行单条分子嵌入计算。"""
    from rdkit import Chem

    model.eval().to(device)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    global_stats = get_global_feature_statistics([mol])

    data = smiles_to_hypergraph(smiles=smiles, mol_id="query", config=config, global_stats=global_stats, device=device)
    if data is None:
        raise RuntimeError("Failed to construct hypergraph from SMILES")

    with torch.no_grad():
        z = model.get_embedding(data.x, data.hyperedge_index, getattr(data, 'hyperedge_attr', None))
        if pool == 'mean':
            g = z.mean(dim=0)
        elif pool == 'max':
            g, _ = z.max(dim=0)
        elif pool == 'sum':
            g = z.sum(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {pool}")
        if normalize:
            g = torch.nn.functional.normalize(g.unsqueeze(0), p=2, dim=1).squeeze(0)
    return g.detach().cpu().numpy().astype(np.float32)


def build_protein_embedding(seq: str,
                            protbert_model: str,
                            device: torch.device,
                            pooling: str = 'mean') -> np.ndarray:
    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        raise ImportError("transformers not installed. Please `pip install transformers`. ") from e

    # Clean sequence and tokenize (ProtBert expects spaced amino acids)
    seq = seq.strip().upper()
    seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    spaced = ' '.join(list(seq))

    tokenizer = AutoTokenizer.from_pretrained(protbert_model, do_lower_case=False)
    model = AutoModel.from_pretrained(protbert_model)
    model.eval().to(device)

    enc = tokenizer(spaced, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        last_hidden = outputs.last_hidden_state  # [1, L, H]
        attn_mask = enc.get('attention_mask', torch.ones(last_hidden.size()[:2], device=last_hidden.device))

        if pooling == 'cls' and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            emb = outputs.pooler_output.squeeze(0)
        else:
            # Mean over non-pad tokens (exclude [CLS]/[SEP] by mask if needed)
            mask = attn_mask.unsqueeze(-1)  # [1, L, 1]
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            emb = summed / denom
            emb = emb.squeeze(0)
    return emb.detach().cpu().numpy().astype(np.float32)


def explain_atoms_kernelshap(smiles: str,
                             hg_model: EnhancedHyperGraphMAE,
                             config: dict,
                             device: torch.device,
                             pool: str,
                             normalize: bool,
                             booster,
                             prot_vec: np.ndarray,
                             use_preproc: bool,
                             standardizer: 'MoleculeStandardizer | None' = None,
                             global_stats: dict | None = None,
                             std_smiles: str | None = None,
                             background: int = 20,
                             nsamples: int = 200,
                             bg_strategy: str = 'random_keep',
                             topk: int = 20) -> dict:
    """对单个分子的原子级进行 KernelSHAP 解释（节点遮蔽）。返回 JSON 友好字典。"""
    try:
        import shap  # type: ignore
    except Exception as e:
        raise ImportError("需要安装 shap：pip install shap") from e

    # 1) 构图，获得 Data（与推理一致）
    if use_preproc:
        if not _PREPROC_AVAILABLE:
            raise RuntimeError("Preprocess module unavailable")
        if standardizer is None or global_stats is None:
            raise RuntimeError("standardizer/global_stats not provided for preprocessing")
        s = std_smiles if std_smiles is not None else (standardizer.standardize_smiles(smiles))
        row = {'smiles': s, 'id': 'query'}
        ret = preproc_process_molecule(row, config=config, global_stats=global_stats)
        if ret is None or ret.get('data') is None:
            raise RuntimeError(f"Hypergraph construction failed for SHAP (preproc): {ret}")
        data = ret['data']
        smiles_used = s
    else:
        data = smiles_to_hypergraph(smiles=smiles, mol_id="query", config=config, global_stats=get_global_feature_statistics([Chem.MolFromSmiles(smiles)]), device=device)
        if data is None:
            raise RuntimeError("Failed to construct hypergraph for SHAP")
        smiles_used = smiles

    x = data.x.to(device)
    he_idx = data.hyperedge_index.to(device)
    he_attr = getattr(data, 'hyperedge_attr', None)
    if he_attr is not None:
        he_attr = he_attr.to(device)

    N = int(x.size(0))
    E = int(he_attr.size(0)) if he_attr is not None and hasattr(he_attr, 'size') else 0
    edge_mask_zeros = torch.zeros(E, dtype=torch.bool, device=device)

    # 2) 预测函数：根据节点keep向量（1=保留，0=遮蔽）返回 XGB margin
    def predict_with_masks(mask_mat: np.ndarray) -> np.ndarray:
        vals = []
        with torch.no_grad():
            for r in mask_mat:
                r = np.asarray(r).reshape(-1)
                keep_mask = torch.as_tensor(r > 0.5, dtype=torch.bool, device=device)
                node_mask = (~keep_mask).to(torch.bool)
                recon_x, edge_pred, z = hg_model.forward(
                    x, he_idx, he_attr, node_mask=node_mask, edge_mask=edge_mask_zeros, eval_mode=True
                )
                # 仅在未遮蔽节点上池化
                g = _pool_with_mask(z, keep_mask, pool)
                if normalize:
                    g = _l2_normalize_vec(g)
                drug_vec = g.detach().cpu().numpy().astype(np.float32)
                feat = fuse_features(drug_vec, prot_vec).astype(np.float32, copy=False)
                m = _xgb_predict_margin(booster, feat)
                vals.append(float(m[0]))
        return np.asarray(vals, dtype=np.float64)

    # 3) 背景/解释点
    L = int(N)
    if L <= 0:
        raise RuntimeError("No atoms (nodes) to explain")
    background_mat = _make_background(L, int(background), bg_strategy, np.random.default_rng(42))
    x0 = np.ones((1, L), dtype=np.float32)

    # 4) KernelSHAP
    explainer = shap.KernelExplainer(predict_with_masks, background_mat)
    shap_values = explainer.shap_values(x0, nsamples=int(nsamples))
    sv = np.asarray(shap_values).reshape(-1)  # [L]

    # 5) 自检与Top-K
    f_full = float(predict_with_masks(np.ones((1, L), dtype=np.float32))[0])
    f_base = float(predict_with_masks(np.zeros((1, L), dtype=np.float32))[0])
    sum_shap = float(sv.sum())

    order = np.argsort(-np.abs(sv))[:int(max(1, topk))]
    # RDKit 原子标签（与构图一致：显式氢）
    atoms_meta = []
    try:
        mol = Chem.MolFromSmiles(smiles_used)
        if mol is not None:
            mol = Chem.AddHs(mol)
            n_atoms = int(mol.GetNumAtoms())
            for i in range(min(L, n_atoms)):
                atoms_meta.append(mol.GetAtomWithIdx(i).GetSymbol())
        # 若解析失败或数量不足，保持 atoms_meta 为空或较短，后续用 '?' 兜底
    except Exception:
        atoms_meta = ['?'] * L
    # 补全长度，确保与节点数对齐
    if len(atoms_meta) < L:
        atoms_meta.extend(['?'] * (L - len(atoms_meta)))
    top_list = [
        {
            'idx': int(i),
            'symbol': (atoms_meta[i] if i < len(atoms_meta) else '?'),
            'shap': float(sv[i]),
            'abs_shap': float(abs(sv[i]))
        }
        for i in order
    ]

    return {
        'granularity': 'atom',
        'num_nodes': int(L),
        'top_contributions': top_list,
        'weights': [float(v) for v in sv.tolist()],
        'symbols': atoms_meta,
        'smiles': smiles_used,
        'sum_abs': float(np.abs(sv).sum()),
        'additivity_check': {
            'f_full': f_full,
            'f_base': f_base,
            'sum_shap': sum_shap,
            'gap': float(sum_shap - (f_full - f_base))
        }
    }


def explain_residues_kernelshap(sequence: str,
                                 protbert_model: str,
                                 device: torch.device,
                                 drug_vec: np.ndarray,
                                 booster,
                                 occlusion: str = 'drop',
                                 background: int = 20,
                                 nsamples: int = 200,
                                 bg_strategy: str = 'random_keep',
                                 topk: int = 20,
                                 pooling: str = 'mean') -> dict:
    """对单个蛋白序列做残基级 KernelSHAP 解释（token遮蔽）。返回 JSON 友好字典。"""
    try:
        import shap  # type: ignore
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        raise ImportError("需要安装 shap 与 transformers 以进行残基解释") from e

    # 1) baseline 编码与池化
    seq = sequence.strip().upper().replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    spaced = ' '.join(list(seq))
    tokenizer = AutoTokenizer.from_pretrained(protbert_model, do_lower_case=False)
    model = AutoModel.from_pretrained(protbert_model)
    model.eval().to(device)

    enc_base = tokenizer(spaced, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=1024)
    enc_base = {k: v.to(device) for k, v in enc_base.items()}
    num_tokens = int(enc_base['input_ids'].size(1))
    # 可解释残基数（不含CLS/SEP）
    L_eff = max(0, num_tokens - 2)
    if L_eff <= 0:
        raise RuntimeError("Sequence too short for SHAP")

    def _pool(outputs, attn_mask_tensor: torch.Tensor, mode: str) -> torch.Tensor:
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        if mode == 'cls':
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output.squeeze(0)
            return last_hidden[:, 0, :].squeeze(0)
        # mean pooling over non-pad tokens
        mask = attn_mask_tensor.unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (summed / denom).squeeze(0)

    # 2) 预测函数：根据残基keep向量（1=保留，0=遮蔽）返回 XGB margin
    def predict_with_masks(mask_mat: np.ndarray) -> np.ndarray:
        vals = []
        with torch.no_grad():
            for r in mask_mat:
                r = np.asarray(r).reshape(-1)
                # 克隆 baseline 编码
                enc = {k: v.clone() for k, v in enc_base.items()}
                if occlusion == 'drop':
                    am = enc['attention_mask']  # [1, L]
                    keep_mask = torch.as_tensor(r > 0.5, dtype=torch.bool, device=device)
                    # 将被遮蔽的残基位置 attention_mask 置0（偏移1，跳过CLS）
                    if L_eff > 0:
                        am[0, 1:1+L_eff][~keep_mask] = 0
                    enc['attention_mask'] = am
                else:  # 'mask' -> 用[MASK]替代
                    input_ids = enc['input_ids']
                    keep_mask = torch.as_tensor(r > 0.5, dtype=torch.bool, device=device)
                    # tokenizer 必须有mask_token
                    mask_id = tokenizer.mask_token_id
                    if mask_id is None:
                        # 回退：将字母替换为X（等价未知），但这里直接替换为 mask_id 不存在时跳过
                        pass
                    else:
                        if L_eff > 0:
                            pos = (~keep_mask).nonzero(as_tuple=True)[0]
                            if pos.numel() > 0:
                                # 偏移1（CLS）
                                input_ids[0, (pos + 1).clamp_min(1).clamp_max(input_ids.size(1)-2)] = mask_id
                    enc['input_ids'] = input_ids
                outputs = model(**enc)
                prot_vec = _pool(outputs, enc['attention_mask'], pooling)
                feat = fuse_features(
                    drug_vec.astype(np.float32, copy=False),
                    prot_vec.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                m = _xgb_predict_margin(booster, feat)
                vals.append(float(m[0]))
        return np.asarray(vals, dtype=np.float64)

    # 3) 背景/解释点
    background_mat = _make_background(L_eff, int(background), bg_strategy, np.random.default_rng(42))
    x0 = np.ones((1, L_eff), dtype=np.float32)

    # 4) KernelSHAP
    explainer = shap.KernelExplainer(predict_with_masks, background_mat)
    shap_values = explainer.shap_values(x0, nsamples=int(nsamples))
    sv = np.asarray(shap_values).reshape(-1)

    # 5) 自检与Top-K
    f_full = float(predict_with_masks(np.ones((1, L_eff), dtype=np.float32))[0])
    f_base = float(predict_with_masks(np.zeros((1, L_eff), dtype=np.float32))[0])
    sum_shap = float(sv.sum())

    order = np.argsort(-np.abs(sv))[:int(max(1, topk))]
    # 残基字母
    letters = list(seq[:L_eff])
    top_list = [
        {
            'pos': int(i),
            'aa': (letters[i] if i < len(letters) else '?'),
            'shap': float(sv[i]),
            'abs_shap': float(abs(sv[i]))
        }
        for i in order
    ]

    return {
        'granularity': 'residue',
        'num_tokens': int(L_eff),
        'top_contributions': top_list,
        'sum_abs': float(np.abs(sv).sum()),
        'additivity_check': {
            'f_full': f_full,
            'f_base': f_base,
            'sum_shap': sum_shap,
            'gap': float(sum_shap - (f_full - f_base))
        }
    }


def explain_residues_occlusion(sequence: str,
                               protbert_model: str,
                               device: torch.device,
                               drug_vec: np.ndarray,
                               booster,
                               occlusion: str = 'drop',
                               residue_max: int = 512,
                               residue_stride: int = 1,
                               batch_size: int = 64,
                               topk: int = 20,
                               pooling: str = 'mean') -> dict:
    """残基级留一法（occlusion）解释：对每个残基单独遮蔽，估计其边际贡献。
    更快，非严格 SHAP。返回与 KernelSHAP 相近的结构，以便前端复用。
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        raise ImportError("需要安装 transformers 以进行残基解释") from e

    # baseline 编码与池化
    seq = sequence.strip().upper().replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    spaced = ' '.join(list(seq))
    tokenizer = AutoTokenizer.from_pretrained(protbert_model, do_lower_case=False)
    model = AutoModel.from_pretrained(protbert_model)
    model.eval().to(device)

    enc_base = tokenizer(spaced, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=1024)
    enc_base = {k: v.to(device) for k, v in enc_base.items()}
    num_tokens = int(enc_base['input_ids'].size(1))
    L_eff = max(0, num_tokens - 2)
    if L_eff <= 0:
        raise RuntimeError("Sequence too short for occlusion")

    def _pool(outputs, attn_mask_tensor: torch.Tensor, mode: str) -> torch.Tensor:
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        if mode == 'cls':
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            return last_hidden[:, 0, :]
        mask = attn_mask_tensor.unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (summed / denom)

    # baseline margin（全保留）
    with torch.no_grad():
        outputs_full = model(**enc_base)
        prot_full = _pool(outputs_full, enc_base['attention_mask'], pooling)  # [1, H]
    feat_full_vec = fuse_features(
        drug_vec.astype(np.float32, copy=False),
        prot_full.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    )
    feat_full = feat_full_vec.reshape(1, -1)
    m_full = float(_xgb_predict_margin(booster, feat_full)[0])

    # 选择要解释的残基位置（0..L_eff-1）
    idx_all = list(range(0, L_eff, max(1, int(residue_stride))))
    if residue_max is not None and int(residue_max) > 0:
        idx_all = idx_all[: int(residue_max)]
    if len(idx_all) == 0:
        return {
            'granularity': 'residue',
            'num_tokens': int(L_eff),
            'weights': [],
            'top_contributions': [],
            'sum_abs': 0.0,
            'method': 'occlusion',
            'additivity_check': {'f_full': m_full, 'f_base': None, 'sum_shap': None, 'gap': None}
        }

    B = max(1, int(batch_size))
    weights = np.zeros((L_eff,), dtype=np.float64)

    mask_id = tokenizer.mask_token_id
    for s in range(0, len(idx_all), B):
        chunk = idx_all[s: s+B]
        bsz = len(chunk)
        # 构造批量输入
        input_ids = enc_base['input_ids'].repeat(bsz, 1)
        attn_mask = enc_base['attention_mask'].repeat(bsz, 1)
        if occlusion == 'drop':
            for j, pos in enumerate(chunk):
                attn_mask[j, 1 + int(pos)] = 0  # 偏移CLS
        else:  # 'mask'
            if mask_id is not None:
                for j, pos in enumerate(chunk):
                    input_ids[j, 1 + int(pos)] = mask_id
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            prot_vecs = _pool(outputs, attn_mask, pooling)  # [bsz, H]
        # 组装 features 并批量 margin 预测
        prot_np = prot_vecs.detach().cpu().numpy().astype(np.float32, copy=False)
        drug_tile = np.tile(drug_vec.astype(np.float32, copy=False), (bsz, 1))
        feats = np.concatenate([drug_tile, prot_np], axis=1)
        margins = _xgb_predict_margin(booster, feats)  # [bsz]
        # 贡献：baseline - masked
        diffs = (m_full - margins)
        for j, pos in enumerate(chunk):
            weights[int(pos)] = float(diffs[j])

    # Top-K 与输出
    letters = list(seq[:L_eff])
    nonzero_idx = [i for i in idx_all if abs(weights[i]) > 0]
    order = np.argsort(-np.abs(weights[nonzero_idx])) if len(nonzero_idx) > 0 else np.array([], dtype=int)
    top_items = []
    if len(order) > 0:
        # order 是在 nonzero_idx 索引上的排序
        top_positions = [nonzero_idx[int(k)] for k in order[:int(max(1, topk))]]
        for i in top_positions:
            top_items.append({
                'pos': int(i),
                'aa': (letters[i] if i < len(letters) else '?'),
                'shap': float(weights[i]),
                'abs_shap': float(abs(weights[i]))
            })

    return {
        'granularity': 'residue',
        'num_tokens': int(L_eff),
        'sequence': seq[:L_eff],
        'weights': [float(v) for v in weights.tolist()],
        'evaluated_positions': [int(i) for i in idx_all],
        'top_contributions': top_items,
        'sum_abs': float(np.sum(np.abs(weights[idx_all]))),
        'method': 'occlusion',
        'additivity_check': {'f_full': m_full, 'f_base': None, 'sum_shap': None, 'gap': None}
    }


def build_protein_embedding_with_model(seq: str,
                                       tokenizer,
                                       model,
                                       device: torch.device,
                                       pooling: str = 'mean') -> np.ndarray:
    """KISS: 复用已加载的 ProtBert 模型进行单条蛋白嵌入计算。"""
    # Clean sequence and tokenize (ProtBert expects spaced amino acids)
    seq = seq.strip().upper()
    seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    spaced = ' '.join(list(seq))

    enc = tokenizer(spaced, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval().to(device)
    with torch.no_grad():
        outputs = model(**enc)
        last_hidden = outputs.last_hidden_state  # [1, L, H]
        attn_mask = enc.get('attention_mask', torch.ones(last_hidden.size()[:2], device=last_hidden.device))

        if pooling == 'cls' and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            emb = outputs.pooler_output.squeeze(0)
        else:
            mask = attn_mask.unsqueeze(-1)  # [1, L, 1]
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            emb = summed / denom
            emb = emb.squeeze(0)
    return emb.detach().cpu().numpy().astype(np.float32)


def load_xgb(model_path: str):
    try:
        import xgboost as xgb
    except Exception as e:
        raise ImportError("xgboost not installed. Please `pip install xgboost`. ") from e

    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def predict_xgb(booster, features: np.ndarray) -> float:
    import xgboost as xgb
    dm = xgb.DMatrix(features.reshape(1, -1))
    prob = float(booster.predict(dm)[0])
    return prob


# ======== Calibration (optional via netcal) ========
 


def _create_calibrator(method: str):
    """Create a binary probability calibrator from netcal by method name."""
    try:
        if method == 'platt':
            from netcal.scaling import LogisticCalibration  # type: ignore
            return LogisticCalibration()
        elif method == 'isotonic':
            from netcal.scaling import IsotonicRegression  # type: ignore
            return IsotonicRegression()
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
    except Exception as e:
        raise ImportError("netcal not installed. Please `pip install netcal`. ") from e


def parse_args():
    p = argparse.ArgumentParser(description="DTI E2E: SMILES + protein sequence -> score")
    p.add_argument('--task', type=str, default='binary', choices=['binary', 'regression'], help='Head task: binary classification or regression')
    # 单样本模式参数
    p.add_argument('--smiles', required=False, type=str, help='Drug SMILES string')
    p.add_argument('--sequence', required=False, type=str, help='Protein amino acid sequence')
    p.add_argument('--hg_ckpt', required=True, type=str, help='Path to HG-MAE checkpoint (.pth)')
    p.add_argument('--hg_config', type=str, default='hydra/version2/configs/max_full.yaml', help='HG-MAE config YAML')
    p.add_argument('--protbert_model', type=str, default='Rostlab/prot_bert_bfd', help='HF model id or local dir for ProtBert')
    p.add_argument('--xgb_model', type=str, default=None, help='Path to trained XGBoost model (JSON or binary). Not required when --train_xgb')
    p.add_argument('--device', type=str, default=None, help='cuda or cpu (auto if not set)')
    p.add_argument('--pool', type=str, default='mean', choices=['mean','max','sum'], help='Drug pooling')
    p.add_argument('--prot_pool', type=str, default='mean', choices=['mean','cls'], help='Protein pooling (ProtBert): mean or cls')
    p.add_argument('--no_norm', action='store_true', help='Disable L2 normalize for drug embedding')
    p.add_argument('--timeout_seconds', type=int, default=None, help='Override hypergraph construction timeout (seconds)')
    p.add_argument('--output', type=str, default=None, help='Optional path to save JSON result (single sample only)')

    # 批量 CSV 模式参数
    p.add_argument('--csv', type=str, default=None, help='CSV file path for batch prediction')
    p.add_argument('--smiles_col', type=str, default='smiles', help='Column name for SMILES in CSV')
    p.add_argument('--sequence_col', type=str, default='sequence', help='Column name for protein sequence in CSV')
    p.add_argument('--id_col', type=str, default=None, help='Optional ID column to carry over')
    p.add_argument('--output_csv', type=str, default=None, help='Path to save CSV with predictions')
    p.add_argument('--skip_invalid', action='store_true', help='Skip invalid rows instead of raising')
    p.add_argument('--label_col', type=str, default=None, help='Optional label column for metrics (expects 0/1)')
    p.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for classification metrics')

    # 预处理控制（默认开启）
    p.add_argument('--use_preprocess', action='store_true', help='Use integrated preprocessing (standardize+stats+hypergraph)')
    p.add_argument('--no_preprocess', action='store_true', default=True, help='Disable integrated preprocessing (fallback to on-the-fly)')
    p.add_argument('--no_standardize', action='store_true', help='Disable molecule standardization in preprocessing')
    p.add_argument('--keep_metals', action='store_true', help='Keep metal-containing molecules during standardization')
    p.add_argument('--max_atoms', type=int, default=200, help='Max atoms allowed during standardization')
    p.add_argument('--stats_sample_size', type=int, default=10000, help='Sample size to compute global stats')

    # 训练 XGBoost 模式
    p.add_argument('--train_xgb', action='store_true', help='Train an XGBoost model from CSV with labels')
    p.add_argument('--xgb_out', type=str, default=None, help='Output path to save trained XGBoost model')
    p.add_argument('--cv5', action='store_true', help='Enable 5-fold cross-validation on training CSV (ignores test_csv)')
    # Optional precomputed embeddings
    p.add_argument('--drug_emb_parquet', type=str, default=None, help='Precomputed drug embeddings parquet (smiles + emb_*)')
    p.add_argument('--prot_emb_parquet', type=str, default=None, help='Precomputed protein embeddings parquet (protein + emb_*)')
    # 默认 8:1:1 划分（train:val:test）
    p.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio for training split (default 0.1; train:val:test = 8:1:1)')
    p.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio for holdout split (default 0.1; set 0 to disable)')
    p.add_argument('--test_csv', type=str, default=None, help='Optional separate CSV as test set (same columns)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for split and XGB')
    p.add_argument('--xgb_max_depth', type=int, default=0)
    p.add_argument('--progress', action='store_true', help='Show progress bars during embedding/test processing')
    p.add_argument('--xgb_max_leaves', type=int, default=1024, help='If >0, use grow_policy=lossguide with max_leaves and set max_depth=0')
    p.add_argument('--eval_period', type=int, default=1, help='Print eval metrics every N rounds (via callback); N>1 reduces console overhead')
    p.add_argument('--xgb_lr', type=float, default=0.1)
    p.add_argument('--xgb_n_round', type=int, default=1000)
    p.add_argument('--xgb_early_stopping', type=int, default=100)
    p.add_argument('--xgb_subsample', type=float, default=0.8)
    p.add_argument('--xgb_colsample', type=float, default=0.8)
    p.add_argument('--xgb_max_bin', type=int, default=1024, help='Histogram bin count (e.g., 256/512/1024). Larger increases compute/memory.')
    p.add_argument('--xgb_reg_lambda', type=float, default=9.0)
    p.add_argument('--xgb_reg_alpha', type=float, default=0.0, help='L1 regularization term (alpha)')
    p.add_argument('--xgb_min_child_weight', type=float, default=7.0, help='Minimum sum of instance weight (Hessian) needed in a child')
    p.add_argument('--xgb_gamma', type=float, default=1.0, help='Minimum loss reduction required to make a further partition of a leaf node')
    p.add_argument('--auto_scale_pos_weight', action='store_true', help='Auto set scale_pos_weight=neg/pos from train set')
    # Optuna 超参搜索（KISS，按需使用）
    p.add_argument('--optuna', action='store_true', help='Use Optuna to tune XGBoost hyperparameters')
    p.add_argument('--n_trials', type=int, default=30, help='Number of Optuna trials')
    p.add_argument('--opt_metric', type=str, default='aucpr', choices=['aucpr', 'auc', 'mse', 'rmse', 'mae', 'r2'], help='Optimization target metric on validation set (use rmse/mae/r2 for regression)')
    p.add_argument('--timeout', type=int, default=None, help='Optional time limit (seconds) for Optuna study')

    # 概率校准（netcal，可选，仅训练时内存校准）
    p.add_argument('--calibration_method', type=str, default='isotonic', choices=['platt','isotonic'], help='Calibration method when fitting')
    p.add_argument('--calibrate_in_train', action='store_true', help='During XGB training: fit calibrator on validation and use in-mem (no saving)')

    # 解释性（单样本优先）
    p.add_argument('--explain_atoms', action='store_true', help='对单样本进行原子级（节点）解释，KernelSHAP + 节点遮蔽')
    p.add_argument('--explain_residues', action='store_true', help='对单样本进行残基级（序列）解释，KernelSHAP + token遮蔽')
    p.add_argument('--prot_occlusion', type=str, default='drop', choices=['drop','mask'], help='残基遮蔽方式：drop=attention_mask置0；mask=替换为[MASK]')
    p.add_argument('--shap_background_strategy', type=str, default='random_keep', choices=['zeros','random_keep','mix'], help='SHAP背景生成策略')
    p.add_argument('--background', type=int, default=20, help='背景样本数')
    p.add_argument('--nsamples', type=int, default=200, help='KernelSHAP 采样预算（越大越稳，越慢）')
    p.add_argument('--shap_topk', type=int, default=20, help='输出Top-K贡献项')
    p.add_argument('--shap_out', type=str, default=None, help='解释结果JSON输出路径（不指定则与--output同目录）')
    # RDKit 原子热力图导出（仅单样本 + 原子解释可用）
    p.add_argument('--viz_atoms_png', type=str, default=None, help='保存RDKit原子热力图到PNG路径（需 --explain_atoms）')
    p.add_argument('--viz_atoms_svg', type=str, default=None, help='保存RDKit原子热力图到SVG路径（需 --explain_atoms）')
    # 残基解释加速/策略
    p.add_argument('--residue_explainer', type=str, default='occlusion', choices=['kernelshap','occlusion'], help='残基解释方法：默认 occlusion（留一法），更快')
    p.add_argument('--residue_max', type=int, default=512, help='残基解释最多考虑前 N 个残基（截断后，不含CLS/SEP）')
    p.add_argument('--residue_stride', type=int, default=1, help='残基解释步长（stride>1 将稀疏抽样残基）')
    p.add_argument('--shap_batch', type=int, default=64, help='解释时的批大小（用于批量前向加速）')
    return p.parse_args()


def _xgb_params_with_gpu_compat(device: torch.device,
                                lr: float, max_depth: int,
                                subsample: float, colsample: float,
                                reg_lambda: float, seed: int,
                                max_bin: int,
                                min_child_weight: float,
                                gamma: float,
                                scale_pos_weight: float | None = None,
                                objective: str | None = None) -> dict:
    """Return XGBoost params that use GPU when available (2.x: device='cuda'; 1.x: tree_method='gpu_hist').
    Also sets objective/eval_metric according to task.
    """
    import xgboost as xgb
    ver = getattr(xgb, '__version__', '1.7.0')
    major = int(ver.split('.')[0]) if ver and ver[0].isdigit() else 1
    # Decide objective/eval_metric
    obj = objective or 'binary:logistic'
    if obj.startswith('reg:'):
        eval_metric = 'rmse'
    else:
        eval_metric = ['aucpr', 'auc']
    params = {
        'objective': obj,
        'eta': float(lr),
        'max_depth': int(max_depth),
        'subsample': float(subsample),
        'colsample_bytree': float(colsample),
        'lambda': float(reg_lambda),
        'min_child_weight': float(min_child_weight),
        'gamma': float(gamma),
        'eval_metric': eval_metric,
        'seed': int(seed),
        # 提速与显存优化（GPU直方图）
        'max_bin': int(max_bin) if max_bin else 256,
        'single_precision_histogram': True,
        'sampling_method': 'gradient_based',
    }
    if device.type == 'cuda':
        if major >= 2:
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'
        else:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
    else:
        params['tree_method'] = 'hist'
    if (scale_pos_weight is not None) and (not obj.startswith('reg:')):
        params['scale_pos_weight'] = float(scale_pos_weight)
    return params


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 补充：回归任务的默认 opt_metric（统一为 RMSE）
    if getattr(args, 'task', 'binary') == 'regression' and args.opt_metric not in ('mse','rmse','mae','r2'):
        args.opt_metric = 'rmse'

    # ========== 训练 XGBoost 模式 ==========
    if args.train_xgb:
        if args.csv is None:
            raise ValueError('Training mode requires --csv dataset input')
        if not args.label_col:
            raise ValueError('Training mode requires --label_col for ground truth labels (0/1)')

        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

        # 读取 CSV
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {args.csv}")
        df = pd.read_csv(csv_path)
        for col in (args.smiles_col, args.sequence_col, args.label_col):
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        # 预加载模型
        hg_model, config = load_hg_model_robust(args.hg_ckpt, args.hg_config, device)
        # Optional runtime override of timeout
        if getattr(args, 'timeout_seconds', None) is not None:
            try:
                config['timeout_seconds'] = int(args.timeout_seconds)
            except Exception:
                pass
        try:
            from transformers import AutoTokenizer, AutoModel
        except Exception as e:
            raise ImportError("transformers not installed. Please `pip install transformers`. ") from e
        tokenizer = AutoTokenizer.from_pretrained(args.protbert_model, do_lower_case=False, local_files_only=True)
        prot_model = AutoModel.from_pretrained(args.protbert_model, local_files_only=True)

        # 预加载预计算嵌入（如提供）
        drug_cache = {}
        prot_cache = {}
        if args.drug_emb_parquet:
            ddf = pd.read_parquet(args.drug_emb_parquet)
            emb_cols = [c for c in ddf.columns if c.startswith('emb_')]
            d_mat = ddf[emb_cols].to_numpy(dtype=np.float32, copy=False)
            for k, vec in zip(ddf['smiles'].astype(str), d_mat):
                drug_cache[str(k)] = vec.astype(np.float32, copy=False)
        if args.prot_emb_parquet:
            pdf = pd.read_parquet(args.prot_emb_parquet)
            emb_cols = [c for c in pdf.columns if c.startswith('emb_')]
            p_mat = pdf[emb_cols].to_numpy(dtype=np.float32, copy=False)
            for k, vec in zip(pdf['protein'].astype(str), p_mat):
                prot_cache[str(k)] = vec.astype(np.float32, copy=False)
        X_list = []
        y_list = []

        # 训练模式下也支持预处理（标准化+全局统计）
        standardizer = None
        global_stats = None
        if (not args.no_preprocess) and _PREPROC_AVAILABLE:
            # 训练模式同样：仅标准化，不归一化
            standardizer = MoleculeStandardizer(remove_metals=not args.keep_metals, max_atoms=args.max_atoms)
            global_stats = build_identity_global_stats()

        # Optional progress bar for embedding
        _train_iter = df.iterrows()
        if args.progress:
            try:
                from tqdm import tqdm  # type: ignore
                _train_iter = tqdm(_train_iter, total=len(df), desc='Embedding (train)')
            except Exception:
                pass
        for idx, row in _train_iter:
            smi = row[args.smiles_col]
            seq = row[args.sequence_col]
            lbl = row[args.label_col]
            if pd.isna(smi) or pd.isna(seq) or pd.isna(lbl):
                if args.skip_invalid:
                    continue
                else:
                    raise ValueError(f"Row {idx}: missing data in required columns")
            smi = str(smi)
            seq = str(seq)
            try:
                if smi in drug_cache:
                    drug_vec = drug_cache[smi]
                else:
                    if standardizer is not None and global_stats is not None:
                        drug_vec = build_drug_embedding_preprocessed(
                            smiles=smi,
                            model=hg_model,
                            config=config,
                            device=device,
                            pool=args.pool,
                            normalize=not args.no_norm,
                            global_stats=global_stats,
                            standardizer=standardizer,
                            mol_id=str(row[args.smiles_col])
                        )
                    else:
                        drug_vec = build_drug_embedding_with_model(
                            smiles=smi,
                            model=hg_model,
                            config=config,
                            device=device,
                            pool=args.pool,
                            normalize=not args.no_norm,
                        )
                    drug_cache[smi] = drug_vec

                if seq in prot_cache:
                    prot_vec = prot_cache[seq]
                else:
                    prot_vec = build_protein_embedding_with_model(
                        seq=seq,
                        tokenizer=tokenizer,
                        model=prot_model,
                        device=device,
                        pooling=args.prot_pool,
                    )
                    prot_cache[seq] = prot_vec

                feat = fuse_features(drug_vec, prot_vec)
                X_list.append(feat)
                if args.task == 'regression':
                    y_list.append(float(lbl))
                else:
                    y_list.append(int(lbl))
            except Exception as e:
                if args.skip_invalid:
                    print(f"Row {idx} failed: {e}", file=sys.stderr)
                    continue
                else:
                    raise

        if len(X_list) == 0:
            raise RuntimeError('No valid samples for training')

        # 确保使用 float32 的 numpy 数组（避免 DataFrame/object 带来的 CPU 转换开销）
        X = np.stack(X_list).astype(np.float32, copy=False)
        # 二分类标签用 float32，避免内部再转换
        y = np.array(y_list, dtype=np.float32)

        # 若启用5折交叉验证（忽略 test_csv/optuna/校准），直接在嵌入特征上划分折训练
        if args.cv5:
            from sklearn.model_selection import StratifiedKFold, KFold
            import xgboost as xgb
            X = np.stack(X_list).astype(np.float32, copy=False)
            y = np.array(y_list, dtype=np.float32)
            n_splits = 5
            if args.task == 'binary' and len(np.unique(y)) > 1:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(args.seed))
                split_iter = kf.split(X, y)
            else:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(args.seed))
                split_iter = kf.split(X)

            fold_metrics = []
            for fold, (tr_idx, val_idx) in enumerate(split_iter, start=1):
                X_tr = X[tr_idx]; y_tr = y[tr_idx]
                X_val = X[val_idx]; y_val = y[val_idx]
                # DMatrix/QuantileDMatrix 按设备自动选择
                try:
                    if device.type == 'cuda':
                        from xgboost import QuantileDMatrix  # type: ignore
                        dtrain = QuantileDMatrix(X_tr, y_tr, max_bin=int(getattr(args, 'xgb_max_bin', 1024)))
                        dval = QuantileDMatrix(X_val, y_val, ref=dtrain, max_bin=int(getattr(args, 'xgb_max_bin', 1024)))
                    else:
                        dtrain = xgb.DMatrix(X_tr, label=y_tr)
                        dval = xgb.DMatrix(X_val, label=y_val)
                except Exception:
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_val, label=y_val)

                spw = None
                if args.task == 'binary':
                    pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
                    if pos > 0:
                        spw = float(neg / max(1, pos))
                params = _xgb_params_with_gpu_compat(
                    device=device,
                    lr=float(args.xgb_lr),
                    max_depth=int(args.xgb_max_depth),
                    subsample=float(args.xgb_subsample),
                    colsample=float(args.xgb_colsample),
                    reg_lambda=float(args.xgb_reg_lambda),
                    seed=int(args.seed),
                    max_bin=int(args.xgb_max_bin),
                    min_child_weight=float(args.xgb_min_child_weight),
                    gamma=float(args.xgb_gamma),
                    scale_pos_weight=spw,
                    objective=('reg:squarederror' if args.task == 'regression' else 'binary:logistic')
                )
                callbacks = []
                if int(args.eval_period) > 1:
                    try:
                        from xgboost.callback import EvaluationMonitor  # type: ignore
                        callbacks.append(EvaluationMonitor(show_stdv=False, period=int(args.eval_period)))
                    except Exception:
                        pass

                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=int(args.xgb_n_round),
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=int(args.xgb_early_stopping),
                    verbose_eval=False,
                    callbacks=callbacks
                )
                # 评估该折
                y_val_pred = booster.predict(dval)
                if args.task == 'regression':
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
                    mae = float(mean_absolute_error(y_val, y_val_pred))
                    r2 = float(r2_score(y_val, y_val_pred)) if len(np.unique(y_val)) > 1 else None
                    fold_metrics.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
                else:
                    from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
                    auc = float(roc_auc_score(y_val, y_val_pred)) if len(np.unique(y_val)) > 1 else None
                    y_hat = (y_val_pred >= float(args.threshold)).astype(int)
                    fold_metrics.append({
                        'fold': fold,
                        'auc': auc,
                        'acc': float(accuracy_score(y_val, y_hat)),
                        'f1': float(f1_score(y_val, y_hat)),
                        'recall': float(recall_score(y_val, y_hat))
                    })

            # 汇总
            def _avg_std(key):
                vals = [m[key] for m in fold_metrics if m.get(key) is not None]
                return (float(np.mean(vals)) if vals else None, float(np.std(vals)) if vals else None)

            summary = {'cv5': True, 'folds': len(fold_metrics), 'fold_metrics': fold_metrics}
            if args.task == 'regression':
                for k in ['rmse', 'mae', 'r2']:
                    mu, sd = _avg_std(k)
                    summary[f'{k}_mean'] = mu
                    summary[f'{k}_std'] = sd
            else:
                for k in ['auc', 'acc', 'f1', 'recall']:
                    mu, sd = _avg_std(k)
                    summary[f'{k}_mean'] = mu
                    summary[f'{k}_std'] = sd

            print(json.dumps(summary, ensure_ascii=False))
            return

        # 可选测试集划分
        test_ratio = float(max(0.0, min(0.9, args.test_ratio)))
        stratify_all = (y if (args.task == 'binary' and len(np.unique(y)) > 1) else None)
        if args.test_csv is not None:
            # 单独的测试集CSV
            df_t = pd.read_csv(Path(args.test_csv))
            for col in (args.smiles_col, args.sequence_col, args.label_col):
                if col not in df_t.columns:
                    raise ValueError(f"Test CSV missing required column: {col}")
            # 复用同一标准化器/统计与缓存
            X_test, y_test = [], []
            _test_iter = df_t.iterrows()
            if args.progress:
                try:
                    from tqdm import tqdm  # type: ignore
                    _test_iter = tqdm(_test_iter, total=len(df_t), desc='Embedding (test)')
                except Exception:
                    pass
            for idx, row in _test_iter:
                smi = str(row[args.smiles_col]) if pd.notna(row[args.smiles_col]) else ''
                seq = str(row[args.sequence_col]) if pd.notna(row[args.sequence_col]) else ''
                lbl = row[args.label_col]
                if not smi or not seq or pd.isna(lbl):
                    if args.skip_invalid:
                        continue
                    else:
                        raise ValueError(f"Test row {idx}: missing data")
                # 药物向量
                if smi in drug_cache:
                    drug_vec = drug_cache[smi]
                else:
                    if standardizer is not None and global_stats is not None:
                        drug_vec = build_drug_embedding_preprocessed(
                            smiles=smi, model=hg_model, config=config, device=device,
                            pool=args.pool, normalize=not args.no_norm,
                            global_stats=global_stats, standardizer=standardizer,
                            mol_id=str(idx)
                        )
                    else:
                        drug_vec = build_drug_embedding_with_model(
                            smiles=smi, model=hg_model, config=config, device=device,
                            pool=args.pool, normalize=not args.no_norm,
                        )
                    drug_cache[smi] = drug_vec
                # 蛋白向量
                if seq in prot_cache:
                    prot_vec = prot_cache[seq]
                else:
                    prot_vec = build_protein_embedding_with_model(
                        seq=seq, tokenizer=tokenizer, model=prot_model, device=device, pooling=args.prot_pool
                    )
                    prot_cache[seq] = prot_vec
                X_test.append(fuse_features(drug_vec, prot_vec))
                if args.task == 'regression':
                    y_test.append(float(lbl))
                else:
                    y_test.append(int(lbl))
            X_test = np.stack(X_test).astype(np.float32, copy=False) if len(X_test)>0 else np.zeros((0, X.shape[1]), dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)
            # 训练/验证均从训练CSV划分
            X_trval, y_trval = X, y
            stratify_trval = (y_trval if (args.task == 'binary' and len(np.unique(y_trval)) > 1) else None)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_trval, y_trval, test_size=float(args.val_ratio), random_state=int(args.seed), stratify=stratify_trval
            )
            # 显式保持 float32，避免在 DMatrix 构建时转换
            X_tr = X_tr.astype(np.float32, copy=False); X_val = X_val.astype(np.float32, copy=False)
            y_tr = y_tr.astype(np.float32, copy=False); y_val = y_val.astype(np.float32, copy=False)
        else:
            # 从同一CSV中划分 test，然后再从剩余中划分 val
            if test_ratio > 0.0:
                X_rest, X_test, y_rest, y_test = train_test_split(
                    X, y, test_size=test_ratio, random_state=int(args.seed), stratify=stratify_all
                )
                # 将 val_ratio 映射到剩余集的相对比例
                denom = max(1e-9, 1.0 - test_ratio)
                val_ratio_rel = min(0.9, max(1e-6, float(args.val_ratio) / denom))
                stratify_rest = (y_rest if (args.task == 'binary' and len(np.unique(y_rest)) > 1) else None)
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_rest, y_rest, test_size=val_ratio_rel, random_state=int(args.seed), stratify=stratify_rest
                )
            else:
                stratify = (y if (args.task == 'binary' and len(np.unique(y)) > 1) else None)
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=float(args.val_ratio), random_state=int(args.seed), stratify=stratify
                )
                X_test = np.zeros((0, X.shape[1]), dtype=np.float32)
                y_test = np.zeros((0,), dtype=np.float32)
            # 显式保持 float32，避免在 DMatrix 构建时转换
            X_tr = X_tr.astype(np.float32, copy=False); X_val = X_val.astype(np.float32, copy=False)
            y_tr = y_tr.astype(np.float32, copy=False); y_val = y_val.astype(np.float32, copy=False)

        # XGBoost 训练参数
        # XGBoost 参数（自动兼容 1.x/2.x GPU）
        spw = None
        if args.task == 'binary' and args.auto_scale_pos_weight:
            pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
            if pos > 0:
                spw = float(neg / pos)
        # --------- Optuna 调参（KISS）---------
        if args.optuna:
            if args.task == 'binary' and len(np.unique(y_val)) < 2:
                raise ValueError('Optuna (classification) requires validation set to contain both classes; adjust --val_ratio or data.')
            import optuna
            # 构建一次 DMatrix，供各 trial 复用
            try:
                if device.type == 'cuda':
                    from xgboost import QuantileDMatrix  # type: ignore
                    qmaxb = int(args.xgb_max_bin) if hasattr(args, 'xgb_max_bin') else 256
                    dtrain = QuantileDMatrix(X_tr, y_tr, max_bin=qmaxb)
                    dval = QuantileDMatrix(X_val, y_val, ref=dtrain, max_bin=qmaxb)
                else:
                    import xgboost as xgb
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_val, label=y_val)
            except Exception:
                import xgboost as xgb
                dtrain = xgb.DMatrix(X_tr, label=y_tr)
                dval = xgb.DMatrix(X_val, label=y_val)

            def objective(trial: optuna.Trial) -> float:
                import xgboost as xgb
                # 采样简洁空间（深度优先）
                lr = trial.suggest_float('eta', 1e-3, 3e-1, log=True)
                max_depth = trial.suggest_int('max_depth', 3, 12)
                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                colsample = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                min_child_weight = trial.suggest_float('min_child_weight', 1.0, 20.0, log=True)
                reg_lambda = trial.suggest_float('lambda', 1e-8, 50.0, log=True)
                reg_alpha = trial.suggest_float('alpha', 1e-8, 10.0, log=True)
                gamma = trial.suggest_float('gamma', 1e-8, 10.0, log=True)

                params = _xgb_params_with_gpu_compat(
                    device=device, lr=lr, max_depth=max_depth,
                    subsample=subsample, colsample=colsample,
                    reg_lambda=reg_lambda, seed=args.seed, max_bin=args.xgb_max_bin,
                    min_child_weight=min_child_weight, gamma=gamma,
                    scale_pos_weight=spw,
                    objective=('reg:squarederror' if args.task == 'regression' else 'binary:logistic')
                )
                params['alpha'] = reg_alpha

                

                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=int(args.xgb_n_round),
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=int(args.xgb_early_stopping),
                    verbose_eval=False,
                )

                y_val_pred = booster.predict(dval)
                # 选择优化指标
                if args.task == 'regression':
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    if args.opt_metric == 'mse':
                        score = float(mean_squared_error(y_val, y_val_pred))
                    elif args.opt_metric == 'rmse':
                        score = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
                    elif args.opt_metric == 'mae':
                        score = float(mean_absolute_error(y_val, y_val_pred))
                    elif args.opt_metric == 'r2':
                        score = float(r2_score(y_val, y_val_pred))
                    else:
                        score = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
                else:
                    if args.opt_metric == 'aucpr':
                        from sklearn.metrics import average_precision_score
                        score = float(average_precision_score(y_val, y_val_pred))
                    else:
                        from sklearn.metrics import roc_auc_score
                        score = float(roc_auc_score(y_val, y_val_pred))
                # 记录最佳迭代轮次供复现
                trial.set_user_attr('best_iteration', int(getattr(booster, 'best_iteration', -1)))
                return score

            # 目标方向：分类多数最大化（auc/aucpr），回归 rmse/mae 最小化、r2 最大化
            direction = 'maximize'
            if args.task == 'regression':
                direction = 'minimize' if args.opt_metric in ('mse','rmse','mae') else 'maximize'
            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=args.seed))
            study.optimize(objective, n_trials=int(args.n_trials), timeout=args.timeout)

            best = study.best_trial
            # 以最优超参重新训练并评估/保存
            bp = best.params
            params = _xgb_params_with_gpu_compat(
                device=device, lr=bp.get('eta', args.xgb_lr), max_depth=bp.get('max_depth', args.xgb_max_depth),
                subsample=bp.get('subsample', args.xgb_subsample), colsample=bp.get('colsample_bytree', args.xgb_colsample),
                reg_lambda=bp.get('lambda', args.xgb_reg_lambda), seed=args.seed, max_bin=args.xgb_max_bin,
                min_child_weight=bp.get('min_child_weight', args.xgb_min_child_weight), gamma=bp.get('gamma', args.xgb_gamma),
                scale_pos_weight=spw,
                objective=('reg:squarederror' if args.task == 'regression' else 'binary:logistic')
            )
            params['alpha'] = bp.get('alpha', 0.0)

            

            import xgboost as xgb
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=int(args.xgb_n_round),
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=int(args.xgb_early_stopping),
                verbose_eval=(10 if args.progress else 50),
            )

            # 评估验证/测试，并可选“在训练时”做内存校准（仅分类）
            y_val_pred_raw = booster.predict(dval)
            from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
            def _r(v, nd=6):
                try:
                    return None if v is None else float(round(float(v), nd))
                except Exception:
                    return v
            metrics = {'best_iteration': int(getattr(booster, 'best_iteration', -1)), 'best_params': bp, 'study_best_value': float(best.value)}
            if args.task == 'regression':
                rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred_raw)))
                mae = float(mean_absolute_error(y_val, y_val_pred_raw))
                r2 = float(r2_score(y_val, y_val_pred_raw)) if len(np.unique(y_val)) > 1 else None
                metrics.update({'val_rmse': _r(rmse), 'val_mae': _r(mae), 'val_r2': _r(r2)})
            else:
                y_val_pred = (y_val_pred_raw >= float(args.threshold)).astype(int)
                auc_val = float(roc_auc_score(y_val, y_val_pred_raw)) if len(np.unique(y_val)) > 1 else None
                metrics.update({'val_auc': _r(auc_val), 'val_acc': _r(accuracy_score(y_val, y_val_pred)), 'val_f1': _r(f1_score(y_val, y_val_pred)), 'val_recall': _r(recall_score(y_val, y_val_pred))})
                # in-memory calibrator (fit on val, apply to val/test)
                y_val_prob_cal = None
                if args.calibrate_in_train and args.task == 'binary':
                    calibrator = _create_calibrator(args.calibration_method)
                    calibrator.fit(y_val_pred_raw, y_val)
                    y_val_prob_cal = calibrator.transform(y_val_pred_raw)
                    y_val_pred_cal = (y_val_prob_cal >= float(args.threshold)).astype(int)
                    auc_val_cal = float(roc_auc_score(y_val, y_val_prob_cal)) if len(np.unique(y_val)) > 1 else None
                    metrics.update({'val_auc_cal': _r(auc_val_cal), 'val_acc_cal': _r(accuracy_score(y_val, y_val_pred_cal)), 'val_f1_cal': _r(f1_score(y_val, y_val_pred_cal)), 'val_recall_cal': _r(recall_score(y_val, y_val_pred_cal)), 'calibration_method': str(args.calibration_method)})
            test_metrics = None
            if X_test.shape[0] > 0:
                dtest = xgb.DMatrix(X_test, label=y_test)
                y_test_pred_raw = booster.predict(dtest)
                if args.task == 'regression':
                    rmse_t = float(np.sqrt(mean_squared_error(y_test, y_test_pred_raw))) if len(y_test) > 0 else None
                    mae_t = float(mean_absolute_error(y_test, y_test_pred_raw)) if len(y_test) > 0 else None
                    r2_t = float(r2_score(y_test, y_test_pred_raw)) if len(np.unique(y_test)) > 1 else None
                    test_metrics = {'test_rmse': _r(rmse_t), 'test_mae': _r(mae_t), 'test_r2': _r(r2_t)}
                else:
                    y_test_pred = (y_test_pred_raw >= float(args.threshold)).astype(int)
                    auc_test = float(roc_auc_score(y_test, y_test_pred_raw)) if len(np.unique(y_test)) > 1 else None
                    test_metrics = {'test_auc': _r(auc_test), 'test_acc': _r(accuracy_score(y_test, y_test_pred)), 'test_f1': _r(f1_score(y_test, y_test_pred)), 'test_recall': _r(recall_score(y_test, y_test_pred))}
                    if args.calibrate_in_train and args.task == 'binary' and y_val_prob_cal is not None:
                        y_test_prob_cal = calibrator.transform(y_test_pred_raw)
                        y_test_pred_cal = (y_test_prob_cal >= float(args.threshold)).astype(int)
                        auc_test_cal = float(roc_auc_score(y_test, y_test_prob_cal)) if len(np.unique(y_test)) > 1 else None
                        test_metrics.update({'test_auc_cal': _r(auc_test_cal), 'test_acc_cal': _r(accuracy_score(y_test, y_test_pred_cal)), 'test_f1_cal': _r(f1_score(y_test, y_test_pred_cal)), 'test_recall_cal': _r(recall_score(y_test, y_test_pred_cal))})
            out_path = args.xgb_out or str(csv_path.with_suffix('.xgb.optuna.json'))
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(out_path)
            payload = {
                'train_samples': int(len(X_tr)),
                'val_samples': int(len(X_val)),
                'test_samples': int(X_test.shape[0]),
                'xgb_model': out_path,
                **metrics
            }
            if test_metrics:
                payload.update(test_metrics)
            print(json.dumps(payload, ensure_ascii=False))
            return
        params = _xgb_params_with_gpu_compat(
            device=device, lr=args.xgb_lr, max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample, colsample=args.xgb_colsample,
            reg_lambda=args.xgb_reg_lambda, seed=args.seed, max_bin=args.xgb_max_bin,
            min_child_weight=args.xgb_min_child_weight, gamma=args.xgb_gamma,
            scale_pos_weight=spw,
            objective=('reg:squarederror' if args.task == 'regression' else 'binary:logistic')
        )
        # CLI 显式设置 L1 正则（alpha）
        params['alpha'] = float(args.xgb_reg_alpha)

        # Prefer QuantileDMatrix on GPU for memory efficiency
        try:
            if device.type == 'cuda':
                from xgboost import QuantileDMatrix  # type: ignore
                # Ensure max_bin consistency between Booster and all QDMs
                qmaxb = int(args.xgb_max_bin) if hasattr(args, 'xgb_max_bin') else 256
                dtrain = QuantileDMatrix(X_tr, y_tr, max_bin=qmaxb)
                # Validation must reference training and use same max_bin
                dval = QuantileDMatrix(X_val, y_val, ref=dtrain, max_bin=qmaxb)
            else:
                dtrain = xgb.DMatrix(X_tr, label=y_tr)
                dval = xgb.DMatrix(X_val, label=y_val)
        except Exception:
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'val')]

        # 若启用 max_leaves（>0），用 lossguide 策略（比深度限制更稳）
        if int(args.xgb_max_leaves) > 0:
            # xgb.train 会读取 params 中的 grow_policy/max_leaves
            booster_params = dict(params)
            booster_params['grow_policy'] = 'lossguide'
            booster_params['max_leaves'] = int(args.xgb_max_leaves)
            booster_params['max_depth'] = 0
        else:
            booster_params = params

        # Optional evaluation monitor (printing every N rounds)
        callbacks = []
        try:
            if int(args.eval_period) > 1:
                callbacks.append(xgb.callback.EvaluationMonitor(period=int(args.eval_period)))
        except Exception:
            pass

        booster = xgb.train(
            booster_params,
            dtrain,
            num_boost_round=int(args.xgb_n_round),
            evals=evals,
            early_stopping_rounds=int(args.xgb_early_stopping),
            verbose_eval=(10 if args.progress else 50),
            callbacks=callbacks
        )

        # 评估验证集 + 可选（分类）在训练时进行内存校准
        y_val_pred_raw = booster.predict(dval)
        # 高精度输出（保留6位小数）
        def _r(v, nd=6):
            try:
                return None if v is None else float(round(float(v), nd))
            except Exception:
                return v
        metrics = {
            'best_iteration': int(getattr(booster, 'best_iteration', booster.best_ntree_limit if hasattr(booster, 'best_ntree_limit') else -1)),
            'best_score': booster.attributes().get('best_score', None)
        }
        if args.task == 'regression':
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred_raw)))
            mae = float(mean_absolute_error(y_val, y_val_pred_raw))
            r2 = float(r2_score(y_val, y_val_pred_raw)) if len(np.unique(y_val)) > 1 else None
            metrics.update({'val_rmse': _r(rmse), 'val_mae': _r(mae), 'val_r2': _r(r2)})
        else:
            from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
            y_val_pred = (y_val_pred_raw >= float(args.threshold)).astype(int)
            auc_val = float(roc_auc_score(y_val, y_val_pred_raw)) if len(np.unique(y_val)) > 1 else None
            metrics.update({
                'val_auc': _r(auc_val),
                'val_acc': _r(accuracy_score(y_val, y_val_pred)),
                'val_f1': _r(f1_score(y_val, y_val_pred)),
                'val_recall': _r(recall_score(y_val, y_val_pred))
            })
            # 内存校准
            if args.calibrate_in_train:
                calibrator = _create_calibrator(args.calibration_method)
                calibrator.fit(y_val_pred_raw, y_val)
                y_val_prob_cal = calibrator.transform(y_val_pred_raw)
                y_val_pred_cal = (y_val_prob_cal >= float(args.threshold)).astype(int)
                auc_val_cal = float(roc_auc_score(y_val, y_val_prob_cal)) if len(np.unique(y_val)) > 1 else None
                metrics.update({
                    'val_auc_cal': _r(auc_val_cal),
                    'val_acc_cal': _r(accuracy_score(y_val, y_val_pred_cal)),
                    'val_f1_cal': _r(f1_score(y_val, y_val_pred_cal)),
                    'val_recall_cal': _r(recall_score(y_val, y_val_pred_cal)),
                    'calibration_method': str(args.calibration_method)
                })

        # 可选：评估测试集
        test_metrics = None
        if X_test.shape[0] > 0:
            dtest = xgb.DMatrix(X_test, label=y_test)
            y_test_pred_raw = booster.predict(dtest)
            if args.task == 'regression':
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                rmse_t = float(np.sqrt(mean_squared_error(y_test, y_test_pred_raw))) if len(y_test) > 0 else None
                mae_t = float(mean_absolute_error(y_test, y_test_pred_raw)) if len(y_test) > 0 else None
                r2_t = float(r2_score(y_test, y_test_pred_raw)) if len(np.unique(y_test)) > 1 else None
                test_metrics = {'test_rmse': _r(rmse_t), 'test_mae': _r(mae_t), 'test_r2': _r(r2_t)}
            else:
                from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
                y_test_pred = (y_test_pred_raw >= float(args.threshold)).astype(int)
                auc_test = float(roc_auc_score(y_test, y_test_pred_raw)) if len(np.unique(y_test)) > 1 else None
                test_metrics = {'test_auc': _r(auc_test), 'test_acc': _r(accuracy_score(y_test, y_test_pred)), 'test_f1': _r(f1_score(y_test, y_test_pred)), 'test_recall': _r(recall_score(y_test, y_test_pred))}
            if y_val_prob_cal is not None:
                y_test_prob_cal = calibrator.transform(y_test_prob)
                y_test_pred_cal = (y_test_prob_cal >= float(args.threshold)).astype(int)
                auc_test_cal = float(roc_auc_score(y_test, y_test_prob_cal)) if len(np.unique(y_test)) > 1 else None
                test_metrics.update({
                    'test_auc_cal': _r(auc_test_cal),
                    'test_acc_cal': _r(accuracy_score(y_test, y_test_pred_cal)),
                    'test_f1_cal': _r(f1_score(y_test, y_test_pred_cal)),
                    'test_recall_cal': _r(recall_score(y_test, y_test_pred_cal))
                })

        # 保存模型
        out_path = args.xgb_out or str(csv_path.with_suffix('.xgb.json'))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(out_path)

        payload = {
            'train_samples': int(len(X_tr)),
            'val_samples': int(len(X_val)),
            'test_samples': int(X_test.shape[0]),
            'xgb_model': out_path,
            **metrics
        }
        if test_metrics:
            payload.update(test_metrics)
        print(json.dumps(payload, ensure_ascii=False))

        return
    
    # 判定是否启用预处理：
    # - 支持显式 --use_preprocess 开关
    # - 或者未设置 --no_preprocess（默认 False 时启用）
    # - 仍需模块可用
    use_preproc = ((getattr(args, 'use_preprocess', False) or (not args.no_preprocess)) and _PREPROC_AVAILABLE)

    # ========== 批量 CSV 模式 ==========
    if args.csv is not None:
        if not args.train_xgb and not args.xgb_model:
            raise ValueError("Prediction mode requires --xgb_model (path to trained XGBoost model)")
        # 读取 CSV
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {args.csv}")

        df = pd.read_csv(csv_path)
        if args.smiles_col not in df.columns or args.sequence_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{args.smiles_col}' and '{args.sequence_col}'")

        # 预加载模型/配置
        hg_model, config = load_hg_model_robust(args.hg_ckpt, args.hg_config, device)
        if getattr(args, 'timeout_seconds', None) is not None:
            try:
                config['timeout_seconds'] = int(args.timeout_seconds)
            except Exception:
                pass

        try:
            from transformers import AutoTokenizer, AutoModel
        except Exception as e:
            raise ImportError("transformers not installed. Please `pip install transformers`. ") from e
        tokenizer = AutoTokenizer.from_pretrained(args.protbert_model, do_lower_case=False)
        prot_model = AutoModel.from_pretrained(args.protbert_model)

        booster = None if args.train_xgb else load_xgb(args.xgb_model)

        # ========== 预处理准备 ==========
        standardizer = None
        global_stats = None
        if use_preproc:
            # 仅做标准化，不做归一化：使用“恒等”统计
            standardizer = MoleculeStandardizer(remove_metals=not args.keep_metals, max_atoms=args.max_atoms)
            global_stats = build_identity_global_stats()

        results = []
        kept_rows = []
        drug_cache = {}
        prot_cache = {}
        y_true = []  # for metrics when label_col is provided
        for idx, row in df.iterrows():
            smi = str(row[args.smiles_col]) if pd.notna(row[args.smiles_col]) else ''
            seq = str(row[args.sequence_col]) if pd.notna(row[args.sequence_col]) else ''
            if not smi or not seq:
                msg = f"Row {idx}: empty smiles or sequence"
                if args.skip_invalid:
                    print(msg, file=sys.stderr)
                    continue
                else:
                    raise ValueError(msg)
            try:
                if use_preproc:
                    if standardizer is None or global_stats is None:
                        raise RuntimeError("Preprocess not initialized correctly")
                    key = standardizer.standardize_smiles(smi) if not args.no_standardize else smi
                    if key in drug_cache:
                        drug_vec = drug_cache[key]
                    else:
                        drug_vec = build_drug_embedding_preprocessed(
                            smiles=smi,
                            model=hg_model,
                            config=config,
                            device=device,
                            pool=args.pool,
                            normalize=not args.no_norm,
                            global_stats=global_stats,
                            standardizer=standardizer,
                            mol_id=str(row[args.id_col]) if args.id_col and args.id_col in df.columns else f"row_{idx}"
                        )
                        drug_cache[key] = drug_vec
                else:
                    if smi in drug_cache:
                        drug_vec = drug_cache[smi]
                    else:
                        drug_vec = build_drug_embedding_with_model(
                            smiles=smi,
                            model=hg_model,
                            config=config,
                            device=device,
                            pool=args.pool,
                            normalize=not args.no_norm,
                        )
                        drug_cache[smi] = drug_vec
                if seq in prot_cache:
                    prot_vec = prot_cache[seq]
                else:
                    prot_vec = build_protein_embedding_with_model(
                        seq=seq,
                        tokenizer=tokenizer,
                        model=prot_model,
                        device=device,
                        pooling=args.prot_pool,
                    )
                    prot_cache[seq] = prot_vec
                features = fuse_features(drug_vec, prot_vec)
                pred = predict_xgb(booster, features)
                if args.task == 'regression':
                    res = {
                        'affinity': float(pred),
                        'drug_dim': int(drug_vec.shape[0]),
                        'protein_dim': int(prot_vec.shape[0])
                    }
                else:
                    decision = int(pred >= args.threshold)
                    res = {
                        'binding_likelihood': float(pred),
                        'decision': decision,
                        'drug_dim': int(drug_vec.shape[0]),
                        'protein_dim': int(prot_vec.shape[0])
                    }
                if args.id_col and args.id_col in df.columns:
                    res[args.id_col] = row[args.id_col]
                # collect label for metrics
                if args.label_col is not None:
                    if args.label_col not in df.columns:
                        raise ValueError(f"CSV missing label column: {args.label_col}")
                    lbl = row[args.label_col]
                    if pd.isna(lbl):
                        if args.skip_invalid:
                            # skip whole row when label is required for metrics
                            continue
                        else:
                            raise ValueError(f"Row {idx}: label is NaN in column '{args.label_col}'")
                    try:
                        lbl_int = int(lbl)
                    except Exception:
                        raise ValueError(f"Row {idx}: label must be 0/1 integer, got '{lbl}'")
                    if lbl_int not in (0, 1):
                        raise ValueError(f"Row {idx}: label must be 0 or 1, got {lbl_int}")
                    y_true.append(lbl_int)
                results.append(res)
                kept_rows.append(idx)
            except Exception as e:
                if args.skip_invalid:
                    print(f"Row {idx} failed: {e}", file=sys.stderr)
                    continue
                else:
                    raise

        # 组织输出
        if len(results) == 0:
            raise RuntimeError("No valid rows processed.")
        # 不支持CSV路径下的落盘校准器训练（统一用 --calibrate_in_train 在内存中）

        out_df = pd.DataFrame(results)
        # 将原始需要的列拼回（保持顺序，避免多余列引入复杂性）
        src_cols = [c for c in [args.smiles_col, args.sequence_col, args.id_col] if c and c in df.columns]
        if src_cols:
            out_df = pd.concat([df.loc[kept_rows, src_cols].reset_index(drop=True), out_df], axis=1)

        if args.output_csv is None:
            # 默认与输入同目录输出
            args.output_csv = str(csv_path.with_suffix('.pred.csv'))
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        summary = {
            'processed': int(len(results)),
            'saved_to': args.output_csv,
            'drug_dim': int(results[0]['drug_dim']),
            'protein_dim': int(results[0]['protein_dim'])
        }

        # 计算整体指标（若提供了标签列）
        if args.label_col is not None and len(y_true) > 0:
            try:
                # 高精度输出（保留6位小数）
                def _r(v, nd=6):
                    try:
                        return None if v is None else float(round(float(v), nd))
                    except Exception:
                        return v
                if args.task == 'regression':
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    y_pred_vals = [r['affinity'] for r in results]
                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_vals)))
                    mae = float(mean_absolute_error(y_true, y_pred_vals))
                    r2 = float(r2_score(y_true, y_pred_vals)) if len(set(y_true)) > 1 else None
                    summary.update({'rmse': _r(rmse), 'mae': _r(mae), 'r2': _r(r2)})
                else:
                    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
                    y_prob = [r['binding_likelihood'] for r in results]
                    y_pred = [int(p >= args.threshold) for p in y_prob]
                    auc_val = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None
                    summary.update({'accuracy': _r(accuracy_score(y_true, y_pred)), 'auc': _r(auc_val), 'f1': _r(f1_score(y_true, y_pred)), 'recall': _r(recall_score(y_true, y_pred)), 'threshold': _r(args.threshold), 'num_pos': int(sum(y_true)), 'num_neg': int(len(y_true) - sum(y_true))})
            except Exception as me:
                # 明确提示指标计算失败原因，但不阻断导出
                summary['metrics_error'] = str(me)

        print(json.dumps(summary, ensure_ascii=False))

        return

    # ========== 单样本模式（默认走预处理；若 --no_preprocess 则回退） ==========
    if not args.smiles or not args.sequence:
        raise ValueError("Single-sample mode requires --smiles and --sequence (or use --csv)")

    hg_model, config = load_hg_model_robust(args.hg_ckpt, args.hg_config, device)
    if getattr(args, 'timeout_seconds', None) is not None:
        try:
            config['timeout_seconds'] = int(args.timeout_seconds)
        except Exception:
            pass
    if use_preproc:
        if not _PREPROC_AVAILABLE:
            raise RuntimeError("Preprocess module unavailable")
        # 初始化标准化器；全局统计使用“恒等”以避免归一化
        standardizer = MoleculeStandardizer(remove_metals=not args.keep_metals, max_atoms=args.max_atoms)
        std_smiles = standardizer.standardize_smiles(args.smiles) if not args.no_standardize else args.smiles
        if std_smiles is None:
            raise ValueError(f"Standardization failed for SMILES: {args.smiles}")
        global_stats = build_identity_global_stats()
        drug_vec = build_drug_embedding_preprocessed(
            smiles=std_smiles,
            model=hg_model,
            config=config,
            device=device,
            pool=args.pool,
            normalize=not args.no_norm,
            global_stats=global_stats,
            standardizer=standardizer,
            mol_id="query"
        )
    else:
        # Fallback：不使用预处理，但复用已加载的 hg_model，避免 from_pretrained 推断 in_dim 失败
        drug_vec = build_drug_embedding_with_model(
            smiles=args.smiles,
            model=hg_model,
            config=config,
            device=device,
            pool=args.pool,
            normalize=not args.no_norm,
        )

    # 2) Protein embedding
    prot_vec = build_protein_embedding(
        seq=args.sequence,
        protbert_model=args.protbert_model,
        device=device,
        pooling=args.prot_pool,
    )

    # 3) Predict
    booster = load_xgb(args.xgb_model)
    features = fuse_features(drug_vec, prot_vec)
    pred = predict_xgb(booster, features)

    if args.task == 'regression':
        out = {
            'affinity': float(pred),
            'drug_dim': int(drug_vec.shape[0]),
            'protein_dim': int(prot_vec.shape[0])
        }
    else:
        out = {
            'binding_likelihood': float(pred),
            'decision': int(pred >= 0.5),
            'drug_dim': int(drug_vec.shape[0]),
            'protein_dim': int(prot_vec.shape[0])
        }
    # 4) 可选：解释性（单样本）
    explanations = {}
    try:
        if getattr(args, 'explain_atoms', False):
            # 选择用于解释的SMILES（与预处理路径一致）
            smiles_explain = (std_smiles if use_preproc else args.smiles) if 'std_smiles' in locals() else args.smiles
            atoms_report = explain_atoms_kernelshap(
                smiles=smiles_explain,
                hg_model=hg_model,
                config=config,
                device=device,
                pool=args.pool,
                normalize=not args.no_norm,
                booster=booster,
                prot_vec=prot_vec,
                use_preproc=use_preproc,
                standardizer=(standardizer if use_preproc else None),
                global_stats=(global_stats if use_preproc else None),
                std_smiles=(std_smiles if use_preproc else None),
                background=int(getattr(args, 'background', 20)),
                nsamples=int(getattr(args, 'nsamples', 200)),
                bg_strategy=str(getattr(args, 'shap_background_strategy', 'random_keep')),
                topk=int(getattr(args, 'shap_topk', 20)),
            )
            explanations['atoms'] = atoms_report
        if getattr(args, 'explain_residues', False):
            if str(getattr(args, 'residue_explainer', 'occlusion')).lower() == 'occlusion':
                residues_report = explain_residues_occlusion(
                    sequence=args.sequence,
                    protbert_model=args.protbert_model,
                    device=device,
                    drug_vec=drug_vec,
                    booster=booster,
                    occlusion=str(getattr(args, 'prot_occlusion', 'drop')),
                    residue_max=int(getattr(args, 'residue_max', 512)),
                    residue_stride=int(getattr(args, 'residue_stride', 1)),
                    batch_size=int(getattr(args, 'shap_batch', 64)),
                    topk=int(getattr(args, 'shap_topk', 20)),
                    pooling=args.prot_pool,
                )
            else:
                residues_report = explain_residues_kernelshap(
                    sequence=args.sequence,
                    protbert_model=args.protbert_model,
                    device=device,
                    drug_vec=drug_vec,
                    booster=booster,
                    occlusion=str(getattr(args, 'prot_occlusion', 'drop')),
                    background=int(getattr(args, 'background', 20)),
                    nsamples=int(getattr(args, 'nsamples', 200)),
                    bg_strategy=str(getattr(args, 'shap_background_strategy', 'random_keep')),
                    topk=int(getattr(args, 'shap_topk', 20)),
                    pooling=args.prot_pool,
                )
            explanations['residues'] = residues_report
    except Exception as e:
        # 避免解释失败阻断主功能；明确告警
        explanations['error'] = str(e)

    if explanations:
        out['explanations'] = explanations

    print(json.dumps(out, ensure_ascii=False))
    # 文件输出
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    if getattr(args, 'shap_out', None) and explanations:
        Path(args.shap_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.shap_out, 'w') as f:
            json.dump(explanations, f, ensure_ascii=False, indent=2)

    # 5) 可选：保存 RDKit 原子热力图（需 atoms 解释）
    try:
        if explanations and 'atoms' in explanations and (args.viz_atoms_png or args.viz_atoms_svg):
            atoms_expl = explanations['atoms']
            smiles_vis = atoms_expl.get('smiles', (std_smiles if use_preproc else args.smiles) if 'std_smiles' in locals() else args.smiles)
            weights = atoms_expl.get('weights')
            saved = _save_rdkit_atom_heatmap(smiles_vis, weights, out_png=args.viz_atoms_png, out_svg=args.viz_atoms_svg)
            # 将保存路径写回输出
            if 'viz' not in out:
                out['viz'] = {}
            out['viz']['atoms'] = saved
            # 同步落盘（若设置了输出）
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # 打印但不阻断
        try:
            print(json.dumps({'viz_atoms': 'failed', 'error': str(e)}, ensure_ascii=False))
        except Exception:
            pass


if __name__ == '__main__':
    # Reduce RDKit chatter if imported indirectly
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
