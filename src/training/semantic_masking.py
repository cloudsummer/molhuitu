"""
语义感知掩码策略 - 基于分子化学结构的智能掩码

这个模块实现了固定的语义掩码策略，能够：
1. 识别分子中的功能团、环系统、链系统
2. 以语义块为单位进行掩码，而不是随机掩码单个原子
3. 混合语义掩码和随机掩码，平衡效果和鲁棒性
4. 保护重要功能团，避免破坏关键化学结构
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from rdkit import Chem

from .masking_strategies import MaskingStrategy
from ..data.molecular_semantics import MolecularSemanticAnalyzer

logger = logging.getLogger(__name__)


class SemanticMasking(MaskingStrategy):
    """
    固定语义掩码策略
    
    特点：
    - 基于化学结构的语义块掩码
    - 混合语义和随机策略
    - 重要功能团保护
    - 简化的配置和实现
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 基础掩码参数
        self.node_mask_ratio = self.config.get('node_mask_ratio', 0.7)
        self.edge_mask_ratio = self.config.get('edge_mask_ratio', 0.7)
        
        # 语义掩码参数
        semantic_config = self.config.get('semantic', {})
        self.semantic_priority = semantic_config.get('semantic_priority', 0.7)  # 70%概率使用语义掩码
        self.block_types = semantic_config.get('block_types', 
                                             ['functional_group', 'ring', 'chain'])
        self.min_block_size = semantic_config.get('min_block_size', 2)
        self.max_block_size = semantic_config.get('max_block_size', 8)
        # 重要功能团处理策略 - 修复逻辑反转问题
        self.preserve_important = semantic_config.get('preserve_important', True)
        self.enhance_important_masking = semantic_config.get('enhance_important_masking', False)
        
        # preserve_important和enhance_important_masking不能同时启用
        if self.preserve_important and self.enhance_important_masking:
            logger.warning("preserve_important and enhance_important_masking cannot be enabled simultaneously. Using preserve_important=True.")
            self.enhance_important_masking = False
        
        # 重要功能团列表
        self.important_groups = set(semantic_config.get('important_groups', [
            'carboxyl', 'amino', 'hydroxyl', 'nitro', 'cyano', 'phosphate'
        ]))
        
        # KISS优化: 语义掩码抖动参数 - 防止过度规律化
        jitter_config = semantic_config.get('jittering', {})
        self.enable_jittering = jitter_config.get('enable', True)
        self.prob_jitter_std = jitter_config.get('prob_jitter_std', 0.1)  # 掩码概率抖动标准差
        self.boundary_jitter_prob = jitter_config.get('boundary_jitter_prob', 0.2)  # 边界抖动概率
        self.ratio_jitter_std = jitter_config.get('ratio_jitter_std', 0.05)  # 掩码比例抖动标准差
        
        # 语义块类型别名映射 (支持单复数形式)
        self.block_type_aliases = {
            'functional_groups': 'functional_group',
            'rings': 'ring', 
            'chains': 'chain'
        }
        
        # 标准化block_types
        self.block_types = [self.block_type_aliases.get(bt, bt) for bt in self.block_types]
        
        # 分子语义分析器
        self.analyzer = MolecularSemanticAnalyzer()
        
        # 预处理缓存配置
        cache_config = self.config.get('cache', {})
        self.cache_file = cache_config.get('file', None)
        self.enable_cache = cache_config.get('enable', True)
        self.semantic_cache = {}
        
        # 加载预处理缓存
        if self.enable_cache and self.cache_file:
            self._load_semantic_cache()
        
        # 可复现性配置
        random_config = self.config.get('random', {})
        self.base_seed = random_config.get('base_seed', 42)
        self.use_deterministic = random_config.get('deterministic', False)
        self.mask_rng_state = None  # 存储当前RNG状态用于重放
        
        # 统计信息 - 修复KeyError：使用defaultdict
        from collections import defaultdict
        self.stats = {
            'semantic_attempts': 0,
            'semantic_successes': 0,
            'random_fallbacks': 0,
            'total_masks_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'block_type_usage': defaultdict(int)  # 自动初始化新键为0
        }
        # 为已知类型预置计数
        for bt in self.block_types:
            self.stats['block_type_usage'][bt] = 0
        
        # 输出策略配置信息
        strategy_info = "preserve_important" if self.preserve_important else ("enhance_important" if self.enhance_important_masking else "neutral")
        logger.info(f"SemanticMasking initialized: semantic_priority={self.semantic_priority}, "
                   f"block_types={self.block_types}, strategy={strategy_info}")
        if self.semantic_cache:
            logger.info(f"Loaded semantic cache with {len(self.semantic_cache)} molecules")
    
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                      hyperedge_attr: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成语义感知的掩码
        
        Args:
            x: 节点特征 [num_nodes, feature_dim]
            hyperedge_index: 超边连接 [2, num_connections]
            hyperedge_attr: 超边属性 [num_edges, edge_dim]
            **kwargs: 额外参数，包括 smiles, mol_info, batch 等
            
        Returns:
            (node_mask, edge_mask) 元组
        """
        num_nodes = x.size(0)
        # 一次性掩码比例覆盖（来自Trainer，通过kwargs传入）
        self._ratio_override = None
        if 'mask_ratio' in kwargs and kwargs['mask_ratio'] is not None:
            try:
                self._ratio_override = float(kwargs['mask_ratio'])
            except Exception:
                self._ratio_override = None
        # 边数量推断：优先使用hyperedge_attr，否则从hyperedge_index推断
        if hyperedge_attr is not None:
            num_edges = hyperedge_attr.size(0)
        elif hyperedge_index is not None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max().item()) + 1
        else:
            num_edges = 0
        device = x.device
        
        self.stats['total_masks_generated'] += 1
        
        # 设置可复现的随机种子
        step = kwargs.get('global_step', 0)
        epoch = kwargs.get('epoch', 0)
        if self.use_deterministic:
            self.set_mask_seed(step, epoch)
        
        # 获取SMILES和批处理信息
        smiles = kwargs.get('smiles', None)
        batch_tensor = kwargs.get('batch', None)  # PyG batch tensor
        
        # 处理批处理情况
        if batch_tensor is not None and smiles is not None and isinstance(smiles, (list, tuple)):
            return self._generate_batch_semantic_masks(
                x, hyperedge_index, hyperedge_attr, smiles, batch_tensor, device
            )
        
        # 单分子情况（向后兼容）
        single_smiles = None
        if smiles is not None:
            if isinstance(smiles, (list, tuple)) and len(smiles) > 0:
                single_smiles = smiles[0]
            else:
                single_smiles = str(smiles)
        
        # 始终尝试生成语义掩码（去掉随机概率判断）
        self.stats['semantic_attempts'] += 1
        
        # 获取分子信息
        mol_info = kwargs.get('mol_info')
        
        # 生成语义掩码
        semantic_masks = self._generate_semantic_masks(
            num_nodes, num_edges, device, single_smiles, mol_info, hyperedge_index
        )
        
        if semantic_masks is not None:
            self.stats['semantic_successes'] += 1
            node_mask, edge_mask = semantic_masks
            
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    nm = int(node_mask.sum())
                    em = int(edge_mask.sum())
                    logger.debug(f"Semantic masking applied: {nm}/{num_nodes} nodes, {em}/{num_edges} edges")
                except Exception:
                    pass
            # 清理覆盖标记
            self._ratio_override = None
            return node_mask, edge_mask
        
        # 回退到随机掩码
        # 显式提示：无SMILES/分子信息时回退到随机掩码
        if single_smiles is None and kwargs.get('mol_info') is None:
            logger.info("Semantic masking fallback: no SMILES/mol_info provided; using random masking")
        self.stats['random_fallbacks'] += 1
        out_masks = self._generate_random_masks(num_nodes, num_edges, device)
        # 清理覆盖标记
        self._ratio_override = None
        return out_masks
    
    def _generate_semantic_masks(self, num_nodes: int, num_edges: int, device: torch.device,
                               smiles: str = None, mol_info: Dict = None, 
                               hyperedge_index: Optional[torch.Tensor] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        生成基于分子语义的掩码，防止结构泄露
        
        Args:
            num_nodes: 节点数量
            num_edges: 边数量  
            device: 设备
            smiles: SMILES字符串
            mol_info: 预计算的分子信息
            hyperedge_index: 超边索引，用于生成结构泄露防护的超边掩码
            
        Returns:
            (node_mask, hyperedge_mask) 或 None（如果无法生成语义掩码）
        """
        # 获取或计算分子语义信息（优先使用缓存）
        if mol_info is None and smiles:
            mol_info = self._get_cached_mol_info(smiles)
            if mol_info is None:
                return None
        
        if not mol_info:
            logger.debug("No molecular information available for semantic masking")
            return None
        
        # 获取语义块
        semantic_blocks = mol_info.get('semantic_blocks', {})
        if not semantic_blocks or not any(semantic_blocks.values()):
            logger.debug("No semantic blocks found in molecular analysis")
            return None
        
        # 生成语义节点掩码
        node_mask = self._create_semantic_node_mask(
            num_nodes, semantic_blocks, mol_info, device
        )
        
        # 生成防结构泄露的超边掩码，确保长度与num_edges一致
        if hyperedge_index is not None:
            hyperedge_mask = self._generate_structure_aware_hyperedge_mask_fixed(
                node_mask, hyperedge_index, num_edges, device
            )
        else:
            # 回退到简单边掩码
            hyperedge_mask = self._generate_edge_mask(num_edges, device)
        
        return node_mask, hyperedge_mask
    
    def _generate_batch_semantic_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                                     hyperedge_attr: torch.Tensor, smiles_list: List[str], 
                                     batch_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为批处理中的每个分子生成语义掩码
        
        Args:
            x: 节点特征 [total_nodes, feature_dim]
            hyperedge_index: 超边连接 [2, num_connections]
            hyperedge_attr: 超边属性 [num_edges, edge_dim]
            smiles_list: 分子SMILES列表
            batch_tensor: PyG批处理张量，指示每个节点属于哪个分子
            device: 设备
            
        Returns:
            (node_mask, edge_mask) 元组
        """
        total_nodes = x.size(0)
        # 批处理场景下的边数量推断：优先用hyperedge_attr，否则从hyperedge_index推断
        if hyperedge_attr is not None:
            total_edges = hyperedge_attr.size(0)
        elif hyperedge_index is not None and hyperedge_index.numel() > 0:
            total_edges = int(hyperedge_index[1].max().item()) + 1
        else:
            total_edges = 0
        
        # 初始化掩码张量
        node_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
        edge_mask = torch.zeros(total_edges, dtype=torch.bool, device=device) if total_edges > 0 else torch.empty(0, dtype=torch.bool, device=device)
        
        # GPU-optimized: 避免CPU转换，完全在GPU上处理
        unique_graphs = batch_tensor.unique()  # 保持在GPU上
        
        # 修复统计口径：跟踪实际的语义掩码成功数
        actual_semantic_successes = 0
        
        # 使用GPU循环替代CPU列表遍历
        for i in range(unique_graphs.size(0)):
            graph_idx_tensor = unique_graphs[i]  # GPU张量
            # 后续需要转换为Python int时才同步
            graph_idx = graph_idx_tensor.item()  # 延迟到必需时刻
            try:
                # 获取当前分子的节点
                graph_node_indices = torch.where(batch_tensor == graph_idx)[0]
                num_graph_nodes = len(graph_node_indices)
                
                # 获取对应的SMILES（检查索引边界）
                if graph_idx < len(smiles_list):
                    current_smiles = smiles_list[graph_idx]
                    
                    # 提取当前图的超边索引（兼容hyperedge_index为None或空）
                    if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                        graph_hyperedge_mask = (batch_tensor[hyperedge_index[0]] == graph_idx)
                        graph_hyperedge_index = hyperedge_index[:, graph_hyperedge_mask] if graph_hyperedge_mask.any() else None
                    else:
                        graph_hyperedge_mask = torch.zeros(0, dtype=torch.bool, device=device)
                        graph_hyperedge_index = None
                    
                    # 为当前分子生成语义掩码（只需节点掩码）
                    semantic_masks = self._generate_semantic_masks(
                        num_graph_nodes, 0, device, current_smiles, None, None
                    )
                    
                    if semantic_masks is not None:
                        graph_node_mask, _ = semantic_masks
                        # 更新全局节点掩码
                        node_mask[graph_node_indices] = graph_node_mask
                        # 修复统计：实际成功生成语义掩码
                        actual_semantic_successes += 1
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Applied semantic mask to molecule {graph_idx}: "
                                f"{int(graph_node_mask.sum().item())}/{num_graph_nodes} nodes"
                            )
                    else:
                        # 显式提示：当前分子缺少SMILES时回退
                        if current_smiles is None:
                            logger.info(f"Semantic masking fallback: molecule {graph_idx} missing SMILES; using random masking")
                        # 回退到随机掩码 - KISS优化: 使用抖动比例
                        jittered_node_ratio, _ = self._get_jittered_ratios()
                        random_mask = torch.rand(num_graph_nodes, device=device) < jittered_node_ratio
                        node_mask[graph_node_indices] = random_mask
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Applied random fallback mask to molecule {graph_idx}: "
                                f"{int(random_mask.sum().item())}/{num_graph_nodes} nodes"
                            )
                        self.stats['random_fallbacks'] += 1
                else:
                    # 如果SMILES不足，使用随机掩码 - KISS优化: 使用抖动比例
                    logger.warning(f"No SMILES available for molecule {graph_idx}, using random mask")
                    jittered_node_ratio, _ = self._get_jittered_ratios()
                    random_mask = torch.rand(num_graph_nodes, device=device) < jittered_node_ratio
                    node_mask[graph_node_indices] = random_mask
                    
                    self.stats['random_fallbacks'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing molecule {graph_idx} in batch: {e}")
                # 错误时使用随机掩码
                graph_node_indices = torch.where(batch_tensor == graph_idx)[0]
                num_graph_nodes = len(graph_node_indices)
                random_mask = torch.rand(num_graph_nodes, device=device) < self.node_mask_ratio
                node_mask[graph_node_indices] = random_mask
                
                # 为错误处理生成结构感知边掩码
                try:
                    if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                        graph_hyperedge_mask = (batch_tensor[hyperedge_index[0]] == graph_idx)
                        graph_hyperedge_index = hyperedge_index[:, graph_hyperedge_mask]
                    else:
                        graph_hyperedge_index = None
                except Exception as edge_error:
                    logger.warning(f"Failed to generate structure-aware edge mask for error case: {edge_error}")
                
                self.stats['random_fallbacks'] += 1
        
        # 结构感知边掩码补齐：将结构掩码补齐到目标edge_mask_ratio - KISS优化: 使用抖动比例
        if total_edges > 0:
            # 一次性全局结构感知边掩码：基于最终node_mask和全量hyperedge_index
            if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                struct_mask_global = self._generate_structure_aware_hyperedge_mask_fixed(
                    node_mask, hyperedge_index, total_edges, device
                )
                if struct_mask_global.numel() == edge_mask.numel():
                    edge_mask |= struct_mask_global
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(
                            f"Global structure-aware edges masked: {int(edge_mask.sum().item())}/{total_edges}"
                        )
                    except Exception:
                        pass

            # 优先从与已遮节点相连的候选超边集合中补齐，保持结构一致性
            candidate_edges = torch.empty(0, dtype=torch.long, device=device)
            if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                try:
                    conn_nodes = hyperedge_index[0]
                    conn_edges = hyperedge_index[1]
                    involved_edges = conn_edges[node_mask[conn_nodes]].unique()
                    # 过滤越界ID
                    candidate_edges = involved_edges[involved_edges < total_edges]
                except Exception:
                    candidate_edges = torch.empty(0, dtype=torch.long, device=device)

            _, jittered_edge_ratio = self._get_jittered_ratios()
            current_masked_edges = edge_mask.sum().item()
            target_masked_edges = int(total_edges * jittered_edge_ratio)
            
            if current_masked_edges < target_masked_edges:
                # 需要补充掩码：在未被结构规则覆盖的边中随机选择
                remaining_edges = target_masked_edges - current_masked_edges
                unmasked_indices = torch.where(~edge_mask)[0]
                # 首选候选集合（与被遮节点相连的超边）
                if candidate_edges.numel() > 0 and unmasked_indices.numel() > 0:
                    in_pool = torch.zeros(total_edges, dtype=torch.bool, device=device)
                    in_pool[candidate_edges] = True
                    primary_pool = unmasked_indices[in_pool[unmasked_indices]]
                else:
                    primary_pool = torch.empty(0, dtype=torch.long, device=device)
                
                if len(unmasked_indices) > 0:
                    # 随机选择需要补充的边
                    pool = primary_pool if primary_pool.numel() > 0 else unmasked_indices
                    if remaining_edges >= len(pool):
                        edge_mask[pool] = True
                    else:
                        perm = torch.randperm(len(pool), device=pool.device)
                        selected_indices = pool[perm[:remaining_edges]]
                        edge_mask[selected_indices] = True
                        
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(f"Edge mask补齐: {current_masked_edges} -> {int(edge_mask.sum())}/{total_edges} "
                                     f"(target: {target_masked_edges})")
                    except Exception:
                        pass
            elif current_masked_edges > target_masked_edges:
                # 结构掩码超出目标：稀释掩码
                excess_edges = current_masked_edges - target_masked_edges
                masked_indices = torch.where(edge_mask)[0]
                if excess_edges > 0 and len(masked_indices) > excess_edges:
                    # 随机取消一些掩码
                    perm = torch.randperm(len(masked_indices), device=masked_indices.device)
                    deselected_indices = masked_indices[perm[:excess_edges]]
                    edge_mask[deselected_indices] = False
                    
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(f"Edge mask稀释: {current_masked_edges} -> {int(edge_mask.sum())}/{total_edges} "
                                     f"(target: {target_masked_edges})")
                    except Exception:
                        pass
        
        # 修复统计信息：基于实际的语义掩码生成成功数
        self.stats['semantic_attempts'] += len(unique_graphs)
        self.stats['semantic_successes'] += actual_semantic_successes
        
        logger.debug(f"Batch semantic masking: {actual_semantic_successes}/{len(unique_graphs)} molecules processed successfully with semantic masks")
        
        return node_mask, edge_mask
    
    
    def _create_semantic_node_mask(self, num_nodes: int, semantic_blocks: Dict[str, List[List[int]]],
                                 mol_info: Dict, device: torch.device) -> torch.Tensor:
        """
        基于语义块创建节点掩码 - 实现真正的70%语义+30%随机混合策略
        
        Args:
            num_nodes: 节点总数
            semantic_blocks: 语义块信息
            mol_info: 完整分子信息
            device: 设备
            
        Returns:
            节点掩码张量
        """
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        masked_atoms = set()
        # KISS优化: 使用抖动比例计算目标掩码数量
        # 根因修复：小分子会出现target=0导致空掩码；改为向上取整并下限为1（当ratio>0）
        import math
        jittered_node_ratio, _ = self._get_jittered_ratios()
        if jittered_node_ratio <= 0 or num_nodes == 0:
            target_mask_count = 0
        else:
            target_mask_count = max(1, min(num_nodes, int(math.ceil(num_nodes * float(jittered_node_ratio)))))
        
        # 预先计算可用的语义块（用于小分子边界处理）
        available_blocks = []
        for block_type in self.block_types:
            if block_type in semantic_blocks:
                for block in semantic_blocks[block_type]:
                    if self._is_valid_block(block, num_nodes):
                        available_blocks.append((block_type, block))
        
        # 统一策略：固定掩码比例，语义/随机按比例拆分
        # 语义部分占比由 semantic_priority 控制（默认0.7）
        semantic_target = int(max(1 if target_mask_count > 0 else 0,
                                  target_mask_count * self.semantic_priority))
        
        # 第一阶段：使用背包算法精确选择语义块（available_blocks已预先计算）
        if available_blocks:
            selected_blocks = self._knapsack_block_selection(
                available_blocks, semantic_target, num_nodes, mol_info, masked_atoms
            )
            
            # 应用选择的语义块 - KISS优化: 边界抖动
            for block_type, block in selected_blocks:
                block_atoms = [atom for atom in block if atom < num_nodes and atom not in masked_atoms]
                
                # KISS优化: 应用边界抖动防止完美块边界记忆
                if self.enable_jittering and len(block_atoms) > 2:
                    jittered_atoms = self._apply_boundary_jittering(block_atoms)
                    for atom_idx in jittered_atoms:
                        node_mask[atom_idx] = True
                        masked_atoms.add(atom_idx)
                else:
                    # 无抖动情况，直接应用整个块
                    for atom_idx in block_atoms:
                        node_mask[atom_idx] = True
                        masked_atoms.add(atom_idx)
                
                if block_atoms:  # 只有实际掩码了原子才记录
                    # defaultdict自动处理新键初始化
                    self.stats['block_type_usage'][block_type] += 1
                    logger.debug(f"Masked {block_type} block: {len(block_atoms)} atoms")
            
            # 计算实际掩码比例
            actual_ratio = len(masked_atoms) / semantic_target if semantic_target > 0 else 0
            logger.debug(f"Knapsack semantic masking: {len(masked_atoms)}/{semantic_target} target "
                        f"({actual_ratio:.1%} accuracy)")
        else:
            logger.debug("No valid semantic blocks found")
        
        # 第二阶段：随机掩码补充到目标数量（带边界保护）
        remaining_atoms = [i for i in range(num_nodes) if i not in masked_atoms]
        if remaining_atoms and len(masked_atoms) < target_mask_count:
            additional_needed = min(target_mask_count - len(masked_atoms), len(remaining_atoms))
            
            if additional_needed > 0:
                try:
                    additional_atoms = random.sample(remaining_atoms, additional_needed)
                    for atom_idx in additional_atoms:
                        node_mask[atom_idx] = True
                    
                    logger.debug(f"Added {additional_needed} random atoms to reach target ratio")
                except ValueError as e:
                    # 理论上不应该发生，但提供恢复机制
                    logger.warning(f"Random sampling failed: {e}, using all remaining atoms")
                    for atom_idx in remaining_atoms:
                        node_mask[atom_idx] = True
        
        # Calculate final masked count without torch.compile graph break
        final_masked = node_mask.sum()
        
        # 修复小分子掩码矛盾：确保ratio>0时至少掩码1个原子（包括1-2原子分子）
        if final_masked == 0 and target_mask_count > 0 and num_nodes > 0:
            logger.warning(f"No atoms masked for {num_nodes}-atom molecule (target={target_mask_count}), masking one atom by rule")
            random_atom = random.randint(0, num_nodes - 1)
            node_mask[random_atom] = True
            final_masked = 1
        
        # 修复Tensor格式化错误：先转换为Python数值
        final_masked_int = int(final_masked.item()) if isinstance(final_masked, torch.Tensor) else final_masked
        logger.debug(f"Final masking: {final_masked_int}/{num_nodes} = {final_masked_int/num_nodes:.1%}")
        
        return node_mask
    
    def _is_valid_block(self, block: List[int], num_nodes: int) -> bool:
        """检查语义块是否有效"""
        if not block or len(block) < self.min_block_size or len(block) > self.max_block_size:
            return False
        
        # 检查原子索引是否在有效范围内
        return all(0 <= atom_idx < num_nodes for atom_idx in block)
    
    def _get_block_priority(self, block_type: str, block: List[int], mol_info: Dict) -> int:
        """
        获取语义块的掩码优先级（数字越小，优先级越高，越容易被掩码）
        
        药物化学合理性考虑：
        - 功能团是药理活性的核心，应该高频掩码以学习其重要性
        - 环系统承载骨架信息，次要掩码
        - 链系统信息含量较低，可适度掩码
        
        Returns:
            优先级分数（0-100，越小越优先掩码）
        """
        # 重新设计的科学合理优先级
        base_priority = {
            'functional_group': 15,  # 功能团最高掩码优先级（药理活性核心）
            'ring': 25,             # 环系统次高优先级（结构骨架）
            'aromatic': 20,         # 芳香系统高优先级（电子特性重要）
            'chain': 35,            # 链系统中等优先级（结构连接）
            'complex': 30           # 复杂结构中等优先级（特殊结构特征）
        }
        
        priority = base_priority.get(block_type, 50)
        
        # 重要功能团的策略处理 - 修复逻辑反转
        if block_type == 'functional_group':
            functional_groups = mol_info.get('functional_group', [])
            for fg_info in functional_groups:
                if isinstance(fg_info, dict):
                    fg_type = fg_info.get('type', '')
                    fg_atoms = fg_info.get('atoms', [])
                    
                    # 检查当前块是否与重要功能团重叠
                    if fg_type in self.important_groups and set(block) & set(fg_atoms):
                        if self.preserve_important:
                            # 保护模式：降低掩码优先级（数值越大优先级越低）
                            priority = min(95, priority + 20)
                            logger.debug(f"Protected important functional group (reduced masking): {fg_type}")
                        elif self.enhance_important_masking:
                            # 学习模式：提高掩码优先级（数值越小优先级越高）
                            priority = max(5, priority - 10)
                            logger.debug(f"Enhanced masking for important functional group: {fg_type}")
                        break
        
        return priority
    
    def _should_mask_block(self, block_type: str, block: List[int], mol_info: Dict) -> bool:
        """
        决定是否应该掩码某个语义块
        
        科学合理的掩码策略：
        - MAE训练中，掩码重要结构有助于学习其表征
        - 功能团是药理活性核心，高概率掩码
        - 维持适当的随机性避免过拟合
        
        Args:
            block_type: 块类型
            block: 原子索引列表
            mol_info: 分子信息
            
        Returns:
            是否掩码该块
        """
        # 重新设计的科学合理掩码概率
        base_probabilities = {
            'functional_group': 0.85,   # 功能团高掩码概率（学习药理活性特征）
            'aromatic': 0.75,          # 芳香系统高掩码概率（电子特性重要）
            'ring': 0.70,              # 环系统中等掩码概率（结构骨架）
            'complex': 0.65,           # 复杂结构中等掩码概率
            'chain': 0.55              # 链系统较低掩码概率（信息含量相对较低）
        }
        
        mask_prob = base_probabilities.get(block_type, 0.60)
        
        # 重要功能团的策略处理 - 修复逻辑反转
        if block_type == 'functional_group':
            functional_groups = mol_info.get('functional_group', [])
            for fg_info in functional_groups:
                if isinstance(fg_info, dict):
                    fg_type = fg_info.get('type', '')
                    fg_atoms = fg_info.get('atoms', [])
                    
                    # 重要功能团策略调整
                    if fg_type in self.important_groups and set(block) & set(fg_atoms):
                        if self.preserve_important:
                            # 保护模式：大幅降低掩码概率
                            mask_prob = max(0.05, mask_prob - 0.30)  # 降低30%掩码概率，最低5%
                            logger.debug(f"Protected important functional group (reduced masking): {fg_type}")
                        elif self.enhance_important_masking:
                            # 学习模式：提高掩码概率
                            mask_prob = min(mask_prob + 0.10, 0.95)  # 增加10%掩码概率，最高95%
                            logger.debug(f"Enhanced masking for important functional group: {fg_type}")
                        break
        
        # KISS优化: 语义掩码抖动 - 防止模式记忆
        if self.enable_jittering:
            # 高斯噪声抖动掩码概率
            prob_noise = np.random.normal(0, self.prob_jitter_std)
            jittered_prob = mask_prob + prob_noise
            # 确保概率在合理范围内
            final_prob = np.clip(jittered_prob, 0.05, 0.95)
            logger.debug(f"Applied probability jittering: {mask_prob:.3f} -> {final_prob:.3f}")
        else:
            # 保留原有的小幅随机扰动作为回退
            random_factor = 0.9 + 0.2 * random.random()  # 0.9-1.1的随机因子
            final_prob = min(mask_prob * random_factor, 0.95)
        
        return random.random() < final_prob
    
    def _apply_boundary_jittering(self, block_atoms: List[int]) -> List[int]:
        """
        KISS优化: 对语义块边界进行随机抖动，防止模式记忆
        
        Args:
            block_atoms: 语义块中的原子索引列表
            
        Returns:
            抖动后的原子索引列表
        """
        if not self.enable_jittering or len(block_atoms) <= 2:
            return block_atoms
            
        jittered_atoms = block_atoms.copy()
        
        # 随机移除一些边界原子 
        if len(jittered_atoms) > 3:
            num_to_remove = max(1, int(len(jittered_atoms) * self.boundary_jitter_prob))
            indices_to_remove = random.sample(range(len(jittered_atoms)), num_to_remove)
            jittered_atoms = [atom for i, atom in enumerate(jittered_atoms) if i not in indices_to_remove]
            
        logger.debug(f"Boundary jittering: {len(block_atoms)} -> {len(jittered_atoms)} atoms")
        return jittered_atoms
    
    def _get_jittered_ratios(self) -> Tuple[float, float]:
        """
        KISS优化: 获取抖动后的掩码比例，防止固定比例记忆；支持一次性覆盖
        
        Returns:
            (jittered_node_ratio, jittered_edge_ratio)
        """
        base_node = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else self.node_mask_ratio
        base_edge = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else self.edge_mask_ratio

        if not self.enable_jittering:
            return base_node, base_edge
            
        # 对掩码比例添加高斯噪声
        node_noise = np.random.normal(0, self.ratio_jitter_std)
        edge_noise = np.random.normal(0, self.ratio_jitter_std)
        
        jittered_node_ratio = np.clip(base_node + node_noise, 0.1, 0.9)
        jittered_edge_ratio = np.clip(base_edge + edge_noise, 0.1, 0.9)
        
        logger.debug(f"Ratio jittering: node {self.node_mask_ratio:.3f}->{jittered_node_ratio:.3f}, "
                    f"edge {base_edge:.3f}->{jittered_edge_ratio:.3f}")
        
        return jittered_node_ratio, jittered_edge_ratio
    
    def _generate_structure_aware_hyperedge_mask(self, node_mask: torch.Tensor, 
                                               hyperedge_index: torch.Tensor, 
                                               device: torch.device) -> torch.Tensor:
        """
        生成防结构泄露的超边掩码
        
        核心思想：被遮原子参与的超边也需要被遮，防止模型通过超边归属推断被遮片段
        
        Args:
            node_mask: 节点掩码 [num_nodes]
            hyperedge_index: 超边索引 [2, num_connections] (node_idx, hyperedge_idx)
            device: 设备
            
        Returns:
            超边掩码 [num_hyperedges]
        """
        # 修复超边空检查：检查hyperedge_index是否为空
        if hyperedge_index is None or hyperedge_index.numel() == 0 or hyperedge_index.size(1) == 0:
            return torch.empty(0, dtype=torch.bool, device=device)
        
        # 假设hyperedge_index格式为 [node_indices, hyperedge_indices]
        node_indices, hyperedge_indices = hyperedge_index[0], hyperedge_index[1]
        
        # 找出所有被遮原子参与的超边
        masked_nodes = node_mask.nonzero(as_tuple=True)[0]
        
        # 修复超边掩码逻辑：基于节点掩码状态决定超边是否掩码
        if hyperedge_indices.numel() > 0:
            # 获取超边的总数（最大超边索引+1）
            max_hyperedge_idx = hyperedge_indices.max().item() if hyperedge_indices.numel() > 0 else -1
            num_hyperedges = max_hyperedge_idx + 1
            
            # 创建正确大小的超边掩码（覆盖所有可能的超边）
            hyperedge_mask = torch.zeros(num_hyperedges, dtype=torch.bool, device=device)
            
            # 遍历被遮原子，标记其参与的超边
            for masked_node in masked_nodes:
                # 找到该原子参与的所有超边索引
                participating_mask = (node_indices == masked_node)
                if participating_mask.any():
                    participating_hyperedges = hyperedge_indices[participating_mask]
                    # 标记这些超边为被掩码
                    if participating_hyperedges.numel() > 0:
                        hyperedge_mask.scatter_(0, participating_hyperedges, True)
        else:
            hyperedge_mask = torch.empty(0, dtype=torch.bool, device=device)
        
        # Skip debug logging to avoid torch.compile graph breaks
        # logger.debug(f"Structure-aware masking: {hyperedge_mask.sum().item()}/{hyperedge_indices.max().item() + 1 if hyperedge_indices.numel() > 0 else 0} hyperedges masked")
        return hyperedge_mask
    
    def _generate_structure_aware_hyperedge_mask_fixed(self, node_mask: torch.Tensor, 
                                                     hyperedge_index: torch.Tensor, 
                                                     num_edges: int, 
                                                     device: torch.device) -> torch.Tensor:
        """
        生成防结构泄露的超边掩码，确保长度与num_edges一致
        
        Args:
            node_mask: 节点掩码 [num_nodes]
            hyperedge_index: 超边索引 [2, num_connections] (node_idx, hyperedge_idx)
            num_edges: 期望的超边掩码长度
            device: 设备
            
        Returns:
            超边掩码 [num_edges]
        """
        # 修复超边空检查：检查hyperedge_index是否为空
        if hyperedge_index is None or hyperedge_index.numel() == 0 or hyperedge_index.size(1) == 0:
            return torch.zeros(num_edges, dtype=torch.bool, device=device)
        
        # 假设hyperedge_index格式为 [node_indices, hyperedge_indices]
        node_indices, hyperedge_indices = hyperedge_index[0], hyperedge_index[1]
        
        # 找出所有被遮原子参与的超边
        masked_nodes = node_mask.nonzero(as_tuple=True)[0]
        
        # 创建固定长度的超边掩码
        hyperedge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        
        # 修复超边掩码逻辑：基于节点掩码状态决定超边是否掩码
        if hyperedge_indices.numel() > 0 and masked_nodes.numel() > 0:
            # 遍历被遮原子，标记其参与的超边
            for masked_node in masked_nodes:
                # 找到该原子参与的所有超边索引
                participating_mask = (node_indices == masked_node)
                if participating_mask.any():
                    participating_hyperedges = hyperedge_indices[participating_mask]
                    # 只对在有效范围内的超边ID设置掩码
                    if participating_hyperedges.numel() > 0:
                        valid_hyperedges = participating_hyperedges[participating_hyperedges < num_edges]
                        if valid_hyperedges.numel() > 0:
                            hyperedge_mask.scatter_(0, valid_hyperedges, True)
        
        return hyperedge_mask
    
    def _knapsack_block_selection(self, available_blocks: List[Tuple[str, List[int]]], 
                                 target_size: int, num_nodes: int, mol_info: Dict, 
                                 masked_atoms: set) -> List[Tuple[str, List[int]]]:
        """
        使用背包算法精确选择语义块，确保掩码数量接近目标±5%
        
        Args:
            available_blocks: 可用的语义块列表 [(block_type, block_atoms), ...]
            target_size: 目标掩码原子数
            num_nodes: 总节点数
            mol_info: 分子信息
            masked_atoms: 已掩码的原子集合
            
        Returns:
            选择的语义块列表
        """
        if not available_blocks or target_size <= 0:
            return []
        
        # 计算每个块的有效大小（去除已掩码和超出边界的原子）
        block_items = []
        for i, (block_type, block) in enumerate(available_blocks):
            valid_atoms = [atom for atom in block if atom < num_nodes and atom not in masked_atoms]
            if not valid_atoms:  # 跳过空块
                continue
                
            size = len(valid_atoms)
            # 计算价值（修复优先级逻辑：数值越小优先级越高，需要反转）
            priority_score = self._get_block_priority(block_type, block, mol_info)
            mask_prob = 1.0 if self._should_mask_block(block_type, block, mol_info) else 0.3
            
            # 修复价值函数：优先级得分越低（越重要），价值应该越高
            # 使用 (100 - priority_score) 反转优先级，确保重要块得到高价值
            inverted_priority = max(1, 101 - priority_score)  # 确保正值，范围 1-100
            value = inverted_priority * mask_prob * size
            
            block_items.append({
                'index': i,
                'block_type': block_type,
                'block': valid_atoms,  # 存储有效原子列表
                'size': size,
                'value': value,
                'priority_score': priority_score,  # 原始优先级得分（越小越重要）
                'inverted_priority': inverted_priority  # 反转后的优先级（越大越重要）
            })
        
        if not block_items:
            return []
        
        # 动态调整误差范围：小分子更宽容，大分子更严格
        if target_size <= 3:
            tolerance_pct = 0.15  # 15%的容忍度
        elif target_size <= 6:
            tolerance_pct = 0.10  # 10%的容忍度  
        else:
            tolerance_pct = 0.05  # 5%的容忍度
        
        tolerance = max(1, int(target_size * tolerance_pct))
        min_target = max(1, target_size - tolerance)
        max_target = target_size + tolerance
        
        # 贪心近似背包算法：按价值密度排序
        block_items.sort(key=lambda x: x['value'] / x['size'], reverse=True)
        
        selected_blocks = []
        current_size = 0
        
        # 第一轮：贪心选择高价值密度的块
        for item in block_items:
            if current_size + item['size'] <= max_target:
                selected_blocks.append((item['block_type'], item['block']))
                current_size += item['size']
                if current_size >= min_target:
                    break
        
        # 第二轮：如果还没达到最小目标，考虑较小的块
        if current_size < min_target:
            remaining_items = [item for item in block_items 
                             if (item['block_type'], item['block']) not in 
                             [(bt, b) for bt, b in selected_blocks]]
            
            # 按大小排序，优先选择能填补空缺的小块
            remaining_items.sort(key=lambda x: abs((min_target - current_size) - x['size']))
            
            for item in remaining_items:
                if current_size + item['size'] <= max_target:
                    selected_blocks.append((item['block_type'], item['block']))
                    current_size += item['size']
                    if current_size >= min_target:
                        break
        
        # 第三轮：精确调整 - 修复语义完整性问题
        if current_size < min_target and target_size >= 3:
            remaining_gap = min_target - current_size
            
            # 优先寻找大小合适的完整小块
            small_remaining_items = [item for item in block_items 
                                   if (item['block_type'], item['block']) not in 
                                   [(bt, b) for bt, b in selected_blocks] and 
                                   item['size'] <= remaining_gap]
            
            if small_remaining_items:
                # 选择最大的完整小块
                best_small_item = max(small_remaining_items, key=lambda x: x['size'])
                selected_blocks.append((best_small_item['block_type'], best_small_item['block']))
                current_size += best_small_item['size']
            else:
                # 作为最后手段，只对chain类型允许部分选择
                chain_items = [item for item in block_items 
                             if (item['block_type'], item['block']) not in 
                             [(bt, b) for bt, b in selected_blocks] and 
                             item['block_type'] == 'chain' and  # 仅chain类型
                             item['size'] > remaining_gap]
                
                if chain_items:
                    # 选择最接近需求的chain块进行部分选择
                    best_chain = min(chain_items, key=lambda x: x['size'] - remaining_gap)
                    partial_block = best_chain['block'][:remaining_gap]  # 只取前面的原子
                    selected_blocks.append((best_chain['block_type'], partial_block))
                    current_size += len(partial_block)
                    logger.debug(f"Applied partial chain selection: {len(partial_block)}/{best_chain['size']} atoms")
                else:
                    logger.debug(f"No suitable blocks for gap filling ({remaining_gap} atoms needed)")
        
        logger.debug(f"Knapsack selected {len(selected_blocks)} blocks, "
                    f"total size: {current_size}/{target_size} "
                    f"({current_size/target_size:.1%})")
        
        return selected_blocks
    
    def _load_semantic_cache(self):
        """加载预处理的语义块缓存"""
        try:
            import pickle
            from pathlib import Path
            
            cache_path = Path(self.cache_file)
            if not cache_path.exists():
                logger.warning(f"Cache file not found: {cache_path}")
                return
            
            with open(cache_path, 'rb') as f:
                self.semantic_cache = pickle.load(f)
            
            logger.info(f"Successfully loaded semantic cache from {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to load semantic cache: {e}")
            self.semantic_cache = {}
    
    def _get_cached_mol_info(self, smiles: str) -> Optional[Dict]:
        """从缓存获取分子信息，如果没有则在线计算"""
        if not smiles:
            return None
        
        # 尝试从缓存获取
        if self.enable_cache and smiles in self.semantic_cache:
            self.stats['cache_hits'] += 1
            cached_data = self.semantic_cache[smiles]
            # 转换为analyzer.analyze_molecule格式
            return {
                'semantic_blocks': cached_data.get('semantic_blocks', {}),
                'functional_group': cached_data.get('functional_group', []),
                'ring_systems': cached_data.get('ring_systems', []),
                'atom_annotations': cached_data.get('atom_annotations', {}),
                'num_atoms': cached_data.get('num_atoms', 0),
                'num_bonds': cached_data.get('num_bonds', 0)
            }
        
        # 缓存未命中，在线计算
        self.stats['cache_misses'] += 1
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            return self.analyzer.analyze_molecule(mol, smiles)
            
        except Exception as e:
            logger.warning(f"Failed to analyze molecule {smiles}: {e}")
            return None
    
    def set_mask_seed(self, step: int, epoch: int = 0):
        """设置掩码生成的随机种子，确保可复现性"""
        if self.use_deterministic:
            # 基于step和epoch生成确定的种子
            mask_seed = self.base_seed + step * 1000 + epoch * 100000
            random.seed(mask_seed)
            torch.manual_seed(mask_seed)
            try:
                np.random.seed(mask_seed)
            except Exception:
                pass
            # CUDA确定性（如可用）
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(mask_seed)
                except Exception:
                    pass
            
            # 保存当前种子用于调试
            self.mask_rng_state = {
                'step': step,
                'epoch': epoch,
                'seed': mask_seed
            }
            
            logger.debug(f"Set mask seed: {mask_seed} (step={step}, epoch={epoch})")
    
    def get_mask_state(self) -> Dict:
        """获取当前掩码RNG状态"""
        return {
            'rng_state': self.mask_rng_state,
            'stats': self.stats.copy()
        }
    
    def _generate_edge_mask(self, num_edges: int, device: torch.device) -> torch.Tensor:
        """生成边掩码（简单随机策略）"""
        if num_edges == 0:
            return torch.empty(0, dtype=torch.bool, device=device)
        base_edge = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else self.edge_mask_ratio
        return torch.rand(num_edges, device=device) < base_edge
    
    def _generate_random_masks(self, num_nodes: int, num_edges: int, 
                             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成随机掩码作为回退策略，应用小分子边界保护
        
        Args:
            num_nodes: 节点数量
            num_edges: 边数量
            device: 设备
            
        Returns:
            (node_mask, edge_mask) 元组
        """
        # 统一的随机掩码策略：严格遵循固定比例（小分子做轻度保守），并确保非空
        base_ratio = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else float(self.node_mask_ratio)
        if num_nodes <= 6:
            target_ratio = min(base_ratio * 0.5, 0.4)
        elif num_nodes <= 12:
            target_ratio = min(base_ratio * 0.7, 0.55)
        else:
            target_ratio = base_ratio

        import math
        k = 0 if target_ratio <= 0 or num_nodes == 0 else max(1, min(num_nodes, int(math.ceil(num_nodes * target_ratio))))
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if k > 0:
            idx = torch.randperm(num_nodes, device=device)[:k]
            node_mask[idx] = True

        edge_mask = self._generate_edge_mask(num_edges, device)
        
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(f"Random masking applied: {int(node_mask.sum())}/{num_nodes} nodes, "
                             f"{int(edge_mask.sum())}/{num_edges} edges")
            except Exception:
                pass
        
        return node_mask, edge_mask
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取掩码策略的统计信息"""
        total_attempts = self.stats['semantic_attempts'] + self.stats['random_fallbacks']
        
        return {
            'total_masks_generated': self.stats['total_masks_generated'],
            'semantic_success_rate': (
                self.stats['semantic_successes'] / max(self.stats['semantic_attempts'], 1)
            ),
            'semantic_usage_rate': (
                self.stats['semantic_attempts'] / max(total_attempts, 1)
            ),
            'random_fallback_rate': (
                self.stats['random_fallbacks'] / max(total_attempts, 1)
            ),
            'block_type_usage': dict(self.stats['block_type_usage']),
            'config': {
                'semantic_priority': self.semantic_priority,
                'node_mask_ratio': self.node_mask_ratio,
                'edge_mask_ratio': self.edge_mask_ratio,
                'block_types': self.block_types,
                'important_groups': list(self.important_groups)
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        from collections import defaultdict
        self.stats = {
            'semantic_attempts': 0,
            'semantic_successes': 0,
            'random_fallbacks': 0,
            'total_masks_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'block_type_usage': defaultdict(int)
        }
        for bt in self.block_types:
            self.stats['block_type_usage'][bt] = 0


def create_semantic_masking(config: Dict = None) -> SemanticMasking:
    """创建语义掩码策略实例"""
    return SemanticMasking(config)
