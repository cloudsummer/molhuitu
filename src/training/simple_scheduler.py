"""
简化的掩码调度器 - 移除课程学习复杂性

这个模块提供一个简化的掩码调度器，专注于：
1. 策略创建和管理
2. 分子分析结果缓存
3. 简单的策略选择
4. 性能监控

移除了原有的复杂功能：
- 课程学习进展管理
- 多臂老虎机策略选择
- 复杂的性能历史追踪
- 动态难度调整
"""

import torch
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
from rdkit import Chem

from .semantic_masking import SemanticMasking, create_semantic_masking
from .masking_strategies import MaskingStrategy, create_masking_strategy
from ..data.molecular_semantics import MolecularSemanticAnalyzer

logger = logging.getLogger(__name__)


class SimpleMaskingScheduler:
    """
    简化的掩码调度器
    
    核心功能：
    - 策略创建和管理
    - 分子分析缓存
    - 基本性能监控
    
    简化设计：
    - 固定策略选择，无动态切换
    - 基本的LRU缓存机制
    - 简单的统计信息收集
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.masking_config = config.get('masking', {})
        
        # 策略管理
        self.strategy_type = self.masking_config.get('strategy', 'semantic')
        self.strategy = None
        
        # 分子分析缓存
        self.molecular_cache = {}
        self.cache_size_limit = self.masking_config.get('cache_size_limit', 1000)
        self.cache_access_order = []  # LRU缓存的访问顺序
        
        # 分子语义分析器
        self.analyzer = MolecularSemanticAnalyzer()
        
        # 性能监控
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'semantic_analysis_time': 0.0,
            'mask_generation_time': 0.0,
            'error_count': 0,
            'strategy_usage': defaultdict(int)
        }
        
        # 初始化策略
        self._initialize_strategy()
        
        logger.info(f"SimpleMaskingScheduler initialized with strategy: {self.strategy_type}")
    
    def _initialize_strategy(self):
        """初始化掩码策略"""
        try:
            if self.strategy_type == 'semantic':
                self.strategy = create_semantic_masking(self.masking_config)
            else:
                self.strategy = create_masking_strategy(self.strategy_type, self.masking_config)
            
            logger.info(f"Successfully initialized {self.strategy_type} masking strategy")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.strategy_type} strategy: {e}")
            
            # 回退到随机策略
            logger.warning("Falling back to random masking strategy")
            self.strategy = create_masking_strategy('random', {'mask_ratio': 0.7})
            self.strategy_type = 'random'
    
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                      hyperedge_attr: torch.Tensor, global_step: int = 0, 
                      max_steps: int = 1000, epoch: int = 0, smiles: str = None,
                      mol: Chem.Mol = None, recent_loss: float = None,
                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成掩码（简化版本）
        
        Args:
            x: 节点特征
            hyperedge_index: 超边连接
            hyperedge_attr: 超边属性
            global_step: 全局步数（保留兼容性）
            max_steps: 最大步数（保留兼容性）
            epoch: 轮次（保留兼容性）
            smiles: SMILES字符串
            mol: RDKit分子对象
            recent_loss: 最近损失（保留兼容性）
            **kwargs: 其他参数
            
        Returns:
            (node_mask, edge_mask) 元组
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        self.stats['strategy_usage'][self.strategy_type] += 1
        
        try:
            # 获取分子信息（使用缓存）
            mol_info = None
            if smiles and self.strategy_type == 'semantic':
                # 如果是批量SMILES（list/tuple），交由策略在批处理中逐个处理，
                # 此处不做整体分子分析，避免对list做hash导致异常。
                if isinstance(smiles, (list, tuple)):
                    mol_info = None
                else:
                    mol_info = self._get_molecular_info(smiles)
            
            # 生成掩码
            mask_start_time = time.time()
            node_mask, edge_mask = self.strategy.generate_masks(
                x, hyperedge_index, hyperedge_attr,
                smiles=smiles, mol_info=mol_info,
                epoch=epoch, **kwargs
            )
            
            self.stats['mask_generation_time'] += time.time() - mask_start_time
            
            # 验证掩码
            self._validate_masks(node_mask, edge_mask, x.size(0), 
                               hyperedge_attr.size(0) if hyperedge_attr is not None else 0)
            
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    nm = int(node_mask.sum())
                    em = int(edge_mask.sum()) if edge_mask.numel() > 0 else 0
                    logger.debug(f"Generated masks: {nm}/{node_mask.size(0)} nodes, {em}/{edge_mask.size(0)} edges")
                except Exception:
                    pass
            
            return node_mask, edge_mask
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Error in mask generation: {e}")
            
            # 紧急回退：生成基本随机掩码
            return self._generate_emergency_masks(x, hyperedge_attr)
        
        finally:
            total_time = time.time() - start_time
            logger.debug(f"Mask generation took {total_time:.4f}s")
    
    def _get_molecular_info(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        获取分子信息（使用LRU缓存）
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            分子分析结果或None
        """
        if not smiles:
            return None
        
        # 检查缓存
        if smiles in self.molecular_cache:
            self.stats['cache_hits'] += 1
            
            # 更新LRU访问顺序
            if smiles in self.cache_access_order:
                self.cache_access_order.remove(smiles)
            self.cache_access_order.append(smiles)
            
            return self.molecular_cache[smiles]
        
        # 缓存未命中，进行分子分析
        self.stats['cache_misses'] += 1
        
        try:
            analysis_start_time = time.time()
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            mol_info = self.analyzer.analyze_molecule(mol, smiles)
            
            self.stats['semantic_analysis_time'] += time.time() - analysis_start_time
            
            # 添加到缓存
            self._add_to_cache(smiles, mol_info)
            
            return mol_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze molecule {smiles}: {e}")
            return None
    
    def _add_to_cache(self, smiles: str, mol_info: Dict[str, Any]):
        """
        添加分子信息到缓存（LRU策略）
        
        Args:
            smiles: SMILES字符串
            mol_info: 分子分析结果
        """
        # 检查缓存大小限制
        while len(self.molecular_cache) >= self.cache_size_limit:
            # 移除最久未使用的项
            if self.cache_access_order:
                oldest_smiles = self.cache_access_order.pop(0)
                if oldest_smiles in self.molecular_cache:
                    del self.molecular_cache[oldest_smiles]
            else:
                break
        
        # 添加新项
        self.molecular_cache[smiles] = mol_info
        self.cache_access_order.append(smiles)
        
        logger.debug(f"Added {smiles} to molecular cache (size: {len(self.molecular_cache)})")
    
    def _validate_masks(self, node_mask: torch.Tensor, edge_mask: torch.Tensor,
                       num_nodes: int, num_edges: int):
        """
        验证掩码的有效性
        
        Args:
            node_mask: 节点掩码
            edge_mask: 边掩码
            num_nodes: 期望的节点数量
            num_edges: 期望的边数量
        """
        # 检查掩码维度
        if node_mask.size(0) != num_nodes:
            raise ValueError(f"Node mask size {node_mask.size(0)} doesn't match num_nodes {num_nodes}")
        
        if edge_mask.size(0) != num_edges:
            raise ValueError(f"Edge mask size {edge_mask.size(0)} doesn't match num_edges {num_edges}")
        
        # 检查掩码类型
        if node_mask.dtype != torch.bool:
            logger.warning(f"Node mask dtype is {node_mask.dtype}, expected bool")
        
        if edge_mask.dtype != torch.bool:
            logger.warning(f"Edge mask dtype is {edge_mask.dtype}, expected bool")
        
        # 检查掩码比例 (避免.item()调用以防torch.compile图断裂)
        node_mask_ratio = node_mask.float().mean()
        edge_mask_ratio = edge_mask.float().mean() if num_edges > 0 else 0.0
        
        # 只在需要记录日志时才转换为标量
        if node_mask_ratio < 0.1 or node_mask_ratio > 0.95:
            logger.warning(f"Unusual node mask ratio: {node_mask_ratio.item():.3f}")
        
        if num_edges > 0 and (edge_mask_ratio < 0.1 or edge_mask_ratio > 0.95):
            logger.warning(f"Unusual edge mask ratio: {edge_mask_ratio.item():.3f}")
    
    def _generate_emergency_masks(self, x: torch.Tensor, 
                                hyperedge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成紧急回退掩码
        
        Args:
            x: 节点特征
            hyperedge_attr: 超边属性
            
        Returns:
            (node_mask, edge_mask) 元组
        """
        num_nodes = x.size(0)
        num_edges = hyperedge_attr.size(0) if hyperedge_attr is not None else 0
        device = x.device
        
        # 简单的随机掩码
        node_mask = torch.rand(num_nodes, device=device) < 0.7
        edge_mask = torch.rand(num_edges, device=device) < 0.7 if num_edges > 0 else torch.empty(0, dtype=torch.bool, device=device)
        
        logger.warning(f"Using emergency random masks: {node_mask.sum().item()}/{num_nodes} nodes")
        
        return node_mask, edge_mask
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Returns:
            性能统计信息字典
        """
        total_calls = max(self.stats['total_calls'], 1)
        cache_total = self.stats['cache_hits'] + self.stats['cache_misses']
        
        summary = {
            'scheduler_stats': {
                'strategy_type': self.strategy_type,
                'total_calls': self.stats['total_calls'],
                'error_rate': self.stats['error_count'] / total_calls,
                'avg_mask_time': self.stats['mask_generation_time'] / total_calls,
                'strategy_usage': dict(self.stats['strategy_usage'])
            },
            'cache_stats': {
                'cache_size': len(self.molecular_cache),
                'cache_limit': self.cache_size_limit,
                'cache_hit_rate': self.stats['cache_hits'] / max(cache_total, 1),
                'total_cache_requests': cache_total,
                'avg_analysis_time': (
                    self.stats['semantic_analysis_time'] / max(self.stats['cache_misses'], 1)
                )
            }
        }
        
        # 添加策略特定的统计信息
        if hasattr(self.strategy, 'get_statistics'):
            summary['strategy_stats'] = self.strategy.get_statistics()
        
        return summary
    
    def clear_cache(self):
        """清空分子分析缓存"""
        self.molecular_cache.clear()
        self.cache_access_order.clear()
        logger.info("Molecular analysis cache cleared")
    
    def save_cache(self, filepath: str):
        """
        保存缓存到文件
        
        Args:
            filepath: 缓存文件路径
        """
        try:
            import pickle
            cache_data = {
                'molecular_cache': self.molecular_cache,
                'cache_access_order': self.cache_access_order,
                'stats': self.stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cache saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_cache(self, filepath: str):
        """
        从文件加载缓存
        
        Args:
            filepath: 缓存文件路径
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.molecular_cache = cache_data.get('molecular_cache', {})
            self.cache_access_order = cache_data.get('cache_access_order', [])
            
            # 可选择性加载统计信息
            saved_stats = cache_data.get('stats', {})
            for key in ['cache_hits', 'cache_misses', 'semantic_analysis_time']:
                if key in saved_stats:
                    self.stats[key] = saved_stats[key]
            
            logger.info(f"Cache loaded from {filepath} (size: {len(self.molecular_cache)})")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")


def create_simple_masking_scheduler(config: Dict) -> SimpleMaskingScheduler:
    """创建简化的掩码调度器"""
    return SimpleMaskingScheduler(config)
