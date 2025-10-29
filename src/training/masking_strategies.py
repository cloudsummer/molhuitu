"""
简化的掩码策略模块 - 只保留核心策略

这个模块只实现两种核心掩码策略：
1. 随机掩码 - 基线策略

注意：语义掩码在 semantic_masking.py 中单独实现
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
from rdkit import Chem
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class MaskingStrategy(ABC):
    """掩码策略的抽象基类"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.mask_ratio = self.config.get('mask_ratio', 0.4)
        self.device = None
        
    @abstractmethod
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor, 
                      hyperedge_attr: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成节点和边掩码
        
        Args:
            x: 节点特征 [num_nodes, feature_dim]
            hyperedge_index: 超边连接 [2, num_connections]
            hyperedge_attr: 超边属性 [num_edges, edge_dim]
            **kwargs: 额外上下文（epoch, 分子信息等）
            
        Returns:
            (node_mask, edge_mask) 元组
        """
        pass
    
    def set_device(self, device: torch.device):
        """设置设备"""
        self.device = device


class RandomMasking(MaskingStrategy):
    """基本随机掩码策略"""
    
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                      hyperedge_attr: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成随机掩码"""
        num_nodes = x.size(0)
        num_edges = hyperedge_attr.size(0) if hyperedge_attr is not None else 0
        
        device = x.device
        
        # 随机节点掩码
        node_mask = torch.rand(num_nodes, device=device) < self.mask_ratio
        
        # 随机边掩码
        if num_edges > 0:
            edge_mask = torch.rand(num_edges, device=device) < self.mask_ratio
        else:
            edge_mask = torch.empty(0, dtype=torch.bool, device=device)
            
        return node_mask, edge_mask





# 策略工厂函数
def create_masking_strategy(strategy_type: str, config: Dict = None) -> MaskingStrategy:
    """创建掩码策略的工厂函数 - 简化版本，只支持随机掩码"""
    strategy_map = {
        'random': RandomMasking,
    }
    
    if strategy_type not in strategy_map:
        logger.warning(f"未知掩码策略: {strategy_type}. 只支持 'random' 策略，使用随机掩码.")
        logger.info("提示：如需语义掩码，请在配置中使用 strategy: 'semantic'")
        strategy_type = 'random'
    
    return strategy_map[strategy_type](config)