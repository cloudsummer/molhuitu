"""
Target Contribution Control (TCC) for Multi-objective Loss Weighting
目标贡献占比控制器 - 自动优化多损失权重平衡

Based on the principle of maintaining target contribution ratios for each loss component
while avoiding the oscillation issues of traditional adaptive weighting methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TargetContributionControl(nn.Module):
    """
    Target Contribution Control (TCC) - 目标贡献占比控制器
    
    核心思想：
    1. 为每个损失组件设定目标贡献占比区间
    2. 通过滞回控制避免权重抖动
    3. 在log域进行权重更新保证数值稳定性
    4. 使用EMA平滑贡献占比计算
    """
    
    def __init__(self, loss_components: List[str], config: Dict):
        """
        初始化TCC控制器
        
        Args:
            loss_components: 损失组件列表，如['reconstruction', 'edge']
            config: TCC配置参数
        """
        super().__init__()
        
        # 修复类型问题：处理OmegaConf ListConfig和普通列表
        if isinstance(loss_components, str):
            logger.error(f"loss_components should be a list, got string: '{loss_components}'")
            raise ValueError(f"loss_components must be a list, not string: '{loss_components}'")
        
        # 将OmegaConf ListConfig转换为普通列表
        try:
            from omegaconf import ListConfig
            if isinstance(loss_components, ListConfig):
                loss_components = list(loss_components)
        except ImportError:
            pass  # omegaconf未安装时忽略
            
        # 检查是否是类似列表的对象
        if not hasattr(loss_components, '__iter__') or isinstance(loss_components, (str, bytes)):
            logger.error(f"loss_components should be iterable, got {type(loss_components)}: {loss_components}")
            raise ValueError(f"loss_components must be iterable, got {type(loss_components)}")
        
        # 转换为列表以确保兼容性
        loss_components = list(loss_components)
        
        if len(loss_components) == 0:
            raise ValueError("loss_components cannot be empty")
            
        self.loss_components = loss_components
        self.num_components = len(loss_components)
        
        # 控制参数（降低权重变化激烈程度）
        self.update_frequency = config.get('update_frequency', 100)  # 降低更新频率
        self.tolerance = config.get('tolerance', 0.08)  # 增加容差，减少不必要调整
        self.adjustment_rate = config.get('adjustment_rate', 0.04)  # 降低调整速率，更稳定
        self.ema_alpha = config.get('ema', 0.9)  # 更平稳的EMA
        self.weight_clip = config.get('weight_clip', [0.1, 20.0])
        self.renorm = config.get('renorm', 'mean_log')
        
        # 根据损失组件数量自动设定目标比例
        self.target_ratios = self._get_adaptive_ratios(loss_components, config)
        
        # 可学习的log权重（初始化为均匀权重）
        initial_log_weights = torch.zeros(self.num_components)
        self.log_weights = nn.Parameter(initial_log_weights)
        
        # 状态变量
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('ema_contributions', torch.zeros(self.num_components))
        self.register_buffer('initialized', torch.tensor(False))
        
        # 损失尺度追踪器 - 解决多任务损失失衡问题
        self.register_buffer('ema_scales', torch.ones(self.num_components))
        self.scale_ema_alpha = config.get('scale_ema', 0.9)
        self.register_buffer('scale_initialized', torch.tensor(False))
        
        logger.info(f"TCC initialized for components: {loss_components}")
        logger.info(f"Target ratios: {self.target_ratios}")
        logger.info(f"Step-based config: update_freq={self.update_frequency}, tolerance={self.tolerance:.3f}")
        logger.info(f"Optimized for fast convergence: adj_rate={self.adjustment_rate:.3f}, ema={self.ema_alpha:.3f}")
        
    def _get_adaptive_ratios(self, components: List[str], config: Dict) -> Dict[str, Tuple[float, float]]:
        """
        根据损失组件数量自适应设定目标比例区间
        """
        adaptive_config = config.get('adaptive_ratios', {})
        
        if self.num_components == 1:
            # 单组件：100%贡献
            return {components[0]: (0.95, 1.0)}
            
        elif self.num_components == 2:
            # 双组件：主要+辅助
            ratios_config = adaptive_config.get('two_components', {
                'primary': [0.65, 0.75],
                'secondary': [0.25, 0.35]
            })
            primary_idx = 0 if 'reconstruction' in components else 0
            secondary_idx = 1 - primary_idx
            
            return {
                components[primary_idx]: tuple(ratios_config['primary']),
                components[secondary_idx]: tuple(ratios_config['secondary'])
            }
            
        elif self.num_components == 3:
            # 三组件：主要+辅助+正则化（明确将 edge 视为辅助，将 descriptor/contrastive 视为正则化）
            ratios_config = adaptive_config.get('three_components', {
                'primary': [0.40, 0.50],
                'auxiliary': [0.30, 0.40],
                'regularization': [0.15, 0.25]
            })

            primary_idx = components.index('reconstruction') if 'reconstruction' in components else 0
            aux_idx = components.index('edge') if 'edge' in components else None
            # Regularization 优先选择 descriptor，没有则对比项，再没有从剩余中挑一个
            if 'descriptor' in components:
                reg_idx = components.index('descriptor')
            elif 'contrastive' in components:
                reg_idx = components.index('contrastive')
            else:
                reg_idx = None

            remaining = [i for i in range(3) if i != primary_idx and i not in [aux_idx, reg_idx] and i is not None]
            if aux_idx is None and remaining:
                aux_idx = remaining.pop(0)
            if reg_idx is None and remaining:
                reg_idx = remaining.pop(0)

            return {
                components[primary_idx]: tuple(ratios_config['primary']),
                components[aux_idx]: tuple(ratios_config['auxiliary']),
                components[reg_idx]: tuple(ratios_config['regularization'])
            }
            
        elif self.num_components == 4:
            # 四组件：主要+辅助+正则化+对比
            ratios_config = adaptive_config.get('four_components', {
                'primary': [0.45, 0.55],
                'auxiliary': [0.20, 0.30],
                'regularization': [0.10, 0.15],
                'contrastive': [0.10, 0.20]
            })
            
            # 智能角色分配
            role_mapping = {}
            for i, comp in enumerate(components):
                if comp == 'reconstruction':
                    role_mapping[i] = 'primary'
                elif comp == 'edge':
                    role_mapping[i] = 'auxiliary'
                elif comp == 'regularization':
                    role_mapping[i] = 'regularization'
                elif comp == 'contrastive':
                    role_mapping[i] = 'contrastive'
                else:
                    # 默认分配
                    if len([r for r in role_mapping.values() if r == 'auxiliary']) == 0:
                        role_mapping[i] = 'auxiliary'
                    else:
                        role_mapping[i] = 'regularization'
            
            return {
                components[i]: tuple(ratios_config[role])
                for i, role in role_mapping.items()
            }
        
        else:
            # 更多组件：均匀分配
            uniform_ratio = 1.0 / self.num_components
            margin = 0.1 * uniform_ratio
            return {
                comp: (uniform_ratio - margin, uniform_ratio + margin)
                for comp in components
            }
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TCC前向传播：计算加权总损失（包含损失尺度归一化）
        
        Args:
            losses: 各损失组件字典
            
        Returns:
            total_loss: 加权总损失
            info: 权重和贡献占比信息（包含尺度信息）
        """
        # 提取原始损失值（鲁棒：缺失的组件用0占位，避免KeyError）
        if len(losses) > 0:
            some_tensor = next(iter(losses.values()))
            device = some_tensor.device if hasattr(some_tensor, 'device') else (
                torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )
            dtype = some_tensor.dtype if hasattr(some_tensor, 'dtype') else torch.float32
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            dtype = torch.float32

        loss_list = []
        for comp in self.loss_components:
            v = losses.get(comp, None)
            if v is None:
                v = torch.tensor(0.0, device=device, dtype=dtype)
            loss_list.append(v)
        raw_loss_values = torch.stack(loss_list)
        
        # 更新EMA损失尺度
        with torch.no_grad():
            raw_loss_items = torch.stack([(losses.get(comp, torch.tensor(0.0, device=device, dtype=dtype))).detach()
                                          for comp in self.loss_components])
            
            if self.scale_initialized:
                self.ema_scales = (
                    self.scale_ema_alpha * self.ema_scales + 
                    (1 - self.scale_ema_alpha) * raw_loss_items
                )
            else:
                self.ema_scales = raw_loss_items.clone()
                self.scale_initialized = torch.tensor(True)
        
        # 损失尺度归一化 - 解决多任务损失失衡的核心
        normalized_loss_values = raw_loss_values / (self.ema_scales + 1e-8)
        
        # 使用归一化损失计算权重（保持TCC原有逻辑）
        current_weights = torch.exp(self.log_weights)
        
        # 计算加权损失（使用归一化损失进行权重平衡计算）
        weighted_normalized_losses = current_weights * normalized_loss_values
        normalized_total_loss = weighted_normalized_losses.sum()
        
        # 计算当前贡献占比（基于归一化损失）
        current_contributions = (weighted_normalized_losses / normalized_total_loss).detach()
        
        # EMA平滑贡献占比
        if self.initialized:
            self.ema_contributions = (
                self.ema_alpha * self.ema_contributions + 
                (1 - self.ema_alpha) * current_contributions
            )
        else:
            self.ema_contributions = current_contributions
            self.initialized = torch.tensor(True)
        
        # 权重更新（滞回控制） - 基于归一化损失的贡献占比
        if self.step_count % self.update_frequency == 0:
            self._update_weights()
        
        self.step_count += 1
        
        # 计算实际输出的总损失
        # 修复：用归一化后的损失参与反传，避免某一项因量纲过大主导训练
        weighted_raw_losses = current_weights * raw_loss_values
        weighted_normalized_losses = current_weights * normalized_loss_values
        raw_total_loss = weighted_raw_losses.sum()     # 仅用于诊断
        total_loss = weighted_normalized_losses.sum()  # 用于反传
        
        # 返回信息（包含尺度追踪信息）
        info = {
            'weights': {comp: current_weights[i].item() for i, comp in enumerate(self.loss_components)},
            'contributions': {comp: current_contributions[i].item() for i, comp in enumerate(self.loss_components)},
            'ema_contributions': {comp: self.ema_contributions[i].item() for i, comp in enumerate(self.loss_components)},
            'targets': self.target_ratios,
            'total_loss': total_loss.item(),
            'raw_total_loss': raw_total_loss.item(),
            # 新增尺度追踪信息
            'raw_losses': {comp: raw_loss_values[i].item() for i, comp in enumerate(self.loss_components)},
            'normalized_losses': {comp: normalized_loss_values[i].item() for i, comp in enumerate(self.loss_components)},
            'ema_scales': {comp: self.ema_scales[i].item() for i, comp in enumerate(self.loss_components)}
        }
        
        return total_loss, info
    
    def _update_weights(self):
        """
        权重更新逻辑 - 滞回控制避免抖动
        """
        with torch.no_grad():
            for i, comp in enumerate(self.loss_components):
                target_low, target_high = self.target_ratios[comp]
                current_contrib = self.ema_contributions[i]
                
                # 计算偏差
                gap_low = target_low - current_contrib
                gap_high = current_contrib - target_high
                
                # 滞回控制：只有超出容差才调整
                if gap_low > self.tolerance:
                    # 低于下界 -> 增加权重
                    adjustment = self.adjustment_rate * gap_low
                    self.log_weights[i] += adjustment
                elif gap_high > self.tolerance:
                    # 高于上界 -> 减少权重
                    adjustment = self.adjustment_rate * gap_high
                    self.log_weights[i] -= adjustment
                # 否则位于死区内，不调整
            
            # 硬边界限制
            self.log_weights.clamp_(
                min=np.log(self.weight_clip[0]),
                max=np.log(self.weight_clip[1])
            )
            
            # 归一化：消除整体尺度漂移
            if self.renorm == 'mean_log':
                mean_log_weight = self.log_weights.mean()
                self.log_weights -= mean_log_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        with torch.no_grad():
            current_weights = torch.exp(self.log_weights)
            return {comp: current_weights[i].item() for i, comp in enumerate(self.loss_components)}
    
    def get_target_ratios(self) -> Dict[str, Tuple[float, float]]:
        """获取目标比例区间"""
        return self.target_ratios.copy()


def create_tcc_controller(loss_components: List[str], config: Dict) -> Optional[TargetContributionControl]:
    """
    创建TCC控制器的工厂函数
    
    Args:
        loss_components: 损失组件列表
        config: TCC配置
        
    Returns:
        TCC控制器实例或None
    """
    if not config.get('enabled', False):
        return None
    
    if len(loss_components) < 1:
        logger.warning("TCC requires at least 1 loss component")
        return None
    
    return TargetContributionControl(loss_components, config)
