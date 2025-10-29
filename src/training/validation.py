"""
统一的验证控制器

整理之前混乱的验证参数控制，提供清晰一致的验证管理接口。

之前的混乱状况：
- trainer.py中有val_interval, val_every_n_steps, eval_every_n_steps三个参数
- optuna_tune.py中硬编码覆盖config['training']['eval_every_n_steps'] = 500
- 验证触发逻辑分散在多处，命名不一致

现在统一为ValidationController管理所有验证逻辑。
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ValidationController:
    """统一的验证控制器"""
    
    def __init__(self, validation_config: Dict[str, Any], max_steps: int):
        """
        初始化验证控制器
        
        Args:
            validation_config: 验证配置字典
            max_steps: 最大训练步数，用于计算默认间隔
        """
        self.enabled = validation_config.get('enabled', True)
        self.interval_steps = validation_config.get('interval_steps', max(1, max_steps // 8))
        self.quick_batches = validation_config.get('quick_batches', 50)
        
        # 调优模式检测：如果间隔步数过大，认为是调优模式
        self.is_tuning_mode = self.interval_steps > max_steps
        
        if self.is_tuning_mode:
            logger.info(f"Validation Controller: 调优模式，验证已禁用 (interval={self.interval_steps})")
        else:
            logger.info(f"Validation Controller: 正常模式，每{self.interval_steps}步验证一次")
    
    def should_validate(self, current_step: int, step_updated: bool = True) -> bool:
        """
        判断当前步是否应该进行验证
        
        Args:
            current_step: 当前训练步数
            step_updated: 优化器是否真正更新了（避免梯度累积时的重复验证）
            
        Returns:
            bool: 是否应该验证
        """
        if not self.enabled or not step_updated:
            return False
            
        if self.is_tuning_mode:
            return False  # 调优模式下不验证
            
        return current_step % self.interval_steps == 0
    
    def get_quick_batches(self) -> Optional[int]:
        """获取快速验证的批数限制"""
        return self.quick_batches if self.quick_batches > 0 else None
    
    def get_status_info(self) -> Dict[str, Any]:
        """获取验证控制器状态信息，用于日志输出"""
        return {
            'enabled': self.enabled,
            'interval_steps': self.interval_steps,
            'quick_batches': self.quick_batches,
            'is_tuning_mode': self.is_tuning_mode
        }