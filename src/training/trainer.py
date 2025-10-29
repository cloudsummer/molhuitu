"""
Trainer class for HyperGraph-MAE model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple, Callable
import logging
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import math

# Optuna import for trial pruning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..utils.logging_utils import setup_logger
from ..utils.memory_utils import log_memory_usage, cleanup_memory
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from .validation import ValidationController
from ..utils.metrics import (
    masked_mae,
    compute_embedding_health_metrics,
    compute_neighbor_consistency_spearman,
)

from .simple_scheduler import SimpleMaskingScheduler, create_simple_masking_scheduler

logger = logging.getLogger(__name__)


class NaNDetectedException(Exception):
    """Exception raised when NaN values are detected during training."""
    pass


class BatchSkippedException(Exception):
    """Exception to indicate a batch should be skipped due to unrecoverable errors."""
    pass


class HyperGraphMAETrainer:
    """
    Trainer for HyperGraph-MAE model with support for:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device = None, optuna_trial=None):
        """
        Initialize trainer.

        Args:
            model: HyperGraph-MAE model
            config: Training configuration
            device: Device to train on
            optuna_trial: Optional Optuna trial object for pruning
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optuna trial for real-time pruning
        self.optuna_trial = optuna_trial

        # Move model to device
        self.model = self.model.to(self.device)

        # Extract configuration
        self.training_config = config.get('training', {})
        self.use_amp = self.training_config.get('use_amp', True)
        self.gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        self.clip_grad_norm = self.training_config.get('gradient_clip_norm', 1.0)
        # 移除旧的混乱验证参数，统一使用ValidationController管理
        # self.val_interval, self.val_every_n_steps, self.val_quick_max_batches 已废弃

        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_callbacks()

        # Enhanced mixed precision training with optimal PyTorch defaults
        self.use_amp_advanced = self.use_amp and torch.cuda.is_available()
        if self.use_amp_advanced:
            # Smart GradScaler & dtype selection
            # BF16-native GPUs (Ampere: A100/H100) support bfloat16 with large dynamic range
            bf16_supported = torch.cuda.is_bf16_supported()
            # Choose autocast dtype: BF16 on supported GPUs, otherwise FP16
            self.autocast_dtype = torch.bfloat16 if bf16_supported else torch.float16
            # Enable GradScaler only when using FP16 (BF16一般不需要)
            self.scaler = GradScaler(enabled=not bf16_supported)

            # TF32状态：优先开启（A100/H100上可显著提速FP32回退路径）
            if torch.cuda.is_available():
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
                tf32_matmul = torch.backends.cuda.matmul.allow_tf32
                tf32_cudnn = torch.backends.cudnn.allow_tf32
                gpu_name = torch.cuda.get_device_name()

                logger.info(f"Mixed precision enabled: autocast dtype={self.autocast_dtype}")
                logger.info(f"GradScaler enabled: {self.scaler.is_enabled()}")
                logger.info(f"TF32 acceleration: matmul={tf32_matmul}, cuDNN={tf32_cudnn}")
                logger.info(f"GPU: {gpu_name}")

                if "A100" in gpu_name or "H100" in gpu_name:
                    # On Ampere/Hopper, prefer BF16 autocast + TF32 fallbacks
                    logger.info("⚡ A100/H100: BF16 autocast + TF32 for FP32 fallbacks; gradient scaling disabled")
                elif "V100" in gpu_name or "T4" in gpu_name:
                    logger.info("📊 V100/T4: FP16 mixed precision with gradient scaling")
        
        else:
            self.scaler = None
            logger.info("Mixed precision disabled - using FP32")

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0
        
        # 优化的批量缓存机制 - 平衡性能和数值稳定性
        # 减少同步间隔防止梯度累积误差，同时保持性能收益
        # 初始化时使用默认值，在train()中根据实际步数重新计算
        self.sync_interval = self.training_config.get('sync_interval', 50)
        
        # 限制缓存大小防止内存泄露和数值精度损失
        self.max_cache_size = min(self.sync_interval, 100)
        self.gpu_loss_cache = []  # 在GPU上缓存损失值
        self.gpu_grad_norm_cache = []  # 在GPU上缓存梯度范数
        self.gpu_metrics_cache = {  # 在GPU上缓存其他指标
            'recon_loss': [],
            'edge_loss': [],
            'contrastive_loss': []
        }
        
        # Initialize masking scheduler if model supports it
        self.masking_scheduler = None
        self._setup_masking_scheduler()
        
        # Recent losses for masking strategy optimization
        self.recent_losses = []
        self.loss_window_size = self.training_config.get('masking_loss_window', 10)
        
        # KISS优化：掩码余弦曲线参数
        self.mask_ratio_min = self.training_config.get('mask_ratio_min', 0.3)
        self.mask_ratio_max = self.training_config.get('mask_ratio_max', 0.7)

        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'step_time': [],
            'masking_stats': []
        }
        
        # NaN detection settings with backward-compatible keys
        nd_cfg = self.training_config.get('nan_detection', {})
        # Fallback to legacy flat keys if present in config.training
        legacy_enabled = self.training_config.get('nan_detection_enabled', None)
        legacy_freq = self.training_config.get('nan_detection_frequency', None)
        legacy_recovery = self.training_config.get('nan_checkpoint_recovery', None)

        self.nan_detection_enabled = bool(nd_cfg.get('enabled', legacy_enabled if legacy_enabled is not None else True))
        self.nan_detection_frequency = int(nd_cfg.get('frequency', legacy_freq if legacy_freq is not None else 50))
        self.nan_checkpoint_recovery = bool(nd_cfg.get('checkpoint_recovery', legacy_recovery if legacy_recovery is not None else True))
        self.last_healthy_step = 0

    def mask_ratio_at(self, step: int, max_steps: int) -> float:
        """
        计算当前训练步的掩码比例（cosine 调度，介于 [mask_ratio_min, mask_ratio_max]）。
        
        - 当 step=0 时返回 mask_ratio_max；随着训练推进，逐步下降到 mask_ratio_min。
        - 若未配置，默认区间来自 __init__ 中的 mask_ratio_min/max。
        """
        import math
        total = max(1, int(max_steps) if max_steps is not None else 1)
        s = max(0, min(int(step), total))
        # 进度 [0,1]
        t = s / total
        # 余弦从 1 -> -1，(1 - cos)/2 从 0 -> 1
        frac = 0.5 * (1.0 - math.cos(math.pi * t))
        hi = float(getattr(self, 'mask_ratio_max', 0.7))
        lo = float(getattr(self, 'mask_ratio_min', 0.3))
        ratio = hi - frac * (hi - lo)
        # 保护边界
        return float(max(0.0, min(1.0, ratio)))

    def _setup_optimizer(self):
        """Setup optimizer with configuration."""
        lr = float(self.training_config.get('learning_rate', 5e-4))
        weight_decay = float(self.training_config.get('weight_decay', 1e-4))
        
        optimizer_config = self.model.configure_optimizers(
            lr=lr,
            weight_decay=weight_decay
        )
        self.optimizer = optimizer_config['optimizer']
        
        # Scheduler is solely managed by Trainer (single source of truth)
        if 'lr_scheduler' in optimizer_config:
            logger.debug("Ignoring model-provided scheduler; Trainer controls scheduling based on training config")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'WarmupCosine')  # KISS优化: 默认使用Warmup+Cosine替代OneCycleLR

        if scheduler_type == 'WarmupCosine':
            # KISS优化: Warmup+Cosine调度器，训练前期温度上升，后期余弦下降
            # Need to know total steps
            self.scheduler = None  # Will be set in train method
        elif scheduler_type == 'OneCycleLR':
            # Legacy scheduler - 保留兼容性
            # Need to know total steps
            self.scheduler = None  # Will be set in train method
        elif scheduler_type == 'CosineAnnealingLR':
            # CosineAnnealingLR not supported in step-based training
            # Use CosineAnnealingWarmRestarts instead
            logger.warning("CosineAnnealingLR not supported in step-based training, using None")
            self.scheduler = None
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 10),
                factor=scheduler_config.get('factor', 0.5)
            )
        else:
            self.scheduler = None

    def _setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = []

        # Early stopping已移至主训练循环中处理，避免双路径冲突
        # 配置通过training.early_stopping在主路径中统一管理

        # Model checkpoint
        checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
        self.callbacks.append(
            ModelCheckpoint(
                directory=checkpoint_dir,
                filename='best_model.pth',
                monitor='val_loss',
                mode='min',
                config=self.config  # Bind config to prevent architecture drift
            )
        )

        # Learning rate monitor
        self.callbacks.append(LearningRateMonitor())

    def _setup_masking_scheduler(self):
        """Setup simple masking scheduler if configured."""
        # Check if model already has a masking scheduler
        if hasattr(self.model, 'masking_scheduler') and self.model.masking_scheduler is not None:
            # Use model's scheduler and link trainer to it
            self.masking_scheduler = self.model.masking_scheduler
            logger.info("Using model's simple masking scheduler")
        elif self.config.get('masking'):
            # Create new simple scheduler from config
            try:
                self.masking_scheduler = create_simple_masking_scheduler(self.config)
                # Also set it in the model if supported
                if hasattr(self.model, 'masking_scheduler'):
                    self.model.masking_scheduler = self.masking_scheduler
                strategy_type = self.config['masking'].get('strategy', 'semantic')
                logger.info(f"Created simple masking scheduler: {strategy_type}")
            except Exception as e:
                logger.warning(f"Failed to create simple masking scheduler: {e}")
                self.masking_scheduler = None
        else:
            logger.info("No masking configuration found, using model's default masking")

    def _check_model_parameters_for_nan(self) -> bool:
        """Check if any model parameters contain NaN values.
        
        Returns:
            bool: True if NaN detected, False otherwise
        """
        if not self.nan_detection_enabled:
            return False
            
        try:
            for name, param in self.model.named_parameters():
                if param.data.isnan().any():
                    logger.error(f"NaN detected in model parameter: {name}")
                    return True
                if param.data.isinf().any():
                    logger.error(f"Inf detected in model parameter: {name}")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error during NaN check: {e}")
            return False
    
    def _check_tensors_for_nan(self, tensors_dict: Dict[str, torch.Tensor], context: str) -> bool:
        """Check if any tensors in a dictionary contain NaN values.
        
        Args:
            tensors_dict: Dictionary of tensor name -> tensor pairs
            context: Context string for error messages
            
        Returns:
            bool: True if NaN detected, False otherwise
        """
        if not self.nan_detection_enabled:
            return False
            
        try:
            for name, tensor in tensors_dict.items():
                if isinstance(tensor, torch.Tensor):
                    if tensor.isnan().any():
                        logger.error(f"NaN detected in {context} tensor '{name}'")
                        return True
                    if tensor.isinf().any():
                        logger.error(f"Inf detected in {context} tensor '{name}'")
                        return True
                elif isinstance(tensor, (int, float)):
                    if math.isnan(tensor) or math.isinf(tensor):
                        logger.error(f"NaN/Inf detected in {context} scalar '{name}': {tensor}")
                        return True
            return False
        except Exception as e:
            logger.warning(f"Error during tensor NaN check in {context}: {e}")
            return False
    
    def _check_gradients_for_nan(self) -> bool:
        """Check if any gradients contain NaN values.
        
        Returns:
            bool: True if NaN detected, False otherwise
        """
        if not self.nan_detection_enabled:
            return False
            
        try:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if param.grad.isnan().any():
                        logger.error(f"NaN detected in gradient of parameter: {name}")
                        return True
                    if param.grad.isinf().any():
                        logger.error(f"Inf detected in gradient of parameter: {name}")
                        return True
            return False
        except Exception as e:
            logger.warning(f"Error during gradient NaN check: {e}")
            return False
    
    def _handle_nan_detection(self, context: str, step: int) -> None:
        """Handle NaN detection by logging details and potentially recovering.
        
        Args:
            context: Context where NaN was detected
            step: Current training step
        """
        logger.error(f"🚨 NaN DETECTED at step {step} in context: {context}")
        logger.error(f"Last healthy step was: {self.last_healthy_step}")
        
        # Log additional diagnostics
        lr = self.optimizer.param_groups[0]['lr']
        logger.error(f"Current learning rate: {lr}")
        
        # Log recent losses for context
        if self.recent_losses:
            recent_loss_str = ", ".join([f"{loss:.6f}" for loss in self.recent_losses[-5:]])
            logger.error(f"Recent losses: {recent_loss_str}")
        
        # Save current state before potential recovery
        try:
            nan_checkpoint_path = f"nan_detected_step_{step}.pth"
            self.save_checkpoint(nan_checkpoint_path, step=step, nan_detected=True)
            logger.info(f"Saved checkpoint before NaN handling: {nan_checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save NaN checkpoint: {e}")
        
        # Attempt automatic recovery if enabled
        if self.nan_checkpoint_recovery:
            try:
                recovery_success = self._attempt_nan_recovery(context, step)
                if recovery_success:
                    logger.info(f"Successfully recovered from NaN detection in {context}")
                    return
            except Exception as recovery_error:
                logger.error(f"NaN recovery failed: {recovery_error}")
        
        # Raise exception to trigger recovery or stop training
        raise NaNDetectedException(f"NaN detected in {context} at step {step}")
    
    def _attempt_nan_recovery(self, context: str, current_step: int) -> bool:
        """Attempt to recover from NaN detection by loading a previous healthy checkpoint.
        
        Args:
            context: Context where NaN was detected
            current_step: Current training step
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting NaN recovery at step {current_step}, context: {context}")
        
        # Find the most recent healthy checkpoint
        recovery_checkpoint = None
        
        # Check for best model checkpoint first
        try:
            from pathlib import Path
            checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
            best_checkpoint_path = checkpoint_dir / "best_model.pth"
            
            if best_checkpoint_path.exists():
                recovery_checkpoint = str(best_checkpoint_path)
                logger.info(f"Found best model checkpoint for recovery: {recovery_checkpoint}")
            else:
                # Look for recent step checkpoints
                checkpoint_pattern = checkpoint_dir / "checkpoint_step_*.pth"
                checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pth"))
                
                if checkpoint_files:
                    # Sort by step number and get the most recent one before current step
                    def extract_step(path):
                        try:
                            return int(path.stem.split('_')[-1])
                        except:
                            return 0
                    
                    checkpoint_files.sort(key=extract_step, reverse=True)
                    for ckpt_file in checkpoint_files:
                        ckpt_step = extract_step(ckpt_file)
                        if ckpt_step < current_step and ckpt_step >= self.last_healthy_step:
                            recovery_checkpoint = str(ckpt_file)
                            logger.info(f"Found recovery checkpoint at step {ckpt_step}: {recovery_checkpoint}")
                            break
                
        except Exception as e:
            logger.warning(f"Error searching for recovery checkpoint: {e}")
        
        if recovery_checkpoint is None:
            logger.error("No suitable recovery checkpoint found")
            return False
        
        # Attempt to load the recovery checkpoint
        try:
            logger.info(f"Loading recovery checkpoint: {recovery_checkpoint}")
            checkpoint_data = torch.load(recovery_checkpoint, map_location=self.device, weights_only=False)
            
            # Load model state
            if 'model_state_dict' in checkpoint_data:
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                logger.info("Model state restored from checkpoint")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("Optimizer state restored from checkpoint")
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint_data and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                logger.info("Scheduler state restored from checkpoint")
            
            # Reset scaler if using mixed precision
            if self.scaler is not None:
                logger.info("Resetting gradient scaler after NaN recovery")
                self.scaler = GradScaler(enabled=self.scaler.is_enabled())
            
            # Update recovery tracking
            recovered_step = checkpoint_data.get('step', self.last_healthy_step)
            self.last_healthy_step = recovered_step
            
            # Clear any cached gradients or loss values that might contain NaN
            self._clear_gpu_caches()
            self.recent_losses.clear()
            
            # Force CUDA cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"NaN recovery successful! Restored to step {recovered_step}")
            logger.warning(f"Training will continue from step {current_step} with restored model state")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load recovery checkpoint {recovery_checkpoint}: {e}")
            return False

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              max_steps: int = None) -> Dict:
        """
        Train the model using step-based control only.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_steps: Maximum training steps

        Returns:
            Training history
        """
        # Always use step-based training
        max_steps_config = self.training_config.get('max_steps')
        total_steps = max_steps or max_steps_config
        
        if total_steps is None:
            raise ValueError("max_steps must be specified for step-based training")
            
        return self._train_step_based(train_loader, val_loader, total_steps)

    def _train_step_based(self, train_loader: DataLoader, val_loader: DataLoader,
                          max_steps: int) -> Dict:
        """
        Step-based training loop - the new preferred method.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_steps: Maximum number of training steps
            
        Returns:
            Training history
        """
        # 初始化统一验证控制器
        validation_config = self.config.get('validation', {})
        self.validation_controller = ValidationController(validation_config, max_steps)
        
        # Parse step-based configuration with safe defaults
        # Backward-compat: support training.save_checkpoint_every
        ckpt_every_n_steps = int(self.training_config.get(
            'ckpt_every_n_steps',
            self.training_config.get('save_checkpoint_every', max(1, max_steps // 4))
        ))
        ckpt_every_n_steps = max(1, ckpt_every_n_steps)
        log_every_n_steps = max(1, self.training_config.get('log_every_n_steps', max(1, max_steps // 20)))  # 默认每5%进度记录一次
        
        # Early stopping configuration with type safety
        early_stopping_config = self.training_config.get('early_stopping', {})
        patience_steps = max(1, int(early_stopping_config.get('patience_steps', max(1, max_steps // 4))))
        min_delta = float(early_stopping_config.get('min_delta', 1e-4))
        
        # Setup scheduler for step-based training
        self._setup_step_based_scheduler(max_steps)
        
        logger.info(f"Starting step-based training for {max_steps} steps")
        validation_status = self.validation_controller.get_status_info()
        logger.info(f"  - Validation: {'Enabled' if validation_status['enabled'] else 'Disabled'} "
                   f"(interval: {validation_status['interval_steps']} steps)")
        logger.info(f"  - Checkpoint every: {ckpt_every_n_steps} steps")
        logger.info(f"  - Log every: {log_every_n_steps} steps")
        logger.info(f"  - Early stopping patience: {patience_steps} steps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        
        # Create infinite data iterator for continuous training
        infinite_loader = self._create_infinite_loader(train_loader)
        
        # Training state
        self.global_step = 0
        self.max_steps = max_steps  # Store for model access
        
        # 重新计算基于实际步数的sync_interval（如果未在配置中指定）
        if 'sync_interval' not in self.training_config:
            default_sync_interval = min(50, max(10, self.max_steps // 20))
            self.sync_interval = default_sync_interval
            logger.info(f"Calculated sync_interval based on max_steps: {self.sync_interval}")
        
        self.samples_seen = 0
        best_val_loss = float('inf')
        steps_without_improvement = 0
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(total=max_steps, desc="Training Steps", 
                   dynamic_ncols=True, ascii=True)
        
        # Main step-based training loop with NaN recovery
        while self.global_step < max_steps:
            try:
                # Get next batch from infinite iterator
                batch = next(infinite_loader)
                
                # Train single step
                step_result = self._train_single_step(batch, val_loader)
                
            except NaNDetectedException as nan_error:
                logger.error(f"NaN detected during training: {nan_error}")
                
                if self.nan_checkpoint_recovery:
                    logger.info("NaN recovery was already attempted. Continuing with current model state.")
                    # Continue training with the recovered model state
                    # Skip this batch and try the next one
                    continue
                else:
                    logger.error("NaN recovery is disabled. Stopping training.")
                    break
            except Exception as e:
                # 使用 exception 记录完整堆栈
                logger.exception(f"Unexpected error during training step {self.global_step}")
                
                # 累计错误计数
                if not hasattr(self, 'consecutive_errors'):
                    self.consecutive_errors = 0
                self.consecutive_errors += 1
                
                # 连续错误超过阈值则终止
                if self.consecutive_errors >= 5:
                    logger.error(f"Too many consecutive errors ({self.consecutive_errors}), aborting training")
                    raise RuntimeError(f"Training aborted due to {self.consecutive_errors} consecutive errors") from e
                
                continue
            
            # Update samples seen if step was completed
            if step_result.get('step_updated', False):
                # Reset error counter on successful step
                self.consecutive_errors = 0
                
                if hasattr(batch, 'batch'):
                    # 正确计算图数量 - 优先使用PyG提供的准确值
                    if hasattr(batch, 'num_graphs'):
                        num_graphs = int(batch.num_graphs)
                    elif hasattr(batch, 'batch') and torch.is_tensor(batch.batch):
                        # 从batch索引推断图数量（最大索引+1），这是真正的图数量
                        num_graphs = int(batch.batch.max().item() + 1)
                    else:
                        # 单图情况的fallback
                        num_graphs = 1
                else:
                    # 单图情况的fallback
                    num_graphs = 1
                self.samples_seen += num_graphs * self.gradient_accumulation_steps
            
            # 统一验证逻辑 - 使用ValidationController判断
            if self.validation_controller.should_validate(self.global_step, step_result.get('step_updated', False)):
                # Pre-validation model health check
                if self.nan_detection_enabled:
                    logger.debug(f"Performing pre-validation health check at step {self.global_step}")
                    try:
                        if self._check_model_parameters_for_nan():
                            self._handle_nan_detection("pre_validation_check", self.global_step)
                    except NaNDetectedException:
                        logger.warning("Skipping validation due to NaN detection")
                        continue
                
                val_losses = self._validate(val_loader, max_batches=self.validation_controller.get_quick_batches())
                
                # Early stopping check and best model saving
                current_val_loss = val_losses.get('loss', float('inf'))
                if current_val_loss < best_val_loss - min_delta:
                    best_val_loss = current_val_loss
                    # Update instance variable for consistent state
                    self.best_val_loss = current_val_loss
                    self.best_step = self.global_step
                    steps_without_improvement = 0
                    
                    # KISS优化: 可观测性日志 - 详细验证结果记录
                    val_recon = val_losses.get('recon_loss', 0.0)
                    val_edge = val_losses.get('edge_loss', 0.0)
                    val_contrast = val_losses.get('contrastive_loss', 0.0)
                    improvement = (self.best_val_loss - current_val_loss) if hasattr(self, 'best_val_loss') and self.best_val_loss != float('inf') else 0.0
                    logger.info(f"Step {self.global_step}: ✓ NEW BEST validation loss: {best_val_loss:.6f} "
                               f"(improved by {improvement:.6f}), components: recon={val_recon:.4f}, "
                               f"edge={val_edge:.4f}, contrast={val_contrast:.4f}")
                    
                    # Save best model checkpoint
                    try:
                        self.save_checkpoint("best_model.pth", is_best=True, val_loss=current_val_loss, step=self.global_step)
                        logger.info(f"Saved best model at step {self.global_step}")
                    except Exception as e:
                        logger.warning(f"Failed to save best model checkpoint: {e}")
                else:
                    # 使用验证控制器的间隔而不是硬编码的eval_every_n_steps
                    interval = self.validation_controller.interval_steps
                    steps_without_improvement += interval
                    
                    # KISS优化: 可观测性日志 - 记录非改进验证结果
                    val_recon = val_losses.get('recon_loss', 0.0)
                    val_edge = val_losses.get('edge_loss', 0.0)
                    stagnation_ratio = steps_without_improvement / patience_steps
                    logger.info(f"Step {self.global_step}: validation loss: {current_val_loss:.6f} "
                               f"(best: {best_val_loss:.6f}), stagnation: {steps_without_improvement}/{patience_steps} "
                               f"({stagnation_ratio:.1%}), components: recon={val_recon:.4f}, edge={val_edge:.4f}")
                    
                if steps_without_improvement >= patience_steps:
                    logger.info(f"Early stopping at step {self.global_step} "
                              f"({steps_without_improvement} steps without improvement)")
                    break
                    
                # Validation-driven LR adjustment（仅当主调度器就是 ReduceLROnPlateau 时）
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_val_loss)
                    logger.debug(f"ReduceLROnPlateau scheduler stepped with val_loss={current_val_loss:.6f}")
                
                # Run callbacks (ModelCheckpoint, EarlyStopping, etc.)
                try:
                    should_stop = self._run_callbacks(step_result, val_losses)
                    if should_stop:
                        logger.info(f"Training stopped by callback at step {self.global_step}")
                        break
                except Exception as e:
                    logger.warning(f"Callback execution failed: {e}")
                
                # Update history with step-based metrics
                self._update_step_history(step_result, val_losses)
                
                # Real-time Optuna pruning check
                if self.optuna_trial and OPTUNA_AVAILABLE:
                    # Use reconstruction MAE as pruning metric (lower is better)
                    pruning_metric = val_losses.get('recon_mae_masked', float('inf'))
                    
                    # Report to Optuna
                    self.optuna_trial.report(pruning_metric, self.global_step)
                    
                    # Check if trial should be pruned
                    if self.optuna_trial.should_prune():
                        logger.info(f"Trial pruned at step {self.global_step} with metric {pruning_metric:.6f}")
                        raise optuna.TrialPruned()
                
            # Step-based checkpointing - only trigger when optimizer truly updated
            if step_result.get('step_updated', False) and self.global_step % ckpt_every_n_steps == 0:
                checkpoint_path = f"checkpoint_step_{self.global_step}.pth"
                self.save_checkpoint(checkpoint_path, step=self.global_step, 
                                   samples_seen=self.samples_seen)
                
            # GPU优化的日志记录 - 减少日志频率，减少格式化开销
            if step_result.get('step_updated', False) and self.global_step % log_every_n_steps == 0:
                self._log_step_metrics(step_result)
            
            # GPU优化的进度条更新 - 大幅减少更新频率，避免频繁GUI/终端刷新
            # 只在同步间隔、日志间隔或关键节点更新进度条
            should_update_progress = (
                self.global_step % self.sync_interval == 0 or  # 同步间隔
                self.global_step % log_every_n_steps == 0 or   # 日志间隔
                step_result.get('step_updated', False) and self.global_step % (log_every_n_steps // 4) == 0 or  # 更频繁的优化步更新
                self.global_step < 100 or  # 训练初期更频繁更新
                self.global_step % 1000 == 0  # 每1000步强制更新
            )
            
            if should_update_progress:
                # 批量计算进度条指标，减少重复计算
                elapsed_time = time.time() - start_time
                steps_per_sec = self.global_step / max(elapsed_time, 1e-8)
                eta_seconds = (max_steps - self.global_step) / max(steps_per_sec, 1e-8)
                
                # 优化字符串格式化，减少不必要的精度
                pbar.set_postfix({
                    'Step': f"{self.global_step}/{max_steps}",
                    'Samples': f"{self.samples_seen//1000}k" if self.samples_seen >= 1000 else str(self.samples_seen),
                    'Steps/s': f"{steps_per_sec:.1f}",  # 降低精度减少格式化开销
                    'ETA': f"{eta_seconds/3600:.1f}h"
                })
            
            # 进度条位置更新：总是更新但减少显示刷新
            pbar.update(1)
            
        pbar.close()
        
        # 恢复最佳模型权重（修复早停后未恢复最佳权重的问题）
        if hasattr(self, 'best_val_loss') and torch.isfinite(torch.tensor(self.best_val_loss)) and self.best_val_loss < float('inf'):
            try:
                checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
                best_checkpoint_path = checkpoint_dir / "best_model.pth"
                if best_checkpoint_path.exists():
                    self.load_checkpoint(str(best_checkpoint_path))
                    logger.info(f"Restored best model (val_loss={self.best_val_loss:.6f}) before exit")
                else:
                    logger.warning(f"Best checkpoint not found at {best_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to restore best checkpoint: {e}")
        
        # GPU优化：训练结束时清空所有缓存并进行最终同步
        self._final_cache_cleanup()
        
        # Clean up resources to prevent file handle leaks
        self._cleanup_resources()
        
        total_time = time.time() - start_time
        
        # 训练完成综合报告
        final_mask_ratio = self.mask_ratio_at(self.global_step, self.max_steps)
        final_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        
        logger.info("=" * 80)
        logger.info("📊 TRAINING COMPLETION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏱️  Duration: {total_time:.2f}s ({total_time/3600:.2f}h)")
        logger.info(f"🚀 Steps: {self.global_step}/{max_steps} ({self.global_step/max_steps:.1%})")
        logger.info(f"📈 Samples: {self.samples_seen:,}")
        logger.info(f"⚡ Speed: {self.global_step/total_time:.2f} steps/sec")
        logger.info(f"🎯 Best validation loss: {best_val_loss:.6f} (step {self.best_step})")
        logger.info(f"📊 Fixed mask ratio: {final_mask_ratio:.3f}")
        logger.info(f"📚 Final learning rate: {final_lr:.2e}")
        logger.info("=" * 80)
        
        return self.history

    def _create_infinite_loader(self, train_loader: DataLoader):
        """Create infinite data iterator with proper resource management.
        
        This implementation avoids creating new worker processes on each epoch,
        preventing file handle leaks during long training runs.
        """
        while True:
            # Create iterator once and reuse until exhausted
            loader_iter = iter(train_loader)
            try:
                while True:
                    yield next(loader_iter)
            except StopIteration:
                # Iterator exhausted, continue to create a new one
                # This is much more efficient than recreating workers every epoch
                logger.debug("DataLoader iterator exhausted, creating new one")
                continue

    def _setup_step_based_scheduler(self, max_steps: int):
        """Setup learning rate scheduler for step-based training."""
        if self.scheduler is not None:
            return  # Already set up
            
        scheduler_config = self.training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'WarmupCosine')  # KISS优化: 默认使用Warmup+Cosine
        
        warmup_ratio = scheduler_config.get('warmup_ratio', 0.1)
        warmup_steps = max(1, int(max_steps * warmup_ratio))  # 避免除零风险
        
        logger.info(f"Setting up {scheduler_type} scheduler for {max_steps} steps (warmup: {warmup_steps})")
        
        if scheduler_type == 'WarmupCosine':
            # KISS优化: Warmup+Cosine调度器
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(step):
                if step < warmup_steps:
                    # 线性warm-up阶段
                    return step / warmup_steps
                else:
                    # 余弦下降阶段 - 避免除零风险
                    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
                    return 0.5 * (1 + math.cos(math.pi * progress))
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        elif scheduler_type == 'WSD':
            from torch.optim.lr_scheduler import LinearLR, ConstantLR, CosineAnnealingLR, SequentialLR
            stable_ratio = float(scheduler_config.get('stable_ratio', 0.7))
            decay_ratio = float(scheduler_config.get('decay_ratio', max(0.0, 1.0 - warmup_ratio - stable_ratio)))
            warmup_steps = max(1, int(max_steps * warmup_ratio))
            stable_steps = max(0, int(max_steps * stable_ratio))
            decay_steps = max(1, int(max_steps - warmup_steps - stable_steps))
            remainder = max_steps - (warmup_steps + stable_steps + decay_steps)
            if remainder != 0:
                decay_steps = max(1, decay_steps + remainder)
            min_lr = float(scheduler_config.get('min_lr', 1e-6))
            start_factor = float(scheduler_config.get('start_factor', 0.01))
            logger.info(f"Setting up WSD scheduler: warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}, min_lr={min_lr}")
            sched_warmup = LinearLR(self.optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps)
            sched_stable = ConstantLR(self.optimizer, factor=1.0, total_iters=stable_steps)
            sched_decay = CosineAnnealingLR(self.optimizer, T_max=decay_steps, eta_min=min_lr)
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[sched_warmup, sched_stable, sched_decay],
                milestones=[warmup_steps, warmup_steps + stable_steps]
            )
        elif scheduler_type == 'OneCycleLR':
            # 确保max_steps足够大，避免OneCycleLR的除零错误
            if max_steps < 100:
                logger.warning(f"max_steps={max_steps} too small for OneCycleLR, using ConstantLR instead")
                from torch.optim.lr_scheduler import ConstantLR
                self.scheduler = ConstantLR(self.optimizer, factor=1.0)
            else:
                # 调整warmup比例确保至少有1步warmup
                safe_pct_start = max(1 / max_steps, warmup_ratio)
                
                # 从配置读取或使用默认值
                div_factor = scheduler_config.get('div_factor', 25.0)
                final_div_factor = scheduler_config.get('final_div_factor', 1000.0)
                
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=float(self.training_config.get('learning_rate', 5e-4)),
                    total_steps=max_steps,
                    pct_start=safe_pct_start,
                    div_factor=div_factor,
                    final_div_factor=final_div_factor
                )
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max_steps,
                eta_min=self.training_config.get('learning_rate', 5e-4) * 0.01
            )
        else:
            # 未知调度器类型，使用WarmupCosine作为安全默认值
            logger.warning(f"Unknown scheduler type '{scheduler_type}', falling back to WarmupCosine")
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (max_steps - warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _train_single_step(self, batch, val_loader=None) -> Dict[str, float]:
        """Train a single step with GPU-optimized loss tracking."""
        self.model.train()
        
        try:
            # Move batch to device
            batch = self._prepare_batch(batch)

            # Forward pass
            outputs = self._forward_pass(batch)
            
            # Check for NaN in forward pass outputs
            if self.nan_detection_enabled and self.global_step % self.nan_detection_frequency == 0:
                forward_tensors = {'forward_output': outputs[0] if isinstance(outputs, (list, tuple)) else outputs}
                if self._check_tensors_for_nan(forward_tensors, "forward_pass"):
                    self._handle_nan_detection("forward_pass", self.global_step)
            
            # Handle different output formats robustly（按末项结构区分 TCC vs 权重字典）
            weights = {}
            tcc_info = None

            # Common heads
            loss = outputs[0]
            recon_loss = outputs[1]
            edge_loss = outputs[2]
            contrastive_loss = torch.tensor(0.0, device=loss.device)

            if len(outputs) >= 6:
                maybe_contrastive = outputs[4]
                maybe_meta = outputs[5]

                # TCC: meta 是包含 contributions 与 weights 的字典
                if isinstance(maybe_meta, dict) and ('contributions' in maybe_meta and 'weights' in maybe_meta):
                    tcc_info = maybe_meta
                    if hasattr(maybe_contrastive, 'dtype'):
                        contrastive_loss = maybe_contrastive
                    weights = tcc_info.get('weights', {})
                else:
                    # 动态/静态权重：meta 为权重字典，contrastive占位为张量或0
                    if hasattr(maybe_contrastive, 'dtype'):
                        contrastive_loss = maybe_contrastive
                    if isinstance(maybe_meta, dict):
                        weights = maybe_meta
            elif len(outputs) == 5:
                # 兼容旧格式：最后一项为权重或占位
                maybe_meta = outputs[4]
                if isinstance(maybe_meta, dict):
                    weights = maybe_meta
                elif hasattr(maybe_meta, 'dtype'):
                    contrastive_loss = maybe_meta
            else:
                # 极端旧格式：使用静态权重占位
                weights = {'recon_weight': 1.0, 'edge_weight': 0.5}
                
            # KISS优化: 可观测性日志 - 详细损失组件跟踪 (每50步记录)
            if self.global_step % 50 == 0 and logger.isEnabledFor(logging.INFO):
                try:
                    total_val = float(loss.detach())
                    recon_val = float(recon_loss.detach().mean())
                    edge_val = float(edge_loss.detach().mean())
                    # 从TCC扩展信息优先读取组件损失；若未真实计算则不打印以免误导
                    desc_val = None
                    if tcc_info is not None:
                        computed_map = tcc_info.get('computed_components', {}) or {}
                        desc_computed = computed_map.get('descriptor', None)
                        comp_losses = tcc_info.get('component_losses', {})
                        if desc_computed is True and isinstance(comp_losses, dict) and 'descriptor' in comp_losses:
                            try:
                                desc_val = float(comp_losses['descriptor'])
                            except Exception:
                                desc_val = None
                    # 构造更清晰的日志：total_norm 与 raw_total 区分
                    raw_total = None
                    if tcc_info is not None and isinstance(tcc_info, dict):
                        try:
                            raw_total = float(tcc_info.get('raw_total_loss'))
                        except Exception:
                            raw_total = None
                    if desc_val is not None:
                        if raw_total is not None:
                            logger.info(
                                f"Step {self.global_step} Loss Components: total_norm={total_val:.4f}, raw_total={raw_total:.4f}, "
                                f"recon={recon_val:.4f}, edge={edge_val:.4f}, descriptor={desc_val:.4f}, weights={weights}"
                            )
                        else:
                            logger.info(
                                f"Step {self.global_step} Loss Components: total_norm={total_val:.4f}, "
                                f"recon={recon_val:.4f}, edge={edge_val:.4f}, descriptor={desc_val:.4f}, weights={weights}"
                            )
                    else:
                        if raw_total is not None:
                            logger.info(
                                f"Step {self.global_step} Loss Components: total_norm={total_val:.4f}, raw_total={raw_total:.4f}, "
                                f"recon={recon_val:.4f}, edge={edge_val:.4f}, weights={weights}"
                            )
                        else:
                            logger.info(
                                f"Step {self.global_step} Loss Components: total_norm={total_val:.4f}, "
                                f"recon={recon_val:.4f}, edge={edge_val:.4f}, weights={weights}"
                            )
                except Exception:
                    pass

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self._backward_pass(loss)

            # Check for NaN in gradients after backward pass
            if self.nan_detection_enabled and self.global_step % self.nan_detection_frequency == 0:
                if self._check_gradients_for_nan():
                    self._handle_nan_detection("gradients", self.global_step)

            # Update weights and global step
            step_updated = False
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step()
                self.global_step += 1
                step_updated = True
                
                # Check model parameters for NaN after optimizer step
                if self.nan_detection_enabled and self.global_step % self.nan_detection_frequency == 0:
                    if self._check_model_parameters_for_nan():
                        self._handle_nan_detection("model_parameters", self.global_step)
                    else:
                        # Update last healthy step if no NaN detected
                        self.last_healthy_step = self.global_step

            # GPU-optimized loss tracking: 延迟同步，批量缓存
            # 将损失保持在GPU上，避免频繁.item()调用
            scaled_total_loss = loss * self.gradient_accumulation_steps
            
            # 优化的GPU缓存：限制缓存大小防止内存泄露
            if len(self.gpu_loss_cache) < self.max_cache_size:
                self.gpu_loss_cache.append(scaled_total_loss.detach())
                self.gpu_metrics_cache['recon_loss'].append(recon_loss.detach())
                self.gpu_metrics_cache['edge_loss'].append(edge_loss.detach())
                if hasattr(contrastive_loss, 'detach'):
                    self.gpu_metrics_cache['contrastive_loss'].append(contrastive_loss.detach())
                else:
                    self.gpu_metrics_cache['contrastive_loss'].append(torch.tensor(0.0, device=loss.device))
            else:
                # 缓存已满，立即同步避免内存泄露
                if self.gpu_loss_cache:
                    cpu_loss = torch.stack(self.gpu_loss_cache).mean().item()
                    self.recent_losses.append(cpu_loss)
                    if len(self.recent_losses) > self.loss_window_size:
                        self.recent_losses = self.recent_losses[-self.loss_window_size:]
                    self._clear_gpu_caches()
                # 添加当前损失
                self.gpu_loss_cache.append(scaled_total_loss.detach())
                self.gpu_metrics_cache['recon_loss'].append(recon_loss.detach())
                self.gpu_metrics_cache['edge_loss'].append(edge_loss.detach())
                if hasattr(contrastive_loss, 'detach'):
                    self.gpu_metrics_cache['contrastive_loss'].append(contrastive_loss.detach())
                else:
                    self.gpu_metrics_cache['contrastive_loss'].append(torch.tensor(0.0, device=loss.device))
            
            # 只有在需要同步时才计算CPU值（用于recent_losses等需要CPU计算的逻辑）
            if self.global_step % self.sync_interval == 0 or step_updated:
                # 批量同步GPU缓存到CPU
                if self.gpu_loss_cache:
                    # 计算平均值并同步一次
                    cpu_loss = torch.stack(self.gpu_loss_cache).mean().item()
                    self.recent_losses.append(cpu_loss)
                    if len(self.recent_losses) > self.loss_window_size:
                        self.recent_losses = self.recent_losses[-self.loss_window_size:]
                    
                    # 清空GPU缓存
                    self._clear_gpu_caches()
                        
                    # 用于返回的当前损失值
                    current_loss = cpu_loss
                else:
                    # 备用：直接计算（但这种情况应该很少发生）
                    current_loss = scaled_total_loss.item()
            else:
                # 非同步步骤：估算当前损失（避免GPU-CPU同步）
                current_loss = self.recent_losses[-1] if self.recent_losses else 0.0
            
            # Collect basic masking statistics if scheduler available
            if self.masking_scheduler is not None and step_updated:
                try:
                    # Simple schedulers don't need complex performance tracking
                    logger.debug(f"Training step {self.global_step}: loss={current_loss:.4f}")
                except Exception as e:
                    logger.debug(f"Failed to log masking stats: {e}")

            # GPU优化的返回结果：避免不必要的.item()调用
            if self.global_step % self.sync_interval == 0 or step_updated:
                # 同步步骤：返回准确的CPU值
                # 提取descriptor损失（优先component_losses，再回退raw_losses）
                desc_loss_val = 0.0
                if tcc_info is not None and isinstance(tcc_info, dict):
                    comp_losses = tcc_info.get('component_losses', {})
                    if isinstance(comp_losses, dict) and 'descriptor' in comp_losses:
                        try:
                            desc_loss_val = float(comp_losses['descriptor'])
                        except Exception:
                            desc_loss_val = 0.0
                    if desc_loss_val == 0.0:
                        raw_losses = tcc_info.get('raw_losses', {})
                        if isinstance(raw_losses, dict) and 'descriptor' in raw_losses:
                            try:
                                desc_loss_val = float(raw_losses['descriptor'])
                            except Exception:
                                desc_loss_val = 0.0
                result = {
                    'loss': current_loss,
                    'recon_loss': recon_loss.item(),
                    'edge_loss': edge_loss.item(),
                    'contrastive_loss': contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0,
                    'descriptor_loss': desc_loss_val,
                    'weights': weights,
                    'step_updated': step_updated
                }
            else:
                # 非同步步骤：返回估算值，减少GPU-CPU传输
                desc_loss_val = 0.0
                if tcc_info is not None and isinstance(tcc_info, dict):
                    comp_losses = tcc_info.get('component_losses', {})
                    if isinstance(comp_losses, dict) and 'descriptor' in comp_losses:
                        try:
                            desc_loss_val = float(comp_losses['descriptor'])
                        except Exception:
                            desc_loss_val = 0.0
                    if desc_loss_val == 0.0:
                        raw_losses = tcc_info.get('raw_losses', {})
                        if isinstance(raw_losses, dict) and 'descriptor' in raw_losses:
                            try:
                                desc_loss_val = float(raw_losses['descriptor'])
                            except Exception:
                                desc_loss_val = 0.0
                result = {
                    'loss': current_loss,
                    'recon_loss': self.gpu_metrics_cache['recon_loss'][-1].item() if self.gpu_metrics_cache['recon_loss'] else 0.0,
                    'edge_loss': self.gpu_metrics_cache['edge_loss'][-1].item() if self.gpu_metrics_cache['edge_loss'] else 0.0,
                    'contrastive_loss': self.gpu_metrics_cache['contrastive_loss'][-1].item() if self.gpu_metrics_cache['contrastive_loss'] else 0.0,
                    'descriptor_loss': desc_loss_val,
                    'weights': weights,
                    'step_updated': step_updated
                }
            
            # Final check for NaN in result values
            if self.nan_detection_enabled:
                loss_dict = {
                    'loss': result['loss'],
                    'recon_loss': result['recon_loss'],
                    'edge_loss': result['edge_loss'],
                    'contrastive_loss': result['contrastive_loss']
                }
                if self._check_tensors_for_nan(loss_dict, "training_losses"):
                    self._handle_nan_detection("training_losses", self.global_step)
            
            # Add TCC information if available
            if tcc_info is not None:
                result['tcc_info'] = tcc_info
                
            return result
            
        except Exception as e:
            logger.error(f"Error in training step {getattr(self, 'global_step', 0)}: {e}")
            # Return zero losses to avoid breaking the training loop
            return {
                'loss': 0.0,
                'recon_loss': 0.0,
                'edge_loss': 0.0,
                'contrastive_loss': 0.0,
                'weights': {},
                'step_updated': False
            }

    def _update_step_history(self, step_losses: Dict[str, float], val_losses: Dict[str, float]):
        """Update history with step-based metrics."""
        # Initialize step-based history if needed
        if 'steps' not in self.history:
            self.history['steps'] = []
            self.history['samples_seen'] = []
            self.history['step_train_loss'] = []
            self.history['step_val_loss'] = []
        
        self.history['steps'].append(self.global_step)
        self.history['samples_seen'].append(self.samples_seen)
        self.history['step_train_loss'].append(step_losses['loss'])
        self.history['step_val_loss'].append(val_losses.get('loss', 0.0))
        
        # Keep legacy history format for compatibility
        if 'train_loss' not in self.history:
            self.history['train_loss'] = []
            self.history['val_loss'] = []
        self.history['train_loss'].append(step_losses['loss'])
        self.history['val_loss'].append(val_losses.get('loss', 0.0))

    def _log_step_metrics(self, step_losses: Dict[str, float]):
        """Log step-based training metrics with TCC contributions."""
        lr = self.optimizer.param_groups[0]['lr']
        log_str = f"Step {self.global_step} | "
        log_str += f"loss={step_losses['loss']:.4f}, "
        log_str += f"recon={step_losses['recon_loss']:.4f}, "
        log_str += f"edge={step_losses['edge_loss']:.4f}"
        # descriptor打印在TCC信息里控制（避免未计算时误显示0.0000）
        tcc_info = step_losses.get('tcc_info')
        if tcc_info is not None:
            computed_map_hdr = tcc_info.get('computed_components', {}) or {}
            if computed_map_hdr.get('descriptor', None) is True:
                comp_losses_hdr = tcc_info.get('component_losses', {}) or {}
                if 'descriptor' in comp_losses_hdr:
                    try:
                        log_str += f", descriptor={float(comp_losses_hdr['descriptor']):.4f}"
                    except Exception:
                        pass
        
        # Add TCC contribution information if available (增强损失尺度监控)
        if tcc_info is not None:
            contributions = tcc_info.get('contributions', {})
            raw_losses = tcc_info.get('raw_losses', {})
            normalized_losses = tcc_info.get('normalized_losses', {})
            ema_scales = tcc_info.get('ema_scales', {})
            computed_map = tcc_info.get('computed_components', {}) or {}
            
            if contributions:
                log_str += ", "
                first = True
                for comp, contrib in contributions.items():
                    # 未真实计算的descriptor不显示贡献占比，避免误导
                    if comp == 'descriptor' and computed_map.get('descriptor', None) is False:
                        continue
                    if not first:
                        log_str += ", "
                    first = False
                    comp_short = comp.replace('_loss', '').replace('reconstruction', 'recon').replace('edge', 'edge').replace('smooth', 'smooth')[:1]
                    log_str += f"c_{comp_short}={contrib:.2f}"
            
            # 每50步详细显示损失尺度信息
            if self.global_step % 50 == 0 and raw_losses and normalized_losses:
                raw_recon = raw_losses.get('reconstruction', 0)
                raw_edge = raw_losses.get('edge', 0)
                # 仅在真实计算descriptor时才显示对应数值
                show_desc = computed_map.get('descriptor', None) is True
                raw_desc = raw_losses.get('descriptor', None) if show_desc else None
                norm_recon = normalized_losses.get('reconstruction', 0)
                norm_edge = normalized_losses.get('edge', 0)
                norm_desc = normalized_losses.get('descriptor', None) if show_desc else None
                scale_recon = ema_scales.get('reconstruction', 1)
                scale_edge = ema_scales.get('edge', 1)
                scale_desc = ema_scales.get('descriptor', None) if show_desc else None
                
                logger.info("TCC Loss Scale Analysis:")
                base_raw = f"  Raw: recon={raw_recon:.6f}, edge={raw_edge:.6f}"
                base_norm = f"  Norm: recon={norm_recon:.6f}, edge={norm_edge:.6f}"
                # 更名：Scale -> EMA（表示损失EMA尺度），避免与权重混淆
                base_scale = f"  EMA: recon={scale_recon:.6f}, edge={scale_edge:.6f}"
                if raw_desc is not None:
                    base_raw += f", descriptor={raw_desc:.6f}"
                if norm_desc is not None:
                    base_norm += f", descriptor={norm_desc:.6f}"
                if scale_desc is not None:
                    base_scale += f", descriptor={scale_desc:.6f}"
                logger.info(base_raw)
                logger.info(base_norm)
                logger.info(base_scale)
                # 目标贡献范围可不固定显示，以 contributions 为准

        log_str += f" | LR: {lr:.6f} | Samples: {self.samples_seen:,}"
        
        logger.info(log_str)

    def _validate(self, val_loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_edge_loss = 0
        total_contrastive_loss = 0
        # Proxy metrics accumulators
        total_recon_mae_masked = 0.0
        total_edge_mae_masked = 0.0
        total_health = {
            'var_mean': 0.0,
            'var_min': 0.0,
            'frac_below_thr': 0.0,
            'cov_offdiag_mean': 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(
                val_loader, 
                desc="Validation", 
                leave=False,
                dynamic_ncols=True,
                ascii=True,
                file=None,
                disable=False
            )
            for batch_idx, batch in enumerate(pbar):
                try:
                    batch = self._prepare_batch(batch)
                    outputs = self._forward_pass(batch)
                    
                    # Check for NaN in forward pass outputs during validation
                    if self.nan_detection_enabled:
                        forward_tensors = {'val_forward_output': outputs[0] if isinstance(outputs, (list, tuple)) else outputs}
                        if self._check_tensors_for_nan(forward_tensors, f"validation_forward_batch_{batch_idx}"):
                            logger.error(f"NaN detected in validation forward pass at batch {batch_idx}")
                            # Skip this batch but continue validation
                            continue
                    
                    # 处理不同的输出格式
                    if len(outputs) == 6:  # TCC enabled with extended info
                        loss, recon_loss, edge_loss, _, contrastive_loss, tcc_info = outputs
                    elif len(outputs) == 5:  # Dynamic weighting enabled
                        loss, recon_loss, edge_loss, _, weights = outputs
                        contrastive_loss = torch.tensor(0.0, device=loss.device)
                    else:  # Static weighting
                        loss, recon_loss, edge_loss, _ = outputs
                        contrastive_loss = torch.tensor(0.0, device=loss.device)

                    # Check for NaN in loss values before accumulation
                    if self.nan_detection_enabled:
                        loss_tensors = {
                            'val_loss': loss,
                            'val_recon_loss': recon_loss,
                            'val_edge_loss': edge_loss,
                            'val_contrastive_loss': contrastive_loss
                        }
                        if self._check_tensors_for_nan(loss_tensors, f"validation_losses_batch_{batch_idx}"):
                            logger.error(f"NaN detected in validation losses at batch {batch_idx}")
                            # Skip this batch but continue validation
                            continue

                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()
                    total_edge_loss += edge_loss.item()
                    total_contrastive_loss += contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0
                    
                except Exception as e:
                    logger.warning(f"Error in validation batch {batch_idx}: {e}")
                    # Skip this batch and continue
                    continue
                
                # Compute proxy metrics: masked MAE and embedding health
                try:
                    he_attr = getattr(batch, 'hyperedge_attr', None)
                    he_idx = getattr(batch, 'hyperedge_index', None)
                    
                    # Generate validation masks using the SAME strategy/config as training
                    # - pass smiles to enable semantic masking (same distribution)
                    # - pass mask_ratio from the same scheduler function
                    mask_ratio = self.mask_ratio_at(self.global_step, getattr(self, 'max_steps', 1000))
                    smiles = getattr(batch, 'smiles', None)
                    val_node_mask, val_edge_mask = self.model._generate_intelligent_masks(
                        batch.x, he_idx, he_attr,
                        global_step=self.global_step,
                        max_steps=getattr(self, 'max_steps', 1000),
                        smiles=smiles,
                        mask_ratio=mask_ratio
                    )

                    # Strong validation for mask consistency
                    if val_edge_mask is not None and he_attr is not None:
                        val_edge_mask = val_edge_mask.reshape(-1).bool()
                        if val_edge_mask.numel() != he_attr.size(0):
                            logger.warning(f"Edge mask size {val_edge_mask.numel()} != edge attr size {he_attr.size(0)}, adjusting...")
                            # Resize mask to match exactly
                            if val_edge_mask.numel() < he_attr.size(0):
                                # Pad with False
                                padding = torch.zeros(he_attr.size(0) - val_edge_mask.numel(), dtype=torch.bool, device=val_edge_mask.device)
                                val_edge_mask = torch.cat([val_edge_mask, padding])
                            else:
                                # Truncate
                                val_edge_mask = val_edge_mask[:he_attr.size(0)]
                    
                    # Use unified model interface for consistent evaluation with proper AMP handling
                    if self.use_amp_advanced:
                        with torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype, cache_enabled=True):
                            recon_x, edge_pred, z_enhanced = self.model(
                                batch.x, he_idx, he_attr,
                                node_mask=val_node_mask, edge_mask=val_edge_mask,
                                eval_mode=True,  # Key: use evaluation mode
                                global_step=self.global_step,
                                max_steps=getattr(self, 'max_steps', 1000)
                            )
                    else:
                        recon_x, edge_pred, z_enhanced = self.model(
                            batch.x, he_idx, he_attr,
                            node_mask=val_node_mask, edge_mask=val_edge_mask,
                            eval_mode=True,  # Key: use evaluation mode
                            global_step=self.global_step,
                            max_steps=getattr(self, 'max_steps', 1000)
                        )

                    # Masked MAE using validation masks with proper mask statistics
                    node_mask_count = val_node_mask.sum().item() if val_node_mask is not None else 0
                    edge_mask_count = val_edge_mask.sum().item() if val_edge_mask is not None and val_edge_mask.numel() > 0 else 0
                    
                    logger.debug(f"Validation masks: {node_mask_count}/{batch.x.size(0)} nodes, {edge_mask_count}/{val_edge_mask.numel() if val_edge_mask is not None else 0} edges")
                    
                    # Calculate MAE with comprehensive fallback logic
                    if val_node_mask is not None:
                        if val_node_mask.sum().item() > 0:
                            # Normal case: valid mask with masked elements
                            mae_value = masked_mae(recon_x, batch.x, val_node_mask)
                            total_recon_mae_masked += mae_value
                            if logger.isEnabledFor(logging.DEBUG):
                                try:
                                    logger.debug(f"Validation recon MAE (masked): {mae_value:.6f}, mask sum: {int(val_node_mask.sum())}")
                                except Exception:
                                    pass
                        else:
                            # Mask exists but all False - compute on full data
                            mae_value = torch.nn.functional.l1_loss(recon_x, batch.x).item()
                            total_recon_mae_masked += mae_value
                            logger.warning(f"Empty node mask (all False) in validation, using full data MAE: {mae_value:.6f}")
                    else:
                        # No mask generated - compute on full data
                        mae_value = torch.nn.functional.l1_loss(recon_x, batch.x).item()
                        total_recon_mae_masked += mae_value
                        logger.warning(f"No node mask in validation, using full data MAE: {mae_value:.6f}")
                    
                    if he_attr is not None and he_attr.size(0) > 0:
                        if val_edge_mask is not None:
                            if val_edge_mask.sum().item() > 0:
                                # Normal case: valid edge mask
                                mae_value = masked_mae(edge_pred, he_attr, val_edge_mask)
                                total_edge_mae_masked += mae_value
                                if logger.isEnabledFor(logging.DEBUG):
                                    try:
                                        logger.debug(f"Validation edge MAE (masked): {mae_value:.6f}, mask sum: {int(val_edge_mask.sum())}")
                                    except Exception:
                                        pass
                            else:
                                # Edge mask exists but all False
                                mae_value = torch.nn.functional.l1_loss(edge_pred, he_attr).item()
                                total_edge_mae_masked += mae_value
                                logger.warning(f"Empty edge mask (all False) in validation, using full data MAE: {mae_value:.6f}")
                        else:
                            # No edge mask generated
                            mae_value = torch.nn.functional.l1_loss(edge_pred, he_attr).item()
                            total_edge_mae_masked += mae_value
                            logger.warning(f"No edge mask in validation, using full data MAE: {mae_value:.6f}")

                    # Embedding health (batch-level node embeddings)
                    health = compute_embedding_health_metrics(z_enhanced)
                    for k in total_health:
                        total_health[k] += float(health.get(k, 0.0))
                except Exception as metric_err:
                    import traceback
                    logger.error(f"Proxy metric computation failed: {metric_err}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    # Continue to next batch instead of silently failing
                    logger.warning("Continuing with zero metrics for this batch")

                num_batches += 1

                # Early stop for quick validation
                if max_batches is not None and num_batches >= max_batches:
                    break

        # Neighbor consistency on sampled validation subset
        try:
            neighbor_stats = compute_neighbor_consistency_spearman(
                self.model,
                getattr(val_loader, 'dataset', None),
                self.device,
                sample_size=int(self.training_config.get('neighbor_consistency_sample', 128)),
            ) if hasattr(val_loader, 'dataset') else {'neighbor_spearman': 0.0}
        except Exception as e:
            logger.debug(f"Neighbor consistency metric failed: {e}")
            neighbor_stats = {'neighbor_spearman': 0.0}

        # Aggregate with debugging info
        results = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'edge_loss': total_edge_loss / num_batches,
            'recon_mae_masked': total_recon_mae_masked / max(1, num_batches),
            'edge_mae_masked': total_edge_mae_masked / max(1, num_batches),
            'emb_var_mean': total_health['var_mean'] / max(1, num_batches),
            'emb_var_min': total_health['var_min'] / max(1, num_batches),
            'emb_frac_below_thr': total_health['frac_below_thr'] / max(1, num_batches),
            'emb_cov_offdiag_mean': total_health['cov_offdiag_mean'] / max(1, num_batches),
        }
        results.update(neighbor_stats)
        
        # Final NaN check on validation results before returning
        if self.nan_detection_enabled:
            if self._check_tensors_for_nan(results, "validation_results"):
                logger.error("NaN detected in final validation results - replacing with safe values")
                # Return safe default values to prevent training crash
                return {
                    'loss': float('inf'),
                    'recon_loss': float('inf'),
                    'edge_loss': float('inf'),
                    'recon_mae_masked': float('inf'),
                    'edge_mae_masked': float('inf'),
                    'emb_var_mean': 0.0,
                    'emb_var_min': 0.0,
                    'emb_frac_below_thr': 1.0,
                    'emb_cov_offdiag_mean': 0.0,
                }
        
        # Debug: Log validation metrics computation
        logger.info(f"Validation computed from {num_batches} batches:")
        logger.info(f"  Raw totals - recon_mae: {total_recon_mae_masked:.6f}, edge_mae: {total_edge_mae_masked:.6f}")
        logger.info(f"  Health totals - var_mean: {total_health['var_mean']:.6f}, cov_offdiag: {total_health['cov_offdiag_mean']:.6f}")
        logger.info(f"  Final metrics - recon_mae_masked: {results['recon_mae_masked']:.6f}, edge_mae_masked: {results['edge_mae_masked']:.6f}")
        
        return results

    def _prepare_batch(self, batch) -> Dict:
        """Prepare batch for training."""
        # Move to device
        batch = batch.to(self.device, non_blocking=True)
        return batch

    def _forward_pass(self, batch):
        """Forward pass through the model with enhanced error handling."""
        try:
            # Ensure all tensors are on the correct device and have correct dtypes
            batch.x = batch.x.float()
            if hasattr(batch, 'hyperedge_attr') and batch.hyperedge_attr is not None:
                batch.hyperedge_attr = batch.hyperedge_attr.float()
            if hasattr(batch, 'hyperedge_index') and batch.hyperedge_index is not None:
                batch.hyperedge_index = batch.hyperedge_index.long()
            
            # Fix: Handle missing mask fields gracefully
            # For intelligent masking, always pass None to let model generate masks dynamically
            # The static masks from batch are only used for fallback cases
            node_mask = None  # Let model generate intelligent masks
            edge_mask = None  # Let model generate intelligent masks
            
            # Get additional information for intelligent masking
            smiles_list = getattr(batch, 'smiles', None)
            
            # 处理SMILES列表 - 支持批处理中的每个分子
            if smiles_list is not None:
                if isinstance(smiles_list, (list, tuple)):
                    # 如果是列表，保持原样供后续处理
                    smiles = smiles_list
                elif hasattr(smiles_list, 'item'):
                    # 处理tensor情况
                    smiles = [smiles_list.item()]
                else:
                    # 单个字符串的情况
                    smiles = [str(smiles_list)]
            else:
                smiles = None
            
            # Get recent loss for strategy optimization
            recent_loss = None
            if len(self.recent_losses) > 0:
                recent_loss = self.recent_losses[-1]
            
            # 计算掩码比例（已在配置中固定为0.7，但命名不带fixed）
            mask_ratio = self.mask_ratio_at(self.global_step, self.max_steps)
            
            # KISS优化: 可观测性日志 - 每100步记录关键指标
            if self.global_step % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                recent_loss_str = f"{recent_loss:.4f}" if recent_loss is not None else "N/A"
                logger.info(f"Step {self.global_step}: mask_ratio={mask_ratio:.3f}, "
                           f"lr={current_lr:.2e}, recent_loss={recent_loss_str}")
            
            if self.use_amp_advanced:
                # Autocast with selected dtype (BF16 on Ampere, else FP16)
                with torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype, cache_enabled=True):
                    outputs = self.model(
                        batch.x,
                        batch.hyperedge_index,
                        batch.hyperedge_attr,
                        node_mask,
                        edge_mask,
                        # epoch removed for step-based training
                        global_step=getattr(self, 'global_step', 0),
                        max_steps=getattr(self, 'max_steps', 1000),
                        smiles=smiles,
                        recent_loss=recent_loss,
                        mask_ratio=mask_ratio,
                        batch=getattr(batch, 'batch', None)
                    )
            else:
                outputs = self.model(
                    batch.x,
                    batch.hyperedge_index,
                    batch.hyperedge_attr,
                    node_mask,
                    edge_mask,
                    # epoch removed for step-based training
                    global_step=getattr(self, 'global_step', 0),
                    max_steps=getattr(self, 'max_steps', 1000),
                    smiles=smiles,
                    recent_loss=recent_loss,
                    mask_ratio=mask_ratio,
                    batch=getattr(batch, 'batch', None)
                )

            return outputs
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e):
                logger.error(f"CUDA/CUBLAS error in forward pass: {e}")
                # Clean up memory and retry without AMP
                torch.cuda.empty_cache()
                
                # Disable AMP for this batch and ensure complete output
                try:
                    outputs = self.model(
                        batch.x.float(),
                        batch.hyperedge_index.long(),
                        batch.hyperedge_attr.float() if batch.hyperedge_attr is not None else None,
                        node_mask,  # Use the safely extracted mask
                        edge_mask,  # Use the safely extracted mask
                        # epoch removed for step-based training
                        global_step=getattr(self, 'global_step', 0),
                        max_steps=getattr(self, 'max_steps', 1000),
                        smiles=smiles,
                        recent_loss=recent_loss,
                        mask_ratio=mask_ratio
                    )
                    return outputs  # Return complete outputs tuple
                except Exception as retry_e:
                    logger.error(f"Retry also failed: {retry_e}")
                    # Raise a special exception to indicate batch should be skipped
                    raise BatchSkippedException("Both original and retry forward pass failed")
            else:
                raise e

    def _backward_pass(self, loss: torch.Tensor):
        """Backward pass with gradient scaling."""
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self):
        """Enhanced optimizer step with GPU-optimized gradient norm tracking."""
        grad_norm = None
        
        if self.use_amp_advanced and self.scaler is not None:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping with enhanced numerical stability and GPU-optimized logging
        if self.clip_grad_norm > 0:
            # Use more stable gradient clipping for mixed precision
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_grad_norm,
                norm_type=2.0
            )
            # 确保grad_norm是float，兼容Tensor和非Tensor返回值
            grad_norm = float(grad_norm_t) if torch.is_tensor(grad_norm_t) else float(grad_norm_t)
            
            # KISS优化: 可观测性日志 - 梯度稳定性监控 (每200步记录)
            if self.global_step % 200 == 0 and grad_norm is not None:
                was_clipped = grad_norm > float(self.clip_grad_norm)
                logger.info(f"Step {self.global_step} Gradient: norm={grad_norm:.4f}, "
                           f"clipped={'Yes' if was_clipped else 'No'}, clip_threshold={self.clip_grad_norm}")
                # 检测梯度异常
                if grad_norm > 10.0:
                    logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")
                elif grad_norm < 1e-6:
                    logger.warning(f"Very small gradient norm detected: {grad_norm:.4e}")
            
            # Enhanced gradient norm monitoring with NaN detection
            if self.nan_detection_enabled and (math.isnan(grad_norm) or math.isinf(grad_norm)):
                self._handle_nan_detection(f"gradient_norm(value={grad_norm})", self.global_step)
            
            # 优化的梯度范数缓存：限制缓存大小，防止内存泄露 - 存储float
            if len(self.gpu_grad_norm_cache) < self.max_cache_size:
                self.gpu_grad_norm_cache.append(grad_norm)
            else:
                # 缓存已满，立即同步并清空
                if self.gpu_grad_norm_cache:
                    avg_grad_norm = sum(self.gpu_grad_norm_cache) / len(self.gpu_grad_norm_cache)
                    max_grad_norm = max(self.gpu_grad_norm_cache)
                    
                    # Enhanced anomaly detection for gradient norms
                    if max_grad_norm > 50.0:
                        logger.error(f"Severe gradient explosion detected (max: {max_grad_norm:.4f}, avg: {avg_grad_norm:.4f})")
                    elif max_grad_norm > 10.0:
                        logger.warning(f"Large gradient norm detected (max: {max_grad_norm:.4f}, avg: {avg_grad_norm:.4f})")
                    
                    self.gpu_grad_norm_cache.clear()
                
                # 添加当前梯度范数
                self.gpu_grad_norm_cache.append(grad_norm)
            
            # 只在同步间隔时记录梯度范数日志
            if self.global_step % self.sync_interval == 0:
                if self.gpu_grad_norm_cache:
                    # 计算缓存期间的平均梯度范数
                    avg_grad_norm = sum(self.gpu_grad_norm_cache) / len(self.gpu_grad_norm_cache)
                    max_grad_norm = max(self.gpu_grad_norm_cache)
                    min_grad_norm = min(self.gpu_grad_norm_cache)
                    
                    # Enhanced gradient norm analysis with anomaly detection
                    if max_grad_norm > 50.0:
                        logger.error(f"Severe gradient explosion detected (max: {max_grad_norm:.4f}, avg: {avg_grad_norm:.4f}, min: {min_grad_norm:.4f}) in last {len(self.gpu_grad_norm_cache)} steps")
                    elif max_grad_norm > 10.0:
                        logger.warning(f"Large gradient norm detected (max: {max_grad_norm:.4f}, avg: {avg_grad_norm:.4f}, min: {min_grad_norm:.4f}) in last {len(self.gpu_grad_norm_cache)} steps")
                    elif avg_grad_norm > 5.0:
                        logger.info(f"Moderate gradient norm (avg: {avg_grad_norm:.4f}, max: {max_grad_norm:.4f}, min: {min_grad_norm:.4f}) in last {len(self.gpu_grad_norm_cache)} steps")
                    elif min_grad_norm < 1e-6:
                        logger.warning(f"Very small gradient norm detected (min: {min_grad_norm:.6f}, avg: {avg_grad_norm:.4f}, max: {max_grad_norm:.4f}) - possible vanishing gradients")
                    else:
                        logger.debug(f"Normal gradient norm (avg: {avg_grad_norm:.4f}, max: {max_grad_norm:.4f}, min: {min_grad_norm:.4f}) in last {len(self.gpu_grad_norm_cache)} steps")
                    
                    # 清空梯度范数缓存
                    self.gpu_grad_norm_cache.clear()
            elif (self.global_step % self.sync_interval) == 0:
                # 仅在同步步长时记录突发异常，减少日志噪音
                if grad_norm > 50.0:  # 极端情况：严重梯度爆炸
                    logger.error(f"Severe gradient explosion detected: {grad_norm:.4f}")
                elif grad_norm > 20.0:  # 梯度爆炸
                    logger.warning(f"Gradient explosion detected: {grad_norm:.4f}")
            elif grad_norm < 1e-6:  # 梯度消失时立即警告
                logger.warning(f"Vanishing gradient detected: {grad_norm:.6f}")

        # Optimizer step with enhanced error handling
        if self.use_amp_advanced and self.scaler is not None:
            # Enhanced scaler step with better inf/nan detection
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Clear gradients efficiently (set_to_none=True is more memory efficient)
        self.optimizer.zero_grad(set_to_none=True)

        # Scheduler step - unified handling for all scheduler types
        if self.scheduler is not None:
            # ReduceLROnPlateau needs validation metrics, handled separately in validation phase
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass  # Will be stepped during validation
            elif hasattr(self.scheduler, 'step'):
                # All other schedulers step per optimization step
                self.scheduler.step()
            else:
                logger.warning(f"Unknown scheduler type: {type(self.scheduler).__name__}")

    def _update_history(self, train_losses: Dict[str, float], val_losses: Optional[Dict[str, float]],
                        step_time: float):
        """Update training history."""
        self.history['train_loss'].append(train_losses['loss'])
        self.history.setdefault('train_recon_loss', []).append(train_losses['recon_loss'])
        self.history.setdefault('train_edge_loss', []).append(train_losses['edge_loss'])
        
        if val_losses is not None:
            self.history['val_loss'].append(val_losses['loss'])
            self.history.setdefault('val_recon_loss', []).append(val_losses['recon_loss'])
            self.history.setdefault('val_edge_loss', []).append(val_losses['edge_loss'])
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        self.history['step_time'].append(step_time)

    def _run_callbacks(self, train_losses: Dict[str, float], val_losses: Optional[Dict[str, float]]) -> bool:
        """Run callbacks and check for early stopping."""
        should_stop = False

        for callback in self.callbacks:
            if hasattr(callback, 'on_step_end'):
                stop = callback.on_step_end(
                    step=self.global_step,
                    logs={
                        'train_loss': train_losses['loss'],
                        'val_loss': val_losses['loss'] if val_losses else None,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    },
                    model=self.model
                )
                if stop:
                    should_stop = True

        # ReduceLROnPlateau调度已移至主训练循环验证后处理，避免重复step
        return should_stop
    
    def _clear_gpu_caches(self):
        """清空GPU缓存的统一方法"""
        self.gpu_loss_cache.clear()
        for cache_list in self.gpu_metrics_cache.values():
            cache_list.clear()

    def _final_cache_cleanup(self):
        """GPU优化：训练结束时的最终缓存清理和同步"""
        try:
            # 清空损失缓存并记录最终统计
            if self.gpu_loss_cache:
                final_loss = torch.stack(self.gpu_loss_cache).mean().item()
                logger.info(f"Final cached loss batch: {final_loss:.6f} (from {len(self.gpu_loss_cache)} steps)")
                self.gpu_loss_cache.clear()
            
            # 清空指标缓存
            for metric_name, cache_list in self.gpu_metrics_cache.items():
                if cache_list:
                    avg_metric = torch.stack(cache_list).mean().item()
                    logger.debug(f"Final cached {metric_name}: {avg_metric:.6f} (from {len(cache_list)} steps)")
            
            # 统一清空所有GPU缓存
            self._clear_gpu_caches()
            
            # 清空梯度范数缓存并记录最终统计
            if self.gpu_grad_norm_cache:
                avg_grad_norm = sum(self.gpu_grad_norm_cache) / len(self.gpu_grad_norm_cache)
                max_grad_norm = max(self.gpu_grad_norm_cache)
                logger.info(f"Final gradient norm stats: avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f} (from {len(self.gpu_grad_norm_cache)} steps)")
                self.gpu_grad_norm_cache.clear()
            
            logger.debug("GPU cache cleanup completed successfully")
            
        except Exception as e:
            logger.warning(f"GPU cache cleanup failed: {e}")

    def _cleanup_resources(self):
        """Clean up resources to prevent file handle leaks."""
        try:
            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clean up logging handlers if possible
            try:
                import logging
                for handler in logging.root.handlers[:]:
                    if hasattr(handler, 'close'):
                        handler.close()
            except Exception:
                pass  # Ignore logging cleanup errors
                
            logger.debug("Resource cleanup completed")
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {e}")

    def save_checkpoint(self, path: str, **kwargs):
        """Save training checkpoint with architecture consistency validation."""
        from pathlib import Path
        
        # 构建完整checkpoint路径，避免路径重复拼接
        path_obj = Path(path)
        if not path_obj.is_absolute():
            checkpoint_dir_str = self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints')
            checkpoint_dir = Path(checkpoint_dir_str)
            
            # 确保checkpoint_dir存在
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 只取文件名进行拼接，避免路径重复
            full_path = checkpoint_dir / path_obj.name
        else:
            full_path = path_obj
        
        # Check for existing checkpoint architecture consistency
        if full_path.exists():
            try:
                with open(str(full_path), 'rb') as f:
                    existing_ckpt = torch.load(f, map_location='cpu')
                existing_config = existing_ckpt.get('config', {})
                
                # Compare critical architecture parameters
                current_model = self.config.get('model', {})
                existing_model = existing_config.get('model', {})
                
                critical_params = ['hidden_dim', 'latent_dim', 'proj_dim', 'heads', 'num_layers']
                for param in critical_params:
                    if (param in current_model and param in existing_model and 
                        current_model[param] != existing_model[param]):
                        raise ValueError(
                            f"Architecture mismatch detected! Cannot overwrite checkpoint with "
                            f"different {param}: current={current_model[param]} vs "
                            f"existing={existing_model[param]}. Use a different checkpoint path."
                        )
                        
                # Check feature dimensions if available
                current_features = self.config.get('features', {})
                existing_features = existing_config.get('features', {})
                if (current_features.get('hyperedge_dim') and existing_features.get('hyperedge_dim') and
                    current_features['hyperedge_dim'] != existing_features['hyperedge_dim']):
                    raise ValueError(
                        f"Feature dimension mismatch! hyperedge_dim: "
                        f"current={current_features['hyperedge_dim']} vs "
                        f"existing={existing_features['hyperedge_dim']}"
                    )
                    
            except (FileNotFoundError, KeyError, RuntimeError):
                # If we can't load/validate existing checkpoint, proceed with warning
                logger.warning(f"Could not validate existing checkpoint at {full_path}, proceeding with save")
        
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_step': self.best_step,
            'config': self.config  # Always bind current config with state_dict
        }
        checkpoint.update(kwargs)

        torch.save(checkpoint, str(full_path))
        logger.info(f"Checkpoint saved to {full_path} with config binding")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = torch.load(f, map_location=self.device)

        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)
        self.best_step = checkpoint.get('best_step', self.best_step)

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from step {self.global_step}")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best metrics from training history."""  
        if not self.history['val_loss']:
            return {}
            
        best_step_idx = np.argmin(self.history['val_loss'])
        metrics = {
            'val_loss': self.history['val_loss'][best_step_idx],
            'train_loss': self.history['train_loss'][best_step_idx],
            'val_recon_loss': self.history.get('val_recon_loss', [0])[best_step_idx] if self.history.get('val_recon_loss') else 0,
            'val_edge_loss': self.history.get('val_edge_loss', [0])[best_step_idx] if self.history.get('val_edge_loss') else 0,
            'val_contrastive_loss': self.history.get('val_contrastive_loss', [0])[best_step_idx] if self.history.get('val_contrastive_loss') else 0,
        }
        
        # 添加TCC特有指标
        if 'tcc_contributions' in self.history and len(self.history['tcc_contributions']) > best_step_idx:
            tcc_contrib = self.history['tcc_contributions'][best_step_idx]
            tcc_weights = self.history['tcc_weights'][best_step_idx] if 'tcc_weights' in self.history else {}
            violation_rate = self.history['tcc_violation_rates'][best_step_idx] if 'tcc_violation_rates' in self.history else 0
            
            # 展开TCC贡献占比
            for comp, contrib in tcc_contrib.items():
                metrics[f'tcc_contrib_{comp}'] = contrib
            
            # 展开TCC权重
            for comp, weight in tcc_weights.items():
                metrics[f'tcc_weight_{comp}'] = weight
                
            metrics['tcc_violation_rate'] = violation_rate
            
        return metrics
