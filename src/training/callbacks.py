import torch
import os

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ModelCheckpoint:
    def __init__(self, dirpath, monitor="val_loss", mode="min"):
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else -float("inf")

    def save_checkpoint(self, model, current_value, step):
        filename = f"checkpoint_step_{step}.pth"
        filepath = os.path.join(self.dirpath, filename)
        torch.save(model.state_dict(), filepath)
        if (self.mode == "min" and current_value < self.best_value) or (
            self.mode == "max" and current_value > self.best_value
        ):
            self.best_value = current_value
            best_path = os.path.join(self.dirpath, "best_model.pth")
            torch.save(model.state_dict(), best_path)
        return filepath


class LearningRateMonitor:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]
