import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EarlyStopCheckpoint:
    val_loss: float
    epoch: int
    model_state_dict: dict


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a patience.

    This implementation saves a *fold-specific* checkpoint dict:
      {'epoch': int, 'val_loss': float, 'model_state_dict': ...}
    """

    def __init__(self, patience: int = 10, delta: float = 0.0, path: str = 'checkpoint.pt', verbose: bool = True):
        self.patience = int(patience)
        self.delta = float(delta)
        self.path = str(path)
        self.verbose = bool(verbose)

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_epoch: Optional[int] = None

        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int):
        score = -float(val_loss)

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, epoch)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: torch.nn.Module, epoch: int):
        """Save model when validation loss decreases."""
        if self.verbose:
            if val_loss < self.val_loss_min:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving checkpoint...')
            else:
                print('Saving checkpoint...')

        ckpt = {
            'epoch': int(epoch),
            'val_loss': float(val_loss),
            'model_state_dict': model.state_dict(),
        }
        torch.save(ckpt, self.path)
        self.val_loss_min = float(val_loss)
        self.best_epoch = int(epoch)

    def load_best(self, model: torch.nn.Module, map_location=None) -> EarlyStopCheckpoint:
        """Load best checkpoint into the given model."""
        ckpt = torch.load(self.path, map_location=map_location)
        model.load_state_dict(ckpt['model_state_dict'])
        return EarlyStopCheckpoint(
            val_loss=float(ckpt.get('val_loss', float('nan'))),
            epoch=int(ckpt.get('epoch', -1)),
            model_state_dict=ckpt.get('model_state_dict', {}),
        )
