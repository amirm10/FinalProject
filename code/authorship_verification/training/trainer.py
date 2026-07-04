"""
Training Module
===============
Complete training loop for the Siamese network.

Implements: AdamW optimizer, warmup scheduling, gradient accumulation,
early stopping, checkpointing, gradient flow verification.

Test coverage: TR-01 (convergence), TR-02 (gradient flow), TR-03 (overfitting).
"""

import os
import time
import json
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from models.losses import ContrastiveLoss
from models.siamese import SiameseNetwork


class EarlyStopping:
    """Stop training when validation loss stops improving (TR-03)."""

    def __init__(self, patience: Optional[int] = None, min_delta: Optional[float] = None):
        self.patience = patience if patience is not None else cfg.training.early_stopping_patience
        self.min_delta = min_delta if min_delta is not None else cfg.training.early_stopping_min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """Full training cycle for the Siamese network."""

    def __init__(
        self,
        model: SiameseNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else cfg.device
        self.loss_fn = loss_fn if loss_fn is not None else ContrastiveLoss()

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            eps=cfg.training.adam_epsilon,
        )

        total_steps = len(train_loader) * cfg.training.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=cfg.training.learning_rate,
            total_steps=total_steps,
            pct_start=min(cfg.training.warmup_steps / max(total_steps, 1), 0.3),
        )

        self.early_stopping = EarlyStopping()
        self.history = {"train_loss": [], "val_loss": [], "learning_rates": [], "epoch_times": []}

    def train(self) -> Dict:
        """Full training loop. Returns training history."""
        print(f"Training on: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("-" * 60)

        # Single source of truth for "best": self.early_stopping.best_loss
        for epoch in range(cfg.training.num_epochs):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate()
            dt = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])
            self.history["epoch_times"].append(dt)

            print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")

            # Checkpoint "best" now matches EarlyStopping's own improvement rule
            es = self.early_stopping
            is_new_best = (
                es.best_loss is None
                or val_loss <= es.best_loss - es.min_delta
            )
            if is_new_best:
                self._save_checkpoint(epoch, val_loss, is_best=True)

            if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)

            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch+1} (best: {self.early_stopping.best_loss:.4f})")
                break

        print("\nTraining complete!")
        self._save_history()
        return self.history

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, n = 0.0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(batch)

            loss = self.loss_fn(outputs["embedding_a"], outputs["embedding_b"], outputs["labels"])
            loss = loss / cfg.training.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * cfg.training.gradient_accumulation_steps
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(batch)
            loss = self.loss_fn(outputs["embedding_a"], outputs["embedding_b"], outputs["labels"])
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def _save_checkpoint(self, epoch, val_loss, is_best):
        d = os.path.join(cfg.paths.root_dir, cfg.paths.checkpoint_dir)
        os.makedirs(d, exist_ok=True)
        name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, os.path.join(d, name))

    def _save_history(self):
        d = os.path.join(cfg.paths.root_dir, cfg.paths.results_dir)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def verify_gradient_flow(self) -> Dict[str, bool]:
        """TR-02: Verify gradients propagate through all layers."""
        self.model.train()
        batch = next(iter(self.train_loader))
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = self.loss_fn(outputs["embedding_a"], outputs["embedding_b"], outputs["labels"])
        loss.backward()

        results = {}
        for name, params in [
            ("bert_unfrozen", self.model.bert_encoder.bert.parameters()),
            ("cnn", [p for c in self.model.style_encoder.convs for p in c.parameters()]),
            ("bilstm", self.model.style_encoder.bilstm.parameters()),
            ("fc", self.model.style_encoder.fc.parameters()),
        ]:
            grads = [p.grad.abs().mean().item() for p in params if p.requires_grad and p.grad is not None]
            results[name] = len(grads) > 0 and max(grads) > 0

        self.optimizer.zero_grad()
        return results


def load_checkpoint(
    model: SiameseNetwork,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load a saved checkpoint."""
    device = device if device is not None else cfg.device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
