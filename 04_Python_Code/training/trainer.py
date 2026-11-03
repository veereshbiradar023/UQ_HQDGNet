"""
Training pipeline with validation tracking, checkpointing, and cosine annealing schedule.
"""
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .losses import JointUQHQDGLoss

class UQHQDGTrainer:
    def __init__(self, model, train_loader, val_loader, config, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = JointUQHQDGLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.training.epochs, eta_min=1e-6)

        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_det_acc': [], 'val_mal_acc': []}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Dynamic GRL schedule alpha = 2 / (1 + exp(-10 * p)) - 1
        p = epoch / max(1, self.config.training.epochs)
        grl_alpha = 2.0 / (1.0 + 2.71828 ** (-10.0 * p)) - 1.0

        for batch in self.train_loader:
            for k in ['patch', 'det_label', 'bbox', 'malignancy_label', 'domain_id']:
                batch[k] = batch[k].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch['patch'], domain_alpha=grl_alpha)
            loss_dict = self.criterion(outputs, batch)
            
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss_dict['total_loss'].item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(1, n_batches)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        correct_det, total_det = 0, 0
        correct_mal, total_mal = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                for k in ['patch', 'det_label', 'bbox', 'malignancy_label', 'domain_id']:
                    batch[k] = batch[k].to(self.device)

                outputs = self.model(batch['patch'], domain_alpha=0.0)
                loss_dict = self.criterion(outputs, batch)
                total_loss += loss_dict['total_loss'].item()
                n_batches += 1

                det_preds = (torch.sigmoid(outputs['det_logit']) >= 0.5).long()
                correct_det += (det_preds == batch['det_label'].long()).sum().item()
                total_det += batch['det_label'].size(0)

                mal_preds = torch.argmax(outputs['malignancy_logit'], dim=-1)
                correct_mal += (mal_preds == batch['malignancy_label']).sum().item()
                total_mal += batch['malignancy_label'].size(0)

        val_loss = total_loss / max(1, n_batches)
        det_acc = correct_det / max(1, total_det)
        mal_acc = correct_mal / max(1, total_mal)
        return val_loss, det_acc, mal_acc

    def fit(self, save_path=None):
        print(f"Starting training for {self.config.training.epochs} epochs on device: {self.device}...")
        for epoch in range(1, self.config.training.epochs + 1):
            tr_loss = self.train_epoch(epoch)
            va_loss, va_det_acc, va_mal_acc = self.validate()

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(va_loss)
            self.history['val_det_acc'].append(va_det_acc)
            self.history['val_mal_acc'].append(va_mal_acc)

            print(f"Epoch {epoch:02d}/{self.config.training.epochs:02d} | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
                  f"Det Acc: {va_det_acc*100:.1f}% | Mal Acc: {va_mal_acc*100:.1f}%")

            if va_loss < self.best_val_loss and save_path is not None:
                self.best_val_loss = va_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  --> Saved new best checkpoint to {save_path}")

        return self.history
