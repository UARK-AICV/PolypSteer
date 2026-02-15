import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import numpy as np
from torchmetrics import AUROC

class KvasirClassifierModule(pl.LightningModule):
    def __init__(self, model_name="convnext_large", num_classes=8, lr=3e-4, weight_decay=1e-4, warmup_epochs=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_auc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_auc = AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_auc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auc", self.train_auc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_auc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", self.val_auc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return (epoch + 1) / self.hparams.warmup_epochs
            # Avoid division by zero if max_epochs == warmup_epochs
            denom = max(1, self.trainer.max_epochs - self.hparams.warmup_epochs)
            progress = (epoch - self.hparams.warmup_epochs) / denom
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
