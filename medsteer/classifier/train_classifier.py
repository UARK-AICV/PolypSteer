import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.plugins.environments import (
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from medsteer.classifier.dataset import KvasirDataModule
from medsteer.classifier.model import KvasirClassifierModule

# SLURM environment patches to prevent automatic multi-node detection in some environments
SLURMEnvironment.detect = lambda: False
TorchElasticEnvironment.validate_settings = lambda self, num_devices, num_nodes: True

def main():
    parser = argparse.ArgumentParser(description="Train Kvasir Classifier")
    parser.add_argument("--csv_path", type=str, default="dataset/kvasir/train/raw.csv")
    parser.add_argument("--data_root", type=str, default="dataset/kvasir/train")
    parser.add_argument("--output_dir", type=str, default="./kvasir_classifier_output")
    parser.add_argument("--model", type=str, default="convnext_large")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datamodule = KvasirDataModule(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    if args.eval_only and args.resume:
        model = KvasirClassifierModule.load_from_checkpoint(args.resume)
    else:
        model = KvasirClassifierModule(
            model_name=args.model,
            num_classes=8,
            lr=args.lr,
            warmup_epochs=args.warmup_epochs
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=True
    )

    early_stop_callback = EarlyStopping(monitor="val_auc", patience=5, mode="max", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    loggers = [
        CSVLogger(args.output_dir, name="csv_logs"),
        TensorBoardLogger(args.output_dir, name="tb_logs")
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.devices > 0 else "cpu",
        devices=args.devices if args.devices > 0 else "auto",
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=loggers,
        precision="16-mixed" if args.devices > 0 else 32,
        enable_progress_bar=True
    )

    if args.eval_only:
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
        print(f"Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
