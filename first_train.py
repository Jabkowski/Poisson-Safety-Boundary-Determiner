import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchinfo import summary

from models.unet import UNetBilinear, UNetTranspose
from py_utils.utils import GridDataset, calc_grad_loss, load_h5_to_torch, plot_validation_results


EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
DATA_PATH = os.getenv("POISSON_DATA_PATH", "data")
MODEL_NAME = os.getenv("POISSON_MODEL_NAME", "unettranspose")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "poisson-safety-boundary")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
LOAD_EXISTING_WEIGHTS = False
TRAIN_FURTHER = False
PRETRAINED_WEIGHTS_PATH = "weights/grads_loss_v3_ext_data_with_size_unettranspose_model_e15.pth"


def build_model(model_name: str) -> nn.Module:
    if model_name == "unetbilinear":
        return UNetBilinear(in_channels=1, out_channels=1)
    if model_name == "unettranspose":
        return UNetTranspose(in_channels=1, out_channels=1)
    raise ValueError(f"Unsupported model: {model_name}")


class PoissonDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, random_state: int = 42):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.random_state = random_state
        self.train_dataset = None
        self.val_dataset = None
        self.h_mean = None
        self.h_std = None

    def setup(self, stage=None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        grids, h = load_h5_to_torch(self.data_path)
        grids_train, grids_val, h_train, h_val = train_test_split(
            grids,
            h,
            test_size=TRAIN_TEST_SPLIT,
            random_state=self.random_state,
        )

        self.h_mean = h_train.mean()
        self.h_std = h_train.std() + 1e-8

        h_train = (h_train - self.h_mean) / self.h_std
        h_val = (h_val - self.h_mean) / self.h_std

        self.train_dataset = GridDataset(grids_train, h_train)
        self.val_dataset = GridDataset(grids_val, h_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


class PoissonLightningModule(LightningModule):
    def __init__(self, model_name: str, learning_rate: float):
        super().__init__()
        self.model = build_model(model_name)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.example_input_array = torch.randn(1, 1, 512, 512)
        self._latest_val_plot = None

        self.save_hyperparameters(
            {
                "model_name": model_name,
                "learning_rate": learning_rate,
            }
        )

    def forward(self, grid):
        return self.model(grid)

    def _shared_step(self, batch, stage: str):
        grid, h_true = batch
        h_pred = self(grid)
        mse, loss_grad, loss_bc = calc_grad_loss(self.criterion, h_pred, h_true)
        loss = mse + loss_grad + 0.1 * loss_bc

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=grid.size(0))
        self.log(f"{stage}_mse", mse, prog_bar=(stage == "val"), on_step=False, on_epoch=True, batch_size=grid.size(0))
        self.log(f"{stage}_grad_loss", loss_grad, on_step=False, on_epoch=True, batch_size=grid.size(0))
        self.log(f"{stage}_boundary_loss", loss_bc, on_step=False, on_epoch=True, batch_size=grid.size(0))

        return loss, h_pred, h_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, h_pred, h_true = self._shared_step(batch, stage="val")
        if batch_idx == 0 and not self.trainer.sanity_checking:
            self._latest_val_plot = (
                h_pred.detach().cpu(),
                h_true.detach().cpu(),
                self.current_epoch + 1,
            )
        return loss

    def on_validation_epoch_end(self):
        if self._latest_val_plot is None:
            return

        os.makedirs("mlflow_results", exist_ok=True)
        h_pred, h_true, epoch_label = self._latest_val_plot
        plot_validation_results(h_pred, h_true, epoch_label=epoch_label, save_plot=True, i=0)
        self._latest_val_plot = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def log_artifacts(logger: MLFlowLogger, artifact_paths):
    if logger.run_id is None:
        return

    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            continue
        if artifact_path.is_dir():
            logger.experiment.log_artifacts(logger.run_id, str(artifact_path), artifact_path.name)
        else:
            logger.experiment.log_artifact(logger.run_id, str(artifact_path))


def main():
    seed_everything(RANDOM_STATE, workers=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")

    model = build_model(MODEL_NAME)
    print(summary(model, input_size=(1, 1, 512, 512)))

    data_module = PoissonDataModule(DATA_PATH, BATCH_SIZE, RANDOM_STATE)
    data_module.setup()

    weights_path = Path(f"weights/grads_loss_v3_ext_data_with_size_{MODEL_NAME}_model_e{EPOCHS}.pth")
    checkpoint_dir = Path("weights/lightning")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lightning_module = PoissonLightningModule(MODEL_NAME, LEARNING_RATE)
    validate_only = LOAD_EXISTING_WEIGHTS and weights_path.is_file() and not TRAIN_FURTHER

    if validate_only:
        state_dict = torch.load(weights_path, map_location="cpu")
        lightning_module.model.load_state_dict(state_dict)
    elif TRAIN_FURTHER and Path(PRETRAINED_WEIGHTS_PATH).is_file():
        state_dict = torch.load(PRETRAINED_WEIGHTS_PATH, map_location="cpu")
        lightning_module.model.load_state_dict(state_dict)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
        log_model=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{MODEL_NAME}" + "-{epoch:02d}-{val_mse:.4f}",
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    mlflow_logger.log_hyperparams(
        {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "data_path": DATA_PATH,
            "train_test_split": TRAIN_TEST_SPLIT,
            "random_state": RANDOM_STATE,
            "tracking_uri": MLFLOW_TRACKING_URI,
        }
    )

    if validate_only:
        trainer.validate(lightning_module, datamodule=data_module)
    else:
        trainer.fit(lightning_module, datamodule=data_module)

        torch.save(lightning_module.model.state_dict(), weights_path)
        print(f"Saved weights to {weights_path}")

    artifact_paths = [weights_path, checkpoint_dir, Path("results")]
    log_artifacts(mlflow_logger, artifact_paths)

    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
