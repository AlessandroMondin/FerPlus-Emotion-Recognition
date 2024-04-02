import lightning as pl
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from plot import plotConfusionMatrix


class FERModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        model: torch.nn.Module = None,
        loss_fn: torch.nn = None,
        optimizer: torch.optim = Adam,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def compute_accuracy(self, outputs, targets):
        _, preds = torch.max(outputs, dim=1)
        corrects = torch.sum(preds == targets).item()
        return corrects / len(targets)

    def on_train_epoch_start(self):
        # Reset accumulators at the start of each epoch
        self.train_loss_accum = 0.0
        self.train_accuracy_accum = 0.0
        self.train_steps = 0

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        accuracy = self.compute_accuracy(outputs, targets)

        # Update accumulators
        self.train_loss_accum += loss.item() * len(targets)
        self.train_accuracy_accum += accuracy * len(targets)
        self.train_steps += len(targets)

        # Calculate running averages
        running_loss = self.train_loss_accum / self.train_steps
        running_accuracy = self.train_accuracy_accum / self.train_steps

        # Log running averages
        self.log("train_loss", running_loss, on_step=True, prog_bar=True)
        self.log("train_accuracy", running_accuracy, on_step=True, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        # Reset accumulators at the start of each validation epoch
        self.val_loss_accum = 0.0
        self.val_accuracy_accum = 0.0
        self.val_steps = 0

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        accuracy = self.compute_accuracy(outputs, targets)

        # Update accumulators
        self.val_loss_accum += loss.item() * len(targets)
        self.val_accuracy_accum += accuracy * len(targets)
        self.val_steps += len(targets)

        # Calculate running averages
        running_loss = self.val_loss_accum / self.val_steps
        running_accuracy = self.val_accuracy_accum / self.val_steps

        # Log running averages
        self.log("val_loss", running_loss, on_step=True, prog_bar=True)
        self.log("val_accuracy", running_accuracy, on_step=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)

        # For accuracy, ensure targets are in the correct form if necessary
        _, preds = torch.max(outputs, dim=1)
        targets_argmax = targets.argmax(dim=1) if targets.dim() > 1 else targets
        acc = accuracy_score(targets_argmax.cpu(), preds.cpu())

        # Log the loss multiplied by the batch size
        self.log(
            "test_running_loss",
            loss.item() * len(targets),
            on_epoch=True,
            prog_bar=True,
        )

        # Log accuracy
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

        # Precision, recall, F1, as before
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_argmax.cpu(), preds.cpu(), average="weighted"
        )
        self.log("test_precision", precision, on_epoch=True, prog_bar=True)
        self.log("test_recall", recall, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)

        # For confusion matrix calculation later
        self.test_step_outputs.append({"preds": preds, "targets": targets_argmax})

    def on_test_epoch_end(self):
        # Aggregate predictions and targets from all test batches
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs], dim=0)

        # Calculate and log confusion matrix
        conf_matrix = confusion_matrix(all_targets.cpu(), all_preds.cpu())
        self.logger.experiment.add_figure(
            "Confusion Matrix", plotConfusionMatrix(conf_matrix), self.current_epoch
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        # T_0 is the number of iterations for the first restart, T_mult is the multiplier for each subsequent cycle (optional)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or "step" for step-wise updates
                "frequency": 1,
            },
        }
