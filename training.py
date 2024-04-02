import lightning as pl
import torch


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FER2013Dataset
from models.resnet50 import ResNet50
from models.vgg13 import VGG13

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from plot import PlotValidationImagesCallback, plotConfusionMatrix


class FERModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        model: torch.nn.Module = None,
        loss_fn: torch.nn = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

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
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        # T_0 is the number of iterations for the first restart, T_mult is the multiplier for each subsequent cycle (optional)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or "step" for step-wise updates
                "frequency": 1,
            },
        }


class FER_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="./data",
        batch_size=32,
        train_transform=None,
        val_transform=None,
        num_workers=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = FER2013Dataset(
                root_dir="/Users/alessandro/datasets/fer2013",
                stage="train",
                transform=self.train_transform,
            )
            self.val_dataset = FER2013Dataset(
                root_dir="/Users/alessandro/datasets/fer2013",
                stage="val",
                transform=self.val_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = FER2013Dataset(
                root_dir="/Users/alessandro/datasets/fer2013",
                stage="test",
                transform=self.val_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def main():

    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            # Resizes the image to the expected input size
            transforms.Resize((224, 224)),
            # Converts the image to a tensor
            transforms.ToTensor(),
            # Normalizes the image using ImageNet's mean and standard deviation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ce_weights = torch.tensor(
        [0.01638, 0.02256, 0.04745, 0.04777, 0.06866, 0.25797, 0.87179, 1.0]
    )
    ce_weights = ce_weights.to("mps")
    model = FERModel(
        model=ResNet50(num_classes=8),
        learning_rate=1e-4,
        loss_fn=torch.nn.CrossEntropyLoss(ce_weights),
    )

    # Initialize the data module
    data_module = FER_DataModule(
        data_dir="/Users/alessandro/datasets/fer2013",
        batch_size=32,
        num_workers=4,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    plot_samples_callback = PlotValidationImagesCallback(
        path="plot_val",
        dataset_path="/Users/alessandro/datasets/fer2013",
        val_transform=val_transform,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # Specify your checkpoint directory
        filename="{epoch}-{step}",  # Include epoch and global step in the filename
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,  # Save checkpoints at every epoch
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=40,
        callbacks=[checkpoint_callback, plot_samples_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )

    # Train the model
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
