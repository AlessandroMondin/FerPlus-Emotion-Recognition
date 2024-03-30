import lightning as pl
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FER2013Dataset
from models.resnet50 import ResNet50

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
        loss: torch.nn = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss

        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        # Convert soft labels to predicted top1 index
        _, preds = torch.max(outputs, dim=1)
        loss = self.loss(outputs, targets)
        targets = targets.argmax(1)

        # Calculate metrics
        acc = accuracy_score(targets.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets.cpu(), preds.cpu(), average="weighted"
        )

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_precision", precision, on_epoch=True, prog_bar=True)
        self.log("test_recall", recall, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)

        # Store preds and targets to calculate confusion matrix later
        self.test_step_outputs.append({"preds": preds, "targets": targets})

    def on_test_epoch_end(self):
        # Aggregate predictions and targets from all batches
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs], dim=0)

        # Calculate and log confusion matrix
        conf_matrix = confusion_matrix(all_targets.cpu(), all_preds.cpu())
        plotConfusionMatrix(conf_matrix)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def main():

    train_transform = transforms.Compose(
        [
            # Converts B&W images to 3-channel images
            transforms.Grayscale(num_output_channels=3),
            # Resizes the image slightly larger than the final size
            transforms.Resize((256, 256)),
            # Randomly crops the image
            transforms.RandomResizedCrop(224),
            # Randomly flips the image horizontally
            transforms.RandomHorizontalFlip(),
            # Randomly rotates the image to a max of 10 degrees
            transforms.RandomRotation(10),
            # Random perspective transformation
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            # Random color jitter
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            # Converts the image to a tensor
            transforms.ToTensor(),
            # Normalizes with ImageNet's values
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            # Converts B&W images to 3-channel images
            transforms.Grayscale(num_output_channels=3),
            # Resizes the image to the expected input size
            transforms.Resize((224, 224)),
            # Converts the image to a tensor
            transforms.ToTensor(),
            # Normalizes the image using ImageNet's mean and standard deviation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Initialize the model
    model = FERModel(
        model=ResNet50(num_classes=10, freeze=True),
        learning_rate=1e-4,
        loss=torch.nn.CrossEntropyLoss(),
    )

    # Initialize the data module
    data_module = FER_DataModule(
        data_dir="/Users/alessandro/datasets/fer2013",
        batch_size=32,
        num_workers=6,
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
        max_epochs=100,
        callbacks=[checkpoint_callback, plot_samples_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )

    # Train the model
    # trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
