import lightning as pl

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from config import (
    TRAIN_TRANSFORM,
    VAL_TRANSFORM,
    OPTIMIZER,
    LEARNING_RATE,
    EPOCHS,
    NUM_WORKERS,
    LOSS_FUNCTION,
)
from data_module import FER_DataModule
from models.resnet50 import ResNet50
from model_module import FERModel
from plot import PlotValidationImagesCallback


def main():

    model = FERModel(
        model=ResNet50(num_classes=8),
        learning_rate=LEARNING_RATE,
        loss_fn=LOSS_FUNCTION,
        optimizer=OPTIMIZER,
    )

    # Initialize the data module
    data_module = FER_DataModule(
        data_dir="/Users/alessandro/datasets/fer2013",
        batch_size=32,
        num_workers=NUM_WORKERS,
        train_transform=TRAIN_TRANSFORM,
        val_transform=TRAIN_TRANSFORM,
    )

    plot_samples_callback = PlotValidationImagesCallback(
        path="plot_val",
        dataset_path="/Users/alessandro/datasets/fer2013",
        val_transform=VAL_TRANSFORM,
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
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, plot_samples_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )

    # Train the model
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
