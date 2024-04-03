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
    BATCH_SIZE,
    PATH_TO_FERP_DATASET,
)
from data_module import FER_DataModule
from models.resnet50 import ResNet50
from model_module import FERModel
from plot import PlotValidationImagesCallback


def main():

    # Lightning Model Module
    model = FERModel(
        model=ResNet50(num_classes=8),
        learning_rate=LEARNING_RATE,
        loss_fn=LOSS_FUNCTION,
        optimizer=OPTIMIZER,
    )

    # Lightning Data Module
    data_module = FER_DataModule(
        data_dir=PATH_TO_FERP_DATASET,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transform=TRAIN_TRANSFORM,
        val_transform=TRAIN_TRANSFORM,
    )

    # Callback to print 4 images after each epoch to understand visually inspect training
    plot_samples_callback = PlotValidationImagesCallback(
        path="plot_val",
        dataset_path="/Users/alessandro/datasets/fer2013",
        val_transform=VAL_TRANSFORM,
    )

    # Callback to store checkpoints after each epoch
    checkpoint_callback = ModelCheckpoint(
        # checkpoint directory
        dirpath="checkpoints",
        filename="{epoch}-{step}",
        # Save all checkpoints
        save_top_k=-1,
        # Save checkpoints at every epoch
        every_n_epochs=1,
    )

    # Initialize a trainer, accellerator to "auto" means it use all the available devices.
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, plot_samples_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
