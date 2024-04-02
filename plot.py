import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from dataset import FER2013Dataset


class PlotValidationImagesCallback(Callback):
    def __init__(
        self,
        path="/mnt/data/validation_images",
        dataset_path="",
        val_transform=None,
        labels=[
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
            "unknown",
            "NF",
        ],
    ):
        super().__init__()
        self.base_path = path
        self.val_loader = DataLoader(
            FER2013Dataset(
                root_dir=dataset_path,
                stage="val",
                transform=val_transform,
            ),
            num_workers=0,
            batch_size=8,
            shuffle=False,
        )
        self.labels = labels

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        epoch_path = os.path.join(self.base_path, f"epoch_{epoch+1}")
        os.makedirs(epoch_path, exist_ok=True)

        device = pl_module.device
        val_batch = next(iter(self.val_loader))

        images, targets = val_batch
        images, targets = images.to(device), targets.to(device)
        outputs = pl_module(images)

        # Select four images for plotting
        images, preds, targets = images[:4], outputs[:4], targets[:4]
        _, pred_ids = torch.topk(input=preds, k=2, dim=1)

        # Assuming we want to plot and save the first 4 images individually
        for i in range(images.shape[0]):
            self.plot_and_save_image(
                images[i], pred_ids[i], targets[i], pl_module, epoch_path, i
            )

    def plot_and_save_image(
        self, image, pred, target, pl_module, epoch_path, image_index
    ):
        _, ax = plt.subplots()
        img = image.permute(1, 2, 0)
        img = img * torch.tensor(
            [0.229, 0.224, 0.225], device=pl_module.device
        ) + torch.tensor([0.485, 0.456, 0.406], device=pl_module.device)
        img = img.clamp(0, 1).cpu().numpy()

        ax.imshow(img)
        pred0, pred1 = [self.labels[idx] for idx in pred.tolist()]
        actual0 = self.labels[target.item()]
        title = f"Top 2 predicted classes: 1: {pred0}, 2: {pred1}\n Label: {actual0}"
        ax.set_title(title, color="red")
        ax.axis("off")

        # Save the image to a file
        file_path = os.path.join(epoch_path, f"image_{image_index}.png")
        plt.savefig(file_path)
        plt.close()


def plotConfusionMatrix(
    conf_matrix,
    class_names=[
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
        "unknown",
        "NF",
    ],
):
    """
    Creates a confusion matrix plot and saves it as 'confusion_matrix.png' in the current directory.

    Parameters:
    - conf_matrix (numpy.ndarray): The confusion matrix to plot.
    - class_names (list of str): Optional. Names of the classes for axis labels.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", pad=20)
    fig.colorbar(cax)

    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticklabels([""] * conf_matrix.shape[1])
        ax.set_yticklabels([""] * conf_matrix.shape[0])

    plt.xlabel("Predicted")
    plt.ylabel("True Label")

    # Loop over data dimensions and create text annotations.
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig("confusion_matrix.png")
    plt.close(fig)
