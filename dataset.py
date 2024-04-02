import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    def __init__(self, root_dir: str = None, stage: str = None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert stage in [
            "train",
            "val",
            "test",
        ], "You can set stage to 'train', 'val' and 'test'"

        self.root_dir = root_dir
        if stage == "train":
            self.root_dir = os.path.join(self.root_dir, "FER2013Train")
        elif stage == "val":
            self.root_dir = os.path.join(self.root_dir, "FER2013Valid")
        else:
            self.root_dir = os.path.join(self.root_dir, "FER2013Test")

        self.frame = pd.read_csv(os.path.join(self.root_dir, "label.csv"))
        self.transform = transform

        # as https://github.com/borarak/emotion-recognition-vgg13/blob/master/trainer.py#L43
        # discarding the last to labels "UNKNOWN" and "NF" at idx 11 and 12
        self.frame["hard_label"] = self.frame.iloc[:, 2:10].apply(
            lambda x: np.argmax(x.values), axis=1
        )

    def __len__(self):
        return len(self.frame)

    def get_weights(self):
        class_counts = self.frame["hard_label"].value_counts()
        # class_counts = class_counts[:-1]
        total_samples = len(self.frame["hard_label"])
        num_classes = len(class_counts)

        # Calculate weight for each class
        weights = total_samples / (class_counts * num_classes)
        max_weight = weights.max()
        weights = weights / max_weight
        weights = np.round(weights, 5)
        weights = weights.to_list()

        return weights

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])

        # soft labels
        # soft_labels = self.frame.iloc[idx, 2:]
        # label = label / torch.sum(label)

        image = Image.open(img_name).convert("RGB")
        hard_label = self.frame.iloc[idx, -1]
        hard_label = torch.tensor(hard_label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, hard_label


# Example usage
if __name__ == "__main__":
    # Assuming you have PIL Images, if you have RGB images
    # you might need to adjust the transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to the input size of your model
            transforms.ToTensor(),
        ]
    )

    train_dataset = FER2013Dataset(
        root_dir="/Users/alessandro/datasets/fer2013",
        stage="train",
        transform=transform,
    )

if __name__ == "__main__":

    ce_weights = FER2013Dataset(
        root_dir="/Users/alessandro/datasets/fer2013",
        stage="train",
        transform=None,
    ).get_weights()

    print(ce_weights)
