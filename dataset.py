import os

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

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handling slicing
            raise NotImplementedError("Slicing is not implemented")

        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = torch.tensor(self.frame.iloc[idx, 2:], dtype=torch.float32)
        label = label / torch.sum(label)
        if self.transform:
            image = self.transform(image)

        return image, label


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
