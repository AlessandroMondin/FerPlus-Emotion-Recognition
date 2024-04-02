import lightning as pl

from torch.utils.data import DataLoader
from config import PATH_TO_FERP_DATASET
from dataset import FER2013Dataset


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
                root_dir=PATH_TO_FERP_DATASET,
                stage="train",
                transform=self.train_transform,
            )
            self.val_dataset = FER2013Dataset(
                root_dir=PATH_TO_FERP_DATASET,
                stage="val",
                transform=self.val_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = FER2013Dataset(
                root_dir=PATH_TO_FERP_DATASET,
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
