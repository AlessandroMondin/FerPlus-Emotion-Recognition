# used to import ssl certificate
import ssl

import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG13_Weights


from .utils import count_parameters, logger

ssl._create_default_https_context = ssl._create_stdlib_context


class VGG13(nn.Module):
    def __init__(
        self,
        num_classes=10,
        weights=VGG13_Weights.DEFAULT,
    ) -> None:
        super().__init__()

        model = models.vgg13(weights=weights)

        model.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, num_classes),
        )

        for param in model.features.parameters():
            param.requires_grad = False

        logger.info(f"Model has {count_parameters(model)} parameters.")
        self.model = model

    def forward(self, x):
        return self.model(x)
