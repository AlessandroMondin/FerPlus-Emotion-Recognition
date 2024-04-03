import ssl

import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

ssl._create_default_https_context = ssl._create_stdlib_context


class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes=10,
        train_last_n_layers=0,
        weights=ResNet50_Weights.DEFAULT,
    ) -> None:
        super().__init__()
        assert train_last_n_layers >= 0, "Must be 0 or higher"
        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50(weights=weights)

        # Freeze all layers in the network
        for param in resnet50.parameters():
            param.requires_grad = False

        # Unfreeze layers, e.g., unfreeze the last convolution block
        for param in resnet50.layer4[-3:].parameters():
            param.requires_grad = True

        # The fully connected layer
        resnet50.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(2048, num_classes),
        )

        self.model = resnet50

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch

    x = torch.rand(4, 3, 224, 224)
    model = ResNet50(num_classes=10)
    out = model(x)
    assert list(out.shape) == [4, 10]
