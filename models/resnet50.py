# used to import ssl certificate
import ssl

import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights


from .utils import count_parameters, logger

ssl._create_default_https_context = ssl._create_stdlib_context


class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes=10,
        freeze=True,
        weights=ResNet50_Weights.DEFAULT,
    ) -> None:
        super().__init__()

        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50(weights=weights)

        # Optionally freeze the pretrained layers
        if freeze:
            for param in resnet50.parameters():
                param.requires_grad = False

        # Remove the last fully connected layer and the average pooling layer
        features = list(resnet50.children())[:-1]
        features = nn.Sequential(*features)

        # The fully connected layer
        fcl = nn.Linear(2048, num_classes)

        self.resnet50 = nn.Sequential(
            features,
            nn.Flatten(start_dim=1),
            fcl,
            # nn.Softmax(dim=1),
        )

        logger.info(f"Model has {count_parameters(self.resnet50)} parameters.")

    def forward(self, x):
        return self.resnet50(x)


if __name__ == "__main__":
    import torch

    x = torch.rand(4, 3, 224, 224)
    model = ResNet50(num_classes=10, freeze=True)
    out = model(x)
    assert list(out.shape) == [4, 10]
