import torch
from torchvision import transforms

OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 1e-4
EPOCHS = 40
NUM_WORKERS = 4

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomAffine(degrees=10),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        # Resizes the image to the expected input size
        transforms.Resize((224, 224)),
        # Converts the image to a tensor
        transforms.ToTensor(),
        # Normalizes the image using ImageNet's mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ce_weights = torch.tensor(
    [0.01638, 0.02256, 0.04745, 0.04777, 0.06866, 0.25797, 0.87179, 1.0]
)
ce_weights = ce_weights.to("mps")

LOSS_FUNCTION = torch.nn.CrossEntropyLoss(ce_weights)

PATH_TO_FERP_DATASET = "/Users/alessandro/datasets/fer2013"
