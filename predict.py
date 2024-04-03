import torch
import argparse
from PIL import Image
from torchvision.transforms import Compose, Grayscale
from models.resnet50 import ResNet50
from torch.nn.functional import softmax

from config import VAL_TRANSFORM


parser = argparse.ArgumentParser(description="Predict the class of an image.")
parser.add_argument("path_to_weights", type=str, help="Path to the model weights file.")
parser.add_argument("path_to_img", type=str, help="Path to the image file.")
args = parser.parse_args()


# Load the model, the reason why the state dict needs to be modified is
# that when you save the model with lightning and your model set as instance
# attribute of the lightning module, each weight is saved as "model.model.*"
# which trigger an error.
model = ResNet50(num_classes=8)
checkpoint = torch.load(args.path_to_weights, map_location="cpu")
new_state_dict = {
    key.replace("model.model.", "model."): value
    for key, value in checkpoint["state_dict"].items()
}
model.load_state_dict(new_state_dict, strict=False)
# Set the model to evaluation mode
model.eval()

# Update VAL_TRANSFORM to include Grayscale
VAL_TRANSFORM = Compose(
    [
        Grayscale(num_output_channels=3),
        *VAL_TRANSFORM.transforms,
    ]
)

labels = [
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
]

# Process the image
img = Image.open(args.path_to_img)
img_transformed = VAL_TRANSFORM(img)
img_transformed = img_transformed.unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    out = model(img_transformed)
    probabilities = softmax(out, dim=1)

# Get the predicted label index
_, predicted_label = torch.max(probabilities, 1)
predicted_label = predicted_label.item()

print(f"Predicted Label: {labels[predicted_label]}")
