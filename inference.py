import os
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from transformers import AutoImageProcessor, AutoModelForImageClassification


# =========================
# 1. Load MNIST test dataset
# =========================
dataset = MNIST(
    root=os.path.expanduser("~/.cache/torchvision"),
    train=False,
    download=True,
)


# =========================
# 2. Load pretrained ResNet-18
# =========================
model_name = "microsoft/resnet-18"

processor = AutoImageProcessor.from_pretrained(model_name)

# The original ResNet-18 is trained on ImageNet with 1000 classes.
# MNIST has only 10 classes, so we replace the final classifier
# with a randomly initialized 10-class classifier.
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=10,
    ignore_mismatched_sizes=True,
)


# =========================
# 3. Select device
# =========================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)
model.eval()

print("Using device:", device)


# =========================
# 4. Preprocess MNIST images
# =========================
def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        # MNIST images are grayscale, but ResNet expects RGB images
        image = image.convert("RGB")

        # MNIST images are 28x28.
        # Resize them to 224x224 to fit ResNet input size.
        image = image.resize((224, 224))

        images.append(image)
        labels.append(label)

    # Hugging Face processor converts images to tensors
    # and applies the normalization expected by ResNet.
    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    labels = torch.tensor(labels)

    return inputs, labels


# =========================
# 5. Create DataLoader
# =========================
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
)


# =========================
# 6. Run inference
# =========================
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in loader:
        # Move input tensors to GPU / MPS / CPU
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        labels = labels.to(device)

        # Forward pass
        outputs = model(**inputs)

        # outputs.logits shape:
        # [batch_size, 10]
        predictions = outputs.logits.argmax(dim=1)

        # Count correct predictions
        correct += (predictions == labels).sum().item()
        total += labels.size(0)


# =========================
# 7. Calculate accuracy
# =========================
accuracy = correct / total

print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy * 100:.2f}%")