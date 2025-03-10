

# To install the dependencies, please run the following command
# pip install -r requirements.txt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from transformers import ViTModel
import numpy as np
import random


# SimCLR requires two augmented views per image.
class SimCLRTransform:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# CIFAR-10 dataset wrapper for SimCLR pre-training.
# Note: We ignore the labels during pre-training.
class CIFAR10SimCLR(Dataset):
    def __init__(self, root='./data', train=True, download=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform if transform is not None else SimCLRTransform()

    def __getitem__(self, index):
        x, _ = self.dataset[index]  # discard label
        xi, xj = self.transform(x)
        return xi, xj

    def __len__(self):
        return len(self.dataset)


# The SimCLR model comprises of an encoder (ViT from Hugging Face) and an MLP projection head.
class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLRModel, self).__init__()
        # Load a pre-trained ViT model from Hugging Face.
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        hidden_dim = self.encoder.config.hidden_size  # typically 768
        # Projection head: a two-layer MLP.
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def encode(self, x):
        # x should have shape (B, 3, 224, 224)
        outputs = self.encoder(x)
        # Use the [CLS] token representation.
        cls_output = outputs.last_hidden_state[:, 0]
        return cls_output

    def forward(self, x):
        features = self.encode(x)
        proj = self.projection_head(features)
        proj = nn.functional.normalize(proj, dim=1)
        return proj

    def forward_pair(self, x1, x2):
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        return z1, z2

# NT-Xent loss for contrastive learning.
def nt_xent_loss(z1, z2, temperature=0.5):
    device = z1.device
    batch_size = z1.size(0)
    N = batch_size
    # Concatenate both sets of projections.
    z = torch.cat([z1, z2], dim=0)  # shape: (2N, d)
    z = nn.functional.normalize(z, dim=1)
    # Compute cosine similarity.
    logits = torch.matmul(z, z.t()) / temperature  # (2N, 2N)
    # Set self-similarities to a very low value for numerical stability.
    mask = torch.eye(2 * N, device=device).bool()
    logits.masked_fill_(mask, float("-inf"))

    # For index i, the positive index is (i + N) % (2N).
    positive_indices = (torch.arange(2 * N, device=device) + N) % (2 * N)
    positive_logits = logits[torch.arange(2 * N, device=device), positive_indices]

    loss = - positive_logits + torch.logsumexp(logits, dim=1)
    return loss.mean()


# Pre-training loop for self-supervised learning using SimCLR.
def pretrain_simclr(epochs=5, batch_size=128, lr=3e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = SimCLRTransform(image_size=224)
    dataset = CIFAR10SimCLR(train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = SimCLRModel(projection_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (xi, xj) in enumerate(loader):
            xi = xi.to(device)
            xj = xj.to(device)

            optimizer.zero_grad()
            z1, z2 = model.forward_pair(xi, xj)
            loss = nt_xent_loss(z1, z2, temperature=0.5)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Pretrain Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(loader)
        print(f"Pretrain Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "simclr_pretrained.pth")
    print("Pre-training complete; model saved as 'simclr_pretrained.pth'")
    return model

# Fine-tuning: Train a linear classifier on top of the frozen encoder.
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder  # Pre-trained ViT encoder.
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, x):
        # For fine-tuning, extract features without gradient updates.
        with torch.no_grad():
            features = self.encoder(x).last_hidden_state[:, 0]
        logits = self.fc(features)
        return logits

def finetune_linear(epochs=5, batch_size=128, lr=3e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load the pre-trained encoder for fine-tuning.
    encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    state_dict = torch.load("simclr_pretrained.pth", map_location=device)
    encoder_state_dict = {}
    # Extract only the encoder weights (the pre-training model saved both encoder and projection head).
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state_dict[k[len("encoder."):]] = v
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
    print("Loaded pre-trained encoder weights. Missing keys:", missing_keys)

    classifier = LinearClassifier(encoder, num_classes=10).to(device)
    optimizer = optim.Adam(classifier.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Fine-tune Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Fine-tune Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    # Evaluate on the test set.
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Fine-tuning Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    # Set random seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Starting self-supervised pre-training with SimCLR...")
    pretrain_simclr(epochs=5, batch_size=128, lr=3e-4, device=device)

    print("\nStarting fine-tuning on CIFAR-10 classification...")
    finetune_linear(epochs=5, batch_size=128, lr=3e-4, device=device)