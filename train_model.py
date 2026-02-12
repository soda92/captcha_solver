import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from solver.ml_solver import SimpleCNN
from solver.light_solver import CaptchaCracker


# 1. Prepare Dataset
class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # (image, label_idx)
        self.cracker = CaptchaCracker()

        # Define classes
        self.classes = sorted(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))
        self.char_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Load data
        files = list(self.root_dir.glob("*.jpeg")) + list(self.root_dir.glob("*.png"))
        print(f"Loading {len(files)} images...")

        for f in files:
            # Handle filenames like "ABCD_1.jpeg" -> label "ABCD"
            label = f.stem.split('_')[0].upper()
            if len(label) != 4: continue
            
            try:
                # Use cracker to get segments
                # Preprocess & Segment using the IMPROVED logic (borders, thickening)
                chars = self.cracker.segment(self.cracker.preprocess(f))

                if len(chars) != 4:
                    # Skip if segmentation failed
                    continue

                for i, char_img in enumerate(chars):
                    char_label = label[i]
                    if char_label in self.char_to_idx:
                        self.samples.append((char_img, self.char_to_idx[char_label]))
            except Exception as e:
                print(f"Error loading {f}: {e}")

        print(f"Created dataset with {len(self.samples)} characters.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# 2. Training Setup
def train():
    # Augmentation to improve robustness
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ]
    )

    # Dataset
    dataset = CaptchaDataset("raw_captchas", transform=train_transform)
    if len(dataset) == 0:
        print("Error: No data found.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    # Determine num classes dynamically if needed, but we stick to fixed set
    # Wait, dataset uses fixed char_to_idx. Just pass len.

    model = SimpleCNN(len(dataset.classes))
    device = torch.device("cpu")  # GPU not needed for this small task
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 120
    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(dataloader):.4f} - Acc: {100 * correct / total:.2f}%"
        )

    # Save
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    train()
