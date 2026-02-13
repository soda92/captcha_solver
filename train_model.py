import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils
import torch.utils.data
from torchvision import transforms
from pathlib import Path
from solver.ml_solver import MLSolver
from solver.utils import ImgUtil

# Training Configuration
OVERSAMPLE = 10
EPOCHS = 70


class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.util = ImgUtil()

        # Get char map from MLSolver helper
        self.solver = MLSolver(model_path="")
        self.char_to_idx = self.solver.char_to_idx

        files = list(self.root_dir.glob("*.jpeg")) + list(self.root_dir.glob("*.png"))
        print(f"Loading {len(files)} images from {root_dir}...")

        for f in files:
            # Handle suffixes: ABCD_1.jpeg -> ABCD
            label_str = f.stem.split("_")[0].upper()
            if len(label_str) != 4:
                continue

            try:
                # Preprocess full image (No segmentation)
                img = self.util.preprocess(f)

                # Convert label to indices
                label_indices = [
                    self.char_to_idx[c] for c in label_str if c in self.char_to_idx
                ]
                if len(label_indices) != 4:
                    continue

                self.samples.append(
                    (img, torch.tensor(label_indices, dtype=torch.long))
                )
            except Exception as e:
                print(f"Error {f}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def train():
    # Strong augmentation for rotation/shear as requested
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
            ),
            transforms.ToTensor(),
        ]
    )

    dataset = CaptchaDataset("raw_captchas", transform=train_transform)
    if len(dataset) == 0:
        print("No data.")
        return

    # Oversample small dataset to ensure convergence
    print(f"Oversampling dataset {OVERSAMPLE} times...")
    dataset = torch.utils.data.ConcatDataset([dataset] * OVERSAMPLE)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    solver = MLSolver(model_path="")
    model = solver.model
    device = torch.device("cpu")
    model.to(device)
    model.train()

    # CTC Loss
    # Blank label is 0.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training CRNN for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward -> (W, B, C)
            preds = model(images)

            # CTC args
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), preds.size(0), dtype=torch.long)
            # targets is (B, 4). CTCLoss expects targets as 1D concatenated tensor, OR (B, 4) if lengths are provided.
            target_lengths = torch.full((batch_size,), 4, dtype=torch.long)

            loss = criterion(preds, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved.")


if __name__ == "__main__":
    train()
