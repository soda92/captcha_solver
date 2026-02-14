import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from solver.ml_solver import MLSolver
from solver.utils import ImgUtil
import os

# Training Configuration
OVERSAMPLE = 5
EPOCHS = 100
MODEL_OUT = "model_math.pth"
DATA_DIR = "num_captchas"


class MathCaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.util = ImgUtil()

        # Get math char map
        self.solver = MLSolver(model_path="", vocab_type="math")
        self.char_to_idx = self.solver.char_to_idx

        files = list(self.root_dir.glob("*.jpeg")) + list(self.root_dir.glob("*.png"))
        print(f"Loading {len(files)} images from {root_dir}...")

        for f in files:
            # Handle suffixes: 8-5_1.jpeg -> 8-5
            label_str = f.stem.split("_")[0]

            # The filenames might be like '8-5', '3+4'.
            # We treat the filename as the ground truth.

            # Append =? to label because images contain it
            label_str += "=?"

            try:
                # Preprocess full image
                img = self.util.preprocess(f)

                # Convert label to indices
                label_indices = []
                valid = True
                for c in label_str:
                    if c in self.char_to_idx:
                        label_indices.append(self.char_to_idx[c])
                    else:
                        print(
                            f"Warning: Character '{c}' in {f.name} not in math vocab."
                        )
                        valid = False
                        break

                if valid and len(label_indices) > 0:
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
    # Strong augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.ToTensor(),
        ]
    )

    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    dataset = MathCaptchaDataset(DATA_DIR, transform=train_transform)
    if len(dataset) == 0:
        print("No data.")
        return

    print(f"Oversampling dataset {OVERSAMPLE} times...")
    dataset = torch.utils.data.ConcatDataset([dataset] * OVERSAMPLE)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    solver = MLSolver(model_path="", vocab_type="math")
    model = solver.model
    device = torch.device("cpu")
    model.to(device)
    model.train()

    print(f"Training Math CRNN for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # targets is 1D tensor for CTCLoss? No, we passed (B, L) tensors in list?
            # DataLoader collates them. Since labels have VARIABLE length (e.g. "8-5" is 3, "10+5" is 4),
            # default collate might fail if not padded.
            # But math captchas here seem to be single digit +/- single digit -> length 3 usually.
            # If lengths vary, default collate fails.
            # We should use a custom collate_fn to pad targets.

            # Check dataset:
            # If filenames are all "X+Y", length is 3.
            # If "10+2", length is 4.
            # Let's assume variable length and fix collate.

            pass  # See fix below

        # Re-implement loop with proper collation
        pass


# Fix Collate
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)

    target_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long)
    targets = torch.cat(labels)  # Flattened 1D tensor for CTCLoss

    return images, targets, target_lengths


def train_fixed():
    # Strong augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.ToTensor(),
        ]
    )

    dataset = MathCaptchaDataset(DATA_DIR, transform=train_transform)
    if len(dataset) == 0:
        print("No data.")
        return

    dataset = torch.utils.data.ConcatDataset([dataset] * OVERSAMPLE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    solver = MLSolver(model_path="", vocab_type="math")
    model = solver.model
    device = torch.device("cpu")
    model.to(device)
    model.train()

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training Math CRNN for {EPOCHS} epochs...")

    try:
        for epoch in range(EPOCHS):
            running_loss = 0.0

            for images, targets, target_lengths in dataloader:
                images = images.to(device)
                targets = targets.to(device)

                preds = model(images)

                batch_size = images.size(0)
                input_lengths = torch.full((batch_size,), preds.size(0), dtype=torch.long)

                loss = criterion(preds, targets, input_lengths, target_lengths)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1} - Loss: {running_loss / len(dataloader):.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    train_fixed()
