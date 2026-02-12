import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from solver.light_solver import CaptchaCracker


# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Input: 1x32x32 (Grayscale)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Pool again: 16x16 -> 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLSolver:
    def __init__(self, model_path="model.pth"):
        self.cracker = CaptchaCracker()  # Use for preprocessing/segmentation
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.classes = sorted(
            list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ")
        )  # Standard charset excluding 01IO
        self.char_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.model = SimpleCNN(len(self.classes))

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            print("Warning: Model not found. Please train first.")

    def solve(self, img_path):
        # Preprocess & Segment
        img = self.cracker.preprocess(img_path)
        chars = self.cracker.segment(img)

        result = ""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts 0-255 to 0.0-1.0
            ]
        )

        for char_img in chars:
            # Prepare tensor
            tensor = transform(char_img).unsqueeze(0).to(self.device)  # Add batch dim

            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs.data, 1)
                idx = predicted.item()
                result += self.idx_to_char.get(idx, "?")

        return result
