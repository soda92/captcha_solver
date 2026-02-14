import torch
import torch.nn as nn
import torch.nn.functional as F
from solver.utils import ImgUtil


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # Input: 1 x 32 x 100
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16 x 50

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8 x 25

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))  # 4 x 24 (Pool H only)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((4, 1))  # 1 x 23 (Pool H to 1)

        # Sequence modeling (optional RNN, here just Linear for simplicity/speed)
        # Input features: 256
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # x: (B, 256, 1, 23)
        # Permute to (W, B, C) -> (23, B, 256)
        x = x.squeeze(2)  # (B, 256, 23)
        x = x.permute(2, 0, 1)  # (23, B, 256)

        x = self.fc(x)  # (23, B, NumClasses)
        return F.log_softmax(x, dim=2)


class MLSolver:
    def __init__(self, model_path="model.pth", vocab_type="alphanumeric"):
        self.util = ImgUtil()
        self.model_path = model_path
        self.device = torch.device("cpu")

        # Define vocabulary
        if vocab_type == "math":
            # 0-8, +, -, =, ? (No 9)
            self.chars = sorted(list("012345678+-=?"))
        else:  # Default Alphanumeric
            self.chars = sorted(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))

        self.classes = ["-"] + self.chars  # Blank at 0
        self.char_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        self.model = CRNN(len(self.classes))

        if model_path:
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                print("Model not loaded (training mode?)")

    def decode(self, preds):
        # preds: (W, B, C) -> argmax -> (W, B) -> list
        preds = preds.argmax(dim=2).detach().cpu().numpy()
        # Assume batch size 1 for now
        seq = preds[:, 0]

        # CTC Decode (Collapse repeats, remove blanks)
        res = []
        prev = 0
        for idx in seq:
            if idx != prev and idx != 0:
                res.append(self.idx_to_char[idx])
            prev = idx
        return "".join(res)

    def solve(self, img_source):
        """
        Solve captcha from file path or file-like object.
        """
        img = self.util.preprocess(img_source)
        # Convert to tensor: (1, 1, 32, 100)
        import torchvision.transforms as transforms

        tf = transforms.ToTensor()
        img_tensor = tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(img_tensor)

        return self.decode(preds)

    def solve_bytes(self, image_bytes):
        """
        Solve captcha from bytes.
        """
        import io

        return self.solve(io.BytesIO(image_bytes))
