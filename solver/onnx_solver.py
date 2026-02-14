import onnxruntime as ort
import numpy as np
import os
from solver.utils import ImgUtil


class ONNXSolver:
    def __init__(self, model_path="model.onnx", vocab_type="alphanumeric"):
        self.util = ImgUtil()
        self.model_path = model_path

        if vocab_type == "math":
            # 0-8, +, -, =, ? (No 9)
            self.chars = sorted(list("012345678+-=?"))
        else:  # Default Alphanumeric
            self.chars = sorted(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))

        self.classes = ["-"] + self.chars
        self.idx_to_char = {i: c for i, c in enumerate(self.classes)}

        if os.path.exists(model_path):
            self.ort_session = ort.InferenceSession(model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        else:
            print("Warning: ONNX model not found.")

    def solve(self, img_path):
        if not hasattr(self, "ort_session"):
            return "ERROR"

        # Preprocess full image (100x32)
        img = self.util.preprocess(img_path)

        # Prepare input
        img_np = np.array(img, dtype=np.float32)
        img_np /= 255.0
        # (1, 1, 32, 100)
        img_np = img_np.reshape(1, 1, 32, 100)

        # Inference
        outputs = self.ort_session.run(None, {self.input_name: img_np})
        # outputs[0] is (SequenceLength, Batch, NumClasses) -> (25, 1, 33)

        preds = outputs[0]
        # Argmax over classes
        preds_idx = np.argmax(preds, axis=2)  # (25, 1)
        seq = preds_idx[:, 0]  # (25,)

        # CTC Decode
        res = []
        prev = 0  # Blank
        for idx in seq:
            if idx != prev and idx != 0:
                res.append(self.idx_to_char[idx])
            prev = idx

        return "".join(res)
