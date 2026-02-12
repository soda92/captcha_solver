import onnxruntime as ort
import numpy as np
import os
from solver.utils import ImgUtil


class ONNXSolver:
    def __init__(self, model_path="model.onnx"):
        self.util = ImgUtil()
        self.model_path = model_path
        self.classes = sorted(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))
        self.idx_to_char = {i: c for i, c in enumerate(self.classes)}

        if os.path.exists(model_path):
            self.ort_session = ort.InferenceSession(model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        else:
            print("Warning: ONNX model not found.")

    def solve(self, img_path):
        # Preprocess & Segment
        img = self.util.preprocess(img_path)
        chars = self.util.segment(img)

        result = ""

        for char_img in chars:
            # Prepare input
            # Convert PIL to Numpy
            img_np = np.array(char_img, dtype=np.float32)
            # Normalize 0-255 -> 0.0-1.0
            img_np /= 255.0
            # Reshape to (1, 1, 32, 32)
            img_np = img_np.reshape(1, 1, 32, 32)

            # Inference
            outputs = self.ort_session.run(None, {self.input_name: img_np})
            # outputs[0] is (Batch, NumClasses)
            predicted_idx = np.argmax(outputs[0], axis=1)[0]

            result += self.idx_to_char.get(predicted_idx, "?")

        return result
