import numpy as np
from PIL import Image
import onnxruntime as ort
import sys
import io

MODEL_BYTES = None
VOCAB_TYPE = "alphanumeric"


class ImgUtil:
    def __init__(self, img_size=(100, 32)):
        self.img_size = img_size  # (W, H)

    def preprocess(self, img_source):
        """
        Standardize image: Grayscale -> Adaptive Threshold -> Resize to Fixed Size (100x32).
        No border cleaning or segmentation.
        """
        img = Image.open(img_source).convert("L")

        # Adaptive Thresholding
        hist = img.histogram()
        # Find the most common pixel value (Background)
        bg_color = hist.index(max(hist))
        # Set threshold slightly below background (Capture faint text)
        threshold = max(bg_color - 10, 50)

        img = img.point(lambda p: 255 if p > threshold else 0)

        # Resize to fixed size for CRNN (100x32)
        img = img.resize(self.img_size, Image.Resampling.NEAREST)

        return img


class ONNXSolver:
    def __init__(self, model_source="model.onnx"):
        self.util = ImgUtil()

        # Use injected bytes if available
        if MODEL_BYTES is not None:
            model_source = MODEL_BYTES

        # Select Vocabulary based on Injected Type
        if VOCAB_TYPE == "math":
            self.chars = sorted(list("0123456789+-=?"))
        else:
            self.chars = sorted(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))

        self.classes = ["-"] + self.chars
        self.idx_to_char = {i: c for i, c in enumerate(self.classes)}

        try:
            self.ort_session = ort.InferenceSession(model_source)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            # print(f"Error loading model: {e}")
            self.ort_session = None

    def solve(self, img_source):
        if not self.ort_session:
            return "ERROR_MODEL_NOT_LOADED"

        # Preprocess full image (100x32)
        img = self.util.preprocess(img_source)

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


if __name__ == "__main__":
    # Read from stdin
    try:
        input_data = sys.stdin.buffer.read()
        if not input_data:
            sys.exit(1)

        solver = ONNXSolver()
        prediction = solver.solve(io.BytesIO(input_data))
        sys.stdout.write(prediction)
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)
