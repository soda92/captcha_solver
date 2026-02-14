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
        Standardize image: Grayscale -> Adaptive Threshold -> Resize (Preserving Aspect Ratio).
        """
        img = Image.open(img_source).convert("L")

        # Adaptive Thresholding
        hist = img.histogram()
        bg_color = hist.index(max(hist))
        threshold = max(bg_color - 10, 50)

        img = img.point(lambda p: 255 if p > threshold else 0)

        # Resize preserving aspect ratio
        target_w, target_h = self.img_size
        w, h = img.size

        # Scale to fixed height
        scale = target_h / h
        new_w = int(w * scale)
        new_h = target_h

        img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

        # Create canvas
        new_img = Image.new("L", (target_w, target_h), 255)  # White background

        # Paste (Center)
        if new_w > target_w:
            img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
            new_img.paste(img, (0, 0))
        else:
            x_pad = (target_w - new_w) // 2
            new_img.paste(img, (x_pad, 0))

        return new_img


class ONNXSolver:
    def __init__(self, model_source="model.onnx"):
        self.util = ImgUtil()

        # Use injected bytes if available
        if MODEL_BYTES is not None:
            model_source = MODEL_BYTES

        # Select Vocabulary based on Injected Type
        if VOCAB_TYPE == "math":
            self.chars = sorted(list("012345678+-=?"))
        else:
            self.chars = sorted(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))

        self.classes = ["-"] + self.chars
        self.idx_to_char = {i: c for i, c in enumerate(self.classes)}

        try:
            self.ort_session = ort.InferenceSession(model_source)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Error loading model: {e}")
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

        result_str = "".join(res)

        if VOCAB_TYPE == "math":
            try:
                # Clean expression
                expr = result_str.replace("=", "").replace("?", "")
                # Simple and safe eval (only allow 0-9, +, -)
                allowed = set("0123456789+-")
                if all(c in allowed for c in expr):
                    return str(eval(expr))
            except Exception:
                pass

        return result_str


def solve(source):
    """
    Top-level helper to solve captcha from path (str) or bytes.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    solver = ONNXSolver()
    return solver.solve(source)


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
