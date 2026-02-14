from PIL import Image


class ImgUtil:
    def __init__(self, img_size=(100, 32)):
        self.img_size = img_size  # (W, H)

    def preprocess(self, img_path):
        """
        Standardize image: Grayscale -> Adaptive Threshold -> Resize (Preserving Aspect Ratio).
        """
        # Handle path vs file-like
        if hasattr(img_path, "read"):
            img = Image.open(img_path).convert("L")
        else:
            img = Image.open(str(img_path)).convert("L")

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
