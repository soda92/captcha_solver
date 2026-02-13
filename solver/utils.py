from PIL import Image


class ImgUtil:
    def __init__(self, img_size=(100, 32)):
        self.img_size = img_size  # (W, H)

    def preprocess(self, img_path):
        """
        Standardize image: Grayscale -> Adaptive Threshold -> Resize to Fixed Size.
        No border cleaning or segmentation.
        """
        img = Image.open(img_path).convert("L")

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
