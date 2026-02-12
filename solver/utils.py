import numpy as np
from PIL import Image


class ImgUtil:
    def __init__(self, char_size=(32, 32)):
        self.char_size = char_size
        self.database = {}  # { 'A': [array, array...], 'B': ... }
        self.loaded = False

    def clean_borders(self, img):
        """
        Removes borders by turning rows/cols that are > 50% black into white.
        Only checks top/bottom 20% and left/right 20% to avoid deleting character parts.
        Assumes img is 'L' mode (0=Black/Text, 255=White/Bg).
        """
        arr = np.array(img)
        h, w = arr.shape

        # Check rows (Top 20% and Bottom 20%)
        limit_h = int(h * 0.2)
        for y in list(range(0, limit_h)) + list(range(h - limit_h, h)):
            # Count black pixels (0)
            black_count = np.sum(arr[y, :] == 0)
            if black_count > w * 0.5:
                arr[y, :] = 255  # Clear row

        # Check cols (Left 20% and Right 20%)
        limit_w = int(w * 0.2)
        for x in list(range(0, limit_w)) + list(range(w - limit_w, w)):
            black_count = np.sum(arr[:, x] == 0)
            if black_count > h * 0.5:
                arr[:, x] = 255  # Clear col

        return Image.fromarray(arr)

    def preprocess(self, img_path):
        """Standardize image: Grayscale -> Binary -> Crop Borders"""
        img = Image.open(img_path).convert("L")

        # Adaptive Thresholding
        hist = img.histogram()
        # Find the most common pixel value (Background)
        bg_color = hist.index(max(hist))
        # Set threshold slightly below background (Capture faint text)
        threshold = max(bg_color - 10, 50)

        img = img.point(lambda p: 255 if p > threshold else 0)
        # Remove massive borders
        img = self.clean_borders(img)
        return img

    def segment(self, img):
        """
        Tries to slice image into exactly 4 chars.
        Strategy: Vertical Projection (detect gaps).
        Fallback: Fixed width slice if gaps aren't clear.
        """
        arr = np.array(img)
        # Invert: Text=1, Bg=0
        arr = np.where(arr == 0, 1, 0)

        # Sum pixels in each column
        proj = np.sum(arr, axis=0)

        # Find columns where projection is 0 (gaps)
        # simplistic gap detection
        is_text = proj > 0

        # Detect starts and ends of text regions
        starts = np.where(np.diff(is_text.astype(int)) == 1)[0] + 1
        ends = np.where(np.diff(is_text.astype(int)) == -1)[0] + 1

        if is_text[0]:
            starts = np.insert(starts, 0, 0)
        if is_text[-1]:
            ends = np.append(ends, len(proj))

        cuts = []
        # Filter noise (regions < 4px wide)
        regions = [(s, e) for s, e in zip(starts, ends) if (e - s) > 4]

        # STRATEGY A: We found exactly 4 regions (Clean separation)
        if len(regions) == 4:
            for s, e in regions:
                cuts.append(img.crop((s, 0, e, img.height)))

        # STRATEGY B: Text is touching/tilted (Fallback to fixed width)
        else:
            w, h = img.size
            # Assuming standard padding, usually text is centered.
            # You might need to trim left/right margins here if they are huge.
            step = w // 4
            for i in range(4):
                cuts.append(img.crop((i * step, 0, (i + 1) * step, h)))

        # Resize all to uniform template size
        return [self.resize_pad(c, self.char_size) for c in cuts]

    def resize_pad(self, img, size):
        """
        Resizes image to fit inside 'size' (w, h) while keeping aspect ratio,
        then pads with black to fill the box.
        """
        # Create white background
        new_img = Image.new("L", size, 255)

        # Calculate resize ratio
        img.thumbnail(size, Image.Resampling.NEAREST)

        # Center the character
        w, h = img.size
        x_pad = (size[0] - w) // 2
        y_pad = (size[1] - h) // 2

        new_img.paste(img, (x_pad, y_pad))
        return new_img
