import os
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import shutil


class CaptchaCracker:
    def __init__(self, templates_dir="templates", char_size=(32, 32)):
        self.templates_dir = Path(templates_dir)
        self.char_size = char_size
        self.database = {}  # { 'A': [array, array...], 'B': ... }
        self.loaded = False

    def clean_borders(self, img):
        """
        Removes borders by turning rows/cols that are > 50% black into white.
        Assumes img is 'L' mode (0=Black/Text, 255=White/Bg).
        """
        arr = np.array(img)
        h, w = arr.shape
        
        # Check rows
        for y in range(h):
            # Count black pixels (0)
            black_count = np.sum(arr[y, :] == 0)
            if black_count > w * 0.5:
                arr[y, :] = 255 # Clear row
                
        # Check cols
        for x in range(w):
            black_count = np.sum(arr[:, x] == 0)
            if black_count > h * 0.5:
                arr[:, x] = 255 # Clear col
                
        return Image.fromarray(arr)

    def preprocess(self, img_path):
        """Standardize image: Grayscale -> Binary -> Crop Borders"""
        img = Image.open(img_path).convert("L")
        # Threshold to binary (adjust 140 if your images are noisy)
        img = img.point(lambda p: 255 if p > 140 else 0)
        # Remove massive borders
        img = self.clean_borders(img)
        # Thicken characters to repair faint/eroded templates
        img = img.filter(ImageFilter.MinFilter(3))
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
        # Create black background
        new_img = Image.new("L", size, 255)

        # Calculate resize ratio
        img.thumbnail(size, Image.Resampling.NEAREST)

        # Center the character
        w, h = img.size
        x_pad = (size[0] - w) // 2
        y_pad = (size[1] - h) // 2

        new_img.paste(img, (x_pad, y_pad))
        return new_img

    def build_templates(self, source_folder):
        """
        One-time run: Reads '7A2B.jpg', cuts it, saves to templates/7, templates/A...
        """
        if self.templates_dir.exists():
            shutil.rmtree(self.templates_dir)
        self.templates_dir.mkdir()

        files = list(Path(source_folder).glob("*.jpeg")) + list(
            Path(source_folder).glob("*.png")
        )
        print(f"Processing {len(files)} labeled images...")

        for file in files:
            label = file.stem.upper()  # "7a2b" -> "7A2B"
            if len(label) != 4:
                continue  # Skip weird filenames

            try:
                chars = self.segment(self.preprocess(file))

                # We expect 4 chars for 4 letters in label
                for i, char_img in enumerate(chars):
                    # Check for empty/garbage templates
                    # After thickening, a real char should have > 50 black pixels
                    if np.sum(np.array(char_img) == 0) < 50:
                        continue

                    char_val = label[i]
                    save_dir = self.templates_dir / char_val
                    save_dir.mkdir(exist_ok=True)

                    # Save as a hash or counter to avoid overwrite
                    # Using count of existing files
                    cnt = len(list(save_dir.glob("*.png")))
                    char_img.save(save_dir / f"{cnt}.png")
            except Exception as e:
                print(f"Skipped {file}: {e}")

        print("Template build complete.")

    def load_db(self):
        """Loads templates into memory for fast matching"""
        self.database = {}
        for char_dir in self.templates_dir.iterdir():
            if char_dir.is_dir():
                char = char_dir.name
                arrays = []
                for img_file in char_dir.glob("*.png"):
                    img = Image.open(img_file).convert("L")
                    arrays.append(np.array(img).flatten())
                self.database[char] = np.array(arrays)
        self.loaded = True
        print(f"Loaded templates for {len(self.database)} characters.")

    def solve(self, img_path):
        if not self.loaded:
            self.load_db()

        # 1. Get the cropped character images
        chars = self.segment(self.preprocess(img_path))
        result = ""

        for char_img in chars:
            # We convert the query image to a raw 2D array (not flattened yet)
            # This allows us to "roll" (shift) pixels easily.
            query_arr = np.array(char_img)

            best_char = "?"
            min_dist = float("inf")

            # 2. The "Wiggle" Check
            # We will test the query image at positions from -2 to +2 in both X and Y
            shifts = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    shifts.append((dy, dx))

            # Pre-calculate shifted versions of the query image
            shifted_queries = []
            for dy, dx in shifts:
                # np.roll shifts the array, 'axis' handles direction
                shifted = np.roll(query_arr, dy, axis=0)
                shifted = np.roll(shifted, dx, axis=1)

                # IMPORTANT: Rolling wraps pixels around (top becomes bottom).
                # We must zero out the wrapped pixels to avoid noise.
                if dy > 0:
                    shifted[:dy, :] = 0
                elif dy < 0:
                    shifted[dy:, :] = 0
                if dx > 0:
                    shifted[:, :dx] = 0
                elif dx < 0:
                    shifted[:, dx:] = 0

                shifted_queries.append(shifted.flatten())

            # 3. Compare against DB
            for char_key, templates in self.database.items():
                # templates shape: (N, 1024) if 32x32

                # Check ALL shifts against ALL templates for this char
                for q_vec in shifted_queries:
                    # Vectorized diff: (Templates - Shifted_Query)
                    diff = templates - q_vec
                    dist = np.sum(diff**2, axis=1)

                    local_min = np.min(dist)
                    if local_min < min_dist:
                        min_dist = local_min
                        best_char = char_key

            result += best_char
        return result

    def analyze_dataset(self, exclude_chars="01ZIO"):
        """
        Checks for ALL standard characters (0-9, A-Z).
        Explicitly reports '0' for missing folders.
        """
        if not self.templates_dir.exists():
            print("Error: Templates folder not found.")
            return

        # 1. Define the full expected charset
        import string

        full_charset = string.digits + string.ascii_uppercase  # "0123...XYZ"
        # Remove the "impossible" characters
        expected_chars = [c for c in full_charset if c not in exclude_chars]

        counts = {}
        missing = []

        print("\n--- DATASET HEALTH REPORT ---")

        # 2. Check every single expected character
        for char in expected_chars:
            char_dir = self.templates_dir / char

            if char_dir.exists() and char_dir.is_dir():
                count = len(list(char_dir.glob("*.png")))
                counts[char] = count
            else:
                count = 0
                missing.append(char)

            # Visual Bar
            bar = "█" * (count // 2) if count > 0 else ""
            status = ""

            if count == 0:
                status = "  🚨 MISSING!"
            elif count < 5:
                status = "  ⚠️  Critical Low"

            print(f"[{char}]: {count:3d} {bar}{status}")

        # 3. Summary
        total_imgs = sum(counts.values())
        print("-" * 40)
        print(f"Total Templates: {total_imgs}")

        if missing:
            print(f"❌ FATAL: You have ZERO examples for: {', '.join(missing)}")
            print("   The solver will crash or always fail on these characters.")
        elif any(c < 5 for c in counts.values()):
            lows = [k for k, v in counts.items() if v < 5]
            print(f"⚠️  WARNING: Very low data (<5) for: {', '.join(lows)}")
        else:
            print("✅ Dataset looks healthy!")
