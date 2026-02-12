import os
import numpy as np
from PIL import Image
from pathlib import Path
import shutil


class CaptchaCracker:
    def __init__(self, templates_dir="templates", char_size=(32, 32)):
        self.templates_dir = Path(templates_dir)
        self.char_size = char_size
        self.database = {}  # { 'A': [array, array...], 'B': ... }
        self.loaded = False

    def preprocess(self, img_path):
        """Standardize image: Grayscale -> Binary -> Crop Borders"""
        img = Image.open(img_path).convert("L")
        # Threshold to binary (adjust 140 if your images are noisy)
        img = img.point(lambda p: 255 if p > 140 else 0)
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
        return [c.resize(self.char_size, Image.Resampling.NEAREST) for c in cuts]

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

        chars = self.segment(self.preprocess(img_path))
        result = ""

        for char_img in chars:
            query = np.array(char_img).flatten()
            best_char = "?"
            min_dist = float("inf")

            # Compare against every character class
            for char_key, templates in self.database.items():
                # Vectorized Euclidean distance against ALL templates of this char
                # (template - query)^2
                diff = templates - query
                dists = np.sum(diff**2, axis=1)

                # Find the best match within this character class
                local_min = np.min(dists)
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

        print(f"\n--- DATASET HEALTH REPORT ---")

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
