import os
import shutil
from pathlib import Path

SRC_DIR = "num_test_images"
DST_DIR = "num_captchas"

def merge_data():
    if not os.path.exists(SRC_DIR):
        print(f"Source directory '{SRC_DIR}' does not exist.")
        return

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
        print(f"Created destination directory '{DST_DIR}'.")

    files = list(Path(SRC_DIR).glob("*.jpeg")) + list(Path(SRC_DIR).glob("*.png"))
    
    if not files:
        print(f"No images found in '{SRC_DIR}'.")
        return

    print(f"Found {len(files)} images in '{SRC_DIR}'. Merging into '{DST_DIR}'...")

    moved_count = 0
    for src_path in files:
        filename = src_path.name
        dst_path = os.path.join(DST_DIR, filename)

        # Handle collisions
        if os.path.exists(dst_path):
            base_full = filename
            # Check if filename already has an extension
            base, ext = os.path.splitext(base_full)
            
            # If the base already ends with _number, we might want to increment that
            # But simpler logic: just append _counter
            
            counter = 1
            while os.path.exists(dst_path):
                # Try to preserve format like 8-5_1.jpeg -> 8-5_2.jpeg if possible?
                # Or just 8-5_1_1.jpeg
                
                # Check if it ends with _d
                import re
                match = re.search(r'_(\d+)$', base)
                if match:
                    # It has a number suffix, verify if we can increment it
                    # But simpler and safer to just append new suffix to avoid logical complexity
                    # creating 8-5_1_1.jpeg is fine
                    new_filename = f"{base}_{counter}{ext}"
                else:
                    new_filename = f"{base}_{counter}{ext}"
                
                dst_path = os.path.join(DST_DIR, new_filename)
                counter += 1
        
        try:
            shutil.move(str(src_path), dst_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {src_path}: {e}")

    print("-" * 30)
    print(f"Successfully moved {moved_count} images.")
    print("-" * 30)

if __name__ == "__main__":
    merge_data()
