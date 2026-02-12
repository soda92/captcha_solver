from PIL import Image
import numpy as np
from pathlib import Path

def ascii_art(img):
    arr = np.array(img)
    lines = []
    h, w = arr.shape
    for y in range(h):
        line = ""
        for x in range(w):
            val = arr[y, x]
            if val < 128: line += "#"
            else: line += "."
        lines.append(line)
    return "\n".join(lines)

templates_dir = Path("templates")
chars_to_check = ['Y', 'E', 'Q', '4']

for char in chars_to_check:
    char_dir = templates_dir / char
    if not char_dir.exists():
        print(f"Template dir for {char} missing!")
        continue
        
    print(f"\n--- Checking Templates for {char} ---")
    files = list(char_dir.glob("*.png"))
    if not files:
        print("No files found.")
        continue
        
    # Check first 3
    for f in files[:3]:
        print(f"File: {f}")
        img = Image.open(f).convert("L")
        print(ascii_art(img))
        
        # Count black pixels
        arr = np.array(img)
        black_count = np.sum(arr < 128)
        print(f"Black Pixels: {black_count}")
