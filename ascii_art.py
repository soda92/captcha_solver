from PIL import Image
import numpy as np
import sys

def img_to_ascii(path, width=100):
    try:
        img = Image.open(path).convert('L')
        # Resize to maintain aspect ratio, roughly
        aspect_ratio = img.height / img.width
        new_height = int(width * aspect_ratio * 0.55)
        img = img.resize((width, new_height))
        
        pixels = np.array(img)
        chars = "@%#*+=-:. " # Dark to light
        
        ascii_str = ""
        for row in pixels:
            for pixel in row:
                idx = pixel // 26 # 255 // 26 is 9, len(chars) is 10
                if idx >= len(chars): idx = len(chars) - 1
                ascii_str += chars[idx]
            ascii_str += "\n"
        
        print(ascii_str)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_to_ascii(sys.argv[1])
    else:
        print("Usage: python ascii_art.py <image_path>")
