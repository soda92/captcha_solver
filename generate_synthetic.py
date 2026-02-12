import os
import random
import string
import math
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps

def generate_synthetic_data(count=2000, output_dir="synthetic_captchas"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    charset = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    font_path = "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf"
    try:
        font = ImageFont.truetype(font_path, 24)
    except:
        print("Font not found, using default")
        font = ImageFont.load_default()

    print(f"Generating {count} synthetic images...")
    
    for i in range(count):
        label = "".join(random.choices(charset, k=4))
        
        # 1. Base Image
        w, h = 100, 30
        img = Image.new("L", (w, h), 255) # White BG
        draw = ImageDraw.Draw(img)
        
        # Draw Text centered
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) // 2
        y = (h - text_h) // 2
        draw.text((x, y - bbox[1]), label, font=font, fill=0) # Black Text
        
        # 2. Wave Distortion
        # y = y + A * sin(x / freq)
        arr = np.array(img)
        rows, cols = arr.shape
        new_arr = np.full_like(arr, 255) # White fill
        
        amplitude = random.uniform(1.0, 3.0) # Wave height
        frequency = random.uniform(0.05, 0.15) # Wave width
        phase = random.uniform(0, 6.28)
        
        for r in range(rows):
            for c in range(cols):
                # Vertical shift
                offset = int(amplitude * math.sin(c * frequency + phase))
                new_r = r + offset
                if 0 <= new_r < rows:
                    new_arr[new_r, c] = arr[r, c]
                    
        # 3. Noise (Salt and Pepper)
        # Add random black dots
        noise_density = 0.02
        noise_mask = np.random.rand(rows, cols) < noise_density
        new_arr[noise_mask] = 0 # Black dots
        
        # 4. Save
        final_img = Image.fromarray(new_arr)
        
        # Add faint noise/blur?
        # Maybe reduce contrast to mimic 'faint' nature of j6ge
        # But our solver handles 'faint' by thresholding.
        # So providing 'solid' characters is fine, the solver will see them as solid.
        # But if the solver expects 'dots', and we train on 'solid', it might overfit to solid?
        # But my thresholding step converts 'faint' to 'solid'.
        # So the model ALWAYS sees 'solid'.
        # Thus, generating 'solid' synthetic data is CORRECT.
        
        save_path = os.path.join(output_dir, f"{label}_{i}.jpeg")
        final_img.save(save_path)
        
        if (i+1) % 500 == 0:
            print(f"Generated {i+1}...")

if __name__ == "__main__":
    generate_synthetic_data()
