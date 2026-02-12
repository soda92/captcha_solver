from PIL import Image, ImageFont, ImageDraw, ImageOps
import numpy as np
from skimage.metrics import structural_similarity as ssim
from solver.light_solver import CaptchaCracker
import string

# 1. Setup
cracker = CaptchaCracker()
font_path = "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf"
try:
    font = ImageFont.truetype(font_path, 64)
except:
    print("Font not found!")
    exit(1)

# 2. Generate Reference Templates (All Chars)
def generate_template(char):
    # High-res generation
    img = Image.new("L", (100, 100), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((100-w)//2, (100-h)//2 - bbox[1]), char, font=font, fill=0)
    
    # Crop to content
    inverted = ImageOps.invert(img)
    crop_bbox = inverted.getbbox()
    crop = img.crop(crop_bbox) if crop_bbox else img
    
    # Resize pad to 32x32 (using White padding as per recent fix)
    return cracker.resize_pad(crop, (32, 32))

# ... imports

def ascii_art(arr):
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

print("Loading dataset templates...")
cracker.load_db()

# 3. Process Query Image
img_path = "yq4e.jpeg"
# Preprocess
raw_img = cracker.preprocess(img_path)
# Segment
query_chars = cracker.segment(raw_img)

print(f"Solving {img_path} with SSIM (using Dataset Templates)...")
expected_labels = ['Y', 'Q', '4', 'E']
result_str = ""

for i, q_img in enumerate(query_chars):
    if i >= 4: break
    q_arr = np.array(q_img)
    expected = expected_labels[i]
    
    print(f"\n--- Segment {i} (Expected: {expected}) ---")
    print(ascii_art(q_arr))

    results = []
    # Compare against all DB templates
    for char_key, tmpl_list in cracker.database.items():
        # tmpl_list is (N, 1024) - it contains multiple examples!
        # We should check ALL examples and take the best score
        best_local_score = -1.0
        
        # We need to treat tmpl_list as a list of arrays.
        # Check shape first
        if len(tmpl_list.shape) == 2:
            # It's (N, 1024)
            for tmpl_flat in tmpl_list:
                tmpl_arr = tmpl_flat.reshape((32, 32))
                score = ssim(q_arr, tmpl_arr, data_range=255)
                if score > best_local_score:
                    best_local_score = score
        else:
            # Maybe just (1024,) if only 1 example? No, load_db usually makes list
            pass
            
        results.append((char_key, best_local_score))
            
    # Sort desc
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top 3 Matches:")
    for char, score in results[:3]:
        print(f"  {char}: {score:.4f}")
        if i == 0: result_str = char
    
    # Check expected
    expected_score = next((s for c, s in results if c == expected), -1.0)
    print(f"Score for '{expected}': {expected_score:.4f}")
    
    # Append best to result
    if i > 0: result_str += results[0][0]

print(f"Final Result: {result_str}")
