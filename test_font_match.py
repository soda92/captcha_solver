from PIL import Image, ImageFont, ImageDraw, ImageOps
import numpy as np
from solver.light_solver import CaptchaCracker

cracker = CaptchaCracker()
cracker.load_db()

font_path = "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf"
try:
    font = ImageFont.truetype(font_path, 64) # Generate large first
except Exception as e:
    print(f"Error loading font: {e}")
    exit(1)

def generate_char_processed(char):
    # 1. Generate large clean char
    img_size = (100, 100)
    img = Image.new("L", img_size, 255) # White bg
    draw = ImageDraw.Draw(img)
    
    # Draw centered-ish
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (100 - w) // 2
    y = (100 - h) // 2
    draw.text((x, y - bbox[1]), char, font=font, fill=0) # Black text
    
    # 2. Crop to content (Simulate 'segment' finding the character)
    # Invert to find bbox of black text
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    if bbox:
        crop = img.crop(bbox)
    else:
        crop = img # Should not happen
        
    # 3. Apply 'resize_pad' to fit in 32x32 (Standardize)
    final_img = cracker.resize_pad(crop, (32, 32))
    
    return final_img

chars_to_test = ['Y', 'Q', '4', 'E']
generated_templates = {}

print("Generating PROCESSED templates...")
for c in chars_to_test:
    img = generate_char_processed(c)
    generated_templates[c] = np.array(img).flatten()
    # Save for debug
    img.save(f"debug_font_processed_{c}.png")

# Compare
img_path = "yq4e.jpeg"
query_chars = cracker.segment(cracker.preprocess(img_path))

print(f"\nComparing yq4e.jpeg segments against generated font templates...")
for i, query_img in enumerate(query_chars):
    if i >= 4: break
    expected = chars_to_test[i]
    print(f"\n--- Segment {i} (Expected: {expected}) ---")
    
    query_arr = np.array(query_img)
    
    # Use simple shift check (center only to start, or small range)
    # We use uint8 overflow distance metric implicitly if we just do (a-b)**2 with numpy uint8 arrays
    # But let's be explicit and cast to int to see REAL pixel differences
    
    shifts = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
    shifted_queries = []
    for dy, dx in shifts:
        shifted = np.roll(query_arr, dy, axis=0)
        shifted = np.roll(shifted, dx, axis=1)
        if dy > 0: shifted[:dy, :] = 0
        elif dy < 0: shifted[dy:, :] = 0
        if dx > 0: shifted[:, :dx] = 0
        elif dx < 0: shifted[:, dx:] = 0
        shifted_queries.append(shifted.flatten())

    best_gen_char = "?"
    best_gen_dist = float("inf")
    
    for gen_char, gen_tmpl in generated_templates.items():
        local_min = float("inf")
        for q_vec in shifted_queries:
            # Cast to int to get TRUE Euclidean/Hamming distance, not overflow
            diff = gen_tmpl.astype(int) - q_vec.astype(int)
            dist = np.sum(diff**2)
            local_min = min(local_min, dist)
            
        print(f"  Dist to Generated '{gen_char}': {local_min}")
        if local_min < best_gen_dist:
            best_gen_dist = local_min
            best_gen_char = gen_char
            
    print(f"  -> Best Match: {best_gen_char} (Dist: {best_gen_dist})")
