from PIL import Image, ImageOps
import numpy as np
from solver.light_solver import CaptchaCracker

cracker = CaptchaCracker()
img_path = "yq4e.jpeg"
# Manually preprocess and cut to get the raw crops (before resize_pad)
img = cracker.preprocess(img_path)

# Replicate segment to get raw crops
arr = np.array(img)
arr_bin = np.where(arr == 0, 1, 0)
proj = np.sum(arr_bin, axis=0)
noise_threshold = 2
is_text = proj > noise_threshold
starts = np.where(np.diff(is_text.astype(int)) == 1)[0] + 1
ends = np.where(np.diff(is_text.astype(int)) == -1)[0] + 1
if len(is_text) > 0 and is_text[0]: starts = np.insert(starts, 0, 0)
if len(is_text) > 0 and is_text[-1]: ends = np.append(ends, len(proj))
regions = [(s, e) for s, e in zip(starts, ends) if (e - s) > 4]

if len(regions) != 4:
    print(f"Warning: Found {len(regions)} regions. Using fallback cuts.")
    w, h = img.size
    step = w // 4
    raw_cuts = [img.crop((i * step, 0, (i + 1) * step, h)) for i in range(4)]
else:
    raw_cuts = [img.crop((s, 0, e, img.height)) for s, e in regions]

def resize_pad_white(img, size):
    new_img = Image.new("L", size, 255) # White BG
    img.thumbnail(size, Image.Resampling.NEAREST)
    w, h = img.size
    x_pad = (size[0] - w) // 2
    y_pad = (size[1] - h) // 2
    new_img.paste(img, (x_pad, y_pad))
    return new_img

print("Original distance (Black Padding) vs New (White Padding) check...")
# We can't easily check distance against DB because DB is built with Black Padding.
# But we can check if the images look 'cleaner'.

for i, crop in enumerate(raw_cuts):
    padded_black = cracker.resize_pad(crop, (32, 32))
    padded_white = resize_pad_white(crop, (32, 32))
    
    # Count black pixels (Text)
    # Black padding adds A LOT of black pixels.
    # White padding only has the text as black.
    
    pb_arr = np.array(padded_black)
    pw_arr = np.array(padded_white)
    
    black_pixels_pb = np.sum(pb_arr == 0)
    black_pixels_pw = np.sum(pw_arr == 0)
    
    print(f"Char {i}: BlackPixels(Old)={black_pixels_pb}, BlackPixels(New)={black_pixels_pw}")
    
    # If New has drastically fewer black pixels, it means Old had a lot of 'Frame' noise.
    # The 'Text' should be roughly the same count (maybe 50-200 pixels).
    # If Old has 500+ black pixels, it's mostly frame.
