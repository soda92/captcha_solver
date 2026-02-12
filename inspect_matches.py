from solver.light_solver import CaptchaCracker
import numpy as np

cracker = CaptchaCracker()
cracker.load_db()
img_path = "yq4e.jpeg"

# Preprocess (which now includes clean_borders)
img = cracker.preprocess(img_path)
chars = cracker.segment(img)
labels = ['Y', 'Q', '4', 'E']

print(f"Segments: {len(chars)}")

for i, char_img in enumerate(chars):
    print(f"\n--- Char {i+1} (Expected: {labels[i]}) ---")
    query_arr = np.array(char_img)
    
    # Check shifts
    shifts = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shifts.append((dy, dx))
            
    shifted_queries = []
    for dy, dx in shifts:
        shifted = np.roll(query_arr, dy, axis=0)
        shifted = np.roll(shifted, dx, axis=1)
        if dy > 0: shifted[:dy, :] = 0
        elif dy < 0: shifted[dy:, :] = 0
        if dx > 0: shifted[:, :dx] = 0
        elif dx < 0: shifted[:, dx:] = 0
        shifted_queries.append(shifted.flatten())

    results = []
    for char_key, templates in cracker.database.items():
        local_min = float("inf")
        # Check all shifts against all templates
        for q_vec in shifted_queries:
            # Explicit calc
            diff = templates.astype(int) - q_vec.astype(int)
            dist = np.sum(diff**2, axis=1)
            local_min = min(local_min, np.min(dist))
            
        results.append((local_min, char_key))
        
    results.sort(key=lambda x: x[0])
    
    for score, char in results[:5]:
        print(f"  {char}: {score}")
