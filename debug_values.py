from solver.light_solver import CaptchaCracker
import numpy as np

cracker = CaptchaCracker()
cracker.load_db()
img_path = "yq4e.jpeg"

chars = cracker.segment(cracker.preprocess(img_path))
query_arr = np.array(chars[0])

print(f"Query Shape: {query_arr.shape}")
print(f"Query Min: {query_arr.min()}, Max: {query_arr.max()}")
print(f"Query Unique Values: {np.unique(query_arr)}")

if 'Y' in cracker.database:
    tmpl = cracker.database['Y'][0] # Get first template for Y
    print(f"Template Y Shape: {tmpl.shape}")
    print(f"Template Y Min: {tmpl.min()}, Max: {tmpl.max()}")
    print(f"Template Y Unique Values: {np.unique(tmpl)}")
    
    # Calculate distance manually
    q_flat = query_arr.flatten()
    diff = tmpl - q_flat
    dist = np.sum(diff**2)
    print(f"Manually calc dist (Y vs Query): {dist}")
