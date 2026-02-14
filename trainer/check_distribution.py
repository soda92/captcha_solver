import os
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

DATA_DIR = "num_captchas"

def check_distribution():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return

    files = list(Path(DATA_DIR).glob("*.jpeg")) + list(Path(DATA_DIR).glob("*.png"))
    print(f"Found {len(files)} images.")

    char_counts = Counter()
    
    for f in files:
        # Filename format: label_suffix.ext or label.ext
        # e.g. 8-5_1.jpeg -> 8-5
        # e.g. 3+4.jpeg -> 3+4
        label = f.stem.split("_")[0]
        
        # We also need to consider that the math dataset appends ' =?' during training
        # But here we just want to see the distribution of the "input" characters
        # present in the filenames.
        
        for char in label:
            char_counts[char] += 1

    print("\nCharacter Distribution:")
    print("-" * 30)
    for char, count in sorted(char_counts.items()):
        print(f"'{char}': {count}")
    print("-" * 30)

    # Plotting
    chars = sorted(char_counts.keys())
    counts = [char_counts[c] for c in chars]

    plt.figure(figsize=(10, 6))
    plt.bar(chars, counts, color='skyblue')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.title('Training Data Character Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add counts on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + 1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_distribution()
