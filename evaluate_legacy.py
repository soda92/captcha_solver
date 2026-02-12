#!/usr/bin/env python3
import os
from pathlib import Path
from solver.light_solver import CaptchaCracker


def evaluate_legacy(test_dir="test_images"):
    print(f"Evaluating Legacy Solver on '{test_dir}'...")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        return

    cracker = CaptchaCracker()
    # Ensure templates are built/loaded. We assume templates folder exists and is populated.
    if not cracker.templates_dir.exists() or not list(cracker.templates_dir.iterdir()):
        print(
            "Templates not found. Please run solver/main.py first to build templates."
        )
        return

    cracker.load_db()

    test_path = Path(test_dir)
    files = (
        list(test_path.glob("*.jpeg"))
        + list(test_path.glob("*.jpg"))
        + list(test_path.glob("*.png"))
    )

    if not files:
        print("No images found in test directory.")
        return

    total = 0
    correct = 0

    print("\nResults (Legacy):")
    print("-" * 40)
    print(f"{'File':<15} | {'Expected':<8} | {'Predicted':<8} | {'Status'}")
    print("-" * 40)

    for f in sorted(files):
        filename = f.name
        label = f.stem.upper()

        if len(label) != 4:
            continue

        try:
            prediction = cracker.solve(str(f))
            is_correct = prediction == label
            status = "✅ OK" if is_correct else "❌ FAIL"

            print(f"{filename:<15} | {label:<8} | {prediction:<8} | {status}")

            total += 1
            if is_correct:
                correct += 1
        except Exception as e:
            print(f"{filename:<15} | {label:<8} | ERROR    | {e}")

    print("-" * 40)
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nLegacy Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    else:
        print("\nNo valid test images processed.")


if __name__ == "__main__":
    evaluate_legacy()
