#!/usr/bin/env python3
import os
from pathlib import Path
from solver.onnx_solver import ONNXSolver


def evaluate(test_dir="test_images", model_path="model.onnx"):
    print(f"Evaluating model '{model_path}' on '{test_dir}'...")

    if not os.path.exists(model_path):
        print("Error: Model not found.")
        return

    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        return

    solver = ONNXSolver(model_path)
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

    print("\nResults:")
    print("-" * 40)
    print(f"{'File':<15} | {'Expected':<8} | {'Predicted':<8} | {'Status'}")
    print("-" * 40)

    for f in sorted(files):
        filename = f.name
        label = f.stem.upper()

        # Skip if filename isn't 4 chars (might be a different naming scheme, but we assume label=filename)
        if len(label) != 4:
            print(f"{filename:<15} | {'?':<8} | {'?':<8} | SKIPPED (Name len != 4)")
            continue

        try:
            prediction = solver.solve(str(f))
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
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.2f}%)")
    else:
        print("\nNo valid test images processed.")


if __name__ == "__main__":
    evaluate()
