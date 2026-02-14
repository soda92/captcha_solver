from solver.onnx_solver import ONNXSolver
import os
from pathlib import Path


def evaluate():
    model_path = "model_math.onnx"
    test_dir = "num_test_images"

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return

    if not os.path.exists(test_dir):
        # Fallback to num_captchas for demo
        test_dir = "num_captchas"

    print(f"Evaluating model '{model_path}' on '{test_dir}'...")

    # Use 'math' vocab
    solver = ONNXSolver(model_path, vocab_type="math")

    files = sorted(
        list(Path(test_dir).glob("*.jpeg")) + list(Path(test_dir).glob("*.png"))
    )
    if not files:
        print("No images found.")
        return

    correct_count = 0
    total_count = 0

    print(f"{'File':<15} | {'Exp':<10} | {'Pred':<10} | {'Calc':<5} | {'Status'}")
    print("-" * 65)

    for f in files:
        expected = f.stem.split("_")[0]  # e.g. 8-5

        pred_expr = solver.solve(str(f))

        # Calculate result
        try:
            # Clean expression (remove =?)
            expr = pred_expr.replace("=", "").replace("?", "")
            if expr:
                val = eval(expr)
            else:
                val = "Err"
        except Exception as e:
            print(e)
            val = "Err"

        is_match = pred_expr == expected
        status = "✅ OK" if is_match else "❌ FAIL"

        if is_match:
            correct_count += 1
        total_count += 1

        print(
            f"{f.name:<15} | {expected:<10} | {pred_expr:<10} | {str(val):<5} | {status}"
        )

    print("-" * 65)
    if total_count > 0:
        print(
            f"Accuracy: {correct_count}/{total_count} ({100 * correct_count / total_count:.2f}%)"
        )


if __name__ == "__main__":
    evaluate()
