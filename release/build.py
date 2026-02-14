import os
import argparse


def inject_model(task):
    if task == "math":
        model_path = "model_math.onnx"
        output_path = "dist/standalone_math.py"
        vocab_type = "math"
    else:
        model_path = "model.onnx"
        output_path = "dist/standalone_alphanumeric.py"
        vocab_type = "alphanumeric"

    source_script = "solver/standalone.py"

    # Create dist dir
    if not os.path.exists("dist"):
        os.makedirs("dist")

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Read script content
    with open(source_script, "r") as f:
        script_content = f.read()

    # Read model bytes
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    # Replace placeholders
    script_content = script_content.replace(
        "MODEL_BYTES = None", f"MODEL_BYTES = {repr(model_bytes)}"
    )

    script_content = script_content.replace(
        'VOCAB_TYPE = "alphanumeric"', f'VOCAB_TYPE = "{vocab_type}"'
    )

    # Write to dist
    with open(output_path, "w") as f:
        f.write(script_content)

    print(f"Successfully injected {task} model into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Standalone Solver")
    parser.add_argument("task", choices=["alphanumeric", "math"], help="Task to build")
    args = parser.parse_args()

    inject_model(args.task)
