import os


def inject_model(
    source_script="solver/standalone.py",
    model_path="model.onnx",
    output_path="dist/standalone.py",
):
    # Create dist dir
    if not os.path.exists("dist"):
        os.makedirs("dist")

    # Read script content
    with open(source_script, "r") as f:
        script_content = f.read()

    # Read model bytes
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    # Replace placeholder
    # We use repr(model_bytes) to get b'...' string representation
    placeholder = "MODEL_BYTES = None"
    replacement = f"MODEL_BYTES = {repr(model_bytes)}"

    if placeholder not in script_content:
        print("Error: Placeholder 'MODEL_BYTES = None' not found in script.")
        return

    new_content = script_content.replace(placeholder, replacement)

    # Write to dist
    with open(output_path, "w") as f:
        f.write(new_content)

    print(f"Successfully injected model into {output_path}")


if __name__ == "__main__":
    inject_model()
