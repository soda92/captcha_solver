import torch
from solver.ml_solver import CRNN, MLSolver


def export():
    # 1. Load Model
    model_path = "model_math.pth"
    solver = MLSolver(model_path, vocab_type="math")  # Helper to get class count
    num_classes = len(solver.classes)
    model = CRNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 2. Create Dummy Input
    # Input shape: (Batch=1, Channel=1, Height=32, Width=100)
    dummy_input = torch.randn(1, 1, 32, 100)

    # 3. Export
    output_path = "model_math.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {1: "batch_size"}},
    )

    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    export()
