from solver.onnx_solver import ONNXSolver
import os

# Check if model exists
if os.path.exists("model.onnx"):
    solver = ONNXSolver("model.onnx")
    print(solver.solve("./yq4e.jpeg"))
else:
    print("Model not found. Run train_model.py and export_onnx.py first.")
    # Fallback to light solver if needed, or just fail
    from solver.light_solver import CaptchaCracker

    cracker = CaptchaCracker()
    cracker.load_db()
    print(cracker.solve("./yq4e.jpeg"))
