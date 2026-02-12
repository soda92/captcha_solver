from solver.ml_solver import MLSolver

solver = MLSolver("model.pth")
result = solver.solve("yq4e.jpeg")
print(f"ML Solution: {result}")
