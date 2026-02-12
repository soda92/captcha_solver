from solver.light_solver import CaptchaCracker
cracker = CaptchaCracker()


# STEP 2: Use it in your RPA loop
cracker.load_db() # Call once at startup
print(cracker.solve("./yq4e.jpeg"))