from solver.light_solver import CaptchaCracker

cracker = CaptchaCracker()

# STEP 1: Run this ONCE to generate your templates folder
# Point this to your folder of named jpegs
cracker.build_templates("./raw_captchas")
cracker.analyze_dataset()
