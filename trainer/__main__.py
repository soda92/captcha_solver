import argparse
import sys

# Training
from trainer.train import train as train_alphanumeric
from trainer.train_math import train_fixed as train_math

# Export
from trainer.export import export as export_alphanumeric
from trainer.export_math import export as export_math

# Evaluation
from trainer.evaluate import evaluate as evaluate_alphanumeric
from trainer.evaluate_math import evaluate as evaluate_math

def main():
    parser = argparse.ArgumentParser(description="Captcha Trainer CLI")
    
    # We use optional arguments so you can do --train and --eval in one command
    parser.add_argument("--train", choices=["alphanumeric", "math"], help="Train the model")
    parser.add_argument("--eval", choices=["alphanumeric", "math"], help="Export and Evaluate the model")
    
    args = parser.parse_args()
    
    if not args.train and not args.eval:
        parser.print_help()
        sys.exit(1)
        
    # 1. Training Phase
    if args.train == "alphanumeric":
        print("\n=== Training Alphanumeric Model ===")
        train_alphanumeric()
    elif args.train == "math":
        print("\n=== Training Math Model ===")
        train_math()
        
    # 2. Evaluation Phase (Export -> Evaluate)
    if args.eval == "alphanumeric":
        print("\n=== Exporting Alphanumeric Model ===")
        export_alphanumeric()
        print("\n=== Evaluating Alphanumeric Model ===")
        evaluate_alphanumeric()
        
    elif args.eval == "math":
        print("\n=== Exporting Math Model ===")
        export_math()
        print("\n=== Evaluating Math Model ===")
        evaluate_math()

if __name__ == "__main__":
    main()
