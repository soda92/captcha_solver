import sys
import matplotlib

matplotlib.use("Qt5Agg")
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QHBoxLayout,
)
from PySide2.QtCore import QThread, Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch

from trainer.train_math import train_fixed, MODEL_OUT
from trainer.export_math import export
from trainer.evaluate_math import evaluate as evaluate_math
from io import StringIO
import contextlib


class TrainingThread(QThread):
    progress_signal = Signal(int, float)
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        self.log_signal.emit("Starting training...")
        train_fixed(self.progress_callback)
        self.log_signal.emit("Training finished.")

    def progress_callback(self, epoch, loss, model):
        self.progress_signal.emit(epoch, loss)

        # Every 2 epochs, export and evaluate
        if epoch % 2 == 0:
            self.log_signal.emit(f"Epoch {epoch}: Exporting and Evaluating...")

            # Save temp state for export
            torch.save(model.state_dict(), MODEL_OUT)

            try:
                # Export to ONNX
                # Capture stdout to avoid clutter
                with contextlib.redirect_stdout(StringIO()):
                    export()

                # Evaluate
                # Capture evaluation output
                capture = StringIO()
                with contextlib.redirect_stdout(capture):
                    evaluate_math()

                output = capture.getvalue()
                # Parse accuracy from output
                # Output format ends with: Accuracy: X/Y (Z%)
                lines = output.strip().split("\n")
                if lines:
                    last_line = lines[-1]
                    if "Accuracy:" in last_line:
                        self.log_signal.emit(f"Epoch {epoch} Eval: {last_line}")
                    else:
                        self.log_signal.emit(f"Epoch {epoch} Eval: Done (See logs)")

            except Exception as e:
                self.log_signal.emit(f"Error during eval: {e}")


class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Progress")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Controls
        controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.start_btn)
        layout.addLayout(controls_layout)

        # Plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.losses = []
        self.epochs = []
        (self.line,) = self.ax.plot([], [], "b-")

        # Log
        self.log_label = QLabel("Ready")
        layout.addWidget(self.log_label)

        self.thread = TrainingThread()
        self.thread.progress_signal.connect(self.update_plot)
        self.thread.log_signal.connect(self.update_log)

    def start_training(self):
        self.start_btn.setEnabled(False)
        self.losses = []
        self.epochs = []
        self.ax.cla()
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        (self.line,) = self.ax.plot([], [], "b-")
        self.canvas.draw()

        self.thread.start()

    def update_plot(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.line.set_data(self.epochs, self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_log(self, message):
        self.log_label.setText(message)
        print(message)  # Also print to console


def main():
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
