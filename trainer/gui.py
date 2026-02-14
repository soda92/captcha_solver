import sys
import matplotlib
matplotlib.use('Qt5Agg')
from PySide2.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QLabel, QHBoxLayout, QFrame)
from PySide2.QtCore import QThread, Signal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
from torch.utils.data import DataLoader
import os
import contextlib
from io import StringIO

from trainer.train_math import train_fixed, MODEL_OUT, MathCaptchaDataset, collate_fn
from trainer.export_math import export
from solver.ml_solver import MLSolver

# Stylesheet
STYLESHEET = """
QMainWindow {
    background-color: #f0f0f0;
}
QLabel {
    font-size: 14px;
    color: #333;
}
QPushButton {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    border-radius: 5px;
}
QPushButton:hover {
    background-color: #0056b3;
}
QPushButton:disabled {
    background-color: #cccccc;
}
QPushButton#stopBtn {
    background-color: #dc3545;
}
QPushButton#stopBtn:hover {
    background-color: #a71d2a;
}
"""

class TrainingThread(QThread):
    progress_signal = Signal(int, float, float) # epoch, loss, accuracy
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self):
        super().__init__()
        self.stop_requested = False
        self.solver_helper = MLSolver(model_path="", vocab_type="math")
        self.idx_to_char = self.solver_helper.idx_to_char
        
        # Prepare Test Loader
        test_dir = "num_test_images"
        if not os.path.exists(test_dir):
            test_dir = "num_captchas"
            
        self.test_dataset = MathCaptchaDataset(test_dir, transform=None) 
        from torchvision import transforms
        self.test_transform = transforms.Compose([transforms.ToTensor()])
        self.test_dataset.transform = self.test_transform
        
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    def run(self):
        self.stop_requested = False
        self.log_signal.emit("Starting training...")
        train_fixed(self.progress_callback)
        self.log_signal.emit("Training finished.")
        self.finished_signal.emit()

    def request_stop(self):
        self.stop_requested = True

    def decode_batch(self, preds):
        preds = preds.argmax(dim=2).detach().cpu().numpy()
        batch_size = preds.shape[1]
        results = []
        for b in range(batch_size):
            seq = preds[:, b]
            res = []
            prev = 0
            for idx in seq:
                if idx != prev and idx != 0:
                    res.append(self.idx_to_char[idx])
                prev = idx
            results.append("".join(res))
        return results

    def decode_target(self, target_flat, target_lengths):
        results = []
        offset = 0
        for length in target_lengths:
            indices = target_flat[offset : offset + length].tolist()
            res = "".join([self.idx_to_char[i] for i in indices])
            results.append(res)
            offset += length
        return results

    def calculate_accuracy(self, model):
        model.eval()
        correct = 0
        total = 0
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for images, targets, target_lengths in self.test_loader:
                images = images.to(device)
                preds = model(images)
                pred_strs = self.decode_batch(preds)
                target_strs = self.decode_target(targets, target_lengths)
                
                for pred, target in zip(pred_strs, target_strs):
                    try:
                        p_clean = pred.replace("=", "").replace("?", "")
                        t_clean = target.replace("=", "").replace("?", "")
                        if p_clean == t_clean:
                            correct += 1
                    except:
                        pass
                    total += 1
        
        model.train()
        return correct / total if total > 0 else 0

    def progress_callback(self, epoch, loss, model):
        if self.stop_requested:
            return False
            
        acc = self.calculate_accuracy(model)
        self.progress_signal.emit(epoch, loss, acc)
        self.log_signal.emit(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2%}")
        return True

class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Progress")
        self.resize(900, 700)
        self.setStyleSheet(STYLESHEET)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Plot
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax1 = self.figure.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        self.setup_plot()

        # Log
        self.log_label = QLabel("Ready")
        self.log_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.log_label)

        # Controls (Bottom)
        controls_layout = QHBoxLayout()
        controls_layout.addStretch()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop & Export")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.thread = TrainingThread()
        self.thread.progress_signal.connect(self.update_plot)
        self.thread.log_signal.connect(self.update_log)
        self.thread.finished_signal.connect(self.on_finished)

    def setup_plot(self):
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss", color='b')
        self.ax1.tick_params(axis='y', labelcolor='b')
        
        self.ax2.set_ylabel("Accuracy", color='g')
        self.ax2.tick_params(axis='y', labelcolor='g')
        self.ax2.set_ylim(0, 1.05)
        
        self.line_loss, = self.ax1.plot([], [], 'b-', label="Loss")
        self.line_acc, = self.ax2.plot([], [], 'g-', label="Accuracy")
        
        self.figure.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=self.ax1.transAxes)

    def start_training(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.losses = []
        self.accuracies = []
        self.epochs = []
        
        self.ax1.cla()
        self.ax2.cla()
        # Re-setup because cla() clears everything including labels
        self.ax2 = self.ax1.twinx() # Recreate twinx
        self.setup_plot()
        self.canvas.draw()
        
        self.thread.start()

    def stop_training(self):
        self.log_label.setText("Stopping...")
        self.thread.request_stop()
        self.stop_btn.setEnabled(False)

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_model()

    def export_model(self):
        self.log_label.setText("Exporting ONNX...")
        QApplication.processEvents()
        try:
            with contextlib.redirect_stdout(StringIO()):
                export()
            self.log_label.setText("Export Complete.")
        except Exception as e:
            self.log_label.setText(f"Export Error: {e}")

    def update_plot(self, epoch, loss, acc):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(acc)
        
        self.line_loss.set_data(self.epochs, self.losses)
        self.line_acc.set_data(self.epochs, self.accuracies)
        
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        self.canvas.draw()

    def update_log(self, message):
        self.log_label.setText(message)
        print(message)

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.request_stop()
            self.thread.wait()
        
        # Ensure export happens on close if data exists
        if os.path.exists(MODEL_OUT):
             self.export_model()
             
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
