import sys
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QHBoxLayout,
)
from PySide2.QtCore import QThread, Signal, Qt
from PySide2.QtCharts import QtCharts
from PySide2.QtGui import QPainter
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
    progress_signal = Signal(int, float, float)  # epoch, loss, accuracy
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

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
        )

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
                    except Exception:
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

        # Chart Setup
        self.chart = QtCharts.QChart()
        self.chart.setTitle("Training Metrics")
        self.chart.setAnimationOptions(QtCharts.QChart.SeriesAnimations)

        self.loss_series = QtCharts.QLineSeries()
        self.loss_series.setName("Loss")
        self.loss_series.setColor(Qt.blue)
        self.chart.addSeries(self.loss_series)

        self.acc_series = QtCharts.QLineSeries()
        self.acc_series.setName("Accuracy")
        self.acc_series.setColor(Qt.green)
        self.chart.addSeries(self.acc_series)

        # Axes
        self.axis_x = QtCharts.QValueAxis()
        self.axis_x.setTitleText("Epoch")
        self.axis_x.setLabelFormat("%d")
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.loss_series.attachAxis(self.axis_x)
        self.acc_series.attachAxis(self.axis_x)

        self.axis_y_loss = QtCharts.QValueAxis()
        self.axis_y_loss.setTitleText("Loss")
        self.axis_y_loss.setLinePenColor(Qt.blue)
        self.axis_y_loss.setLabelsColor(Qt.blue)
        self.chart.addAxis(self.axis_y_loss, Qt.AlignLeft)
        self.loss_series.attachAxis(self.axis_y_loss)

        self.axis_y_acc = QtCharts.QValueAxis()
        self.axis_y_acc.setTitleText("Accuracy")
        self.axis_y_acc.setLinePenColor(Qt.green)
        self.axis_y_acc.setLabelsColor(Qt.green)
        self.axis_y_acc.setRange(0, 1.05)
        self.chart.addAxis(self.axis_y_acc, Qt.AlignRight)
        self.acc_series.attachAxis(self.axis_y_acc)

        self.chart_view = QtCharts.QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chart_view)

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

    def start_training(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.loss_series.clear()
        self.acc_series.clear()
        self.axis_x.setRange(0, 10)  # Initial range
        self.axis_y_loss.setRange(0, 5)  # Initial range

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
        self.loss_series.append(epoch, loss)
        self.acc_series.append(epoch, acc)

        # Adjust axes
        self.axis_x.setRange(0, epoch + 1)

        # Auto-scale Loss Y-Axis
        current_max = self.axis_y_loss.max()
        if loss > current_max:
            self.axis_y_loss.setRange(0, loss * 1.1)
        elif current_max > 5 and loss < current_max * 0.5:
            # Zoom in if loss drops significantly
            # Find max in recent history ideally, but simple approach:
            pass

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
