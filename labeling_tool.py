import sys
import os
import requests
from datetime import datetime
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
)
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt, QTimer
from solver.onnx_solver import ONNXSolver


class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Captcha Labeling Tool")
        self.raw_dir = "raw_captchas"
        self.test_dir = "test_images"
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # Model
        self.solver = None
        if os.path.exists("model.onnx"):
            try:
                self.solver = ONNXSolver("model.onnx")
                print("Model loaded.")
            except Exception as e:
                print(f"Error loading model: {e}")

        # Current Image Data
        self.current_image_data = None

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image Display
        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 100)
        self.image_label.setStyleSheet("border: 1px solid gray; background: white;")
        layout.addWidget(self.image_label)

        # Input Field
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Label:"))
        self.input_field = QLineEdit()
        self.input_field.setMaxLength(4)
        self.input_field.returnPressed.connect(self.save_image)  # Enter to save
        input_layout.addWidget(self.input_field)
        layout.addLayout(input_layout)

        # Options
        self.save_test_cb = QCheckBox("Save to Test Dir")
        layout.addWidget(self.save_test_cb)

        # Buttons
        btn_layout = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch New (Ctrl+F)")
        self.fetch_btn.setShortcut("Ctrl+F")
        self.fetch_btn.clicked.connect(self.fetch_image)
        btn_layout.addWidget(self.fetch_btn)

        self.save_btn = QPushButton("Save (Enter)")
        self.save_btn.clicked.connect(self.save_image)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Initial Fetch
        self.fetch_image()

    def set_controls_enabled(self, enabled):
        self.fetch_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.input_field.setEnabled(enabled)
        if enabled:
            self.input_field.setFocus()
            self.input_field.selectAll()

    def fetch_image(self):
        try:
            # Disable controls during fetch/wait
            self.set_controls_enabled(False)

            # Generate URL with timestamp
            # Format: Thu Feb 12 2026 15:16:32 GMT+0800 (China Standard Time)
            now = datetime.now()
            # Simple approximation of the format
            date_str = now.strftime(
                "%a %b %d %Y %H:%M:%S GMT+0800 (China Standard Time)"
            )
            url = f"http://192.168.1.230:10085/phis/app/login/voCode?data={date_str}"

            self.status_label.setText("Fetching...")
            QApplication.processEvents()

            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                self.current_image_data = response.content

                # Display Image
                pixmap = QPixmap()
                pixmap.loadFromData(self.current_image_data)
                self.image_label.setPixmap(pixmap.scaled(200, 60, Qt.KeepAspectRatio))

                # Predict
                self.predict_label()
                self.status_label.setText("Fetched. Wait...")

                QTimer.singleShot(
                    7000,
                    lambda: [
                        self.set_controls_enabled(True),
                        self.status_label.setText("Ready."),
                    ],
                )
            else:
                self.status_label.setText(f"Error: {response.status_code}")
                self.set_controls_enabled(True)

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.set_controls_enabled(True)

    def predict_label(self):
        if not self.solver or not self.current_image_data:
            return

        try:
            # Save to temp file for solver
            temp_path = "temp_labeling.jpeg"
            with open(temp_path, "wb") as f:
                f.write(self.current_image_data)

            pred = self.solver.solve(temp_path)
            self.input_field.setText(pred)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            print(f"Prediction error: {e}")

    def save_image(self):
        label = self.input_field.text().strip().upper()
        if len(label) != 4:
            self.status_label.setText("Label must be 4 characters!")
            return

        if not self.current_image_data:
            return

        # Determine target directory
        target_dir = self.test_dir if self.save_test_cb.isChecked() else self.raw_dir

        # Check if file exists, append suffix if needed
        base_name = label
        filename = f"{base_name}.jpeg"
        save_path = os.path.join(target_dir, filename)

        counter = 1
        while os.path.exists(save_path):
            filename = f"{base_name}_{counter}.jpeg"
            save_path = os.path.join(target_dir, filename)
            counter += 1

        try:
            with open(save_path, "wb") as f:
                f.write(self.current_image_data)
            self.status_label.setText(f"Saved {filename} to {target_dir}")
            self.fetch_image()  # Auto fetch next
        except Exception as e:
            self.status_label.setText(f"Save Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingTool()
    window.show()
    sys.exit(app.exec_())
