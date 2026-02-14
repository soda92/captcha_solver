import os
from PySide2.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QApplication,
)
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt, QTimer
from solver.onnx_solver import ONNXSolver
from labeling_tool.session_manager import SessionManager
from typing import Dict


class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Captcha Labeling Tool")
        self.raw_dir = "raw_captchas"
        self.num_dir = "num_captchas"
        self.test_dir = "test_images"
        self.num_test_dir = "num_test_images"
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.exists(self.num_dir):
            os.makedirs(self.num_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.num_test_dir):
            os.makedirs(self.num_test_dir)

        # Session Manager
        self.session_manager = SessionManager()
        self.source_map = self.session_manager.get_sources()

        # Load Models
        self.solvers: Dict[str, ONNXSolver] = {}
        if os.path.exists("model.onnx"):
            try:
                self.solvers["alphanumeric"] = ONNXSolver(
                    "model.onnx", vocab_type="alphanumeric"
                )
                print("Alphanumeric Model loaded.")
            except Exception as e:
                print(f"Error loading alphanumeric model: {e}")

        if os.path.exists("model_math.onnx"):
            try:
                self.solvers["math"] = ONNXSolver("model_math.onnx", vocab_type="math")
                print("Math Model loaded.")
            except Exception as e:
                print(f"Error loading math model: {e}")

        # Current Image Data
        self.current_image_data = None

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Top Controls (Source Selection & Count)
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(self.source_map.keys()))
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        top_layout.addWidget(self.source_combo)

        self.count_label = QLabel("Count: 0")
        top_layout.addWidget(self.count_label)

        layout.addLayout(top_layout)

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
        self.input_field.setMaxLength(10)
        font = self.input_field.font()
        font.setPointSize(20)
        self.input_field.setFont(font)
        self.input_field.returnPressed.connect(self.save_image)
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

        # Restore State
        self.restore_selection()
        self.update_count()

        # Initial Fetch
        QTimer.singleShot(100, self.fetch_image)  # Slight delay to let UI show

    def restore_selection(self):
        last_source = self.session_manager.get_last_source()
        if last_source:
            index = self.source_combo.findText(last_source)
            if index >= 0:
                self.source_combo.setCurrentIndex(index)

    def on_source_changed(self):
        self.session_manager.set_last_source(self.source_combo.currentText())
        self.update_count()
        self.fetch_image()

    def update_count(self):
        source_label = self.source_combo.currentText()
        source_key = self.source_map.get(source_label)
        target_dir = self.session_manager.get_save_dir(source_key)

        count = 0
        if os.path.exists(target_dir):
            count = len(
                [
                    name
                    for name in os.listdir(target_dir)
                    if os.path.isfile(os.path.join(target_dir, name))
                ]
            )

        self.count_label.setText(f"Count: {count}")

    def set_controls_enabled(self, enabled):
        self.fetch_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.input_field.setEnabled(enabled)
        self.source_combo.setEnabled(enabled)
        if enabled:
            self.input_field.setFocus()
            self.input_field.selectAll()

    def fetch_image(self):
        try:
            self.set_controls_enabled(False)
            source_label = self.source_combo.currentText()
            source_key = self.source_map.get(source_label)

            if not source_key:
                # Fallback or empty
                self.set_controls_enabled(True)
                return

            self.status_label.setText(f"Fetching ({source_label})...")
            QApplication.processEvents()

            # Use SessionManager to fetch
            try:
                response = self.session_manager.fetch_captcha(source_key)

                if response.status_code == 200:
                    self.current_image_data = response.content

                    pixmap = QPixmap()
                    pixmap.loadFromData(self.current_image_data)
                    self.image_label.setPixmap(
                        pixmap.scaled(200, 60, Qt.KeepAspectRatio)
                    )

                    self.predict_label(source_key)

                    self.status_label.setText("Fetched. Wait...")
                    QTimer.singleShot(
                        500,
                        lambda: [
                            self.set_controls_enabled(True),
                            self.status_label.setText("Ready."),
                        ],
                    )
                else:
                    self.status_label.setText(f"Error: {response.status_code}")
                    self.set_controls_enabled(True)

            except Exception as e:
                self.status_label.setText(f"Network Error: {e}")
                self.set_controls_enabled(True)

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.set_controls_enabled(True)

    def predict_label(self, source_key):
        solver = self.solvers.get(source_key)
        if not solver or not self.current_image_data:
            self.input_field.clear()
            return

        try:
            temp_path = "temp_labeling.jpeg"
            with open(temp_path, "wb") as f:
                f.write(self.current_image_data)
            pred = solver.solve(temp_path)
            pred = pred.replace("=", "").replace("?", "")
            self.input_field.setText(pred)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Prediction error: {e}")
            self.input_field.clear()

    def save_image(self):
        self.set_controls_enabled(False)
        label = self.input_field.text().strip().upper()

        source_label = self.source_combo.currentText()
        source_key = self.source_map.get(source_label)

        # Basic validation
        if len(label) < 1:
            self.status_label.setText("Label cannot be empty!")
            self.set_controls_enabled(True)
            return

        if not self.current_image_data:
            self.set_controls_enabled(True)
            return

        # Determine target directory
        default_dir = self.session_manager.get_save_dir(source_key)

        if self.save_test_cb.isChecked():
            # If default dir is the math dir, use math test dir
            if default_dir == self.num_dir:
                target_dir = self.num_test_dir
            else:
                target_dir = self.test_dir
        else:
            target_dir = default_dir
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

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
            self.status_label.setText(f"Saved {filename} to {target_dir}. Wait...")
            self.update_count()  # Update count after save
            QTimer.singleShot(500, lambda: self.fetch_image())
        except Exception as e:
            self.status_label.setText(f"Save Error: {e}")
            self.set_controls_enabled(True)
