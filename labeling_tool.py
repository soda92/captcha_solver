import sys
import os
import requests
import time
from datetime import datetime
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox)
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import Qt
from io import BytesIO
from solver.onnx_solver import ONNXSolver

class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Captcha Labeling Tool")
        self.raw_dir = "raw_captchas"
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            
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
        self.input_field.returnPressed.connect(self.save_image) # Enter to save
        input_layout.addWidget(self.input_field)
        layout.addLayout(input_layout)
        
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

    def fetch_image(self):
        try:
            # Generate URL with timestamp
            # Format: Thu Feb 12 2026 15:16:32 GMT+0800 (China Standard Time)
            now = datetime.now()
            # Simple approximation of the format
            date_str = now.strftime("%a %b %d %Y %H:%M:%S GMT+0800 (China Standard Time)")
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
                self.status_label.setText("Fetched.")
                self.input_field.setFocus()
                self.input_field.selectAll()
            else:
                self.status_label.setText(f"Error: {response.status_code}")
                
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

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
            
        # Check if file exists, append suffix if needed
        filename = f"{label}.jpeg"
        save_path = os.path.join(self.raw_dir, filename)
        
        counter = 1
        while os.path.exists(save_path):
            # Check if identical? Or just skip?
            # User might be correcting a label, or adding duplicate.
            # We assume adding new data.
            # Actually, raw_captchas usually relies on unique filenames for unique labels?
            # If "ABCD.jpeg" exists, and we have a NEW image for "ABCD".
            # We should probably rename the old one or name this one "ABCD_1.jpeg"?
            # My training script uses `f.stem.upper()`. "ABCD_1" -> "ABCD_1" -> len!=4.
            # So naming MUST be exactly 4 chars?
            # NO. `label = f.stem.upper()`.
            # `if len(label) != 4: continue`.
            # So "ABCD_1.jpeg" would be SKIPPED by training script!
            
            # I must rename to something that parses to 4 chars?
            # Or modify training script to handle suffixes.
            # Wait, `raw_captchas` usually contains `ABCD.jpeg`.
            # If I have multiple ABCD, I can't name them all `ABCD.jpeg`.
            
            # Current `train_model.py` logic:
            # `label = f.stem.upper()`
            # `if len(label) != 4: continue`
            
            # This is a limitation of the current training script.
            # It only supports ONE image per label?
            # Or filenames like `ABCD.jpeg`?
            # If I have `raw_captchas/2cb8.jpeg` and I add another `2cb8`.
            # I can't.
            
            # I should verify this.
            # `raw_captchas` has `2a42.jpeg`, `2dhp.jpeg`...
            # It seems they are unique captchas.
            
            # If I encounter a duplicate label (e.g. `2CB8` again), I can't save it with the current naming convention?
            # Unless the label IS the captcha text.
            # Captcha text is usually random. Duplicate text is rare (1/36^4).
            # So overwriting or skipping might be acceptable?
            # Or I should update `train_model.py` to handle `ABCD_1.jpeg` -> `ABCD`.
            pass
            
            # For now, I will assume unique labels.
            # If collision, I'll warn.
            if os.path.exists(save_path):
                 reply = QMessageBox.question(self, "Overwrite?", 
                                            f"File {filename} exists. Overwrite?", 
                                            QMessageBox.Yes | QMessageBox.No)
                 if reply == QMessageBox.No:
                     return
                 break # Proceed to overwrite
        
        try:
            with open(save_path, "wb") as f:
                f.write(self.current_image_data)
            self.status_label.setText(f"Saved {filename}")
            self.fetch_image() # Auto fetch next
        except Exception as e:
            self.status_label.setText(f"Save Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingTool()
    window.show()
    sys.exit(app.exec_())
