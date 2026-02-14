import sys
from PySide2.QtWidgets import QApplication
from labeling_tool.gui import LabelingTool

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingTool()
    window.show()
    sys.exit(app.exec_())
