import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QWidget
)
from PyQt5.QtGui import QPixmap, QImage, QBrush, QPalette, QColor
from PyQt5.QtCore import Qt


def cartoon_to_sketch(image_path):
    """
    Convert a cartoon image to a sketch.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    inverted = 255 - gray
    inv_blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - inv_blurred, scale=256.0)
    return sketch


def color_regions_with_outlines(sketch, min_area=500):
    """
    Detect and color regions, while retaining sketch outlines.
    """
    inverted_sketch = cv2.bitwise_not(sketch)
    binary = cv2.adaptiveThreshold(
        inverted_sketch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width = sketch.shape
    colored_image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            random_color = np.random.randint(0, 256, size=3).tolist()
            cv2.drawContours(colored_image, [contour], -1, random_color, thickness=cv2.FILLED)

    outlines = cv2.bitwise_not(sketch)
    outlines_colored = cv2.cvtColor(outlines, cv2.COLOR_GRAY2BGR)
    final_image = cv2.addWeighted(colored_image, 0.9, outlines_colored, 0.7, 0)
    return final_image


class SketchColorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Sketch and Color Generator")
        self.setGeometry(100, 100, 900, 700)

        # Apply class-range gradient background
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    spread:pad,
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff6f61, stop:0.3 #fddb3a, stop:0.6 #76d7c4, stop:1 #3498db
                );
            }
        """)

        # Main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Upload Button with advanced UI styling
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border-radius: 12px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #27ae60;
            }
        """)
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        # Labels for displaying images
        self.sketch_label = QLabel("Sketch Image")
        self.sketch_label.setAlignment(Qt.AlignCenter)
        self.sketch_label.setStyleSheet("border: 2px dashed #bdc3c7;")
        self.layout.addWidget(self.sketch_label)

        self.color_label = QLabel("Colored Image")
        self.color_label.setAlignment(Qt.AlignCenter)
        self.color_label.setStyleSheet("border: 2px dashed #bdc3c7;")
        self.layout.addWidget(self.color_label)

        # Save Button with modern UI styling
        self.save_button = QPushButton("Save Images")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: #ecf0f1;
                border-radius: 12px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.save_button.clicked.connect(self.save_images)
        self.save_button.setDisabled(True)
        self.layout.addWidget(self.save_button)

        self.original_image = None
        self.sketch_image = None
        self.color_image = None

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.sketch_image = cartoon_to_sketch(file_path)
            self.color_image = color_regions_with_outlines(self.sketch_image)

            self.display_images()

    def display_images(self):
        if self.sketch_image is not None:
            sketch_qimg = self.convert_cv_to_qt(self.sketch_image)
            self.sketch_label.setPixmap(QPixmap.fromImage(sketch_qimg).scaled(
                self.width() // 2, self.height() // 2, Qt.KeepAspectRatio
            ))

        if self.color_image is not None:
            color_qimg = self.convert_cv_to_qt(self.color_image)
            self.color_label.setPixmap(QPixmap.fromImage(color_qimg).scaled(
                self.width() // 2, self.height() // 2, Qt.KeepAspectRatio
            ))

        self.save_button.setDisabled(False)

    def save_images(self):
        if self.sketch_image is not None and self.color_image is not None:
            cv2.imwrite("sketch_output.jpg", self.sketch_image)
            cv2.imwrite("colored_output.jpg", self.color_image)

    def convert_cv_to_qt(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

    def resizeEvent(self, event):
        self.display_images()
        super().resizeEvent(event)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = SketchColorApp()
    main_window.show()
    sys.exit(app.exec_())
