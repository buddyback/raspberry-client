#!/usr/bin/env python3
"""
PyQt6 Test Application
A simple test file demonstrating PyQt6 functionality.
"""
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QHBoxLayout
)
from PyQt6.QtCore import Qt


class TestWindow(QMainWindow):
    """Main application window for PyQt testing."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Test Application")
        self.setGeometry(100, 100, 400, 300)  # x, y, width, height

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Add a welcome label
        self.welcome_label = QLabel("Welcome to PyQt6 Test")
        self.welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.welcome_label.setStyleSheet("font-size: 18px; margin-bottom: 15px;")
        self.main_layout.addWidget(self.welcome_label)

        # Add a text input with label
        self.input_layout = QHBoxLayout()
        self.input_label = QLabel("Enter text:")
        self.text_input = QLineEdit()
        self.input_layout.addWidget(self.input_label)
        self.input_layout.addWidget(self.text_input)
        self.main_layout.addLayout(self.input_layout)

        # Add a display label to show input text
        self.display_label = QLabel("Text will appear here")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet("font-style: italic; margin: 10px;")
        self.main_layout.addWidget(self.display_label)

        # Add buttons
        self.update_button = QPushButton("Update Text")
        self.update_button.clicked.connect(self.update_text)
        self.main_layout.addWidget(self.update_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_text)
        self.main_layout.addWidget(self.clear_button)

        # Add a status label at the bottom
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.main_layout.addWidget(self.status_label)

    def update_text(self):
        """Update the display label with text from input"""
        text = self.text_input.text()
        if text:
            self.display_label.setText(text)
            self.status_label.setText("Text updated")
        else:
            self.status_label.setText("Please enter some text")

    def clear_text(self):
        """Clear the input and display"""
        self.text_input.clear()
        self.display_label.setText("Text will appear here")
        self.status_label.setText("Cleared")


def main():
    """Main function to run the test application"""
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()