import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget, \
    QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from default_call import ask_chatgpt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatGPT LLM Interface")
        self.setGeometry(100, 100, 600, 400)

        # 배경색 설정
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        self.setPalette(palette)

        layout = QVBoxLayout()

        # 쿼리 입력 레이블
        self.label = QLabel("Enter your query:")
        self.label.setFont(QFont("Arial", 14))
        layout.addWidget(self.label)

        # 쿼리 입력 필드와 버튼을 수평 레이아웃에 추가
        input_layout = QHBoxLayout()

        # 쿼리 입력 필드
        self.query_input = QLineEdit(self)
        self.query_input.setFont(QFont("Arial", 12))
        self.query_input.setPlaceholderText("Type your question here...")
        input_layout.addWidget(self.query_input)

        # 제출 버튼
        self.submit_button = QPushButton("↑", self)
        self.submit_button.setFont(QFont("Arial", 12))
        self.submit_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 5px; border: none; border-radius: 5px;")
        self.submit_button.clicked.connect(self.on_submit)
        input_layout.addWidget(self.submit_button)

        layout.addLayout(input_layout)  # 수평 레이아웃을 추가

        # 결과 출력 필드
        self.result_output = QTextEdit(self)
        self.result_output.setReadOnly(True)
        self.result_output.setFont(QFont("Arial", 12))
        self.result_output.setStyleSheet(
            "background-color: white; border: 1px solid #ccc; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.result_output)

        # 중앙 위젯 설정
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_submit(self):
        query = self.query_input.text()
        if query:
            completion = ask_chatgpt(query)
            result = completion.choices[0].message.content
            self.result_output.setPlainText(result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
