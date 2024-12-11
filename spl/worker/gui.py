# spl/worker/gui.py
import sys
import asyncio
import threading
import logging
from PyQt5 import QtWidgets, QtCore
from .gui_config import get_private_key, set_private_key, get_config, update_config
from .logging_config import logger, set_gui_handler
from .main_logic import main
from .config import args
from logging import Handler, LogRecord
import os

from .db_client import have_connected_once, verify_db_connection_and_auth

class LogSignal(QtCore.QObject):
    new_log = QtCore.pyqtSignal(str)

class GuiLogHandler(Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record: LogRecord):
        msg = self.format(record)
        self.signal.new_log.emit(msg)

class LoginDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Private Key")
        layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel("Enter Private Key:")
        self.edit = QtWidgets.QLineEdit()
        self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.button = QtWidgets.QPushButton("Save")
        self.button.clicked.connect(self.save_key)
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def save_key(self):
        private_key = self.edit.text().strip()
        if private_key:
            set_private_key(private_key)
            self.accept()

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        layout = QtWidgets.QFormLayout()

        self.config = get_config()

        self.db_url_edit = QtWidgets.QLineEdit(self.config.get("db_url", ""))
        self.docker_url_edit = QtWidgets.QLineEdit(self.config.get("docker_engine_url", ""))
        # CHANGED: rename from api_key to private_key
        self.private_key_edit = QtWidgets.QLineEdit(self.config.get("private_key", "")) 
        self.private_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.subnet_id_edit = QtWidgets.QLineEdit(str(self.config.get("subnet_id", "")))

        layout.addRow("DB URL:", self.db_url_edit)
        layout.addRow("Docker Engine URL:", self.docker_url_edit)
        layout.addRow("Private Key:", self.private_key_edit)  # CHANGED: was API Key
        layout.addRow("Subnet ID:", self.subnet_id_edit)

        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def save_settings(self):
        new_db_url = self.db_url_edit.text().strip()
        new_docker_url = self.docker_url_edit.text().strip()
        new_private_key = self.private_key_edit.text().strip()
        new_subnet_id = self.subnet_id_edit.text().strip()

        try:
            if new_subnet_id:
                new_subnet_id = int(new_subnet_id)
            else:
                new_subnet_id = None
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Subnet ID", "Subnet ID must be an integer.")
            return

        update_data = {
            "db_url": new_db_url,
            "docker_engine_url": new_docker_url,
            "private_key": new_private_key,
            "subnet_id": new_subnet_id
        }

        update_config({k:v for k,v in update_data.items() if v is not None and v != ""})

        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, log_signal):
        super().__init__()
        self.setWindowTitle("Panthalia Worker")
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.setCentralWidget(self.text_edit)

        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        settings_action = QtWidgets.QAction("Edit Settings", self)
        settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(settings_action)

        self.log_signal = log_signal
        self.log_signal.new_log.connect(self.update_log)

    def update_log(self, msg):
        self.text_edit.append(msg)

    def show_settings(self):
        dlg = SettingsDialog()
        dlg.exec_()

def run_worker_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

def run_gui(args):
    os.environ["DOCKER_ENGINE_URL"] = args.docker_engine_url
    private_key = get_private_key()
    app = QtWidgets.QApplication(sys.argv)

    if not private_key:
        dlg = LoginDialog()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            sys.exit(0)
        private_key = get_private_key()

    loop = asyncio.new_event_loop()
    worker_thread = threading.Thread(target=run_worker_loop, args=(loop,), daemon=True)
    worker_thread.start()

    log_signal = LogSignal()
    gui_handler = GuiLogHandler(log_signal)
    gui_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    gui_handler.setFormatter(formatter)
    set_gui_handler(gui_handler)

    window = MainWindow(log_signal)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
