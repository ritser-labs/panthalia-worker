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
import os
import signal

# Register global signal handlers in the main (GUI) thread.
def global_shutdown_handler(signum, frame):
    logger.info(f"Received signal {signum}; initiating shutdown (global handler).")
    from .shutdown_flag import set_shutdown_requested
    set_shutdown_requested(True)

signal.signal(signal.SIGINT, global_shutdown_handler)
signal.signal(signal.SIGTERM, global_shutdown_handler)

class LogSignal(QtCore.QObject):
    new_log = QtCore.pyqtSignal(str)

class GuiLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        try:
            msg = self.format(record)
            self.signal.new_log.emit(msg)
        except RuntimeError:
            # The LogSignal has been deleted (likely because the GUI is closing),
            # so simply ignore the log message.
            pass

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
        # Use "private_key" (not "api_key")
        self.private_key_edit = QtWidgets.QLineEdit(self.config.get("private_key", ""))
        self.private_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.subnet_id_edit = QtWidgets.QLineEdit(str(self.config.get("subnet_id", "")))

        layout.addRow("DB URL:", self.db_url_edit)
        layout.addRow("Docker Engine URL:", self.docker_url_edit)
        layout.addRow("Private Key:", self.private_key_edit)
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

        # Save the updated configuration
        update_config({k: v for k, v in update_data.items() if v is not None and v != ""})
        
        # ---- NEW CODE: update the global args and DB adapter ----
        from .config import args, load_config
        config = load_config()
        args.private_key = config.get("private_key")
        args.db_url = config.get("db_url")
        args.docker_engine_url = config.get("docker_engine_url")
        args.subnet_id = config.get("subnet_id")
        from .db_client import db_adapter
        db_adapter.base_url = args.db_url.rstrip('/')
        db_adapter.private_key = args.private_key
        from .logging_config import logger
        logger.info("Configuration updated. New private key and connection parameters applied.")
        # ---- end NEW CODE ----

        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, log_signal):
        super().__init__()
        self.setWindowTitle("Panthalia Worker")
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.setCentralWidget(self.text_edit)
        
        # Optionally add a status bar if you want to show shutdown messages
        self.statusBar().showMessage("")
        
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        settings_action = QtWidgets.QAction("Edit Settings", self)
        settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(settings_action)

        self.log_signal = log_signal
        self.log_signal.new_log.connect(self.update_log)
        self.worker_thread = None  # This will be set later

    def update_log(self, msg):
        self.text_edit.append(msg)

    def show_settings(self):
        dlg = SettingsDialog()
        dlg.exec_()

    # NEW: Override closeEvent to trigger graceful shutdown without closing immediately
    def closeEvent(self, event):
        from .shutdown_flag import set_shutdown_requested
        # Trigger shutdown
        set_shutdown_requested(True)
        # Instead of disabling the entire window, just update the status bar and log.
        self.statusBar().showMessage("Shutdown initiated – please wait until shutdown is complete...")
        self.text_edit.append("Shutdown initiated – waiting for all in-flight tasks to complete...")
        # Ignore the close event so that the window remains visible until the shutdown completes.
        event.ignore()



def run_worker_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

def run_gui(args):
    os.environ["DOCKER_ENGINE_URL"] = args.docker_engine_url
    private_key = get_private_key()
    app = QtWidgets.QApplication(sys.argv)

    # Connect the aboutToQuit signal to trigger shutdown.
    app.aboutToQuit.connect(lambda: __import__("spl.worker.shutdown_flag", fromlist=["set_shutdown_requested"]).set_shutdown_requested(True))

    if not private_key:
        dlg = LoginDialog()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            sys.exit(0)
        private_key = get_private_key()

    loop = asyncio.new_event_loop()
    worker_thread = threading.Thread(target=run_worker_loop, args=(loop,))
    worker_thread.start()

    log_signal = LogSignal()
    gui_handler = GuiLogHandler(log_signal)
    gui_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    gui_handler.setFormatter(formatter)
    set_gui_handler(gui_handler)

    window = MainWindow(log_signal)
    # Attach the worker thread reference to the window (if needed later).
    window.worker_thread = worker_thread
    window.resize(800, 600)
    window.show()

    # --- NEW CODE: QTimer to poll worker thread status ---
    shutdown_timer = QtCore.QTimer()
    shutdown_timer.setInterval(500)  # check every 500 ms

    def check_shutdown():
        # Once the worker thread has finished, quit the application.
        if not worker_thread.is_alive():
            app.quit()

    shutdown_timer.timeout.connect(check_shutdown)
    shutdown_timer.start()
    # --- end NEW CODE ---

    exit_code = app.exec_()
    # At this point, the worker thread should have completed graceful shutdown.
    worker_thread.join()
    sys.exit(exit_code)
