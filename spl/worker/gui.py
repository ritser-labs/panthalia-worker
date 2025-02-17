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
    update_target_price_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        layout = QtWidgets.QFormLayout()

        self.config = get_config()

        self.db_url_edit = QtWidgets.QLineEdit(self.config.get("db_url", ""))
        self.docker_url_edit = QtWidgets.QLineEdit(self.config.get("docker_engine_url", ""))
        self.private_key_edit = QtWidgets.QLineEdit(self.config.get("private_key", ""))
        self.private_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.subnet_id_edit = QtWidgets.QLineEdit(str(self.config.get("subnet_id", "")))
        self.limit_price_edit = QtWidgets.QLineEdit(str(self.config.get("limit_price", "")))

        # Initially display a placeholder.
        self.target_price_label = QtWidgets.QLabel("Fetching...")

        layout.addRow("DB URL:", self.db_url_edit)
        layout.addRow("Docker Engine URL:", self.docker_url_edit)
        layout.addRow("Private Key:", self.private_key_edit)
        layout.addRow("Subnet ID:", self.subnet_id_edit)
        layout.addRow("Limit Price:", self.limit_price_edit)
        layout.addRow("Reference Target Price:", self.target_price_label)

        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

        self.update_target_price_signal.connect(self.set_target_price)
        self.fetch_and_update_target_price()

    def set_target_price(self, price):
        self.target_price_label.setText(str(price))

    def fetch_and_update_target_price(self):
        try:
            future = asyncio.run_coroutine_threadsafe(self.fetch_target_price(), worker_loop)
            def done_callback(fut):
                try:
                    result = fut.result()
                except Exception:
                    result = "Error"
                self.update_target_price_signal.emit(str(result))
            future.add_done_callback(done_callback)
        except Exception as e:
            self.target_price_label.setText("Error fetching target price")

    async def fetch_target_price(self):
        from .db_client import db_adapter
        from ..models.schema import DOLLAR_AMOUNT
        subnet_db = await db_adapter.get_subnet(args.subnet_id)
        target_price = getattr(subnet_db, 'target_price', None)
        if target_price is not None:
            # Convert the stored integer value into dollars for display purposes.
            return target_price / DOLLAR_AMOUNT
        return "N/A"

    def save_settings(self):
        new_db_url = self.db_url_edit.text().strip()
        new_docker_url = self.docker_url_edit.text().strip()
        new_private_key = self.private_key_edit.text().strip()
        new_subnet_id = self.subnet_id_edit.text().strip()
        new_limit_price = self.limit_price_edit.text().strip()

        try:
            new_subnet_id = int(new_subnet_id) if new_subnet_id else None
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Subnet ID", "Subnet ID must be an integer.")
            return

        # Force the user to provide a limit price.
        if not new_limit_price:
            QtWidgets.QMessageBox.warning(self, "Missing Limit Price", "Limit Price is required.")
            return

        try:
            # Convert input from dollars to float first
            new_limit_price_float = float(new_limit_price)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Limit Price", "Limit Price must be a number.")
            return

        # Multiply by DOLLAR_AMOUNT to convert dollars to the scaled integer
        from ..models.schema import DOLLAR_AMOUNT
        new_limit_price_int = int(new_limit_price_float * DOLLAR_AMOUNT)

        update_data = {
            "db_url": new_db_url,
            "docker_engine_url": new_docker_url,
            "private_key": new_private_key,
            "subnet_id": new_subnet_id,
            "limit_price": new_limit_price_int  # store as integer
        }

        update_config({k: v for k, v in update_data.items() if v is not None and v != ""})

        from .config import args, load_config
        config = load_config()
        args.private_key = config.get("private_key")
        args.db_url = config.get("db_url")
        args.docker_engine_url = config.get("docker_engine_url")
        args.subnet_id = config.get("subnet_id")
        args.limit_price = config.get("limit_price")
        from .db_client import db_adapter
        db_adapter.base_url = args.db_url.rstrip('/')
        db_adapter.private_key = args.private_key
        from .logging_config import logger
        logger.info("Configuration updated. New private key and connection parameters applied.")
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

        # Track whether graceful shutdown has been initiated
        self.shutdown_initiated = False

    def update_log(self, msg):
        self.text_edit.append(msg)

    def show_settings(self):
        dlg = SettingsDialog()
        dlg.exec_()

    def closeEvent(self, event):
        from .shutdown_flag import set_shutdown_requested

        if not self.shutdown_initiated:
            # FIRST ATTEMPT: Inform the user about the graceful shutdown.
            info_box = QtWidgets.QMessageBox(self)
            info_box.setIcon(QtWidgets.QMessageBox.Information)
            info_box.setWindowTitle("Graceful Shutdown Initiated")
            info_box.setText(
                "A graceful shutdown attempt is being initiated.\n\n"
                "The worker will try to complete all in-flight tasks before closing."
            )
            info_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            info_box.exec_()

            # Now initiate graceful shutdown.
            self.shutdown_initiated = True
            set_shutdown_requested(True)
            self.statusBar().showMessage("Graceful shutdown initiated – waiting for all in-flight tasks to complete...")
            self.text_edit.append("Graceful shutdown initiated – waiting for all in-flight tasks to complete...")
            event.ignore()
        else:
            # SECOND ATTEMPT: Confirm forceful exit of the entire process.
            reply = QtWidgets.QMessageBox.question(
                self,
                "Force Close Worker",
                "Force closing will immediately terminate the program. This may cause you to fail assigned tasks.\n\nAre you sure you want to forcefully close the worker?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                # Forcefully terminate the entire process.
                os._exit(0)
            else:
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
        from .gui import LoginDialog  # Import here if needed.
        dlg = LoginDialog()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            sys.exit(0)
        private_key = get_private_key()

    # Create a new asyncio event loop for the worker and store it in a global variable.
    loop = asyncio.new_event_loop()
    global worker_loop
    worker_loop = loop
    worker_thread = threading.Thread(target=run_worker_loop, args=(loop,))
    worker_thread.start()

    # Setup GUI logging.
    from .gui import LogSignal, GuiLogHandler, MainWindow  # Ensure these are imported appropriately.
    log_signal = LogSignal()
    gui_handler = GuiLogHandler(log_signal)
    gui_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    gui_handler.setFormatter(formatter)
    set_gui_handler(gui_handler)

    window = MainWindow(log_signal)
    window.worker_thread = worker_thread
    window.resize(800, 600)
    window.show()

    # QTimer to poll the worker thread status.
    shutdown_timer = QtCore.QTimer()
    shutdown_timer.setInterval(500)
    shutdown_timer.timeout.connect(lambda: app.quit() if not worker_thread.is_alive() else None)
    shutdown_timer.start()

    exit_code = app.exec_()
    worker_thread.join()
    sys.exit(exit_code)
