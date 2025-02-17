# spl/worker/system_check.py
import shutil
import sys

def check_system_dependencies(gui_mode: bool):
    """
    Check if Docker and NVIDIA Container Toolkit are installed.
    If any dependency is missing, show an error (GUI or CLI) and exit.
    """
    missing = []
    if not shutil.which("docker"):
        missing.append("Docker")
    if not shutil.which("nvidia-container-cli"):
        missing.append("NVIDIA Container Toolkit (nvidia-container-cli)")

    if missing:
        message = "Missing required dependency(ies): " + ", ".join(missing)
        if gui_mode:
            # Display error dialog in GUI mode
            from PyQt5 import QtWidgets
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication(sys.argv)
            QtWidgets.QMessageBox.critical(None, "Missing Dependencies", message)
        else:
            sys.stderr.write(message + "\n")
        sys.exit(1)
