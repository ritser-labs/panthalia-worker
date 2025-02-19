import shutil
import sys
import platform

def check_system_dependencies(gui_mode: bool):
    """
    Check if Docker and NVIDIA Container Toolkit are installed.
    If any dependency is missing, show an error (GUI or CLI) and exit.
    """
    missing = []

    # Check for Docker
    if not shutil.which("docker"):
        missing.append("Docker")

    # Check for NVIDIA Container Toolkit (nvidia-container-cli) on non-Windows platforms.
    if platform.system() != "Windows":
        if not shutil.which("nvidia-container-cli"):
            missing.append("NVIDIA Container Toolkit (nvidia-container-cli)")
    else:
        # On Windows, consider using an alternative check if needed.
        # For instance, you might run a test container to verify GPU support.
        pass

    if missing:
        message = "Missing required dependency(ies): " + ", ".join(missing)
        if gui_mode:
            try:
                from PyQt5 import QtWidgets
            except ImportError:
                sys.stderr.write("PyQt5 is required for GUI mode.\n")
                sys.exit(1)
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication(sys.argv)
            QtWidgets.QMessageBox.critical(None, "Missing Dependencies", message)
        else:
            sys.stderr.write(message + "\n")
        sys.exit(1)
