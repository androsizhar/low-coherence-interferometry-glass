try:
    from PySide6 import QtCore, QtGui, QtWidgets

    Signal = QtCore.Signal
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets

        Signal = QtCore.pyqtSignal
    except ImportError as exc:
        raise ImportError(
            "Falta una biblioteca de escritorio. Con el entorno activado ejecute: "
            "python -m pip install -r requirements.txt  (si falla, use Python 3.12 o 3.13). "
            "El núcleo funciona sin la interfaz."
        ) from exc
