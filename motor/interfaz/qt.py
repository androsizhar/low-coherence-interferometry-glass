try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise ImportError(
        "Falta PySide6. Con el entorno activado ejecute: "
        "python -m pip install -r requirements.txt  (si falla, use Python 3.12 o 3.13). "
        "El núcleo funciona sin la interfaz."
    ) from exc

Signal = QtCore.Signal

__all__ = ["QtCore", "QtGui", "QtWidgets", "Signal"]
