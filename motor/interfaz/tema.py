CLARO = {
    "fondo": "#f3f6fa",
    "panel": "#ffffff",
    "texto": "#162a42",
    "borde": "#b3c1d1",
    "alterno": "#edf2f7",
}
OSCURO = {
    "fondo": "#121b2b",
    "panel": "#1c293c",
    "texto": "#edf3fb",
    "borde": "#53647a",
    "alterno": "#25344a",
}
ACENTO = "#1688b7"


def hoja_de_estilo(oscuro):
    c = OSCURO if oscuro else CLARO
    return f"""
        QMainWindow, QWidget {{
            background: {c["fondo"]}; color: {c["texto"]}; font-family: Arial; font-size: 13px;
        }}
        QLabel#titulo {{ font-size: 24px; font-weight: bold; }}
        QLabel#estado_resultado {{
            border: 1px solid {c["borde"]}; border-radius: 10px; padding: 3px 12px;
        }}
        QGroupBox {{
            border: 1px solid {c["borde"]}; border-radius: 6px; margin-top: 12px; padding: 10px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        QLineEdit, QPlainTextEdit, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {c["panel"]}; color: {c["texto"]}; border: 1px solid {c["borde"]};
            border-radius: 4px; padding: 5px;
            selection-background-color: #1764a3; selection-color: white;
        }}
        QTableWidget {{ alternate-background-color: {c["alterno"]}; gridline-color: {c["borde"]}; }}
        QHeaderView::section {{
            background: {c["alterno"]}; color: {c["texto"]}; padding: 7px; border: 1px solid {c["borde"]};
        }}
        QPushButton {{
            background: {c["panel"]}; border: 1px solid {c["borde"]}; border-radius: 5px; padding: 8px 12px;
        }}
        QPushButton:hover {{ border: 1px solid {ACENTO}; }}
        QPushButton:disabled {{ color: #8993a1; }}
        QCheckBox::indicator {{
            width: 14px; height: 14px; border: 1px solid {c["borde"]}; border-radius: 3px;
            background: {c["panel"]};
        }}
        QCheckBox::indicator:checked {{ background: {ACENTO}; }}
        QTabBar::tab {{ padding: 11px 22px; border-bottom: 3px solid transparent; }}
        QTabBar::tab:selected {{ border-bottom: 3px solid {ACENTO}; font-weight: bold; }}
        QProgressBar {{ border: 1px solid {c["borde"]}; border-radius: 3px; text-align: center; }}
        QProgressBar::chunk {{ background: {ACENTO}; }}
    """
