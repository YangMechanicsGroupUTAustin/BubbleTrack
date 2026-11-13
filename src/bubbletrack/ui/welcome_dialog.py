"""First-launch onboarding dialog showing the 3-step workflow."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class WelcomeDialog(QDialog):
    """First-launch tutorial overlay showing the 3-step workflow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to BubbleTrack")
        self.setMinimumSize(500, 350)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("<h2>Welcome to BubbleTrack v3.0</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Step-by-step workflow guide
        steps = [
            (
                "1. Load Images",
                "Select a folder containing your high-speed camera images "
                "(TIFF, PNG, JPG) or open a video file (AVI, MP4, MOV, MKV).",
            ),
            (
                "2. Fit",
                "Auto compares every frame with a bubble-free reference frame: "
                "no threshold to tune. Optionally click the bubble once (Seed), "
                "then press Fit to measure every frame.",
            ),
            (
                "3. Review & Export",
                "Check the frames the Review panel lists, fix the few that need "
                "it, then Export R(t) to MAT, CSV or Excel.",
            ),
        ]
        for step_title, desc in steps:
            step_label = QLabel(f"<b>{step_title}</b><br>{desc}")
            step_label.setWordWrap(True)
            layout.addWidget(step_label)

        layout.addStretch()

        self._dont_show = QCheckBox("Don't show this again")
        layout.addWidget(self._dont_show)

        btn = QPushButton("Get Started")
        btn.setDefault(True)
        btn.clicked.connect(self.accept)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

    @property
    def dont_show_again(self) -> bool:
        """Return True if the user checked 'Don't show this again'."""
        return self._dont_show.isChecked()
