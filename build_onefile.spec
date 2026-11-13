# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for BubbleTrack.

Cross-platform:
  - Windows: onefile .exe  (all binaries packed into a single executable)
  - macOS:   onedir .app   (proper .app bundle — fast launch, no temp extraction)
"""

import os
import sys

block_cipher = None

ROOT = os.path.abspath('.')
SRC = os.path.join(ROOT, 'src', 'bubbletrack')
IS_MAC = sys.platform == 'darwin'

# --- Icon: .ico for Windows, .icns for macOS ---
icon_ico = os.path.join(SRC, 'resources', 'icon.ico')
icon_icns = os.path.join(SRC, 'resources', 'icon.icns')

if IS_MAC and os.path.exists(icon_icns):
    icon_file = icon_icns
elif os.path.exists(icon_ico):
    icon_file = icon_ico
else:
    icon_file = None

a = Analysis(
    [os.path.join(SRC, 'app.py')],
    pathex=[os.path.join(ROOT, 'src')],
    binaries=[],
    datas=[
        (os.path.join(SRC, 'resources', 'style.qss'), os.path.join('bubbletrack', 'resources')),
        (os.path.join(SRC, 'resources', 'icon.ico'), os.path.join('bubbletrack', 'resources')),
    ],
    hiddenimports=[
        # UI
        'bubbletrack.ui.main_window',
        'bubbletrack.ui.header_bar',
        'bubbletrack.ui.status_bar',
        'bubbletrack.ui.left_panel',
        'bubbletrack.ui.image_panel',
        'bubbletrack.ui.image_source',
        'bubbletrack.ui.tab_bar',
        'bubbletrack.ui.measure_panel',
        'bubbletrack.ui.classic_controls',
        'bubbletrack.ui.edge_options',
        'bubbletrack.ui.fit_controls',
        'bubbletrack.ui.manual_tab',
        'bubbletrack.ui.export_dialog',
        'bubbletrack.ui.settings_dialog',
        'bubbletrack.ui.radius_chart',
        'bubbletrack.ui.frame_scrubber',
        'bubbletrack.ui.widgets',
        'bubbletrack.ui.shortcuts',
        'bubbletrack.ui.image_compare',
        'bubbletrack.ui.welcome_dialog',
        # Controller
        'bubbletrack.controller.base',
        'bubbletrack.controller.controller',
        'bubbletrack.controller.file_controller',
        'bubbletrack.controller.pretune_controller',
        'bubbletrack.controller.manual_controller',
        'bubbletrack.controller.auto_controller',
        'bubbletrack.controller.export_controller',
        'bubbletrack.controller.display_mixin',
        'bubbletrack.controller.worker',
        'bubbletrack.controller.autotune_worker',
        'bubbletrack.controller.settings_controller',
        # Model
        'bubbletrack.model.anomaly',
        'bubbletrack.model.auto_optimize',
        'bubbletrack.model.cache',
        'bubbletrack.model.circle_fit',
        'bubbletrack.model.config',
        'bubbletrack.model.constants',
        'bubbletrack.model.conventions',
        'bubbletrack.model.detection',
        'bubbletrack.model.export',
        'bubbletrack.model.image_io',
        'bubbletrack.model.presets',
        'bubbletrack.model.removing_factor',
        'bubbletrack.model.report',
        'bubbletrack.model.session',
        'bubbletrack.model.state',
        'bubbletrack.model.undo',
        'bubbletrack.model.autotune',
        'bubbletrack.model.quality',
        # Core
        'bubbletrack.event_bus',
        'bubbletrack.logging_config',
        # Third-party
        'pyqtgraph',
        'openpyxl',
        'fpdf',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt5", "PySide2", "PySide6",
        "PyQt6.QtWebEngine", "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets",
        "PyQt6.Qt3D", "PyQt6.QtBluetooth", "PyQt6.QtMultimedia",
        "PyQt6.QtNetwork", "PyQt6.QtSql", "PyQt6.QtTest",
        "PyQt6.QtQml", "PyQt6.QtQuick",
        "matplotlib",
        "IPython", "sphinx", "docutils", "nbformat", "dask",
        "black", "yapf_third_party", "astroid",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if IS_MAC:
    # ---- macOS: onedir mode → .app bundle (fast launch) ---- #
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name='BubbleTrack',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=icon_file,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name='BubbleTrack',
    )

    app = BUNDLE(
        coll,
        name='BubbleTrack.app',
        icon=icon_icns if os.path.exists(icon_icns) else None,
        bundle_identifier='com.utaustin.bubbletrack',
        info_plist={
            'CFBundleShortVersionString': '3.0.0',
            'NSHighResolutionCapable': True,
        },
    )
else:
    # ---- Windows: onefile .exe ---- #
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='BubbleTrack',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        console=False,
        icon=icon_file,
    )
