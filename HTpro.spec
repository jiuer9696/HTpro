# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['HTpro_ui.py'],
    pathex=[],
    binaries=[],
    # 补充项目数据文件
    datas=[
        ('logo', 'logo'),
        ('halftoning_algorithms', 'halftoning_algorithms'),
        ('analysis', 'analysis'),
        ('PredictiveReversibleHalftoning', 'PredictiveReversibleHalftoning'),
    ],
    # 补充必要的隐藏导入
    hiddenimports=[
        'PyQt5.sip',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.QtCore',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib',
        'numpy',
        'PIL',
        'matplotlib.pyplot',
        'importlib.metadata',
        'pkg_resources'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 排除不需要的库
    excludes=['streamlit'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HTpro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 关键修改：关闭控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo\\logo.ico'],  # 图标路径正确
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HTpro',
)