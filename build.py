import os
import sys
import subprocess
from pathlib import Path


def build_app(debug=False):
    """打包HTpro应用程序"""
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()

    # 切换到项目目录
    os.chdir(project_root)

    # 确保logo目录存在
    logo_dir = project_root / "logo"
    if not logo_dir.exists():
        logo_dir.mkdir(exist_ok=True)
        print("警告: logo目录不存在，请确保图标文件已准备就绪")

    # 构建PyInstaller命令
    cmd = [
        'pyinstaller',
        '--name=HTpro',
        '--onedir',  # 生成文件夹形式的分发包
        '--clean',  # 清理临时文件
        '--noconfirm'  # 不询问覆盖
    ]

    # 根据调试模式决定是否显示控制台
    if debug:
        print("注意：使用调试模式，将显示控制台窗口")
        cmd.append('--console')  # 调试模式显示控制台
    else:
        cmd.append('--windowed')  # 正常模式不显示控制台

    # 添加图标（如果存在）
    icon_path = logo_dir / "logo.ico"
    if icon_path.exists():
        cmd.append(f'--icon={icon_path}')
    else:
        print("警告: 未找到logo/logo.ico图标文件")

    # 添加数据文件和目录
    data_dirs = [
        'logo',
        'halftoning_algorithms',
        'analysis',
        'PredictiveReversibleHalftoning'
    ]
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            cmd.append(f'--add-data={data_dir}{os.pathsep}{data_dir}')
        else:
            print(f"警告: 数据目录 {data_dir} 不存在")

    # 添加streamlit数据文件（如果存在）
    try:
        import streamlit
        streamlit_path = Path(streamlit.__file__).parent
        if streamlit_path.exists():
            cmd.append(f'--add-data={str(streamlit_path)}{os.pathsep}streamlit')
            print(f"已添加streamlit资源: {streamlit_path}")
        else:
            print("警告: 未找到streamlit安装目录")
    except ImportError:
        print("警告: 未安装streamlit模块")

    # 添加隐藏导入
    hidden_imports = [
        # PyQt5相关
        'PyQt5.sip',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.QtCore',

        # 数据处理相关
        'numpy',
        'PIL',

        # 可视化相关
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_qt5agg',

        # streamlit相关
        'streamlit',
        'streamlit.web.server',
        'streamlit.web.server.routes',
        'streamlit.web.server.server',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.state',

        # 其他可能需要的模块
        'importlib.metadata',
        'pkg_resources'
    ]

    for hidden_import in hidden_imports:
        cmd.append(f'--hidden-import={hidden_import}')

    # 添加主程序文件
    cmd.append('HTpro_ui.py')

    print("执行打包命令:")
    print(' '.join(cmd))
    print("-" * 50)

    try:
        # 执行打包命令
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("标准输出:")
        print(result.stdout)
        print("打包完成！可执行文件位于 dist/HTpro/ 目录下")

    except subprocess.CalledProcessError as e:
        print("打包过程中出现错误:")
        print("错误输出:")
        print(e.stderr)
        print("标准输出:")
        print(e.stdout)
        sys.exit(1)
    except FileNotFoundError:
        print("错误: 未找到pyinstaller命令，请确保已安装PyInstaller")
        print("安装命令: pip install pyinstaller")
        sys.exit(1)


def create_spec_file():
    """创建spec文件用于更高级的配置"""
    # 尝试获取streamlit安装路径
    streamlit_path = ""
    try:
        import streamlit
        streamlit_path = str(Path(streamlit.__file__).parent)
    except ImportError:
        pass

    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['HTpro_ui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('logo', 'logo'),
        ('halftoning_algorithms', 'halftoning_algorithms'),
        ('analysis', 'analysis'),
        ('PredictiveReversibleHalftoning', 'PredictiveReversibleHalftoning'),
        {f"('{streamlit_path}', 'streamlit')," if streamlit_path else ""}
    ],
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
        'streamlit',
        'streamlit.web.server',
        'streamlit.web.server.routes',
        'streamlit.web.server.server',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.state',
        'importlib.metadata',
        'pkg_resources'
    ],
    hookspath=[],
    hooksconfig=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HTpro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo/logo.ico'  # 可选: 指定图标文件路径
)
'''

    with open('HTpro.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)

    print("已创建HTpro.spec文件，可用于高级配置打包")


if __name__ == '__main__':
    print("HTpro 应用程序打包工具")
    print("=" * 50)

    # 检查命令行参数
    debug_mode = '--debug' in sys.argv
    create_spec = '--spec' in sys.argv

    if create_spec:
        create_spec_file()
        print("使用方法:")
        print("  pyinstaller HTpro.spec  # 使用spec文件打包")
    else:
        print(f"正在开始打包...{'（调试模式）' if debug_mode else ''}")
        build_app(debug=debug_mode)
        print("\n提示:")
        print("  如需创建spec配置文件，请运行: python build.py --spec")
        print("  如需调试模式打包（显示控制台），请运行: python build.py --debug")
