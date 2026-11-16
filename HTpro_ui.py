import io
import numpy as np
from PIL import Image, ImageOps
import sys
import os
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGridLayout, QMessageBox, QFrame, QSizePolicy,
    QCheckBox, QScrollArea, QStackedWidget, QProgressDialog
)
from PyQt5.QtCore import Qt
import logging
import matplotlib.pyplot as plt

# 假设所有算法模块都存在
from halftoning_algorithms.youxu import ordered_dither
from halftoning_algorithms.误差扩散2 import floyd_steinberg_dithering
from halftoning_algorithms.vacmain import vac_dithering
from halftoning_algorithms.bippcca import bippcca_dithering
from halftoning_algorithms.EDODF import edodf_dithering
from halftoning_algorithms.DBS10 import efficient_dbs_with_padding, nasanen_hvs_filter_precise, floyd_steinberg_dither
from halftoning_algorithms.eventonemain import perform_halftoning
from PredictiveReversibleHalftoning.inference_single import run

# 假设所有分析模块都存在
from analysis.R import compute_R_function, plot_R_function, analyze_R_function
from analysis.D3 import compute_D_function, plot_polar_diagram, analyze_directional_pattern
from analysis.RAPSD import calculate_rapsd, plot_rapsd, analyze_rapsd
from analysis.floyA import calculate_rapsd_anisotropy, plot_anisotropy_db, preprocess_image, analyze_anisotropy

# 确保matplotlib中文字体正常显示（使用微软雅黑）
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# 配置日志
logging.basicConfig(
    filename='halftone_analysis.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class HalftoneProcessing(QWidget):
    """半色调处理模块 - 跨系统兼容设计"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.image = None
        self.result = None
        # 定义统一的按钮字体和高度(使用微软雅黑)
        self.button_font = QFont("Microsoft YaHei", 12)
        self.button_height = 40
        # 添加最大显示尺寸设置
        self.max_display_width = 1000
        self.max_display_height = 800
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 操作区域 - 卡片式设计
        operate_frame = QFrame()
        operate_frame.setStyleSheet("""
            QFrame {
                background-color: #E0ECFA;  
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                padding: 15px;
            }
        """)

        operate_layout = QHBoxLayout(operate_frame)
        operate_layout.setContentsMargins(10, 10, 15, 10)
        operate_layout.setSpacing(10)
        # 设置布局伸缩因子，提高适应性
        operate_layout.setStretch(0, 1)
        operate_layout.setStretch(1, 3)
        operate_layout.setStretch(2, 2)

        algo_label = QLabel("半色调算法：")
        # 使用微软雅黑字体
        font = QFont("Microsoft YaHei", 14, QFont.Bold)
        algo_label.setFont(font)
        algo_label.setMinimumWidth(100)
        algo_label.setMaximumWidth(180)
        algo_label.setStyleSheet("text-overflow: ellipsis;")
        operate_layout.addWidget(algo_label)

        self.algo_combo = QComboBox()
        self.algo_combo.setFont(QFont("Microsoft YaHei", 12))
        self.algo_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 12px;
                background-color: white;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #C0C0C0;
            }
            QComboBox::drop-down {
                border-left: 1px solid #E0E0E0;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: url(:/icons/down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #E8F3FF;
                selection-color: #165DFF;
            }
        """)

        self.algo_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        algorithms = [
            '请选择', '有序抖动', '误差扩散', 'VAC算法',
            'BIPPCCA算法', 'ETS算法', 'EDODF算法',
            'DBS算法', 'CNN-HalftoneNet算法'
        ]
        self.algo_combo.addItems(algorithms)
        self.algo_combo.setCurrentIndex(0)
        self.algo_combo.currentIndexChanged.connect(self.on_algo_changed)
        operate_layout.addWidget(self.algo_combo)

        # 按钮区域 - 右对齐
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.setAlignment(Qt.AlignRight)

        self.select_img_btn = QPushButton("选择图片", self)
        self.select_img_btn.setFont(self.button_font)
        self.select_img_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333333;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #F5F5F5;
                border-color: #C0C0C0;
            }
            QPushButton:pressed {
                background-color: #E8E8E8;
            }
        """)
        self.select_img_btn.setFixedHeight(self.button_height)
        self.select_img_btn.clicked.connect(self.open_image)
        button_layout.addWidget(self.select_img_btn)

        self.process_btn = QPushButton("处理图片", self)
        self.process_btn.setFont(self.button_font)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4080FF;
            }
            QPushButton:pressed {
                background-color: #0E42D2;
            }
            QPushButton:disabled {
                background-color: #C9CDD4;
                color: #FFFFFF;
                cursor: not-allowed;
            }
        """)
        self.process_btn.setFixedHeight(self.button_height)
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)

        operate_layout.addLayout(button_layout)
        main_layout.addWidget(operate_frame)

        # 图像显示区域 - 卡片式设计
        orig_frame = QFrame()
        orig_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                padding: 15px;
            }
        """)
        orig_layout = QVBoxLayout(orig_frame)
        orig_layout.setContentsMargins(10, 10, 10, 10)
        orig_layout.setSpacing(15)

        # 添加带logo的标题
        orig_title_container = QFrame()
        orig_title_container.setStyleSheet("background-color: transparent;")
        orig_title_layout = QHBoxLayout(orig_title_container)
        orig_title_layout.setContentsMargins(0, 0, 0, 0)
        orig_title_layout.setSpacing(10)

        # 添加logo
        logo_pixmap = QPixmap("logo/tu.png")  # 假设logo文件路径
        if not logo_pixmap.isNull():
            # 调整logo大小
            logo_pixmap = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap)
            orig_title_layout.addWidget(logo_label)

        # 标题文字 - 使用微软雅黑
        orig_label = QLabel("原图/半色调图")
        font = QFont("Microsoft YaHei", 14, QFont.Bold)
        orig_label.setFont(font)
        orig_label.setStyleSheet(
            "color: #333333; padding-bottom: 5px; border-bottom: 1px solid #F0F0F0; text-overflow: ellipsis;")
        orig_title_layout.addWidget(orig_label)
        orig_title_layout.addStretch()  # 推到左侧

        orig_layout.addWidget(orig_title_container)

        # 图像显示滚动区域
        self.orig_scroll = QScrollArea()
        self.orig_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #F0F0F0;
                border-radius: 8px;
                background-color: #FAFAFA;
            }
            QScrollBar:vertical {
                border: none;
                background: #FAFAFA;
                width: 10px;
                margin: 5px 0;
            }
            QScrollBar::handle:vertical {
                background: #DDDDDD;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #BBBBBB;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #FAFAFA;
                height: 10px;
                margin: 0 5px;
            }
            QScrollBar::handle:horizontal {
                background: #DDDDDD;
                min-width: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #BBBBBB;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        self.orig_scroll.setWidgetResizable(True)
        self.orig_scroll.setMinimumSize(800, 530)
        orig_layout.addWidget(self.orig_scroll)

        # 图像居中容器
        center_container = QFrame()
        center_container.setStyleSheet("background-color: transparent;")
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setAlignment(Qt.AlignCenter)

        self.orig_img = QLabel()
        self.orig_img.setStyleSheet("background-color: transparent;")
        self.orig_img.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.orig_img)
        self.orig_scroll.setWidget(center_container)

        # 保存按钮布局
        orig_btn_layout = QHBoxLayout()
        orig_btn_layout.addStretch()
        self.save_img_btn = QPushButton("保存效果图")
        self.save_img_btn.setFont(self.button_font)
        self.save_img_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333333;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #F5F5F5;
                border-color: #C0C0C0;
            }
            QPushButton:pressed {
                background-color: #E8E8E8;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #B0B0B0;
                border: 1px solid #E0E0E0;
                cursor: not-allowed;
            }
        """)
        self.save_img_btn.setFixedHeight(self.button_height)
        self.save_img_btn.clicked.connect(self.save_image_result)
        self.save_img_btn.setEnabled(False)
        orig_btn_layout.addWidget(self.save_img_btn)
        orig_layout.addLayout(orig_btn_layout)

        main_layout.addWidget(orig_frame)
        main_layout.addStretch()

    def on_algo_changed(self, index):
        self.process_btn.setEnabled(self.image is not None and index > 0)

    def open_image(self):
        # 支持TIFF格式图片选择
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.tiff *.tif)"
        )
        if file_path:
            try:
                self.image = Image.open(file_path).convert('L')
                self.image = ImageOps.exif_transpose(self.image)  # 处理图像旋转
                # 显示图片时使用带缩放限制的方式
                self.show_pil_image(self.image, self.orig_img, scale=True)
                self.process_btn.setEnabled(self.algo_combo.currentIndex() > 0)
                self.save_img_btn.setEnabled(False)
            except Exception as e:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("错误")
                msg.setText(f"打开图片失败：{str(e)}")
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()

    def process_image(self):
        if self.image is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("提示")
            msg.setText("请先选择图片")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
            return

        algo = self.algo_combo.currentText()
        if algo == '请选择':
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("提示")
            msg.setText("请先选择半色调算法")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
            return

        try:
            if algo == '有序抖动':
                self.result = ordered_dither(self.image)
            elif algo == '误差扩散':
                self.result = floyd_steinberg_dithering(self.image)
            elif algo == 'VAC算法':
                self.result = vac_dithering(self.image)
            elif algo == 'BIPPCCA算法':
                self.result = bippcca_dithering(self.image)
            elif algo == 'ETS算法':
                image_np = np.array(self.image, dtype=np.uint8)
                result_np = perform_halftoning(image_np, 600, 600, 1)
                self.result = Image.fromarray(result_np)
            elif algo == 'EDODF算法':
                self.result = edodf_dithering(self.image, h=0.5, serpentine=False)
            elif algo == 'DBS算法':
                img_gray = self.image.convert('L')
                f = np.array(img_gray) / 255.0
                hvs_filter = nasanen_hvs_filter_precise(size=31)
                init_g = floyd_steinberg_dither(f)
                g_final = efficient_dbs_with_padding(f, init_g, hvs_filter, max_iters=10)
                self.result = Image.fromarray((g_final * 255).astype(np.uint8))
            elif algo == 'CNN-HalftoneNet算法':
                method = 'ours'
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_input_path = temp_file.name
                    self.image.save(temp_input_path)
                with tempfile.TemporaryDirectory() as temp_output_dir:
                    run(input_path=temp_input_path, output_dir=temp_output_dir, method=method)
                    halftone_path = os.path.join(temp_output_dir, 'output_halftone_q.png')
                    if not os.path.exists(halftone_path):
                        raise FileNotFoundError(f"模型未生成结果文件：{halftone_path}")
                    self.result = Image.open(halftone_path).convert('RGB')
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
            else:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("提示")
                msg.setText("该算法暂未实现")
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()
                return

            if self.result is None:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("警告")
                msg.setText(f"{algo}算法未返回结果")
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()
                return

            # 显示处理结果
            self.show_pil_image(self.result, self.orig_img, scale=True)
            self.save_img_btn.setEnabled(True)

        except Exception as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("处理错误")
            msg.setText(f"应用半色调算法时出错：{str(e)}")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()

    def show_pil_image(self, pil_img, label, scale=True):
        w, h = pil_img.size

        # 计算缩放比例，确保图像不会超过最大显示尺寸
        if scale:
            scale_width = self.max_display_width / w
            scale_height = self.max_display_height / h
            scale_factor = min(scale_width, scale_height, 1.0)  # 不放大图像，只缩小

            # 计算新尺寸
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)

            # 调整图像大小
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        # 转换图像格式并显示
        if pil_img.mode == '1':
            pil_img = pil_img.convert('L')
        if pil_img.mode == 'L':
            qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_Grayscale8)
        elif pil_img.mode == 'RGB':
            qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGB888)
        elif pil_img.mode == 'RGBA':
            qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGBA8888)
        else:
            pil_img = pil_img.convert('RGB')
            qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        label.setFixedSize(w, h)
        label.setAlignment(Qt.AlignCenter)

    def save_image_result(self):
        if self.result is not None:
            # 支持保存为TIFF格式
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存效果图", "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp);;TIFF Files (*.tiff *.tif)"
            )
            if file_path:
                try:
                    # 处理TIFF保存的特殊情况
                    if file_path.lower().endswith(('.tiff', '.tif')):
                        # 对于TIFF格式，确保使用合适的参数
                        self.result.save(file_path, format='TIFF')
                    else:
                        self.result.save(file_path)

                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("保存成功")
                    msg.setText("效果图已成功保存！")
                    msg.setStyleSheet(self.get_message_box_style())
                    msg.exec_()
                except Exception as e:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("保存失败")
                    msg.setText(f"保存效果图时发生错误：{str(e)}")
                    msg.setStyleSheet(self.get_message_box_style())
                    msg.exec_()
        else:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("提示")
            msg.setText("没有可保存的效果图")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()

    def get_message_box_style(self):
        """统一的消息框样式"""
        return """
            QMessageBox {
            background-color: white; 
            color: #333333;
            font-family: "Microsoft YaHei";
            font-size: 18px;
             min-width: 400px;
            min-height: 200px;
        }
        QMessageBox QLabel {
            color: #333333; 
            font-family: "Microsoft YaHei";
            font-size: 18px;
            padding: 10px;
        }
            QPushButton {
                background-color: #ECECEC; 
                color: #333333;
                border: 1px solid #DCDCDC;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 100px;
                font-family: "Microsoft YaHei";
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #DCDCDC; 
                border-color: #CBCBCB;
            }
            QPushButton:pressed {
                background-color: #CBCBCB;
            }
            QMessageBox QWidget {
                background-color: white;
            }
        """


class HalftoneAnalysis(QWidget):
    """半色调质量分析模块 - 跨系统兼容设计"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.analysis_image = None
        self.analysis_image_path = None  # 保存图像路径用于缓存
        self.current_analysis_figs = {}
        self.is_analysis_result = False
        self.analysis_results = {}
        self.analysis_texts = {}
        self.analysis_cache = {}  # 分析结果缓存
        self.button_font = QFont("Microsoft YaHei", 12)
        self.button_height = 40
        self.max_analysis_width = 400
        self.max_analysis_height = 300

        # 启用拖拽功能
        self.setAcceptDrops(True)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 操作区域 - 卡片式设计
        operate_frame = QFrame()
        operate_frame.setStyleSheet("""
            QFrame {
                background-color:#E0ECFA ;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                padding: 15px;
            }
        """)
        operate_layout = QHBoxLayout(operate_frame)
        operate_layout.setContentsMargins(10, 10, 10, 10)
        operate_layout.setSpacing(15)
        # 设置布局伸缩因子
        operate_layout.setStretch(0, 1)
        operate_layout.setStretch(1, 5)
        operate_layout.setStretch(2, 2)

        analysis_label = QLabel("质量分析：")
        # 使用微软雅黑字体
        font = QFont("Microsoft YaHei", 14, QFont.Bold)
        analysis_label.setFont(font)
        analysis_label.setMinimumWidth(100)
        analysis_label.setMaximumWidth(150)
        analysis_label.setStyleSheet("text-overflow: ellipsis;")
        operate_layout.addWidget(analysis_label)

        # 分析选项复选框区域
        checkboxes_frame = QFrame()
        checkboxes_frame.setStyleSheet("background-color: transparent;")
        checkboxes_frame.setMinimumHeight(self.button_height + 40)
        checkboxes_frame.setMaximumHeight(self.button_height + 60)
        checkboxes_layout = QGridLayout(checkboxes_frame)
        checkboxes_layout.setContentsMargins(0, 0, 0, 20)
        checkboxes_layout.setSpacing(20)
        checkboxes_layout.setColumnStretch(0, 1)
        checkboxes_layout.setColumnStretch(1, 1)
        checkboxes_layout.setAlignment(Qt.AlignCenter)

        self.analysis_checkboxes = {}
        analysis_types = [
            '空间关联函数 R(r)',
            '方向分布函数 D(R₁R₂)',
            '径向平均功率谱密度P(f)',
            '各向异性 A(f)'
        ]

        for idx, analysis_type in enumerate(analysis_types):
            checkbox = QCheckBox(analysis_type)
            checkbox.setFont(QFont("Microsoft YaHei", 11))
            checkbox.setFixedHeight(40)
            checkbox.setMinimumWidth(200)
            checkbox.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Preferred
            )
            checkbox.setChecked(True)
            checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                color: #333;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px; 
                height: 18px;
                border: 1px solid #165DFF;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #165DFF;
            }
            QCheckBox::indicator:checked::after {
                content: "✔";
                color: white; 
                font-family: "Segoe UI", "FontAwesome";
                font-size: 14px; 
                position: absolute;
                top: 1px; 
                left: 3px;
            }
            """)

            checkbox.stateChanged.connect(self.on_analysis_checkbox_changed)
            self.analysis_checkboxes[analysis_type] = checkbox

            row = idx // 2
            col = idx % 2
            checkboxes_layout.addWidget(checkbox, row, col, alignment=Qt.AlignLeft)

        checkboxes_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 创建一个包含左边距的布局
        checkboxes_wrapper = QHBoxLayout()
        checkboxes_wrapper.setContentsMargins(20, 0, 0, 0)
        checkboxes_wrapper.addWidget(checkboxes_frame)
        operate_layout.addLayout(checkboxes_wrapper)

        # 分析按钮区域
        analysis_buttons_layout = QHBoxLayout()
        analysis_buttons_layout.setSpacing(12)

        self.select_analysis_img_btn = QPushButton("选择半色调图片", self)
        self.select_analysis_img_btn.setFont(self.button_font)
        self.select_analysis_img_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333333;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #F5F5F5;
                border-color: #C0C0C0;
            }
            QPushButton:pressed {
                background-color: #E8E8E8;
            }
        """)
        self.select_analysis_img_btn.setFixedHeight(self.button_height)
        self.select_analysis_img_btn.clicked.connect(self.open_analysis_image)

        self.analyze_btn = QPushButton("分析图片", self)
        self.analyze_btn.setFont(self.button_font)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 110px;
            }
            QPushButton:hover {
                background-color: #4080FF;
            }
            QPushButton:pressed {
                background-color: #0E42D2;
            }
            QPushButton:disabled {
                background-color: #C9CDD4;
                color: #FFFFFF;
                cursor: not-allowed;
            }
        """)
        self.analyze_btn.setFixedHeight(self.button_height)
        self.analyze_btn.clicked.connect(self.process_analysis)
        self.analyze_btn.setEnabled(False)

        analysis_buttons_layout.addWidget(self.select_analysis_img_btn)
        analysis_buttons_layout.addWidget(self.analyze_btn)
        operate_layout.addLayout(analysis_buttons_layout)

        main_layout.addWidget(operate_frame)

        # 统一的滚动区域
        self.unified_scroll_area = QScrollArea()
        self.unified_scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #F0F0F0;
                border-radius: 8px;
                background-color: #FAFAFA;
            }
            QScrollBar:vertical {
                border: none;
                background: #FAFAFA;
                width: 10px;
                margin: 5px 0;
            }
            QScrollBar::handle:vertical {
                background: #DDDDDD;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #BBBBBB;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #FAFAFA;
                height: 10px;
                margin: 0 5px;
            }
            QScrollBar::handle:horizontal {
                background: #DDDDDD;
                min-width: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #BBBBBB;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        self.unified_scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.unified_scroll_area, stretch=1)

        # 滚动区域内容容器
        scroll_content = QFrame()
        scroll_content.setStyleSheet("background-color: transparent;")
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(15, 15, 15, 15)
        self.scroll_layout.setSpacing(20)
        self.unified_scroll_area.setWidget(scroll_content)

        # 半色调图像显示区域
        image_display_container = QFrame()
        image_display_container.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            padding: 15px;
        """)
        image_display_layout = QVBoxLayout(image_display_container)
        image_display_layout.setContentsMargins(10, 10, 10, 10)
        image_display_layout.setSpacing(10)

        # 添加带logo的半色调图像标题
        halftone_title_container = QFrame()
        halftone_title_container.setStyleSheet("background-color: transparent;")
        halftone_title_layout = QHBoxLayout(halftone_title_container)
        halftone_title_layout.setContentsMargins(0, 0, 0, 0)
        halftone_title_layout.setSpacing(10)

        # 添加logo
        logo_pixmap = QPixmap(resource_path("logo/tu.png"))
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap)
            halftone_title_layout.addWidget(logo_label)

        # 标题文字 - 使用微软雅黑
        self.halftone_title = QLabel("半色调图像")
        font = QFont("Microsoft YaHei", 14, QFont.Bold)
        self.halftone_title.setFont(font)
        self.halftone_title.setStyleSheet(
            "color: #333333; padding-bottom: 5px; border-bottom: 1px solid #F0F0F0; text-overflow: ellipsis;")
        self.halftone_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        halftone_title_layout.addWidget(self.halftone_title)
        halftone_title_layout.addStretch()

        image_display_layout.addWidget(halftone_title_container)

        # 图像显示区域
        image_center_container = QFrame()
        image_center_container.setStyleSheet("background-color: #FAFAFA; border-radius: 6px;")
        image_center_layout = QHBoxLayout(image_center_container)
        image_center_layout.setContentsMargins(0, 0, 0, 0)
        image_center_layout.setAlignment(Qt.AlignCenter)

        # 创建一个容器用于同时容纳图像和提示标签
        image_container = QFrame()
        image_container.setStyleSheet("background-color: transparent;")
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        image_container_layout.setAlignment(Qt.AlignCenter)

        self.orig_img = QLabel()
        self.orig_img.setStyleSheet("background-color: transparent;")
        self.orig_img.setAlignment(Qt.AlignCenter)
        self.orig_img.setVisible(False)  # 初始隐藏

        # 添加提示标签
        self.placeholder_label = QLabel("请上传灰度均匀分布的半色调块\n推荐尺寸：128*128")
        self.placeholder_label.setStyleSheet("color: #AAAAAA; font-size: 14px; font-family: 'Microsoft YaHei';")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setMinimumHeight(100)

        # 将图像标签和提示标签都添加到容器中
        image_container_layout.addWidget(self.orig_img)
        image_container_layout.addWidget(self.placeholder_label)

        image_center_layout.addWidget(image_container)

        image_display_layout.addWidget(image_center_container, stretch=1)
        self.scroll_layout.addWidget(image_display_container)

        # 分析结果容器
        self.results_container = QFrame()
        self.results_container.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            padding: 15px;
        """)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(10, 10, 10, 10)
        self.results_layout.setSpacing(10)

        # 添加带logo的分析结果标题
        analysis_title_container = QFrame()
        analysis_title_container.setStyleSheet("background-color: transparent;")
        analysis_title_layout = QHBoxLayout(analysis_title_container)
        analysis_title_layout.setContentsMargins(0, 0, 0, 0)
        analysis_title_layout.setSpacing(10)

        # 添加logo
        logo_pixmap = QPixmap(resource_path("logo/fenxi.png"))
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap)
            analysis_title_layout.addWidget(logo_label)

        # 标题文字 - 使用微软雅黑
        self.analysis_results_label = QLabel("分析结果")
        font = QFont("Microsoft YaHei", 14, QFont.Bold)
        self.analysis_results_label.setFont(font)
        self.analysis_results_label.setStyleSheet(
            "color: #333333; padding-bottom: 5px; border-bottom: 1px solid #F0F0F0; text-overflow: ellipsis;")
        self.analysis_results_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        analysis_title_layout.addWidget(self.analysis_results_label)
        analysis_title_layout.addStretch()

        self.results_layout.addWidget(analysis_title_container)

        # 分析结果网格容器
        self.results_grid_container = QFrame()
        self.results_grid_container.setStyleSheet("background-color: transparent;")
        self.results_grid_layout = QGridLayout(self.results_grid_container)
        self.results_grid_layout.setSpacing(20)
        self.results_grid_layout.setContentsMargins(5, 5, 5, 5)
        self.results_grid_layout.setColumnStretch(0, 1)
        self.results_grid_layout.setColumnStretch(1, 1)
        self.results_layout.addWidget(self.results_grid_container)

        # 保存按钮布局
        save_btn_layout = QHBoxLayout()
        save_btn_layout.addStretch()
        self.save_analysis_btn = QPushButton("保存分析结果")
        self.save_analysis_btn.setFont(self.button_font)
        self.save_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333333;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #F5F5F5;
                border-color: #C0C0C0;
            }
            QPushButton:pressed {
                background-color: #E8E8E8;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #B0B0B0;
                border: 1px solid #E0E0E0;
                cursor: not-allowed;
            }
        """)
        self.save_analysis_btn.setFixedHeight(self.button_height)
        self.save_analysis_btn.clicked.connect(self.save_analysis_result)
        self.save_analysis_btn.setEnabled(False)

        save_btn_layout.addWidget(self.save_analysis_btn)
        self.results_layout.addLayout(save_btn_layout)

        self.results_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_layout.addWidget(self.results_container, stretch=1)

        self.scroll_layout.addStretch()

    def show_halftone_image(self, pil_img, label, scale=True):
        """半色调图像显示专用方法"""
        try:
            # 恢复原始的最大显示尺寸参数
            max_display_width = 1000
            max_display_height = 800

            w, h = pil_img.size

            # 缩放图像以适应显示区域
            if scale:
                scale_width = max_display_width / w
                scale_height = max_display_height / h
                scale_factor = min(scale_width, scale_height, 1.0)  # 不放大图像，只缩小

                # 计算新尺寸
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)

                # 调整图像大小
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                w, h = new_w, new_h

            # 转换图像格式为QImage
            if pil_img.mode == '1':
                pil_img = pil_img.convert('L')

            if pil_img.mode == 'L':
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_Grayscale8)
            elif pil_img.mode == 'RGB':
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGB888)
            elif pil_img.mode == 'RGBA':
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGBA8888)
            else:
                pil_img = pil_img.convert('RGB')
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGB888)

            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(qimg)
            label.setPixmap(pixmap)
            label.setFixedSize(w, h)
            label.setAlignment(Qt.AlignCenter)
            label.setScaledContents(False)

            # 显示图像，隐藏提示
            label.setVisible(True)
            self.placeholder_label.setVisible(False)

        except Exception as e:
            logging.error(f"半色调图像显示失败：{str(e)}", exc_info=True)
            self.placeholder_label.setText(f"半色调图像显示失败：{str(e)}")
            self.placeholder_label.setVisible(True)
            label.setVisible(False)

    def process_analysis(self):
        if self.analysis_image is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("提示")
            msg.setText("请先选择半色调图片")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
            return

        selected_analyses = [name for name, checkbox in self.analysis_checkboxes.items() if checkbox.isChecked()]
        if not selected_analyses:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("提示")
            msg.setText("请至少选择一种分析类型")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
            return

        try:
            # 清除之前的分析结果
            for i in reversed(range(self.results_grid_layout.count())):
                widget = self.results_grid_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            image_np = np.array(self.analysis_image)

            # 确保图像是二维数组（灰度图）
            if len(image_np.shape) > 2:
                image_np = np.mean(image_np, axis=2).astype(np.uint8)

            # 显示进度对话框
            progress = QProgressDialog("正在分析图像...", "取消", 0, len(selected_analyses), self)
            progress.setWindowTitle("处理中")
            progress.setLabelText("正在分析......")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            # 设置进度对话框的样式
            progress.setStyleSheet("""
                QProgressDialog {
                    background-color: white;
                    border: 1px solid #E0E0E0;
                    border-radius: 8px;
                    font-family: "Microsoft YaHei";
                    font-size: 12px;
                    min-width: 320px;
                    min-height: 150px;
                }
                QProgressDialog QLabel {
                    color: #333333;
                    font-size: 12px;
                    padding: 10px;
                }
                QProgressBar {
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                    text-align: center;
                    height: 12px;
                    margin: 0 10px 15px 10px;
                }
                QProgressBar::chunk {
                    background-color: #165DFF;
                    border-radius: 3px;
                }
                QPushButton {
                    background-color: #ECECEC;
                    color: #333333;
                    border: 1px solid #DCDCDC;
                    border-radius: 6px;
                    padding: 8px 16px;
                    min-width: 100px;
                    font-family: "Microsoft YaHei";
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #DCDCDC;
                    border-color: #CBCBCB;
                }
                QPushButton:pressed {
                    background-color: #CBCBCB;
                }
            """)
            progress.show()

            # 动态生成选中的分析结果
            for idx, analysis in enumerate(selected_analyses):
                # 检查是否取消
                if progress.wasCanceled():
                    self.reset_analysis_results()
                    return
                progress.setValue(idx)

                # 分析结果子容器
                result_container = QFrame()
                result_container.setStyleSheet("""
                    background-color: #F9F9F9;
                    border-radius: 8px;
                    padding: 15px;
                """)
                result_layout = QVBoxLayout(result_container)
                result_layout.setContentsMargins(5, 5, 5, 5)
                result_layout.setSpacing(10)

                result_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

                # 分析标题 - 使用微软雅黑
                title_label = QLabel(analysis)
                title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
                title_label.setAlignment(Qt.AlignCenter)
                title_label.setWordWrap(True)
                title_label.setFixedHeight(50)
                title_label.setStyleSheet("color: #333333; text-overflow: ellipsis;")
                result_layout.addWidget(title_label)

                # 分析图像
                img_label = QLabel()
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setMinimumSize(300, 200)
                img_label.setMaximumSize(self.max_analysis_width, self.max_analysis_height)
                img_label.setStyleSheet("background-color: white; border-radius: 6px;")
                img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                img_label.setScaledContents(True)
                result_layout.addWidget(img_label, alignment=Qt.AlignCenter)
                self.analysis_results[analysis] = img_label

                # 分析文本 - 使用微软雅黑
                text_label = QLabel()
                text_label.setFont(QFont("Microsoft YaHei", 11))
                text_label.setWordWrap(True)
                text_label.setAlignment(Qt.AlignTop)
                text_label.setMinimumHeight(100)
                text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                text_label.setStyleSheet("color: #555555; padding: 5px;")
                result_layout.addWidget(text_label)
                self.analysis_texts[analysis] = text_label

                # 添加到网格布局
                row = idx // 2
                col = idx % 2
                self.results_grid_layout.addWidget(result_container, row, col)

                # 检查缓存
                cache_key = (self.analysis_image_path, analysis)
                if cache_key in self.analysis_cache:
                    # 使用缓存结果
                    fig, text = self.analysis_cache[cache_key]
                    self.current_analysis_figs[analysis] = fig
                    buf = io.BytesIO()
                    fig.savefig(
                        buf,
                        format="png",
                        dpi=150,
                        bbox_inches='tight',
                        pad_inches=0
                    )
                    buf.seek(0)
                    result_image = Image.open(buf)
                    self.show_pil_image(result_image, img_label)
                    text_label.setText(text)
                    continue

                # 执行分析
                fig = None
                text = ""
                try:
                    if analysis == '空间关联函数 R(r)':
                        radii, R_values = compute_R_function(image_np)
                        fig = plot_R_function(radii, R_values)
                        fig.set_size_inches(4, 3)
                        text = analyze_R_function(radii, R_values)
                    elif analysis == '方向分布函数 D(R₁R₂)':
                        # 验证参数有效性
                        r1, r2 = 1, 15
                        if not (0 < r1 < r2):
                            raise ValueError(f"无效的参数 r1={r1}, r2={r2}，需满足 0 < r1 < r2")

                        angles, D_values = compute_D_function(image_np, r1=r1, r2=r2)
                        fig = plot_polar_diagram(angles, D_values, delta_a_deg=10, r1=r1, r2=r2)
                        fig.set_size_inches(4, 3)
                        text = analyze_directional_pattern(angles, D_values, r1=r1, r2=r2)
                    elif analysis == '径向平均功率谱密度P(f)':
                        # 确保输入是二值图像
                        binary_img = (image_np > 127).astype(np.uint8)

                        # 调用计算函数并检查结果
                        f_rho, P_rapsd, counts = calculate_rapsd(binary_img)

                        # 验证数据有效性
                        if f_rho is None or P_rapsd is None or len(f_rho) == 0 or len(P_rapsd) == 0:
                            raise ValueError("RAPSD计算返回空数据")

                        # 生成图像
                        fig = plot_rapsd(f_rho, P_rapsd)
                        fig.set_size_inches(4, 3)

                        # 获取分析文本
                        text_result = analyze_rapsd(f_rho, P_rapsd, counts)
                        text = text_result[0] if isinstance(text_result, tuple) else text_result

                    elif analysis == '各向异性 A(f)':
                        binary_image = preprocess_image(image_np, threshold=None, crop_center=True)
                        f_rho, A_values = calculate_rapsd_anisotropy(
                            binary_image, delta_rho=0.035, f_rho_min=1e-6
                        )
                        fig = plot_anisotropy_db(f_rho, A_values, title="各向异性频谱 (dB)")
                        fig.set_size_inches(4, 3)
                        text_result = analyze_anisotropy(f_rho, A_values)
                        text = text_result[0] if isinstance(text_result, tuple) else text_result

                    # 显示分析结果
                    if fig is not None:
                        self.current_analysis_figs[analysis] = fig
                        buf = io.BytesIO()
                        fig.savefig(
                            buf,
                            format="png",
                            dpi=150,
                            bbox_inches='tight',
                            pad_inches=0.1
                        )
                        buf.seek(0)
                        result_image = Image.open(buf)
                        self.show_pil_image(result_image, img_label)

                        # 缓存结果
                        self.analysis_cache[cache_key] = (fig, text)
                    else:
                        img_label.setText(f"无法生成{analysis}图像")

                    # 处理文本结果
                    if not text:
                        text = f"{analysis}分析完成，但未返回分析文本"
                    text_label.setText(text)

                except Exception as e:
                    error_msg = f"{analysis}分析出错: {str(e)}"
                    img_label.setText(error_msg)
                    text_label.setText(f"错误详情: {str(e)}\n请检查图像是否合适或算法实现是否正确")
                    logging.error(error_msg, exc_info=True)

                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("分析警告")
                    msg.setText(error_msg)
                    msg.setStyleSheet(self.get_message_box_style())
                    msg.exec_()

            # 完成进度
            progress.setValue(len(selected_analyses))
            self.save_analysis_btn.setEnabled(True)
            self.is_analysis_result = True

        except Exception as e:
            logging.error(f"执行图像分析时出错：{str(e)}", exc_info=True)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("分析错误")
            msg.setText(f"执行图像分析时出错：{str(e)}")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()

    def on_analysis_checkbox_changed(self):
        any_checked = any(checkbox.isChecked() for checkbox in self.analysis_checkboxes.values())
        self.analyze_btn.setEnabled(self.analysis_image is not None and any_checked)

    def open_analysis_image(self):
        # 支持TIFF格式图片选择
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择半色调图片", "", "Images (*.png *.jpg *.jpeg *.tiff *.tif)"
        )
        if file_path:
            self._load_image_from_path(file_path)

    def _load_image_from_path(self, file_path):
        """从路径加载图像的通用方法，用于文件选择和拖拽功能"""
        try:
            # 验证文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 尝试打开图像
            self.analysis_image = Image.open(file_path).convert('L')
            self.analysis_image = ImageOps.exif_transpose(self.analysis_image)

            # 验证图像有效性
            if self.analysis_image.size[0] == 0 or self.analysis_image.size[1] == 0:
                raise ValueError("加载的图像为空或无效")

            # 保存图像路径用于缓存
            self.analysis_image_path = file_path

            # 显示图像
            self.show_halftone_image(self.analysis_image, self.orig_img, scale=True)

            # 更新UI状态
            any_checked = any(checkbox.isChecked() for checkbox in self.analysis_checkboxes.values())
            self.analyze_btn.setEnabled(any_checked)
            self.reset_analysis_results()
            self.save_analysis_btn.setEnabled(False)

        except Exception as e:
            logging.error(f"打开图片失败：{str(e)}", exc_info=True)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("错误")
            msg.setText(f"打开图片失败：{str(e)}")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
            self.placeholder_label.show()
            self.orig_img.setVisible(False)
            self.analysis_image = None
            self.analysis_image_path = None

    def reset_analysis_results(self):
        # 清除网格布局中的所有部件
        for i in reversed(range(self.results_grid_layout.count())):
            widget = self.results_grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # 清除缓存和结果
        self.analysis_results.clear()
        self.analysis_texts.clear()

        # 关闭所有matplotlib图表，防止内存泄漏
        plt.close('all')
        self.current_analysis_figs.clear()

        # 更新UI状态
        self.save_analysis_btn.setEnabled(False)
        self.is_analysis_result = False

    def show_pil_image(self, pil_img, label, scale=True):
        """优化的图像显示方法，确保图像正确显示"""
        try:
            w, h = pil_img.size

            # 缩放图像以适应显示区域
            if scale:
                # 始终计算缩放因子，确保填充容器空间同时保持比例
                scale_factor = min(self.max_analysis_width / w, self.max_analysis_height / h)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)

                # 确保缩放后尺寸至少为容器的95%，避免过小
                if new_w < self.max_analysis_width * 0.95 or new_h < self.max_analysis_height * 0.95:
                    scale_factor = min(self.max_analysis_width * 0.95 / w, self.max_analysis_height * 0.95 / h)
                    new_w, new_h = int(w * scale_factor), int(h * scale_factor)

                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                w, h = new_w, new_h

            # 转换图像格式为QImage
            if pil_img.mode == '1':
                pil_img = pil_img.convert('L')

            if pil_img.mode == 'L':
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_Grayscale8)
            elif pil_img.mode == 'RGB':
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGB888)
            elif pil_img.mode == 'RGBA':
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGBA8888)
            else:
                pil_img = pil_img.convert('RGB')
                qimg = QImage(pil_img.tobytes(), w, h, QImage.Format_RGB888)

            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(qimg)
            if pixmap.isNull():
                raise ValueError("无法将图像转换为QPixmap")

            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(w, h)
            label.setScaledContents(True)

            # 显示图像，隐藏提示
            label.setVisible(True)
            self.placeholder_label.setVisible(False)

        except Exception as e:
            logging.error(f"显示图像失败：{str(e)}", exc_info=True)
            self.placeholder_label.setText(f"显示图像失败：{str(e)}")
            self.placeholder_label.setVisible(True)
            label.setVisible(False)

    def save_analysis_result(self):
        if self.is_analysis_result and self.current_analysis_figs:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("保存选项")
            msg_box.setText("请保存:")
            msg_box.setStyleSheet(self.get_message_box_style())

            # 添加"全部保存"按钮
            save_all_btn = msg_box.addButton("全部保存", QMessageBox.ActionRole)
            # 添加自定义文本为"取消"的按钮
            cancel_btn = msg_box.addButton("取消", QMessageBox.RejectRole)

            msg_box.exec_()

            if msg_box.clickedButton() == cancel_btn:
                return

            to_save = []
            if msg_box.clickedButton() == save_all_btn:
                to_save = list(self.current_analysis_figs.keys())
            else:
                to_save = [name for name, cb in self.analysis_checkboxes.items()
                           if cb.isChecked() and name in self.current_analysis_figs]

            if not to_save:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("提示")
                msg.setText("没有选中要保存的分析结果")
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()
                return

            dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
            if not dir_path:
                return

            success_count = 0
            for analysis_type in to_save:
                try:
                    file_path = f"{dir_path}/{analysis_type}.png"
                    self.current_analysis_figs[analysis_type].savefig(
                        file_path, dpi=300, bbox_inches='tight'
                    )
                    success_count += 1
                except Exception as e:
                    logging.error(f"保存{analysis_type}时出错：{str(e)}", exc_info=True)
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("保存失败")
                    msg.setText(f"保存{analysis_type}时出错：{str(e)}")
                    msg.setStyleSheet(self.get_message_box_style())
                    msg.exec_()

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("保存完成")
            msg.setText(f"成功保存{success_count}/{len(to_save)}个分析结果到目录：\n{dir_path}")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
        else:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("提示")
            msg.setText("没有可保存的分析结果")
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()

    def get_message_box_style(self):
        """统一的消息框样式"""
        return """
            QMessageBox {
                background-color: white; 
                color: #333333;
                font-family: "Microsoft YaHei";
                font-size: 18px;
                min-width: 400px;
                min-height: 200px;
            }
            QMessageBox QLabel {
                color: #333333; 
                font-family: "Microsoft YaHei";
                font-size: 18px;
            }
            QPushButton {
                background-color: #ECECEC; 
                color: #333333;
                border: 1px solid #DCDCDC;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 100px;
                font-family: "Microsoft YaHei";
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #DCDCDC; 
                border-color: #CBCBCB;
            }
            QPushButton:pressed {
                background-color: #CBCBCB;
            }
            QMessageBox QWidget {
                background-color: white;
            }
        """

    # 拖拽功能实现
    def dragEnterEvent(self, event):
        """拖拽进入事件处理"""
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                event.acceptProposedAction()

    def dropEvent(self, event):
        """拖拽释放事件处理"""
        file_path = event.mimeData().urls()[0].toLocalFile()
        self._load_image_from_path(file_path)


class HalftoneApp(QWidget):
    """主应用窗口 - 跨系统兼容设计"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HTpro")
        self.setMinimumSize(1200, 800)
        self.dragging = False
        self.drag_position = None
        # 定义统一的按钮字体和高度(使用微软雅黑)
        self.button_font = QFont("Microsoft YaHei", 12)
        self.button_height = 50
        # 设置窗口图标
        self.setWindowIcon(QIcon(resource_path("logo/logo.ico")))

        self.init_ui()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.isMaximized():
            self.dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() == Qt.LeftButton and not self.isMaximized():
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 侧边导航栏 - 现代设计
        sidebar = QFrame()
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2C3E50;
                border-right: 1px solid #233140;
            }
        """)
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 30, 0, 0)
        sidebar_layout.setSpacing(0)

        # 应用标题 - 使用微软雅黑
        title_label = QLabel("HTpro")
        font = QFont("Microsoft YaHei", 16, QFont.Bold)
        title_label.setFont(font)
        title_label.setStyleSheet("""
            color: white; 
            padding: 15px; 
            text-align: center; 
            text-overflow: ellipsis; 
            white-space: nowrap; 
            overflow: hidden;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(60)
        sidebar_layout.addWidget(title_label)

        # 导航按钮 - 使用微软雅黑
        self.nav_buttons = []
        self.processing_btn = QPushButton("图像半色调处理")
        self.processing_btn.setObjectName("navBtn")
        self.processing_btn.setFont(self.button_font)
        self.processing_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495E;
                color: white;
                text-align: left;
                padding: 0 20px;
                border: none;
                border-left: 4px solid #165DFF;
                height: %dpx;
            }
            QPushButton:hover {
                background-color: #4A6988;
            }
            QPushButton:checked {
                background-color: #34495E;
                border-left: 4px solid #165DFF;
            }
        """ % self.button_height)
        self.processing_btn.setCheckable(True)
        self.processing_btn.clicked.connect(lambda: self.switch_tab(0))
        self.processing_btn.setStyleSheet("text-overflow: ellipsis;")
        self.nav_buttons.append(self.processing_btn)

        self.analysis_btn = QPushButton("半色调质量分析")
        self.analysis_btn.setObjectName("navBtn")
        self.analysis_btn.setFont(self.button_font)
        self.analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #BDC3C7;
                text-align: left;
                padding: 0 20px;
                border: none;
                border-left: 4px solid transparent;
                height: %dpx;
            }
            QPushButton:hover {
                background-color: #4A6988;
                color: white;
            }
            QPushButton:checked {
                background-color: #34495E;
                color: white;
                border-left: 4px solid #165DFF;
            }
        """ % self.button_height)
        self.analysis_btn.setCheckable(True)
        self.analysis_btn.clicked.connect(lambda: self.switch_tab(1))
        self.analysis_btn.setStyleSheet("text-overflow: ellipsis;")
        self.nav_buttons.append(self.analysis_btn)

        for btn in self.nav_buttons:
            sidebar_layout.addWidget(btn)
        sidebar_layout.addStretch()

        # 主内容区背景
        content_bg = QFrame()
        content_bg.setStyleSheet("background-color: #F5F7FA;")
        content_layout = QVBoxLayout(content_bg)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 主内容区
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("background-color: transparent;")

        # 创建两个模块并关联
        self.processing_module = HalftoneProcessing(self)
        self.processing_module.setStyleSheet("background-color: transparent;")
        self.analysis_module = HalftoneAnalysis(self)
        self.analysis_module.setStyleSheet("background-color: transparent;")

        self.stacked_widget.addWidget(self.processing_module)
        self.stacked_widget.addWidget(self.analysis_module)

        content_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_bg, 1)

        # 默认显示第一页
        self.switch_tab(0)

    def switch_tab(self, index):
        self.stacked_widget.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
            if i == index:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #34495E;
                        color: white;
                        text-align: left;
                        padding: 0 20px;
                        border: none;
                        border-left: 4px solid #165DFF;
                        height: %dpx;
                    }
                    QPushButton:hover {
                        background-color: #4A6988;
                    }
                """ % self.button_height)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2C3E50;
                        color: #BDC3C7;
                        text-align: left;
                        padding: 0 20px;
                        border: none;
                        border-left: 4px solid transparent;
                        height: %dpx;
                    }
                    QPushButton:hover {
                        background-color: #4A6988;
                        color: white;
                    }
                """ % self.button_height)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序图标
    app_icon = QIcon(resource_path("logo/logo.ico"))
    app.setWindowIcon(app_icon)  # 设置应用程序图标

    # 使用微软雅黑作为全局字体
    app.setFont(QFont("Microsoft YaHei", 10))

    # 全局样式设置，确保所有窗口组件默认使用白色背景
    app.setStyleSheet("""
        QWidget {
            background-color: white;
        }
    """)

    window = HalftoneApp()
    window.show()
    sys.exit(app.exec_())
