# HTpro · 数字加网及评价系统

> 一套面向数字半色调（Digital Halftoning）的算法工具箱与可视化桌面应用：内置 7 种加网算法、4 种点过程分析指标，并提供 PyQt5 图形界面与一键打包。

[![CI](https://github.com/jiuer9696/HTpro/actions/workflows/ci.yml/badge.svg)](https://github.com/jiuer9696/HTpro/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/jiuer9696/HTpro)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)]()

---

## 目录

- [特性](#特性)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [模型权重](#模型权重)
- [项目结构](#项目结构)
- [算法说明](#算法说明)
- [点过程分析](#点过程分析)
- [打包为可执行文件](#打包为可执行文件)
- [开发与贡献](#开发与贡献)
- [许可证](#许可证)

## 特性

- **7 种半色调算法**：有序抖动、Floyd–Steinberg 误差扩散、VAC 蓝噪声、BIPPCCA 绿噪声、DBS 迭代优化、Even Toned Screening、EDODF
- **4 种点过程分析**：配对相关函数 R(r)、方向分布函数 D(α)、径向平均功率谱密度 RAPSD、各向异性 A(f)
- **图形界面**：PyQt5 桌面应用，拖拽选图、切换算法、实时查看处理结果与分析曲线
- **深度学习加网**：集成 Predictive Reversible Halftoning 推理流程
- **一键打包**：`build.py` 基于 PyInstaller 生成 Windows 分发包

## 环境要求

| 项目 | 版本 |
| --- | --- |
| Python | 3.9 及以上（推荐 3.9 / 3.10） |
| 操作系统 | Windows 10 / 11（已验证） |
| 依赖 | 见 [requirements.txt](./requirements.txt) |

> `torch==2.7.0` 体积较大，若只用传统算法可先不装，界面中相关入口会自动跳过。

## 安装

```bash
git clone https://github.com/jiuer9696/HTpro.git
cd HTpro

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

国内网络可用镜像加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始

图形界面：

```bash
python HTpro_ui.py
```

界面使用流程：

1. 用 **选择文件** 按钮导入一张图片
2. 在下拉菜单中切换半色调算法，调节参数后生成加网结果
3. 切换到分析页，查看 R(r)、D(α)、RAPSD、A(f) 曲线与诊断结论

单独调用某个算法（不启动界面）：

```python
from PIL import Image
from halftoning_algorithms.vacmain import vac_dithering

img = Image.open("logo/tu.png").convert("L")
result = vac_dithering(img)
result.save("out.png")
```

## 模型权重

深度学习加网依赖预训练权重：

```
PredictiveReversibleHalftoning/checkpoints/ours_stage2.pth.tar   (~730 MB)
```

由于超出 GitHub 单文件 100 MB 限制，**权重不随仓库分发**。请按以下方式获取后放到该路径：

1. 从 [Releases](https://github.com/jiuer9696/HTpro/releases) 页面下载（体积原因需单独发布）
2. 或使用上游项目提供的权重

未放置权重时，传统算法与点过程分析功能仍可正常使用。

## 项目结构

```
HTpro/
├── HTpro_ui.py                     # PyQt5 主界面与交互入口
├── build.py                        # PyInstaller 打包脚本
├── HTpro.spec                      # PyInstaller 配置
├── requirements.txt
├── analysis/                       # 点过程分析
│   ├── R.py                        # 配对相关函数 R(r)
│   ├── D3.py                       # 方向分布函数 D(α)
│   ├── RAPSD.py                    # 径向平均功率谱密度 P(f)
│   └── floyA.py                    # 各向异性 A(f)
├── halftoning_algorithms/          # 半色调算法
│   ├── youxu.py                    # 有序抖动
│   ├── 误差扩散2.py                # Floyd–Steinberg 误差扩散
│   ├── vacmain.py                  # VAC 蓝噪声
│   ├── bippcca.py                  # BIPPCCA 绿噪声
│   ├── EDODF.py                    # EDODF 绿噪声
│   ├── DBS10.py                    # DBS 迭代优化
│   ├── eventone.py / eventonemain.py  # Even Toned Screening
│   ├── generate_bippcca_threshold_matrix.py
│   └── generate_vac_threshold_matrix.py
├── PredictiveReversibleHalftoning/ # 深度学习加网
│   ├── inference_single.py
│   ├── models/
│   ├── utils/
│   └── checkpoints/                # 权重目录（不入库）
└── logo/                           # 图标与示意图
```

## 算法说明

| 算法 | 文件 | 原理 | 优点 | 局限 | 典型适用 |
| --- | --- | --- | --- | --- | --- |
| 有序抖动 | [`youxu.py`](./halftoning_algorithms/youxu.py) | 用预定义阈值矩阵（如 Bayer 矩阵）分块比较生成二值图像 | 实现简单、速度快、适合硬件实现 | 易产生周期性棋盘格纹理与莫尔条纹，细节还原有限 | 对速度要求高、视觉质量要求低的场合 |
| 误差扩散 | [`误差扩散2.py`](./halftoning_algorithms/误差扩散2.py) | 将每像素量化误差按权重扩散至未处理邻域，实现全局灰度平衡 | 灰度还原能力强，视觉自然 | 可能产生"蠕虫状"伪影与方向性结构 | 打印、显示等对图像质量有要求的应用 |
| VAC | [`vacmain.py`](./halftoning_algorithms/vacmain.py) | 迭代优化点分布，填补稀疏区（Void）、消除密集簇（Cluster），生成蓝噪声 | 点分布均匀，高频能量主导，伪影极少 | 算法复杂、计算量大 | 高质量图像输出、艺术印刷 |
| BIPPCCA | [`bippcca.py`](./halftoning_algorithms/bippcca.py) | 块内像素对相关性优化，生成局部聚集的绿噪声图案 | 生成速度快，局部聚集特性明显 | 块结构可能带来方向性伪影，边界平滑性略差 | 效率优先、中间调表现优先 |
| Even Toned Screening | [`eventonemain.py`](./halftoning_algorithms/eventonemain.py) | 动态调整阈值并结合点间距离反馈，抑制低频结构 | 点分布接近理想蓝噪声，方向性伪影极少 | 实现较复杂，参数调节需经验 | 高质量打印、均匀性要求高 |
| EDODF | [`EDODF.py`](./halftoning_algorithms/EDODF.py) | 在误差扩散基础上引入输出依赖反馈，增强像素聚集 | 聚集性可控，适合深色调表现 | 可能引入方向性伪影，参数敏感 | 高质量深色调输出 |
| DBS | [`DBS10.py`](./halftoning_algorithms/DBS10.py) | 基于感知误差最小化的贪心迭代，每次尝试翻转/交换直至收敛 | 视觉质量最佳，空间与频谱均衡 | 计算量极大，速度慢 | 高端印刷、科研等极高要求场景 |

## 点过程分析

| 指标 | 文件 | 原理 | 作用 |
| --- | --- | --- | --- |
| 配对相关函数 R(r) | [`R.py`](./analysis/R.py) | 统计距离 r 的点对出现概率与随机分布的比值 | 判断点分布的空间均匀性、聚集性或排斥性 |
| 方向分布函数 D(α) | [`D3.py`](./analysis/D3.py) | 统计特定方向子区间内点密度与整体环形区域密度之比 | 检测各向异性，揭示方向性伪影 |
| 径向平均功率谱密度 P(f) | [`RAPSD.py`](./analysis/RAPSD.py) | 对图像做 2D 傅里叶变换，统计各频率环带平均能量 | 区分白噪声 / 蓝噪声 / 绿噪声等频谱特性 |
| 各向异性 A(f) | [`floyA.py`](./analysis/floyA.py) | 统计同一频率环带内不同方向能量分布的均匀性 | 辅助空间域分析，综合评价各向异性 |

## 打包为可执行文件

```bash
pip install pyinstaller
python build.py
```

产物位于 `dist/HTpro/`，双击 `HTpro.exe` 运行。加 `--debug` 参数可生成带控制台的调试版本：

```bash
python build.py --debug
```

## 开发与贡献

- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)
- 新算法请放在 `halftoning_algorithms/`，新指标放在 `analysis/`，并在 README 表格中补充一行
- 详细流程见 [CONTRIBUTING.md](./CONTRIBUTING.md)，行为准则见 [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- 安全漏洞请按 [SECURITY.md](./SECURITY.md) 私密上报，不要公开提 Issue

## 许可证

本项目基于 [Apache License 2.0](./LICENSE) 开源。

---

如有问题欢迎提 [Issue](https://github.com/jiuer9696/HTpro/issues)。
