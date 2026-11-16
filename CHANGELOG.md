# 更新日志

本项目所有重要变更记录于此。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 待补充

## [1.0.0] - 2026-11-16

### Added

- 7 种半色调算法：有序抖动、Floyd–Steinberg 误差扩散、VAC、BIPPCCA、Even Toned Screening、EDODF、DBS
- 4 种点过程分析指标：配对相关函数 R(r)、方向分布函数 D(α)、径向平均功率谱密度 RAPSD、各向异性 A(f)
- 基于 PyQt5 的图形界面 `HTpro_ui.py`，支持选图、算法切换、结果预览与曲线展示
- 集成 Predictive Reversible Halftoning 深度学习加网推理
- `build.py` 一键打包脚本（PyInstaller，onedir 模式）
- 阈值矩阵生成脚本：`generate_bippcca_threshold_matrix.py`、`generate_vac_threshold_matrix.py`

### Notes

- 预训练权重 `ours_stage2.pth.tar`（约 730 MB）不随仓库分发，获取方式见 README「模型权重」章节

[Unreleased]: https://github.com/jiuer9696/HTpro/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jiuer9696/HTpro/releases/tag/v1.0.0
