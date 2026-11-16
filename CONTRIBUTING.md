# 贡献指南

感谢你对 **HTpro** 的关注！欢迎提交 Issue 和 Pull Request。

## 行为准则

参与本项目即表示你同意遵守 [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)。

## 报告问题

- **Bug**：使用 [Bug Report](https://github.com/jiuer9696/HTpro/issues/new?template=bug_report.yml) 模板，附上复现步骤、图片样本、报错日志
- **新功能 / 新算法**：使用 [Feature Request](https://github.com/jiuer9696/HTpro/issues/new?template=feature_request.yml) 模板，说明动机与论文出处
- **安全漏洞**：**不要**公开提 Issue，请见 [SECURITY.md](./SECURITY.md)

提交前请先搜索已有 Issue，避免重复。

## 提交代码

```bash
git clone https://github.com/jiuer9696/HTpro.git
cd HTpro
git switch -c feat/your-algorithm

# 开发并验证
python -m compileall -q .
python HTpro_ui.py

git add -A
git commit -m "feat(halftoning): 新增 xxx 算法"
git push -u origin feat/your-algorithm
```

PR 要求：

- 关联对应 Issue（`Closes #12`）
- 一个 PR 只做一件事
- 新增算法需提供**处理结果对比图**与**点过程分析曲线**
- 涉及新算法时同步更新 README 的算法表格

## 提交信息规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：

| 类型 | 说明 |
| --- | --- |
| `feat` | 新功能（如新增算法） |
| `fix` | Bug 修复 |
| `docs` | 文档 |
| `perf` | 性能优化（算法提速） |
| `refactor` | 重构 |
| `test` | 测试 |
| `build` | 打包脚本、依赖 |
| `ci` | CI 配置 |
| `chore` | 杂项 |

示例：

```
feat(analysis): 为 RAPSD 增加对数坐标选项

分析页新增"对数坐标"开关，用于观察高频段能量衰减。
同时把网格密度从 50 提升到 100。

Closes #18
```

## 代码约定

- 缩进 4 空格，UTF-8，LF 换行（见 `.editorconfig`）
- 算法函数保持纯函数风格：输入 `PIL.Image`，输出 `PIL.Image` 或 `numpy.ndarray`
- 新算法放 `halftoning_algorithms/`，新指标放 `analysis/`
- 新文件不要引入与现有依赖冲突的版本（优先复用 `requirements.txt` 已有库）
- 不要在代码中硬编码本地绝对路径，资源路径统一相对于项目根目录

## 目录归属

| 目录 | 内容 |
| --- | --- |
| `halftoning_algorithms/` | 各类半色调算法实现 |
| `analysis/` | 点过程分析指标 |
| `PredictiveReversibleHalftoning/` | 深度学习加网 |
| `logo/` | 图标与示意图 |

---

再次感谢！🎉
