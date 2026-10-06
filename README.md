# Learning Joint Recovery of Multi-Agent Graph Optimization for Satellite–Ground Scheduling

## New STK dataset and completed quality evaluation

The [STK material directory](stk/JointRecovery_STK_20261005/) adds 12 physical sources, 72 weighted satellite–ground conflict graphs, frozen code and checkpoints, and completed experimental reports. This update leaves the earlier manuscript unchanged. The [final findings](stk/JointRecovery_STK_20261005/reports/JOINTRECOVERY_QUALITY_RESULTS/FINAL_QUALITY_SIGNOFF_ZH.md) report a small JR improvement over internal greedy ranking, a substantial quality gap to published CHILS-p1, and N/A complete quality for incomplete HiGHS outputs; remaining-head is fixed to zero. The [five-point quality/time curves](stk/JointRecovery_STK_20261005/reports/JOINTRECOVERY_QUALITY_FIGURES/test/quality_curves_test.pdf) come from independent actual executions, with missing outputs and completion rates retained.

See the [dataset README](stk/JointRecovery_STK_20261005/README.md) and [reproduction guide](stk/JointRecovery_STK_20261005/reports/PUBLICATION_REPRODUCTION_GUIDE_ZH.md). Full raw data and execution records are split into ten [Release archives (`stk-jointrecovery-20261006`)](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/tag/stk-jointrecovery-20261006); extract all archives into `stk/JointRecovery_STK_20261005` because their internal paths are relative to that dataset root. Git already provides the readable scientific materials, graphs, contacts, final checkpoints and frozen sources; the release supplies the larger evidence records.

**当前版本的代码、实验结果和论文，供专家审阅。** AAMAS 2027，提交号 1979。本文与实验冻结于 2026-10-04；本仓库仅包含与审阅有关的科学材料。

## 阅读与下载

- [论文 PDF：8 页正文 + 1 页参考文献](paper/main_v4.pdf)
- [科学补充 PDF](paper/supplementary_v4.pdf)
- [当前版本说明与结果局限（中文）](docs/CURRENT_VERSION_ZH.md)
- [完整下载：PDF、源码图源包、代码/结果/模型复现包](https://github.com/Mister-Ryder/AAMAS2027_JointRecovery/releases/tag/v4-current-paper-20261004)

## 关键代码与结果

| 内容 | 入口 |
|---|---|
| 联合恢复模型 | [src/joint_recovery/v4_residual_model.py](src/joint_recovery/v4_residual_model.py) |
| 训练 | [experiments/v4_residual_fit.py](experiments/v4_residual_fit.py) |
| 真实在线比较 | [experiments/v4_residual_online_validation_execute.py](experiments/v4_residual_online_validation_execute.py) |
| 在线结果主表 | [residual_online_table.csv](results/v4_residual_online/independent_analysis_01/residual_online_table.csv) |
| 公开已发表算法结果主表 | [published_original_table.csv](results/v4_unit_public_original/independent_audit_01/published_original_table.csv) |
| 原始标签 | [results/v4_residual_labels/bulk_03](results/v4_residual_labels/bulk_03) |
| 15 次训练记录与模型 | [results/v4_residual_fit/bulk_01](results/v4_residual_fit/bulk_01) |
| 可编辑论文及图源 | [paper/main_v4.tex](paper/main_v4.tex)、[paper/v4_figures](paper/v4_figures) |

已完成 96 图标签采集、5 个变体 × 3 个随机种子训练、1,512 次真实在线决策，以及 40 个公开图上的 2,040 次已发表求解器运行。结果表中使用 ↑/↓ 标明指标方向，方法引用见论文。

当前学习策略尚未稳定优于强贪心对照；公开表是求解器比较，不能作为学习模型公开迁移优势的证据。公开学习迁移与独立确认实验尚未执行。完整结果、局限和适用范围见论文讨论与中文说明。

## 复现与版本

依赖见 [pyproject.toml](pyproject.toml)；安装 `python -m pip install -e .`。实际运行参数保存在 [docs/research_v4](docs/research_v4)，标签、训练历史、checkpoint 和聚合结果保存在 `results/`。路径和环境差异的说明见 [REPRODUCTION.md](REPRODUCTION.md)。公开算法需从其官方发布获取，外部可执行文件和过程日志不在本仓库。

论文编译使用 `paper/` 中的 AAMAS 类文件：对 `main_v4.tex` 运行 LaTeX、BibTeX，再运行 LaTeX 两次。

版本标签：`v4-current-paper-20261004`。原本地科学源码提交：`0401629138c707c32e241f8265559d4717e37854`。本仓库以精选材料建立独立的公开快照，不包含无关开发历史；[MANIFEST.json](MANIFEST.json) 保留交付科学包内文件的 SHA256，[FINAL_CHECK.json](release/current_version/FINAL_CHECK.json) 保留 PDF、页数与交付包校验值。

## Publication receipt

[Completed public upload and asset SHA256 receipt](stk/JointRecovery_STK_20261005/reports/GITHUB_STK_PUBLICATION_ZH.md). Version `stk-jointrecovery-20261006` contains the scientific snapshot; this receipt records the completed Release upload.
