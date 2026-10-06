# 独立测试比较图的执行契约

从数据集根目录运行：

```text
python code/submitted_comparison_figures.py --results-root reports/SUBMITTED_METHOD_RESULTS
```

输入仅为 `submitted_comparison_report.py` 生成的本地封存报告：`REPORT_MANIFEST.json`、`main_source_equal.csv`、`main_per_physical_source.csv` 和 `capacity_comparator_differences.csv`。程序先检查 TEST 完成标记；未完成时写 `WAITING`，不读取 CSV，不读取原始运行行、云端或实时测试输出。

输出位于 `results-root/figures/`：`submitted_comparison_2x2.png`（360 dpi）、`submitted_comparison_2x2.pdf`（矢量）、`FIGURE_CAPTION_ZH.md` 和 `FIGURE_MANIFEST.json`。PDF 使用嵌入字体，英文轴适合科研复用，中文说明解释计量、归因和局限。

四个面板使用相同方法颜色与线型：A 为完整调用过程严格按时交付增益；B 为 Capacity 两种训练种子均值与四个参照的有符号差值；C 为按时完整返回率；D 为预定 wide 预算下四个物理来源的差值。D 在四来源等权总体结果上统一选择 CHILS-p1 和 HiGHS 中较强的参照，不按来源分别换对手，不挑选预算。HiGHS 明确是通用 MILP 软件参照，并非另一个专门 MWIS 论文算法。

曲线只连接三个实际测量预算点，未经平滑；阴影是两个拟合种子的最小值至最大值，不是置信区间。失败、迟到和负结果保留。主指标单位是原始接触时长秒，均以 ↑ 标注方向；不把共同贪心前缀或原生搜索收益归因给学习模型。

必要检查仅执行一次合成布局 smoke，并检查导出图片的可读性。该输入带 `guard_synthetic_fixture=true`，图片加显著合成水印，状态为 `SYNTHETIC_LAYOUT_GUARD_ONLY`，不可当作实验结果。完成代码后没有读取真实 TEST CSV，也没有启动额外实验。
# 结果用途与方法可见名称更新

本四联图展示短预算效率与压力诊断；充裕预算质量曲线由独立质量工具生成。图中本文方法明确为 JointRecovery（JR，原键 FullCapacity）；Independent-replacement+CHILS（原 FullP1）只是对照。映射由 jointrecovery_method_names.py 提供，原始键和所有数字保持不变。
