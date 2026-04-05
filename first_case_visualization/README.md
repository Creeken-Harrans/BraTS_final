# 00 First Case Visualization

这一阶段近乎原样迁移自旧 `BraTS` 项目，目标不变：

- 先建立对 BraTS 原始 3D 数据的直觉
- 先肉眼确认四模态和标签是否共空间对齐
- 先看清楚病灶、切片、overlay、bbox、强度分布分别长什么样

它不是训练必须步骤，但它仍然是新项目里最值得先做的认知准备。

## 输入和输出

默认输入目录：

- `BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

默认输出目录：

- `first_case_visualization/output`

典型输出包括：

- `case_summary.txt`
- `case_summary.json`
- `seg_labels_summary.txt`
- `intensity_stats.csv`
- 多张切片、overlay、bbox、直方图 PNG

## 运行方式

当前这一阶段先保留为独立脚本，不依赖新的 CLI。

在 `BraTS_final` 根目录运行：

```bash
python first_case_visualization/visualize_first_case.py
```

常见参数：

```bash
python first_case_visualization/visualize_first_case.py \
  --data-root BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --output-dir first_case_visualization/output
```

## 对应代码

- [visualize_first_case.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/first_case_visualization/visualize_first_case.py)

## 它到底在做什么

脚本会：

1. 递归扫描原始 BraTS 数据根目录
2. 自动找出按排序后的第一个合法病例目录
3. 识别 `t1 / t1ce / t2 / flair / seg`
4. 加载 3D NIfTI volume
5. 检查 shape、spacing、affine 是否一致
6. 输出文本摘要、JSON 摘要、强度统计
7. 生成三视图、montage、overlay、bbox、标签分布等图片

## 建议阅读顺序

1. `output/case_summary.txt`
2. `output/seg_labels_summary.txt`
3. `output/modalities_mid_slices.png`
4. `output/t1_montage.png`
5. `output/t1ce_montage.png`
6. `output/t2_montage.png`
7. `output/flair_montage.png`
8. `output/seg_montage.png`
9. `output/flair_overlay_best_slices.png`
10. `output/t1ce_overlay_best_slices.png`
11. `output/tumor_bbox_views.png`
12. `output/intensity_histograms.png`
13. `output/seg_label_distribution.png`

## 和旧版本相比的变化

这次迁移只做了必要改动：

- 默认数据目录从 `BraTS-Dataset` 改为 `BraTS_final_Dataset`
- 项目根目录解析改成适配 `BraTS_final`
- README 里的旧 CLI 入口说明移除，先保留为独立脚本运行

可视化思路、输出结构和生成结果类型保持基本不变。
