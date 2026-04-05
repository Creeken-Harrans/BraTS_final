# Data Contract

这份文档近乎原样迁移自旧 `BraTS` 项目，定义的是 `BraTS_final` 当前仍然沿用的底层数据契约。

它不是建议格式，而是后续预处理、训练、推理能够稳定工作的前提。

## 1. 目录契约

训练目录必须命名为：

- `Dataset220_BraTS2020`

并且至少包含：

- `imagesTr/`
- `labelsTr/`
- `dataset.json`

在当前项目边界里，它默认位于：

- `BraTS_final_Dataset/nnUNet_raw/Dataset220_BraTS2020`

## 2. 文件命名契约

每个 case 必须有 4 个模态文件：

- `{case_id}_0000.nii.gz`
- `{case_id}_0001.nii.gz`
- `{case_id}_0002.nii.gz`
- `{case_id}_0003.nii.gz`

标签文件：

- `{case_id}.nii.gz`

## 3. 通道语义契约

当前项目要求：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

训练目录和后续推理输入目录都必须严格遵守这条顺序契约。

## 4. 标签语义契约

原始 BraTS 标签：

- `0 = background`
- `1 = NCR/NET`
- `2 = ED`
- `4 = ET`

项目内部训练标签：

- `0 -> 0`
- `2 -> 1`
- `1 -> 2`
- `4 -> 3`

## 5. region 契约

`dataset.json` 中当前维持的 region 定义为：

- `whole_tumor = [1, 2, 3]`
- `tumor_core = [2, 3]`
- `enhancing_tumor = [3]`

并且：

- `regions_class_order = [1, 2, 3]`

## 6. 空间几何契约

同一病例内：

- 4 个模态必须共空间对齐
- 标签必须和图像共空间对齐

脚本会在转换阶段检查：

- size
- spacing
- origin
- direction

## 7. 当前默认路径

原始 BraTS 数据默认读取：

- `BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

转换后 raw 数据默认写入：

- `BraTS_final_Dataset/nnUNet_raw`

## 8. 当前状态说明

虽然你的目标是后续直接复用已经提取好的固定数据集，但在当前这一步，项目仍然保留这份转换脚本和原始数据契约说明，作为后续迁移的兼容基础。
