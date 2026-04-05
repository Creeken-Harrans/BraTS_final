# 01 Data Preparation

这一阶段近乎原样迁移自旧 `BraTS` 项目。

它负责把原始 BraTS 病例转换成当前项目训练链路承认的原始训练目录，也就是：

- 从原始 BraTS 表示
- 到 `Dataset220_BraTS2020`

当前这一步先保留旧项目的数据准备逻辑和数据契约。后续因为你的数据集是固定的，项目还会进一步演化为“直接复用已经提取好的数据”，但这一步暂时还没有把那层简化写进去。

## 输入、输出和入口

默认输入：

- `BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

默认输出：

- `BraTS_final_Dataset/nnUNet_raw/Dataset220_BraTS2020`

其结构至少包括：

- `imagesTr/`
- `labelsTr/`
- `dataset.json`

## 运行方式

当前这一阶段先保留为独立脚本，不依赖新的 CLI。

在 `BraTS_final` 根目录运行：

```bash
conda run -n pytorch python data_preparation/scripts/prepare_brats2020_for_project.py
```

显式指定路径时：

```bash
conda run -n pytorch python data_preparation/scripts/prepare_brats2020_for_project.py \
  --src-root BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --project-raw BraTS_final_Dataset/nnUNet_raw
```

如果目标目录已经完整存在，脚本会直接复用。

只有在你确实要重建时，才使用：

```bash
conda run -n pytorch python data_preparation/scripts/prepare_brats2020_for_project.py --force
```

## 对应代码

- [prepare_brats2020_for_project.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/scripts/prepare_brats2020_for_project.py)
- [data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/docs/data_contract.md)
- [raw_dataset.json](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/metadata/raw_dataset.json)

## 当前保留的数据契约

通道顺序保持不变：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

标签转换保持不变：

- `0 -> 0`
- `2 -> 1`
- `1 -> 2`
- `4 -> 3`

region 定义保持不变：

- `whole_tumor = [1, 2, 3]`
- `tumor_core = [2, 3]`
- `enhancing_tumor = [3]`

## 和旧版本相比的变化

这次迁移只做了必要改动：

- 默认原始数据目录从 `BraTS-Dataset/archive/...` 改为 `BraTS_final/BraTS_final_Dataset/archive/...`
- 默认输出目录从 `BraTS-Dataset/nnUNet_raw` 改为 `BraTS_final/BraTS_final_Dataset/nnUNet_raw`
- 文档里的旧 CLI 入口说明移除，先保留为独立脚本运行
- 脚本的项目根目录发现逻辑改成适配 `BraTS_final`
