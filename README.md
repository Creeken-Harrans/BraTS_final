# BraTS_final

BraTS2020 分割项目。唯一入口文档是这个根 `README.md`，子目录不再维护各自 README。

## 项目边界

这个仓库当前包含 3 条主线：

- 数据准备：把原始 BraTS2020 训练集转成项目使用的 `nnUNet_raw/Dataset220_BraTS2020`
- 训练与验证：训练 3D UNet，产出 fold 级 checkpoint 和 validation 结果
- 推理与评估：基于已有 checkpoint 抽样预测、评估、生成报告

默认配置：

- dataset: `Dataset220_BraTS2020`
- model: `BraTSFixed3DUNet`
- configuration: `fixed_3d_fullres`
- default folds: `0 1 2 3 4`

## 目录结构

- `run.py`: 根命令入口
- `cli.py`: CLI 定义与命令分发
- `project.py`: 项目路径、数据集、结果目录等统一解析
- `data_preparation/`: 原始数据转换脚本与数据契约
- `training/src/`: 训练、数据、模型、优化与监控代码
- `validation/src/`: validation、best-config、抽样预测、可视化
- `evaluation/src/`: 指标评估与报告生成
- `training_results/`: 训练与验证运行产物，不纳入版本控制
- `evaluation/results/`: 推理评估运行产物，不纳入版本控制
- `first_case_visualization/`: 独立的首病例可视化脚本

## 先决条件

至少要满足这几件事：

- Python 环境安装完 [pyproject.toml](/home/Creeken/Desktop/machine-learning-test/BraTS_final/pyproject.toml) 里的依赖
- 预处理数据和 GT 路径能被 [project.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/project.py) 正确解析
- 如果要训练或验证，`BraTS_final_Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/` 或外部回退目录下要有 `dataset.json`、`splits_final.json`、训练 case 和 `gt_segmentations`
- 如果要生成报告，raw dataset 目录下要有 `imagesTr/` 与 `labelsTr/`

建议先跑：

```bash
python run.py doctor
```

它会打印项目根目录、数据集、模型、配置、预处理目录、训练样本目录和 GT 目录，先确认路径是不是你想要的。

## 常用流程

完整主线通常是：

```bash
python run.py train-all --npz
python run.py find-best-config
python run.py predict --sample-training-cases 12 --sample-seed 123
python run.py evaluate --pred-dir evaluation/results/predict_training_sample_n12_seed123_fold0
```

如果只跑一个 fold：

```bash
python run.py train --fold 0
python run.py validate --fold 0
python run.py evaluate --fold 0
```

开始一轮新的自动推理评估前，可以先清理自动产物：

```bash
python run.py clean-last-results
```

## CLI

总帮助：

```bash
python run.py --help
```

当前命令：

- `doctor`
- `train`
- `train-all`
- `validate`
- `find-best-config`
- `predict`
- `evaluate`
- `clean-last-results`

### `doctor`

查看当前项目配置和关键路径。

```bash
python run.py doctor
```

会打印：

- 项目根目录
- 数据集名
- 模型名
- 配置名
- 预处理目录
- 实际训练样本目录
- 实际 GT 标注目录

### `train`

训练单个 fold。若已有 checkpoint，会自动恢复；若该 fold 已完成，则默认跳过。

```bash
python run.py train --fold 0
python run.py train --fold 0 --restart-training
python run.py train --fold 0 --validation-only
python run.py train --fold 0 --pretrained-weights /path/to/checkpoint.pth
python run.py train --fold 0 --val-best --npz
```

参数：

- `--fold`: 必填，fold 编号
- `--npz`: 验证时保留概率图
- `--validation-only`: 不训练，只加载 checkpoint 做完整验证
- `--restart-training`: 清理当前 fold 产物后重新开始
- `--pretrained-weights`: 指定预训练权重
- `--disable-checkpointing`: 不写 checkpoint
- `--val-best`: 验证时优先使用 `checkpoint_best.pth`

默认输出到 `training_results/fold<fold>/`。

### `train-all`

按默认 folds 顺序依次执行 `train`。

```bash
python run.py train-all
python run.py train-all --restart-training
python run.py train-all --validation-only --val-best
python run.py train-all --npz
```

参数与 `train` 基本一致，但不需要传 `--fold`。运行前会先打印每个 fold 的计划动作。

### `validate`

使用已有 checkpoint 对单个 fold 跑完整验证，不启动训练循环。

```bash
python run.py validate --fold 0
python run.py validate --fold 0 --val-best
python run.py validate --fold 0 --npz
```

参数：

- `--fold`: 必填，fold 编号
- `--npz`: 导出概率图
- `--val-best`: 优先使用 `checkpoint_best.pth`

默认输出：

- 分割结果：`training_results/fold<fold>/validation/`
- validation summary：`training_results/fold<fold>/validation/summary.json`
- fold summary 副本：`training_results/fold<fold>/summary.json`

### `find-best-config`

从已有结果里选出 best config，并写出后续推理所需的信息文件。

```bash
python run.py find-best-config
python run.py find-best-config --search-root training_results
python run.py find-best-config --search-root /abs/path/to/summary_parent
```

它会：

- 找到当前可用的 validation folds
- 自动聚合 shared folds 的 cross-validation 结果
- 输出 best config 结果
- 写出 `inference_information.json`
- 写出 `inference_instructions.txt`
- 写出 `postprocessing.json`

参数：

- `--search-root`: 可选，手动指定扫描目录或 `summary.json` 所在位置；不传时走项目默认流程

### `predict`

从训练集随机抽样若干 case 做推理，并自动评估。

```bash
python run.py predict --sample-training-cases 12 --sample-seed 123
python run.py predict --sample-training-cases 12 --sample-seed 123 --val-best --npz
python run.py predict --sample-training-cases 12 --sample-seed 123 --output-dir evaluation/results/demo_predict --overwrite
```

说明：

- `predict` 不暴露 `--fold`
- 会自动选择可用 checkpoint
- 若 `find-best-config` 已生成 `inference_information.json`，会优先使用其中推荐的 folds
- 会自动生成 `sample_selection.json` 和 `summary.json`
- 当前代码会在写预测文件前自动创建输出目录

参数：

- `--sample-training-cases`: 必填，抽样数量
- `--sample-seed`: 必填，随机种子
- `--npz`: 保留概率图
- `--val-best`: 优先使用 `checkpoint_best.pth`
- `--overwrite`: 覆盖已有输出目录
- `--output-dir`: 自定义输出目录

默认输出位于 `evaluation/results/`。

### `evaluate`

对某个验证目录或预测目录做指标评估，并自动生成完整报告。

```bash
python run.py evaluate
python run.py evaluate --fold 0
python run.py evaluate --pred-dir evaluation/results/demo_predict
```

默认行为：

- 自动从 `evaluation/results/` 中找到最近一次 `predict` 产出的目录
- 直接对这个目录做评估
- 把 summary 写回 `evaluation/results/`
- 自动生成对应的报告目录

参数：

- `--fold`: 使用 `training_results/fold<fold>/validation/` 作为预测目录
- `--pred-dir`: 显式指定预测目录
- `--gt-dir`: 自定义 GT 目录
- `--output-file`: 自定义 summary 输出路径
- `--raw-dataset-dir`: 报告读取原始影像时使用的数据目录
- `--report-output-dir`: 自定义报告输出目录
- `--sample-selection-file`: 给报告附带抽样清单
- `--num-processes`: 评估并行进程数
- `--chill`: 允许预测目录不是完整全集

### `clean-last-results`

清理自动管理的推理评估产物。

```bash
python run.py clean-last-results
```

它会清理：

- `evaluation/results/` 下最近一次 `predict` 产出的目录
- 对应自动生成的 `evaluation/results/<pred_dir_name>_summary.json`
- 对应自动生成的报告目录
- `training_results/` 下的 `inference_information.json`
- `training_results/` 下的 `inference_instructions.txt`
- `training_results/` 下的 `postprocessing.json`
- 默认 folds 对应的 cross-validation 汇总目录

不会删除 fold 训练 checkpoint 或 validation 主产物。

## 独立脚本

这两个脚本不走 `run.py`，但仍然是仓库的一部分。

### 首病例可视化

用于快速看 BraTS 原始病例的四模态、分割、overlay、bbox 和强度分布。

```bash
python first_case_visualization/visualize_first_case.py
python first_case_visualization/visualize_first_case.py \
  --data-root BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --output-dir first_case_visualization/output
```

对应代码：

- [visualize_first_case.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/first_case_visualization/visualize_first_case.py)

### 数据准备

把原始 BraTS2020 训练数据转换成项目使用的 `nnUNet_raw/Dataset220_BraTS2020`。

```bash
conda run -n pytorch python data_preparation/scripts/prepare_brats2020_for_project.py
conda run -n pytorch python data_preparation/scripts/prepare_brats2020_for_project.py \
  --src-root BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --project-raw BraTS_final_Dataset/nnUNet_raw
```

如需强制重建：

```bash
conda run -n pytorch python data_preparation/scripts/prepare_brats2020_for_project.py --force
```

对应代码和契约：

- [prepare_brats2020_for_project.py](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/scripts/prepare_brats2020_for_project.py)
- [data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/docs/data_contract.md)
- [raw_dataset.json](/home/Creeken/Desktop/machine-learning-test/BraTS_final/data_preparation/metadata/raw_dataset.json)

## 代码组织

- `training/src/core/`: 训练配置、运行时路径、主训练器
- `training/src/data/`: 数据集、标签管理、数据增强、loader
- `training/src/models/`: BraTS 训练/推理模型定义
- `training/src/optimization/`: 损失函数、优化器、调度器、预训练权重
- `training/src/monitoring/`: 训练日志与异步绘图
- `validation/src/inference.py`: 滑窗推理、恢复原空间、分割写出
- `validation/src/find_best_config.py`: best-config 选择与推理说明生成
- `validation/src/predict.py`: 抽样预测
- `validation/src/visualization.py`: overlay、调试图、训练进度图
- `evaluation/src/metrics.py`: 指标计算、summary 序列化、cross-validation 聚合
- `evaluation/src/generate_evaluation_report.py`: 评估报告生成

## 其他说明

- 所有命令都支持 `--help`
- 运行产物主要写入 `training_results/`、`evaluation/results/`、`first_case_visualization/output/`，这些目录默认不纳入版本控制
- 如果代码结构和文档冲突，以这个根 README 和 `python run.py --help` 为准
