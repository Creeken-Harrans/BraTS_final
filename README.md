# BraTS_final

BraTS2020 分割项目。命令行统一入口是根目录的 `run.py`。

默认配置：

- dataset: `Dataset220_BraTS2020`
- model: `BraTSFixed3DUNet`
- configuration: `fixed_3d_fullres`
- default folds: `0 1 2 3 4`

主要目录：

- `training/`: 训练逻辑
- `validation/`: 验证、best-config、抽样预测
- `evaluation/`: 指标评估与报告生成
- `training_results/`: 训练与验证运行产物，不纳入版本控制

## CLI

总帮助：

```bash
python run.py --help
```

当前 CLI 子命令一共 8 个：

- `doctor`
- `train`
- `train-all`
- `validate`
- `find-best-config`
- `predict`
- `evaluate`
- `clean-last-results`

建议先跑：

```bash
python run.py doctor
```

它会打印项目根目录、数据集、预处理目录、训练样本目录和 GT 目录，用来确认环境是否对齐。

## 常用流程

完整主线通常是：

```bash
python run.py train-all --npz
python run.py find-best-config
python run.py predict --sample-training-cases 12 --sample-seed 123
python run.py evaluate --pred-dir evaluation/results/predict_training_sample_n12_seed123_fold0
```

如果只想处理一个 fold：

```bash
python run.py train --fold 0
python run.py validate --fold 0
python run.py evaluate --fold 0
```

## 命令详情

### `doctor`

查看当前项目配置和数据路径。

```bash
python run.py doctor
```

输出内容包括：

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
- `--validation-only`: 不训练，只加载 checkpoint 跑完整验证
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

执行旧项目风格的 best-config 选择主流程。

它会：

- 找到当前可用的 validation folds
- 自动聚合 shared folds 的 cross-validation 结果
- 输出 best config 结果
- 写出 `inference_information.json`
- 写出 `inference_instructions.txt`

```bash
python run.py find-best-config
python run.py find-best-config --search-root training_results
python run.py find-best-config --search-root /abs/path/to/summary_parent
```

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
- 它会自动选择可用 checkpoint
- 若 `find-best-config` 已生成 `inference_information.json`，会优先使用其中推荐的 folds
- 会自动生成 `sample_selection.json` 和 `summary.json`

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
```

默认行为：

- 自动从 `evaluation/results/` 中找到最近一次 `predict` 产出的目录
- 直接对这个目录做评估
- 把 summary 写回 `evaluation/results/`
- 自动生成对应的 `report/` 目录

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

要求：

- 不传参数时会自动读取最近一次 `predict` 结果
- 也可以手动传 `--fold` 或 `--pred-dir`
- 不再需要单独执行 `report`

### `clean-last-results`

删除最近一次按默认规则生成的 `predict` / `evaluate` 结果，以及 `find-best-config` 生成的自动产物。

```bash
python run.py clean-last-results
```

它会清理：

- `evaluation/results/` 下最近一次 `predict` 产出的目录
- 对应自动生成的 `evaluation/results/<pred_dir_name>_summary.json`
- 对应自动生成的报告目录
- `training_results/` 下的 `inference_information.json`、`inference_instructions.txt`、`postprocessing.json`
- 默认 folds 对应的 cross-validation 汇总目录

说明：

- 只针对自动管理的 `find-best-config` / `predict` / `evaluate` 结果
- 不会删除 `training_results/` 下的训练或 validation 产物
- 主要用于开始新一轮推理评估前清空上一轮自动产物

## 其他说明

- 所有命令都支持 `--help`
- 训练、验证、预测相关产物主要写入 `training_results/` 和 `evaluation/results/`，这些目录默认不纳入版本控制
- 原始 BraTS 数据转换脚本仍保留在 `data_preparation/scripts/prepare_brats2020_for_project.py`，供以后重建 `nnUNet_raw` 时使用；它不属于当前 CLI 子命令集合
