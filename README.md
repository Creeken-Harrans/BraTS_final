# BraTS_final

BraTS2020 分割项目，当前统一通过根目录的 `run.py` 作为命令行入口。

默认数据集与训练配置：

- dataset: `Dataset220_BraTS2020`
- model: `BraTSFixed3DUNet`
- configuration: `fixed_3d_fullres`
- default folds: `0 1 2 3 4`

常用目录：

- `training/`：训练代码
- `validation/`：验证与预测代码
- `evaluation/`：评估与报告代码
- `training_results/`：训练和验证输出
- `preprocess/`：预处理配置与元数据

## CLI

统一入口：

```bash
python run.py --help
```

建议先跑一次环境检查：

```bash
python run.py doctor
```

它会打印项目根目录、当前数据集、预处理目录、训练样本目录和 GT 目录，便于先确认路径是否对。

### 常用工作流

1. 查看当前配置和数据路径

```bash
python run.py doctor
```

2. 训练单个 fold

```bash
python run.py train --fold 0
```

3. 只对已有 checkpoint 跑验证

```bash
python run.py validate --fold 0
```

4. 训练完多个 fold 后合并交叉验证结果

```bash
python run.py accumulate-cv --folds 0 1 2 3 4
```

5. 从训练集抽样做一次预测并自动评估

```bash
python run.py predict --sample-training-cases 12 --sample-seed 123
```

6. 给某个预测目录生成评估报告

```bash
python run.py report --pred-dir evaluation/results/predict_training_sample_n12_seed123_fold0
```

### 命令说明

#### `doctor`

查看项目当前使用的关键路径和配置。

```bash
python run.py doctor
```

适合在训练前先确认：

- 当前数据集名
- 预处理元数据目录
- 实际训练样本目录
- 实际 GT 标注目录

#### `train`

训练一个 fold。若检测到已有 checkpoint，会自动恢复；若该 fold 已标记完成，则默认跳过。

```bash
python run.py train --fold 0
python run.py train --fold 0 --restart-training
python run.py train --fold 0 --pretrained-weights /path/to/checkpoint.pth
python run.py train --fold 0 --validation-only
python run.py train --fold 0 --val-best --npz
```

常用参数：

- `--fold`：必填，指定 fold 编号
- `--validation-only`：不训练，只加载 checkpoint 做完整验证
- `--restart-training`：清理当前 fold 下的训练产物后重新开始
- `--pretrained-weights`：加载外部权重作为初始化
- `--disable-checkpointing`：训练过程中不写 checkpoint
- `--val-best`：验证时优先使用 `checkpoint_best.pth`
- `--npz`：验证时额外导出概率图 `.npz`

默认输出目录是 `training_results/fold<fold>/`。

#### `train-all`

按 `project_config.json` 里的默认 folds 依次执行 `train`。

```bash
python run.py train-all
python run.py train-all --restart-training
python run.py train-all --validation-only --val-best
```

这个命令会先打印每个 fold 的计划动作，例如从头训练、恢复训练、只验证或跳过已完成 fold。

#### `validate`

使用已有 checkpoint 对单个 fold 跑验证，不启动训练循环。

```bash
python run.py validate --fold 0
python run.py validate --fold 0 --val-best
python run.py validate --fold 0 --npz
```

常用场景：

- 训练结束后补跑一次完整验证
- 比较 `final` 和 `best` checkpoint 的验证结果
- 导出概率图供后处理或分析

验证输出默认写到 `training_results/fold<fold>/validation/`，同时会生成对应 `summary.json`。

#### `find-best-config`

扫描 `training_results/` 和 `evaluation/results/` 下的 `summary.json`，按指定指标排序。

```bash
python run.py find-best-config
python run.py find-best-config --metric foreground_mean.FN
python run.py find-best-config --metric mean.(3,).Dice --top-k 10
python run.py find-best-config --output-file evaluation/results/best_config_ranking.json
```

常用参数：

- `--search-root`：指定扫描目录或直接给 `summary.json`
- `--metric`：点路径形式的指标名，默认 `foreground_mean.Dice`
- `--top-k`：输出前几名
- `--lower-is-better`：显式按升序排序
- `--output-file`：把完整排名写到 JSON

#### `predict`

从训练集样本中随机抽取若干 case 做推理，并自动生成 `summary.json`。

```bash
python run.py predict --sample-training-cases 12 --sample-seed 123
python run.py predict --sample-training-cases 12 --sample-seed 123 --fold 0
python run.py predict --sample-training-cases 12 --sample-seed 123 --val-best --npz
python run.py predict --sample-training-cases 12 --sample-seed 123 --output-dir evaluation/results/demo_predict --overwrite
```

说明：

- 不传 `--fold` 时，会自动选择当前可用且验证 Dice 最优的 fold
- 会输出采样清单 `sample_selection.json`
- 会自动评估预测结果并生成 `summary.json`

常用参数：

- `--sample-training-cases`：必填，抽样数量
- `--sample-seed`：必填，随机种子
- `--fold`：显式指定使用哪个 fold 的 checkpoint
- `--val-best`：优先使用 `checkpoint_best.pth`
- `--output-dir`：自定义输出目录
- `--overwrite`：允许覆盖已有非空目录
- `--npz`：额外导出概率图 `.npz`

#### `evaluate`

对某个预测目录单独做指标评估。

```bash
python run.py evaluate --fold 0
python run.py evaluate --pred-dir evaluation/results/demo_predict
python run.py evaluate --pred-dir evaluation/results/demo_predict --gt-dir /path/to/gt_segmentations
```

使用规则：

- `--fold` 和 `--pred-dir` 二选一，至少提供一个
- 只给 `--fold` 时，默认评估 `training_results/fold<fold>/validation/`
- 不提供 `--output-file` 时，会写到 `evaluation/results/`

常用参数：

- `--gt-dir`：自定义 GT 目录
- `--output-file`：自定义 summary 输出路径
- `--num-processes`：评估并行进程数
- `--chill`：允许预测目录不是完整全集

#### `report`

基于预测目录和 `summary.json` 生成报告；若缺少 `summary.json`，会先自动评估。

```bash
python run.py report --fold 0
python run.py report --pred-dir evaluation/results/demo_predict
python run.py report --pred-dir evaluation/results/demo_predict --summary-file evaluation/results/demo_predict/summary.json
```

常用参数：

- `--fold`：直接对某个 fold 的验证目录生成报告
- `--pred-dir`：指定任意预测目录
- `--summary-file`：复用已有评估结果
- `--gt-dir`：自动评估时使用的 GT 目录
- `--raw-dataset-dir`：报告里读取原始影像时使用的数据目录
- `--output-dir`：自定义报告输出目录
- `--sample-selection-file`：给抽样预测报告附带样本清单

#### `accumulate-cv`

把多个 fold 的验证预测合并到一个目录，再重新评估得到交叉验证整体结果。

```bash
python run.py accumulate-cv --folds 0 1 2 3 4
python run.py accumulate-cv --folds 0 1 2 --output-dir evaluation/results/cv_folds_0_1_2
python run.py accumulate-cv --folds 0 1 2 3 4 --num-processes 4
```

常用参数：

- `--folds`：要合并的 fold 列表
- `--output-dir`：输出目录，默认在 `evaluation/results/`
- `--num-processes`：重新评估时的并行进程数
- `--no-overwrite`：若输出目录已存在则不覆盖

## 备注

- 根入口文件是 `run.py`
- 所有命令都可以先加 `--help` 查看完整参数
- 训练、验证、预测相关输出默认会写到 `training_results/` 或 `evaluation/results/`
