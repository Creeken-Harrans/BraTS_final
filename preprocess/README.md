# Preprocess

这一阶段不再保留旧项目里的自动 fingerprint 和自动 planning 主链。

在 `BraTS_final` 里，`preprocess` 的角色已经改成：

- 固定 BraTS 数据集
- 固定预处理参数
- 固定输出组织方式

也就是说，它现在是一个固定预处理层，而不是一个为任意数据集自动设计训练方案的系统。

## 当前原则

新项目不再把下面这些内容视为运行时主流程的一部分：

- 自动提取 `dataset_fingerprint.json`
- 自动生成 `ProjectPlans.json`
- 多 configuration 选择
- planner 驱动的动态建网

运行时真正需要的是：

- `dataset.json`
- `splits_final.json`
- 固定的预处理配置

## 目录说明

- `config/`
  固定预处理配置和模型输入配置。
- `metadata/`
  当前正式元数据和历史参考文件。
- `logs/`
  预处理日志目录。

## 当前正式文件

- `metadata/dataset.json`
- `metadata/splits_final.json`
- `config/preprocess_config.json`
- `config/model_input_config.json`

## 当前参考文件

这些文件只作为旧系统迁移时的来源归档，不应该再被新主流程直接读取：

- `metadata/reference_ProjectPlans.json`
- `metadata/reference_dataset_fingerprint.json`
- `metadata/reference_nnUNetPlans.json`

## 固定下来的关键参数

当前从旧 `ProjectPlans.json` 提炼出来的固定 3D 配置是：

- target spacing: `[1.0, 1.0, 1.0]`
- patch size: `[128, 128, 128]`
- batch size: `2`
- data identifier: `brats_fixed_3d_fullres`

## 下一步方向

后续这里会补一个新的固定预处理脚本，例如：

- `scripts/preprocess_brats.py`

它只做一条固定路线：

- 从 `BraTS_final_Dataset/nnUNet_raw/Dataset220_BraTS2020` 读取
- 输出到 `BraTS_final_Dataset/nnUNet_preprocessed/Dataset220_BraTS2020`
- 不再经过 fingerprint / planner / plans handler
