# Preprocess Guide

这份文档是 `BraTS_final` 的简化版预处理说明。

## 1. 旧系统和新系统的区别

旧 `BraTS` 项目的 `preprocess` 负责：

1. 提取 fingerprint
2. 根据 fingerprint 生成 plans
3. 根据 plans 执行 preprocess

新 `BraTS_final` 项目不再保留前两步作为运行时能力。

现在的策略是：

1. 数据集固定
2. 预处理参数固定
3. 网络结构固定

因此预处理层只需要负责：

- 把 raw 数据转换成固定训练输入表示
- 保持输出目录和训练输入契约稳定

## 2. 目前保留的正式输入

- `data_preparation` 生成的 `Dataset220_BraTS2020`
- `metadata/dataset.json`
- `metadata/splits_final.json`
- `config/preprocess_config.json`

## 3. 目前降级为参考的历史文件

- `reference_dataset_fingerprint.json`
- `reference_ProjectPlans.json`
- `reference_nnUNetPlans.json`

这些文件的作用只剩下：

- 解释固定参数的来源
- 方便人工核对历史配置

它们不再是新主流程的代码依赖。

## 4. 当前固定配置来源

当前固定配置来自旧项目 `ProjectPlans.json` 的 `3d_fullres`：

- spacing: `[1.0, 1.0, 1.0]`
- patch size: `[128, 128, 128]`
- batch size: `2`
- 3D `PlainConvUNet`

## 5. 后续实现目标

后续新脚本只需要覆盖一条固定链路：

- 读取 raw case
- 做固定的裁剪、归一化、重采样
- 保存固定格式的预处理结果
- 准备 `gt_segmentations`

不再支持：

- planner 自动决策
- 多 configuration 切换
- 动态网络设计
