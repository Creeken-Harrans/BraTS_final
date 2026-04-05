# Training

训练代码统一收进 `training/src/`，并按功能拆分：

- `training/src/core/`：训练配置、运行时路径、主训练器
- `training/src/data/`：数据集、采样器、标签管理、数据增强
- `training/src/optimization/`：损失函数、优化器、调度器、预训练权重加载
- `training/src/monitoring/`：训练日志与异步绘图
- `training/src/utils.py`：训练流程共用小工具

训练命令入口仍然是仓库根目录的 `run.py` / `cli.py`。
