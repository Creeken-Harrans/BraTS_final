# Validation

验证相关代码已单独整理到 `validation/src/`：

- `validation/src/inference.py`：滑窗推理、分割恢复、case 级指标汇总
- `validation/src/visualization.py`：overlay、调试图、训练进度图
- `validation/src/find_best_config.py`：扫描 `summary.json`，按指标找出当前最优结果
- `validation/src/predict.py`：抽样训练集 case，运行预测并输出评估结果

命令行入口：

```bash
python run.py find-best-config
python run.py find-best-config --metric foreground_mean.Dice
python run.py find-best-config --search-root training_results evaluation/results
python run.py predict --sample-training-cases 12 --sample-seed 123
```
