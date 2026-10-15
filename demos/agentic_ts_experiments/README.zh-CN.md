# Agentic Time Series 手写练习

本目录用于自己重新实现实验。参考答案在项目根目录的 `examples/agentic_ts_experiments/`。
这里先放说明文件，Python 文件由你按学习进度逐个创建。

建议顺序：

1. `_common.py`：设置项目路径、读取数据、调用原预测引擎、计算指标。
2. `recipes.py`：先实现 drift 和季节 naive，再加入其他模型和变换。
3. `01_forecast_comparison.py`：统一时间切分，比较模型，再实现历史内选模。
4. `quality_policies.py`、`02_quality_ablation.py`：固定预测器，比较清洗策略。
5. 最后参考输入准备、作者 Agent 适配、核验与报告脚本。

本目录与参考目录的层级相同，所以在这里创建的 `_common.py` 中，
`Path(__file__).resolve().parents[2]` 仍然指向项目根目录。
结果路径也应根据当前脚本目录构造，使输出保存在本练习目录内。

## 运行自己的代码

在项目根目录的 PowerShell 中执行。先自行创建相应脚本，再运行：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe demos\agentic_ts_experiments\01_forecast_comparison.py --quick
```

## 用 Git 查看和提交

Git 跟踪文件，空文件夹不会单独进入版本控制。这个 README 可以保留目录。

新文件最初出现在 `git status` 中；普通 `git diff` 默认不显示未跟踪文件。
如果希望先用普通 diff 查看新文件，可以登记“准备添加”，此时尚未暂存文件内容：

```powershell
git add -N -- demos/agentic_ts_experiments/README.zh-CN.md
git diff -- demos/agentic_ts_experiments/
```

每完成一段练习，可以按下面的顺序暂存、检查和提交：

```powershell
git add -- demos/agentic_ts_experiments/
git diff --cached -- demos/agentic_ts_experiments/
git status --short
git commit -m "Add handwritten time series experiments"
```

`git add` 后应使用 `git diff --cached` 查看待提交内容。
`git commit` 会提交所有已暂存文件，因此提交前用 `git status` 检查暂存区。
