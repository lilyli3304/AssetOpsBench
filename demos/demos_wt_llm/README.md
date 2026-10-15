# 自己动手练习

这里用于参照 `examples/learning_no_llm_answers/`，重新独立实现各课练习。

建议沿用课号命名，例如 `01_tool_client.py`、`02_asset_registry.py`，方便对照。

在项目根目录运行自己写的文件，例如：

```powershell
uv run python demos/demos_wt_llm/01_tool_client.py
```

上面的 Python 文件需要自己创建；本目录目前只有这份说明。

Git 不记录空目录；这份 README 让目录可以随文件一起提交。新增练习文件后，在项目根目录执行：

```powershell
git status --short -- demos/demos_wt_llm/
git add demos/demos_wt_llm/
git diff --cached -- demos/demos_wt_llm/
git commit -m "Add personal learning demos"
```

新文件在 `git add` 前显示为未跟踪文件，不出现在普通 `git diff` 中。加入暂存区后用 `git diff --cached` 查看；已跟踪文件后续的未暂存修改用普通 `git diff` 查看。
