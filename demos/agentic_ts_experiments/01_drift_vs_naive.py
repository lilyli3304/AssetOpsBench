
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from recipes_test_01 import recipes

# 找到仓库根目录, 让 python 能导入原来的 servers 包
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from servers.tsfm.engine.composition import build_forecaster

def predict(recipe, train, horizon):
    """根据配置即recipe, 用历史数据即train拟合模型，再预测未来 horizon 个点"""
    # 1. 把历史数据转换为模型接收的Series
    y = pd.Series(np.asarray(train, dtype=float))

    # 2. 复用AssetOpsBench; 把配置转换成模型对象.
    model = build_forecaster(recipe)

    # 3. 指定预测未来第 1,2,...,horizon 个点
    fh = list(range(1, horizon + 1))

    # 4. 只在历史数据上拟合, y->fh
    model.fit(y, fh=fh)

    # 5. 预测并转换为一组数组
    prediction = model.predict(fh=fh)
    return np.asarray(prediction, dtype=float).ravel()

# 绘图函数
def save_plot(values, train, comparison, scores):
    import matplotlib
    matplotlib.use("Agg")  # 只保存图片，不弹出窗口
    import matplotlib.pyplot as plt

    split = len(train)
    future_x = np.arange(split, len(values))
    score_map = {row["method"]: row["MAE"] for row in scores}

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True
    )

    # 左图：最近48点历史，以及未来真实值和预测。
    start = max(0, split - 48)
    axes[0].plot(
        np.arange(start, len(values)),
        values.iloc[start:],
        color="#334155", label="Actual", linewidth=2,
    )
    axes[0].axvspan(
        split - 0.5, len(values) - 0.5,
        color="#e0f2fe", alpha=0.7,
    )
    axes[0].axvline(split - 0.5, color="gray", linestyle=":")
    axes[0].set_title("Recent history and forecast")

    # 右图：放大测试窗口。
    axes[1].plot(
        future_x, comparison["actual"], "o-",
        color="#334155", label="Actual", linewidth=2,
    )
    axes[1].set_title(f"Forecast window ({len(comparison)} points)")

    # 把两种算法的预测画到左右两张图上。
    colors = {
        "original_drift": "#ea580c",
        "seasonal_naive": "#2563eb",
    }
    for name, color in colors.items():
        label = f"{name} (MAE={score_map[name]:.4f})"
        for ax in axes:
            ax.plot(
                future_x, comparison[name], "o--",
                color=color, label=label,
            )

    for ax in axes:
        ax.set_xlabel("Observation index")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "drift_vs_naive.png"

    fig.savefig(image_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\n图片已保存：{image_path}")

def main():
    # 构造240点数据：固定水平 + 24点周期 + 缓慢上升。
    # 暂时不加噪声、缺失，先理解最基本的预测流程。
    t = np.arange(240)
    values = pd.Series(
        20 + 4 * np.sin(2 * np.pi * t / 24) + 0.02 * t
    )

    # 最后6点作为测试答案，其余作为历史。
    horizon = 6
    train = values.iloc[:-horizon]
    truth = values.iloc[-horizon:].to_numpy()

    comparison = pd.DataFrame({
        "step": range(1, horizon + 1),
        "actual": truth,
    })

    scores = []

    # 对两个方案使用完全相同的历史和测试答案。
    for name, recipe in recipes(sp=24).items():
        prediction = predict(recipe, train, horizon)

        comparison[name] = prediction

        mae = float(np.mean(np.abs(truth - prediction)))
        scores.append({"method": name, "MAE": mae})

    print(f"历史点数：{len(train)}")
    print(f"测试点数：{len(truth)}")

    print("\n逐点比较：")
    print(comparison.round(4).to_string(index=False))

    print("\n平均绝对误差，越小越好：")
    print(pd.DataFrame(scores).round(4).to_string(index=False))

    save_plot(values, train, comparison, scores)

if __name__ == "__main__":
    main()







