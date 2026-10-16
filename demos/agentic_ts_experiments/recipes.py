
def recipes(sp=24):
    """返回两个预测方案; 这里只定义配置, 不训练模型"""

    model_class = "sktime.forecasting.naive.NaiveForecaster"

    return {
        # 根据历史首尾的平均变化, 向未来外推
        "original_drift": {
            "estimator": {
                "sktime_class": model_class,
                "params": {"strategy": "drift"},
            }
        },

        # 重复最近一个周期；sp=24 表示一个周期有24点。
        "seasonal_naive": {
            "estimator": {
                "sktime_class": model_class,
                "params": {"strategy": "last", "sp": sp},
            }
        },
    }