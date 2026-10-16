"""只改新文件中的 recipe，就可以练习替换算法、组合变换和模型。"""
from copy import deepcopy

NAIVE = "sktime.forecasting.naive.NaiveForecaster"
ES = "sktime.forecasting.exp_smoothing.ExponentialSmoothing"


def recipes(sp: int) -> dict:
    drift = {"estimator": {"sktime_class": NAIVE, "params": {"strategy": "drift"}}}
    seasonal = {"estimator": {"sktime_class": NAIVE, "params": {"strategy": "last", "sp": sp}}}
    detrended = deepcopy(seasonal)
    detrended["transforms"] = [
        {"sktime_class": "sktime.transformations.series.detrend.Detrender", "params": {}}
    ]
    result = {
        "original_drift": drift,  # 原学习脚本 11 的模型
        "original_es_sp_only": {"estimator": {"sktime_class": ES, "params": {"sp": sp}}},
        "seasonal_naive": seasonal,
        "detrend_seasonal": detrended,
        "theta": {"estimator": {"sktime_class": "sktime.forecasting.theta.ThetaForecaster",
                                 "params": {"sp": sp, "deseasonalize": sp > 1}}},
        "holt_winters": {"estimator": {"sktime_class": ES, "params": {
            "sp": sp, "trend": "add", "seasonal": "add" if sp > 1 else None,
            "damped_trend": True,
        }}},
        "mean_ensemble": {"ensemble": {"combine": "mean", "members": [
            {"name": "drift", **drift["estimator"]},
            {"name": "seasonal", **seasonal["estimator"]},
        ]}},
    }
    return result


# 最小策略选择实验：候选集合固定，不能看外层测试误差选择模型。
SELECTOR_CANDIDATES = ["original_drift", "seasonal_naive", "detrend_seasonal", "theta", "holt_winters"]
