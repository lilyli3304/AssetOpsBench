"""参考答案共用的小工具；不修改仓库原有模块。"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "src/couchdb/scenarios_data/shared"
OUTPUT = HERE / "_outputs"

# 所有相对路径按仓库根目录解释；直接运行单个文件也能找到 src 下的包。
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
load_dotenv(ROOT / ".env")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def plain(value):
    """Pydantic、dataclass 和 numpy 数据转为可打印的普通值。"""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def show(title, value):
    print(f"\n=== {title} ===")
    print(json.dumps(value, indent=2, ensure_ascii=False, default=plain))


def checked(value):
    """解包 result 并识别业务错误，避免把失败返回误当成成功数据。"""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, dict) and set(value) == {"result"}:
        value = value["result"]
    if isinstance(value, dict):
        if value.get("error"):
            raise RuntimeError(str(value["error"]))
        if value.get("success") is False or value.get("status") == "error":
            raise RuntimeError(str(value))
    return value


def call(tu, name, **arguments):
    """仍通过仓库的 ToolUniverse.run() 调用真实 MCP 工具。"""
    result = checked(tu.run({"name": name, "arguments": arguments}))
    show(name, result)
    return result


def open_tools(names, env=None):
    """沿用当前 Python 环境启动服务，省去子进程再次 uv sync 的步骤。"""
    from mcphub import ToolUniverse

    server_commands = {
        name: [sys.executable, "-m", f"servers.{name}.main"] for name in names
    }
    child_env = {"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    child_env.update(env or {})
    return ToolUniverse(servers=server_commands, env=child_env)


def output_dir(lesson):
    path = OUTPUT / lesson
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=plain),
                    encoding="utf-8")
    return path


def create_practice_database(name):
    """只创建指定练习库，已存在则保留；不删除或重建任何库。"""
    import requests

    if name not in {"aob_learning_failure_mode", "aob_learning_workorder"}:
        raise ValueError("只允许本答案约定的两个练习数据库名称")
    url = os.environ.get("COUCHDB_URL", "http://localhost:5984").rstrip("/")
    auth = (os.environ.get("COUCHDB_USERNAME", "admin"),
            os.environ.get("COUCHDB_PASSWORD", "password"))
    response = requests.put(f"{url}/{name}", auth=auth, timeout=10)
    if response.status_code != 412:  # 412 = 已存在
        response.raise_for_status()
    show("练习数据库", name)


def prepare_tsfm(lesson):
    """使用仓库现有 MemoryStore，并从随仓库目录数据播种；不下载模型。"""
    os.environ["TSFM_STORE"] = "memory"
    os.environ["TSFM_WORKDIR"] = str(output_dir(lesson))
    from servers.tsfm import main as tsfm
    from servers.tsfm.stores import feature_store, model_store

    for kind, collection, id_field, prefix in [
        ("model", model_store.collection_name(), "model_id", "model"),
        ("feature", feature_store.collection_name(), "feature_id", "feature"),
    ]:
        docs = json.loads((DATA / f"tsfm/{kind}_catalog.json").read_text(encoding="utf-8"))
        for doc in docs:
            doc.setdefault("_id", f"{prefix}:{doc[id_field]}")
            store = tsfm._STORE if kind == "model" else tsfm._FEATURE_STORE
            store.put(collection, doc)
    return tsfm


def make_series(path, dirty=False):
    """固定随机种子的合成数据：趋势 + 24 点周期 + 少量噪声。"""
    import numpy as np
    import pandas as pd

    t = np.arange(240)
    values = 20 + 4 * np.sin(2 * np.pi * t / 24) + 0.02 * t
    values += np.random.default_rng(42).normal(0, 0.1, len(t))
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(t), freq="h"),
        "value": values,
    })
    if dirty:
        frame.loc[[5, 25], "value"] = np.nan
    frame.to_csv(path, index=False)
    return frame
