"""03：发现实测传感器，查询时间范围，翻页，统计，比较传感器清单。"""
# src/servers/iot/main.py

import argparse
from datetime import datetime, timedelta

from _common import call, open_tools, show

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="MAIN")
    parser.add_argument("--asset", default="Chiller 6")
    args = parser.parse_args()
    base = {"site_name": args.site, "asset_id": args.asset}
    with open_tools(["iot"]) as tu:
        tu.load_tools(["iot"])
        # @mcp.tool(title="List Installed Sensors")
        # def installed_sensors(
        installed = call(tu, "iot.installed_sensors", **base)["sensors"]
        # @mcp.tool(title="List Measured Sensors")
        # def measured_sensors(
        measured = call(tu, "iot.measured_sensors", **base)["sensors"]
        show("传感器清单比较", {
            "两边都有": sorted(set(installed) & set(measured)),
            "登记但未测到": sorted(set(installed) - set(measured)),
            "测到但未登记": sorted(set(measured) - set(installed)),
        })
        if not measured:
            raise RuntimeError("设备没有实测字段，请换一台有 IoT 数据的设备")
        sensor = measured[0]  # 从发现结果取名，避免猜传感器字段。
        extent = call(tu, "iot.stream_extent", **base, sensor=sensor)
        start = datetime.fromisoformat(extent["start_time"])
        last = datetime.fromisoformat(extent["end_time"])
        interval = extent["approx_interval_seconds"] or 1
        # [start, end) 左闭右开；末条记录后加 1 微秒以便包含末条。
        end = min(start + timedelta(seconds=max(interval, 1) * 12),
                  last + timedelta(microseconds=1))
        query = {**base, "start": start.isoformat(), "end": end.isoformat(),
                 "sensors": [sensor], "limit": 3}
        # @mcp.tool(title="Get Sensor History")
        # def history(
        page = call(tu, "iot.history", **query)
        rows = list(page["observations"])
        if page["has_more"]:
            page = call(tu, "iot.history", **query, cursor=page["next_cursor"])
            rows.extend(page["observations"])
        show("已读两页的记录数（不是全窗口总数）", len(rows))
        # @mcp.tool(title="Sensor Statistics")
        # def sensor_stats(
        call(tu, "iot.sensor_stats", **base, sensor=sensor,
             start=query["start"], end=query["end"])
        # @mcp.tool(title="Latest Reading")
        # def latest_reading(
        call(tu, "iot.latest_reading", **base, sensor=sensor)
        # @mcp.tool(title="Sensor Coverage")
        # def sensor_coverage(
        call(tu, "iot.sensor_coverage", **base)
        # @mcp.tool(title="Find Assets By Sensors")
        # def find_assets_by_sensors(
        call(tu, "iot.find_assets_by_sensors", site_name=args.site,
             sensors=[sensor], source="measured", match="all")

if __name__ == "__main__":
    main()