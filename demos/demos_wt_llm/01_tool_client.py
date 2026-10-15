"""发现工具, 读参数schema, 关键词搜索, 手动调用."""
from _common import call, open_tools, show

def main():
    with open_tools(["iot"]) as tu:
        count = tu.load_tools(servers=["iot"])
        show("发现的工具数量", count)
        show("iot 工具列表", tu.list_tools())
        show("asset_ids 的参数说明", tu.tool_specification("iot.asset_ids"))
        show("按关键词查工具", [t["name"] for t in tu.find_tools("asset")])
        result = call(tu,"iot.asset_ids",site_name="MAIN")
        print(f"取出设备列表:{result['assets']}")

if __name__ == "__main__":
    main()