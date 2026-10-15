
import argparse

from _common import call, open_tools

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="MAIN")
    parser.add_argument("--asset", default="Chiller 6")
    args = parser.parse_args()
    with open_tools(["iot"]) as tu:
        tu.load_tools(["iot"])
        call(tu, "iot.sites")
        call(tu, "iot.asset_ids", site_name=args.site)
        call(tu, "iot.assets", site_name=args.site)
        call(tu, "iot.assets", site_name=args.site, assettype="CHILLER")
        call(tu, "iot.asset_detail", site_name=args.site, asset_id=args.asset)
        call(tu, "iot.installed_sensors", site_name=args.site, asset_id=args.asset)

if __name__ == "__main__":
    main()
