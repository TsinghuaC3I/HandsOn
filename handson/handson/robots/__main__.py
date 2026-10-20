"""Inspect robot descriptions: python -m handson.robots --help."""
import argparse
import json

from .assets import _catalog, get_robot_asset, list_robot_assets, validate_robot_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=list_robot_assets())
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--simulation", action="store_true", help="apply an explicit estimated dynamics profile")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check", action="store_true")
    operation.add_argument("--prepare-urdf", metavar="OUTPUT")
    operation.add_argument("--install-from", metavar="DIRECTORY", help="import authorized user-obtained assets; no download")
    operation.add_argument("--sources", action="store_true", help="show external model sources and availability")
    args = parser.parse_args()
    if args.simulation and not args.prepare_urdf:
        parser.error("--simulation requires --prepare-urdf")
    try:
        if args.install_from:
            from .external import external_catalog, install_external_assets
            if args.robot not in external_catalog():
                parser.error("--install-from requires an external --robot")
            print(json.dumps(install_external_assets(args.robot, args.install_from), indent=2))
        elif args.sources:
            from .external import external_catalog, external_status
            selected = [args.robot] if args.robot else external_catalog()
            print(json.dumps({key: external_status(key) for key in selected}, indent=2))
        elif args.check:
            print(json.dumps(validate_robot_assets(args.robot), indent=2))
        elif args.prepare_urdf:
            if not args.robot:
                parser.error("--prepare-urdf requires --robot")
            print(get_robot_asset(args.robot, side=args.side).prepare_urdf(args.prepare_urdf, simulation=args.simulation))
        elif args.robot:
            asset = get_robot_asset(args.robot, side=args.side)
            print(json.dumps({**vars(asset), "path": str(asset.path), "source_record": str(asset.source_record)}, indent=2))
        else:
            for name in list_robot_assets():
                spec = _catalog()["robots"][name]
                print(f"{name:24} {spec['kind']:4} {spec['model']} [{spec.get('distribution', 'bundled')}]")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
