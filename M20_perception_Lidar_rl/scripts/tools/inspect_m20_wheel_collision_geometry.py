"""Inspect wheel collision geometry from the actual M20 USD loaded by Isaac."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pxr import Gf, Usd, UsdGeom, UsdPhysics


USD_PATH = "/home/ubuntu/m20_perceptive_final/deep_robotics_model/M20/usd/M20.usd"
WHEELS = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def main():
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError("failed to open " + USD_PATH)
    found = {wheel: [] for wheel in WHEELS}
    for wheel in WHEELS:
        instance = stage.GetPrimAtPath(f"/M20/{wheel}/collisions")
        assert instance.IsInstance(), wheel
        prototype = instance.GetPrototype()
        for prim in Usd.PrimRange(prototype):
            collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
            if not collision_api and not prim.IsA(UsdGeom.Cylinder) and not prim.IsA(UsdGeom.Mesh):
                continue
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            scale = Gf.Transform(transform).GetScale()
            item = {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "scale": tuple(float(x) for x in scale),
                "collision_api": collision_api,
            }
            if prim.IsA(UsdGeom.Cylinder):
                cylinder = UsdGeom.Cylinder(prim)
                item["radius"] = float(cylinder.GetRadiusAttr().Get())
                item["height"] = float(cylinder.GetHeightAttr().Get())
            elif prim.IsA(UsdGeom.Capsule):
                capsule = UsdGeom.Capsule(prim)
                item["radius"] = float(capsule.GetRadiusAttr().Get())
                item["height"] = float(capsule.GetHeightAttr().Get())
            found[wheel].append(item)
    for wheel, items in found.items():
        print("WHEEL_LINK=" + wheel)
        for item in items:
            print("COLLISION_PRIM=" + repr(item))
        collision_shapes = [item for item in items if item["type"] in ("Cylinder", "Capsule")]
        assert len(collision_shapes) == 1, (wheel, items)
        shape = collision_shapes[0]
        radial_scale = max(abs(shape["scale"][0]), abs(shape["scale"][2]))
        radius = shape["radius"] * radial_scale
        width = shape["height"] * abs(shape["scale"][1])
        print("OFFICIAL_WHEEL_COLLISION_SHAPE=" + shape["type"])
        print(f"OFFICIAL_WHEEL_RADIUS_{wheel[:2].upper()}={radius:.9f}")
        print(f"OFFICIAL_WHEEL_WIDTH_{wheel[:2].upper()}={width:.9f}")
        assert abs(radius - 0.09) < 1.0e-7
        assert abs(width - 0.054) < 1.0e-7
    print("OFFICIAL_WHEEL_ASSET_PATH=" + USD_PATH)
    print("OFFICIAL_WHEEL_GEOMETRY=PASS")


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
