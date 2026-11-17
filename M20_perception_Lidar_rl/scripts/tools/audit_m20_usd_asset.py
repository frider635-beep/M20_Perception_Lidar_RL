#!/usr/bin/env python3
"""Dump physics-relevant properties from the M20 USD without launching Isaac Sim."""

from __future__ import annotations

import argparse
import json

from pxr import Usd, UsdGeom, UsdPhysics


def value(attribute):
    result = attribute.Get() if attribute else None
    if result is None:
        return None
    if hasattr(result, "__iter__") and not isinstance(result, str):
        return list(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd")
    args = parser.parse_args()
    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {args.usd}")

    output = {"rigid_bodies": [], "joints": [], "collisions": []}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            output["rigid_bodies"].append(
                {
                    "path": path,
                    "mass": value(mass.GetMassAttr()),
                    "center_of_mass": value(mass.GetCenterOfMassAttr()),
                    "diagonal_inertia": value(mass.GetDiagonalInertiaAttr()),
                    "principal_axes": value(mass.GetPrincipalAxesAttr()),
                }
            )
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            output["joints"].append(
                {
                    "path": path,
                    "body0": [str(item) for item in joint.GetBody0Rel().GetTargets()],
                    "body1": [str(item) for item in joint.GetBody1Rel().GetTargets()],
                    "axis": value(joint.GetAxisAttr()),
                    "lower_limit": value(joint.GetLowerLimitAttr()),
                    "upper_limit": value(joint.GetUpperLimitAttr()),
                    "local_pos0": value(joint.GetLocalPos0Attr()),
                    "local_pos1": value(joint.GetLocalPos1Attr()),
                    "local_rot0": value(joint.GetLocalRot0Attr()),
                    "local_rot1": value(joint.GetLocalRot1Attr()),
                }
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            transform = UsdGeom.Xformable(prim)
            output["collisions"].append(
                {
                    "path": path,
                    "type": prim.GetTypeName(),
                    "xform_ops": [str(item) for item in transform.GetOrderedXformOps()],
                    "radius": value(prim.GetAttribute("radius")),
                    "height": value(prim.GetAttribute("height")),
                    "size": value(prim.GetAttribute("size")),
                    "extent": value(prim.GetAttribute("extent")),
                }
            )
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
