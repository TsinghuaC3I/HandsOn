"""Inspect source GLTF -> OBJ -> native USD UVs and normal basis handedness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

from handson.tasks.eval_gen07_stack_boxes.visual_appearance import ASSET_DIR, BODY_NAMES, SOURCE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gltf = json.loads((SOURCE_DIR / "scene.gltf").read_text())
    binary = (SOURCE_DIR / "scene.bin").read_bytes()

    def accessor(index):
        a = gltf["accessors"][index]
        view = gltf["bufferViews"][a["bufferView"]]
        assert not a.get("sparse")
        count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        dtype = np.dtype({5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]])
        return np.ndarray((a["count"],count), dtype=dtype, buffer=binary,
            offset=view.get("byteOffset",0)+a.get("byteOffset",0),
            strides=(view.get("byteStride",dtype.itemsize*count),dtype.itemsize))

    mesh = next(m for m in gltf["meshes"] if m["name"] == "Box_big__CardboardBoxes__0")
    primitive = mesh["primitives"][0]
    attributes = primitive["attributes"]
    positions = accessor(attributes["POSITION"]).astype(float)
    normals = accessor(attributes["NORMAL"]).astype(float)
    tangents = accessor(attributes["TANGENT"])
    source_uv = accessor(attributes["TEXCOORD_0"])
    triangles = accessor(primitive["indices"]).reshape(-1,3)
    expected_st = source_uv.copy(); expected_st[:,1] = 1-expected_st[:,1]
    obj_uv = np.array([[float(x) for x in l.split()[1:3]] for l in (ASSET_DIR/"visuals/base.obj").read_text().splitlines() if l.startswith("vt ")])
    obj_error = float(np.abs(obj_uv-expected_st).max())
    assert obj_error < 1e-7
    stage = Usd.Stage.Open(str(args.stage))
    report = {"source_component": mesh["name"], "source_uv_count": len(source_uv), "triangle_count": len(triangles),
        "obj_st_formula": "(source_u, 1-source_v)", "max_obj_st_error": obj_error,
        "source_tangent_w_values": np.unique(tangents[:,3]).tolist(), "usd_meshes": [],
        "texture_samples": [], "normal_basis_test": {}, "passed": False,
        "scope": "Source/OBJ/native-USD face-corner and pixel coordinate audit. Triangle-derivative tangent basis parity is measured; source averaged glTF tangent interpolation versus RTX generated tangents is not certified."}
    for name in BODY_NAMES:
        variant = "base" if name == BODY_NAMES[0] else "small"
        path = f"/World/Task/{name}/{name}/visuals/gen5_{variant}_visual_mesh/gen5_{variant}_visual_mesh"
        mesh = UsdGeom.Mesh.Get(stage,path)
        uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        actual = np.asarray(uv.ComputeFlattened())
        expected = expected_st[triangles.ravel()]
        error = float(np.abs(actual-expected).max())
        assert actual.shape == (132,2) and error == 0.0
        report["usd_meshes"].append({"path":path,"face_corner_count":len(actual),"max_st_error":error,"uv_interpolation":uv.GetInterpolation(),
            "authored_tangent_primvar_present":any("tangent" in a.GetName().lower() for a in mesh.GetPrim().GetAttributes())})

    # Sample all 44 source triangle centroids in the actual image files.
    # glTF image coordinates start at the top; USD st uses the opposite V origin.
    raw_centroids = source_uv[triangles].astype(float).mean(axis=1)
    usd_centroids = expected_st[triangles].astype(float).mean(axis=1)
    for file in sorted((SOURCE_DIR/"textures").glob("*.png")):
        pixels = np.asarray(Image.open(file).convert("RGB"))
        h,w = pixels.shape[:2]
        raw_pixel = np.floor(raw_centroids*np.array([w,h])).astype(int)
        converted = usd_centroids.copy(); converted[:,1] = 1-converted[:,1]
        usd_pixel = np.floor(converted*np.array([w,h])).astype(int)
        assert np.array_equal(raw_pixel,usd_pixel)
        colors = pixels[raw_pixel[:,1].clip(0,h-1),raw_pixel[:,0].clip(0,w-1)]
        report["texture_samples"].append({"file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
            "shape":list(pixels.shape),"sample_count":44,"same_image_pixels":True,
            "sample_pixel_coordinates":raw_pixel.tolist(),"sample_rgb_sha256":hashlib.sha256(colors.tobytes()).hexdigest()})

    def basis(ids, uv):
        p = positions[ids]; tex = uv[ids].astype(float)
        deriv = np.linalg.solve(np.stack([tex[1]-tex[0],tex[2]-tex[0]]),np.stack([p[1]-p[0],p[2]-p[0]]))
        n = normals[ids].mean(axis=0); n /= np.linalg.norm(n)
        t = deriv[0]-n*np.dot(n,deriv[0]); t /= np.linalg.norm(t)
        b = np.cross(n,t)*np.sign(np.dot(np.cross(n,t),deriv[1]))
        return np.column_stack([t,b,n])

    normal_pixels = np.asarray(Image.open(SOURCE_DIR/"textures/CardboardBoxes_normal.png").convert("RGB"),dtype=float)/255
    h,w = normal_pixels.shape[:2]
    compensated, uncorrected, reversed_b = [], [], []
    for index,ids in enumerate(triangles):
        a,b = basis(ids,source_uv),basis(ids,expected_st)
        u,v = raw_centroids[index]; encoded = normal_pixels[min(h-1,int(v*h)),min(w-1,int(u*w))]
        normal = 2*encoded-1
        compensated.append(float(np.max(np.abs(a@normal-b@(normal*np.array([1,-1,1]))))))
        uncorrected.append(float(np.max(np.abs(a@normal-b@normal))))
        reversed_b.append(float(np.dot(a[:,1],b[:,1])))
    assert max(compensated)<1e-6 and max(reversed_b)<-.999999
    report["normal_basis_test"]={"triangle_count":44,"max_compensated_world_normal_error":max(compensated),
        "max_uncorrected_world_normal_error":max(uncorrected),"max_original_vs_usd_bitangent_dot":max(reversed_b),
        "required_decode_scale":[2,-2,2,1],"required_decode_bias":[-1,1,-1,0]}
    report["passed"] = True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":True,"max_obj_st_error":obj_error,"normal_basis_test":report["normal_basis_test"]},indent=2))


if __name__ == "__main__":
    main()
