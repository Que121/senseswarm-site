#!/usr/bin/env python3
"""Build clean 2D occupancy maps of the HKUST Main Building from the SLABIM BIM meshes.

SLABIM (Liu et al., ICRA 2025, arXiv:2502.16856) ships the as-designed BIM of the HKUST Main Building as
per-storey, per-category meshes: floors.ply, walls.ply, columns.ply, doors.ply (PLY exported by Rhino,
coordinates in metres). We turn one storey into an occupancy grid by exploiting that category split, which
mirrors the IfcSpace / IfcWall / IfcDoor semantics of a BIM:

  free  = the floor slab footprint (floors.ply, projected to XY)  MINUS  the walls and columns
          (walls.ply + columns.ply, projected to XY)
  doors = doors.ply projected and dilated, used to re-open the wall where a doorway is, so rooms connect
          through their real doorways (the doors are a separate category, so the walls would otherwise seal
          each room)

Every element is rasterized the same way: each PLY face is triangulated with earcut (robust ear-clipping)
and the triangles are filled in XY. Vertical wall/column surfaces project to zero area and vanish, while the
horizontal slab caps fill the true footprint, so there are no fan or section artifacts (stray triangles or
diagonals across rooms).

We keep the largest connected free component as the explorable map. No artificial corridors are added: a
room is connected only where the building's own door geometry connects it.

Floors used: 1F to 5F. On 1F the BIM floor slab splits into the main concourse plus two detached
lecture-theatre slabs sitting 16 to 22 m away (separate levels); the connected map therefore keeps the main
concourse (~16000 m^2, CC 0.67) but not the detached theatres, because we do not fabricate a corridor that
the source geometry does not contain. 2F to 5F are each a single large connected floor (10000 to 24000 m^2,
CC 0.89 to 0.98). See benchmark/hkust/METHOD.md.

Source: HuggingFace dataset BobH62/SLABIM (BIM.zip, public). Repo: HKUST-Aerial-Robotics/SLABIM.
"""
import argparse, os, json
import numpy as np
import cv2
import scipy.ndimage as ndi
import mapbox_earcut as earcut
from plyfile import PlyData

HERE = os.path.dirname(os.path.abspath(__file__))


def load_polys(path):
    """Load a Rhino-exported PLY (tolerating the trailing `element material` block that trips trimesh's
    binary reader) as (vertices Nx3, faces) where faces is the list of ORIGINAL polygon vertex-index arrays
    (not triangulated), so concave faces can be filled correctly."""
    p = PlyData.read(path)
    ve = p["vertex"].data
    v = np.stack([ve["x"], ve["y"], ve["z"]], 1).astype(float)
    faces = [np.asarray(fc) for fc in p["face"].data["vertex_indices"]]
    return v, faces


def _project(verts, faces, xmin, ymin, res, H, W):
    """Project polygon faces to XY and rasterize. Each face is triangulated with earcut (a robust ear-clipping
    triangulator) rather than a fan, so concave / curved faces (e.g. the semicircular lecture theatre) fill
    cleanly without the stray triangles a fan or even-odd fillPoly produces."""
    g = np.zeros((H, W), np.uint8)
    for fc in faces:
        ring = np.ascontiguousarray(verts[fc][:, :2], dtype=np.float64)
        try:
            idx = earcut.triangulate_float64(ring, np.array([len(ring)]))
        except Exception:
            continue
        for t in range(0, len(idx), 3):
            tri = ring[[idx[t], idx[t + 1], idx[t + 2]]]
            p = np.stack([((tri[:, 0] - xmin) / res + 2), ((tri[:, 1] - ymin) / res + 2)], 1).astype(np.int32)
            cv2.fillConvexPoly(g, p, 1)
    return g


def floor_occupancy(mesh_dir, res=0.10, door_dilate_m=1.2):
    """Return (occupancy uint8 255=obstacle/0=free, info). Free = floor minus walls/columns, doors re-opened,
    largest connected component (no artificial bridging). Every element is rasterized by earcut-triangulated
    XY projection: vertical wall/column surfaces project to zero area and vanish, while the slab caps fill the
    true footprint, so there are no section-ordering artifacts."""
    fl_v, fl_f = load_polys(os.path.join(mesh_dir, "floors.ply"))
    dr_v, dr_f = load_polys(os.path.join(mesh_dir, "doors.ply"))
    wl_v, wl_f = load_polys(os.path.join(mesh_dir, "walls.ply"))
    cl_v, cl_f = load_polys(os.path.join(mesh_dir, "columns.ply"))
    allv = np.vstack([fl_v[:, :2], wl_v[:, :2]])
    xmin, ymin = allv.min(0)
    xmax, ymax = allv.max(0)
    W = int((xmax - xmin) / res) + 4
    H = int((ymax - ymin) / res) + 4
    floor = _project(fl_v, fl_f, xmin, ymin, res, H, W)                              # walkable floor slab
    wall = _project(wl_v, wl_f, xmin, ymin, res, H, W) | _project(cl_v, cl_f, xmin, ymin, res, H, W)
    door = _project(dr_v, dr_f, xmin, ymin, res, H, W)
    d = max(2, int(round(door_dilate_m / res)))
    door = cv2.dilate(door, np.ones((d, d), np.uint8))
    free = ((floor > 0) & ((wall == 0) | (door > 0))).astype(np.uint8)
    lbl, _ = ndi.label(free)
    sz = np.bincount(lbl.ravel())[1:]
    big = lbl == (1 + int(sz.argmax()))
    occ = np.where(big, 0, 255).astype(np.uint8)
    info = dict(res_m=res, shape=[H, W], width_m=round(xmax - xmin, 1), height_m=round(ymax - ymin, 1),
                free_area_m2=round(int(big.sum()) * res * res), floor_area_m2=round(int(floor.sum()) * res * res),
                largest_cc=round(big.sum() / free.sum(), 3))
    return occ, info


def save(occ, info, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{name}.npy"), occ)
    json.dump(info, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=2)
    from PIL import Image
    Image.fromarray(np.flipud(np.where(occ > 0, 0, 255).astype(np.uint8)), "L").save(os.path.join(out_dir, f"{name}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bim", default="/mnt/data1/hkust_slabim/BIM", help="SLABIM BIM dir with <X>F/mesh/*.ply")
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark", "hkust"))
    a = ap.parse_args()
    for fl in ["1F", "2F", "3F", "4F", "5F"]:                     # 1F keeps the main concourse (detached theatres aside)
        occ, info = floor_occupancy(os.path.join(a.bim, fl, "mesh"), a.res)
        info["floor"] = fl
        save(occ, info, a.out, f"hkust_{fl.lower()}")
        print(f"hkust_{fl.lower()}: {info['width_m']}x{info['height_m']} m  free={info['free_area_m2']} m2  cc={info['largest_cc']}")


if __name__ == "__main__":
    main()
