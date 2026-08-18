#!/usr/bin/env python3
"""Compose real KTH floor plans into large connected CORRIDOR buildings for the exploration benchmark.

A single KTH Campus Valhallavagen floor saturates well before 1000 robots, so we build three buildings
of growing area by composing many real floors with a deterministic, reproducible rule:

  1. Floor pool. Floors come from the GT-FILLED maps (kth_maps/gt_filled/GTF_<id>.npy, built by
     kth_gt_raster.py from SenseExpo's human-corrected GT.bmp drawings - the 92 successful conversions),
     restricted to LANDSCAPE floors of CONSISTENT height (12 to 16 m tall, aspect > 1.9). Consistent height
     is what makes the stitch clean: every floor becomes a uniform-height corridor band, so bands abut
     without ragged gaps. Floors are drawn from DIVERSE buildings (de-duplicated by KTH building folder, at
     most 4 floors per building) so the corridor is varied rather than repeated near-identical plans.
     Selection is seeded and reproducible.

  2. One corridor (a "layer"). Floors are chained left to right and SPINE-ALIGNED: each floor is placed so
     its main corridor row (the grid row carrying the most free space) sits on one common horizontal line.
     At every seam ONE straight rectangular corridor segment is carved on that common spine line, so the
     primary corridors fuse into a single straight corridor running the whole strip - clean, axis-aligned,
     no organic doorway nibbles. All original walls are kept.

  3. Stacking into 1 / 2 / 3 layers. The single, double and triple buildings stack 1, 2, 3 such corridors.
     Layers are joined by UNIFORM vertical shafts: a 3 m wide straight rectangular cut every ~24 m along
     each seam, each placed at the locally shortest free-to-free crossing - regular, square, staircase-like.
     Area scales ~1:2:3, giving three robot-count regimes.

Output per building: a uint8 occupancy array (255 = obstacle, 0 = free; simulator convention), a metre-axis
metadata file, and a pixel-exact PNG (wall black, free white). The largest connected free component is
reported; the simulator's coverage metric is taken over that reachable component.
"""
import argparse, os, json, random, math
from collections import defaultdict
import numpy as np
import scipy.ndimage as ndi
from gen_kth import load_floor, build_grid
from gen_houseexpo import decompose, deploy_cell, deploy_box

HERE = os.path.dirname(os.path.abspath(__file__))
GT_TILES = os.path.join(HERE, "kth_maps", "xml_tiles")   # ORIGINAL KTH XML vector rasterizations
GT_RES = 0.10
REPAIR = False   # vector tiles are clean by construction; GT.bmp pools (gt_filled_*) need repair_walls


def floor_pool(tiles_dir=GT_TILES):
    """All accepted GT-filled floor ids: GTF_*.npy minus EXCLUDE.txt."""
    excl = set()
    ex = os.path.join(tiles_dir, "EXCLUDE.txt")
    if os.path.exists(ex):
        excl = {l.split("#")[0].strip() for l in open(ex) if l.split("#")[0].strip()}
    ids = [f[4:-4] for f in sorted(os.listdir(tiles_dir)) if f.startswith("GTF_") and f.endswith(".npy")]
    return [i for i in ids if i not in excl]


def _building_key(fid):
    """XML-pool ids are <building-folder>_<floor> (A0043006_50023964): the folder IS the building. GT.bmp
    ids group sibling floors by drawing number with the last digit dropped."""
    if fid.startswith("A00"):
        return fid.split("_")[0]
    digits = "".join(c for c in fid.split("_")[0] if c.isdigit())
    return digits[:-1] if len(digits) > 1 else fid


def repair_walls(t, res, thick_m=0.30, close_m=0.40, blob_m=0.50):
    """Normalize the drawing style so every tile reads cleanly at benchmark scale: (1) directional closing
    heals sub-door breaks (< close_m) in dashed/comb wall chains, (2) walls are SKELETONIZED and redrawn at
    a uniform thick_m (kills the blobby variable-thickness look), (3) massive solid cores (half-width >=
    blob_m: shafts, courtyard fills, thick structural masses) are kept as drawn. Door gaps (drawn >= 0.5 m)
    are untouched; free space can only grow where over-thick walls slim down."""
    from skimage.morphology import skeletonize
    k = max(2, int(round(close_m / res)))
    wall = ndi.binary_closing(t, np.ones((1, k), bool))
    wall = ndi.binary_closing(wall, np.ones((k, 1), bool))
    d = ndi.distance_transform_edt(wall)
    blobs = ndi.binary_propagation(d > blob_m / res, mask=wall)  # full extent of the thick masses
    r = max(1, int(round(thick_m / res / 2)))
    skel = ndi.binary_dilation(skeletonize(wall), ndi.generate_binary_structure(2, 2), iterations=r)
    new = skel | blobs
    # the closing can seal narrow drawn doors: re-open any originally-free ribbon that separates two free
    # components (same mechanism as the rasterizer's door re-opening), then fill sub-3m2 isolated slivers
    from kth_gt_raster import _adjacency
    free0 = ~t
    for _ in range(6):
        flbl, fn = ndi.label(~new)
        if fn <= 1:
            break
        plbl, pn = ndi.label(free0 & new)
        if pn == 0:
            break
        adj = _adjacency(plbl, flbl)
        open_ids = [p for p, fs in adj.items() if len(fs) >= 2]
        if not open_ids:
            break
        new &= ~np.isin(plbl, open_ids)
    # everything still disconnected is not a room (raw v6 free space is a single component and all drawn
    # doors were re-opened above): exterior "moat" bands freed by thinning the border fill, or slivers.
    flbl, fn = ndi.label(~new)
    if fn > 1:
        sz = ndi.sum(np.ones_like(flbl), flbl, range(1, fn + 1))
        new |= (flbl > 0) & (flbl != 1 + int(np.argmax(sz)))
    return new


def gt_tile(fid, res, tiles_dir=GT_TILES):
    """Load one GT-filled floor (GTF_<id>.npy, 1 = obstacle at 0.10 m) as a bool wall grid at res,
    cropped to the free footprint + 1 cell so it composes tightly in the stitcher."""
    occ = np.load(os.path.join(tiles_dir, f"GTF_{fid}.npy"))
    if abs(res - GT_RES) > 1e-6:
        import cv2
        s = GT_RES / res
        occ = cv2.resize(occ, (int(round(occ.shape[1] * s)), int(round(occ.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
    occ = occ.astype(bool)
    if REPAIR:
        occ = repair_walls(occ, res)
    ys, xs = np.where(~occ)
    i0, j0 = max(0, ys.min() - 1), max(0, xs.min() - 1)
    return occ[i0:min(occ.shape[0] - 1, ys.max() + 1) + 1, j0:min(occ.shape[1] - 1, xs.max() + 1) + 1].copy()


def select_floors(res, n, seed, h_lo=10.0, h_hi=18.0, min_aspect=1.9,
                  w_lo=28.0, w_hi=84.0, max_per_building=4, band_m=1.0, min_rect=0.88, tiles_dir=GT_TILES):
    """Pick n landscape floors of NEAR-IDENTICAL height from diverse buildings: gather candidates in a
    loose band, then keep only the densest band_m-tall height window (near-even tile tops = no big dead
    bands above short tiles in the strip). The window auto-widens in 0.5 m steps if n floors can't be
    drawn from it. Returns list of bool wall grids."""
    def candidates(rect):
        cand = []
        for fid in floor_pool(tiles_dir):
            t = gt_tile(fid, res, tiles_dir)
            h_m, w_m = t.shape[0] * res, t.shape[1] * res
            if not (h_lo <= h_m <= h_hi and w_m > min_aspect * h_m and w_lo < w_m < w_hi):
                continue
            # RECTANGULARITY: notched/winged footprints leave deep black bands above and below the
            # strip; keep floors whose free extent spans most of the bbox height in most columns
            free = ~t
            has = free.any(axis=0)
            if has.mean() < 0.95:
                continue
            depth = np.where(has, free.shape[0] - np.argmax(free, axis=0)
                             - np.argmax(free[::-1], axis=0), 0)
            if np.percentile(depth[has], 25) / t.shape[0] < rect:
                continue
            cand.append((fid, t, _building_key(fid), h_m))
        return cand

    # prefer strict rectangularity + tight height window; relax stepwise only if the pool starves
    for rect, per_b in ((min_rect, max_per_building), (min_rect - 0.05, max_per_building),
                        (min_rect - 0.1, max_per_building + 2)):
        cand = candidates(rect)
        bw = band_m
        while bw <= (h_hi - h_lo):
            hs = sorted(c[3] for c in cand)
            lo = max(hs, key=lambda h: sum(1 for x in hs if h <= x <= h + bw), default=None)
            pick = [c for c in cand if lo is not None and lo <= c[3] <= lo + bw]
            random.Random(seed).shuffle(pick)
            per, out = defaultdict(int), []
            for fid, t, b, _h in pick:
                if per[b] >= per_b:
                    continue
                per[b] += 1
                out.append(t)
                if len(out) >= n:
                    return out
            bw += 0.5
    raise RuntimeError(f"could not draw {n} floors even after relaxing; widen height/width bands")


def _spine(t):
    """Row index of the floor's main corridor = the row carrying the most free cells."""
    return int((~t).sum(axis=1).argmax())


def build_corridor(tiles, res, corr_h_m=2.4):
    """Chain landscape floors left to right, TOP-ALIGNED: the roofline is a straight edge and, with the
    height-windowed tile selection, the floor line is nearly straight too (no big dead bands above or
    below any tile). At every seam carve ONE straight rectangular corridor band (corr_h_m tall) at the
    row where the two neighbours' free space comes closest - clean, axis-aligned, shortest cut.
    -> tight wall grid."""
    ch = max(2, int(round(corr_h_m / res)))
    gap = max(3, int(round(0.3 / res)))                       # inter-tile wall at REAL thickness
    H = max(t.shape[0] for t in tiles) + 8
    W = sum(t.shape[1] for t in tiles) + gap * (len(tiles) + 2) + 16
    c = np.ones((H, W), bool)
    x, prev, pp = 5, None, None
    for t in tiles:
        yi = 3                                                # even roofline
        c[yi:yi + t.shape[0], x:x + t.shape[1]] = t
        if prev is not None:                                  # shortest straight corridor across the seam
            best = None
            r0 = yi + ch // 2 + 2
            r1 = yi + min(prev.shape[0], t.shape[0]) - ch // 2 - 2
            for r in range(r0, r1, 2):
                band = slice(r - ch // 2, r + ch // 2 + 1)
                ja = next((j for j in range(x - 1, pp[1] - 1, -1) if (~c[band, j]).any()), None)
                jb = next((j for j in range(x, x + t.shape[1]) if (~c[band, j]).any()), None)
                if ja is None or jb is None:
                    continue
                if best is None or jb - ja < best[0]:
                    best = (jb - ja, band, ja, jb)
            if best is not None:
                _, band, ja, jb = best
                c[band, ja:jb + 1] = False
        prev, pp, x = t, (yi, x), x + t.shape[1] + gap
    f = ~c
    ys, xs = np.where(f)
    return c[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]


def _vlink(cv, S, res, spacing_m=24.0, reach_m=16.0, door_w_m=3.0, search_m=8.0):
    """Cut UNIFORM straight vertical shafts (door_w_m wide) across an inter-layer seam at grid row S:
    one shaft every spacing_m, each placed at the locally shortest free-to-free vertical crossing."""
    spacing = max(4, int(round(spacing_m / res)))
    reach = max(4, int(round(reach_m / res)))
    door_w = max(2, int(round(door_w_m / res)))
    search = max(4, int(round(search_m / res)))
    H, W = cv.shape

    def gap(jj):
        ia = next((i for i in range(S, max(0, S - reach), -1) if not cv[i, jj]), None)
        ib = next((i for i in range(S, min(H, S + reach)) if not cv[i, jj]), None)
        return (ib - ia, jj, ia, ib) if ia is not None and ib is not None else None

    cnt = 0
    for xc in range(spacing // 2, W, spacing):
        cands = [g for g in (gap(jj) for jj in range(max(1, xc - search), min(W - 1, xc + search))) if g]
        if not cands:
            continue
        _g, jj, ia, ib = min(cands)
        cv[ia:ib + 1, max(0, jj - door_w // 2):jj + door_w // 2 + 1] = False
        cnt += 1
    return cnt


def stack_layers(strips, res, vgap=2):
    """Stack corridor strips into a multi-layer building, densely interconnected at each seam."""
    W = max(s.shape[1] for s in strips)
    H = sum(s.shape[0] for s in strips) + vgap * (len(strips) + 1) + 4
    cv = np.ones((H, W), bool)
    y, seams = vgap + 2, []
    for k, s in enumerate(strips):
        xo = (W - s.shape[1]) // 2
        cv[y:y + s.shape[0], xo:xo + s.shape[1]] = s
        if k > 0:
            seams.append(y - vgap // 2 - 1)
        y += s.shape[0] + vgap
    links = sum(_vlink(cv, S, res) for S in seams)
    f = ~cv
    ys, xs = np.where(f)
    return cv[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3], links


def connect_components(wall, res, max_link_m=6.0, door_w_m=1.5, max_iter=200):
    """Connectivity fallback: merge every residual free component into the largest one through its EXACT
    shortest gap (distance transform, no sampling), as long as that gap is below max_link_m. Components
    that would need a longer link refill solid, so the map the swarm sees is exactly the reachable set."""
    import cv2
    rad = max(1, int(round(0.5 * door_w_m / res)))
    for _ in range(max_iter):
        lbl, n = ndi.label(~wall)
        if n <= 1:
            break
        sizes = np.bincount(lbl.ravel())[1:]
        big = 1 + int(sizes.argmax())
        dist, (iy, ix) = ndi.distance_transform_edt(lbl != big, return_indices=True)
        best = None
        for c in range(1, n + 1):
            if c == big:
                continue
            cy, cx = np.where(lbl == c)
            k = int(np.argmin(dist[cy, cx]))
            if best is None or dist[cy[k], cx[k]] < best[0]:
                best = (dist[cy[k], cx[k]], cy[k], cx[k])
        d, y0, x0 = best
        if d * res > max_link_m:
            wall |= (lbl > 0) & (lbl != big)                 # unreachable-by-doorway: refill
            break
        i1, j1 = int(iy[y0, x0]), int(ix[y0, x0])
        # axis-aligned L-shaped carve: straight rectangular openings only
        wall[max(0, y0 - rad):y0 + rad + 1, max(0, min(x0, j1) - rad):max(x0, j1) + rad + 1] = False
        wall[max(0, min(y0, i1) - rad):max(y0, i1) + rad + 1, max(0, j1 - rad):j1 + rad + 1] = False
    return wall


def largest_cc_fraction(wall):
    lbl, _ = ndi.label(~wall)
    sz = np.bincount(lbl.ravel())[1:]
    return sz.max() / sz.sum() if len(sz) else 0.0


def build_building(n_layers, floors_per_layer, res, seed):
    """Compose an n_layers corridor building. Floors are selected PER LAYER (seeded differently), so
    layers may reuse floors in a different order - exactly how real multi-storey buildings repeat their
    floorplans - and the clean-tile pool never starves. Within a layer floors are distinct."""
    strips = [build_corridor(select_floors(res, floors_per_layer, seed + 7 * L), res)
              for L in range(n_layers)]
    wall, links = stack_layers(strips, res)
    wall = connect_components(wall, res)                     # short-doorway connectivity fallback
    free = ~wall
    info = dict(n_layers=n_layers, floors_per_layer=floors_per_layer, res_m=res, seed=seed,
                shape=list(wall.shape), width_m=round(wall.shape[1] * res, 1), height_m=round(wall.shape[0] * res, 1),
                free_area_m2=round(int(free.sum()) * res * res), vertical_links=links,
                largest_cc=round(largest_cc_fraction(wall), 4))
    return wall, info


def save_building(wall, info, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    occ = np.where(wall, 255, 0).astype(np.uint8)            # 255 obstacle, 0 free (simulator convention)
    np.save(os.path.join(out_dir, f"{name}.npy"), occ)
    json.dump(info, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=2)
    from PIL import Image                                    # pixel-exact PNG, wall black / free white, no decoration
    Image.fromarray(np.flipud(np.where(wall, 0, 255).astype(np.uint8)), "L").save(os.path.join(out_dir, f"{name}.png"))
    return info


# LARGE-FLOOR scenes: whole real floors stacked as storeys (no side-by-side tile collage). EVERY layer in
# a scene comes from a DIFFERENT building - no repeated or sibling floorplans anywhere in one map.
# Scene definitions. Entry syntax: "<floor_id>[|fx][|fy][|r<start>:<end>]" - fx/fy mirror the tile
# (disguises floors of the same building reused in another storey), r crops rows. Layers are stacked and
# width-normalized, so every scene is a FILLED RECTANGLE; double and triple hit the golden ratio (1.618).
LARGE_SCENES = {
    # one real floor, as built (perfect rectangle, 133 x 39 m)
    "kth_single": [["A0043035_50015847"]],
    # landscape golden: ~116 x 71.6 m
    "kth_double": [["A0043035_50015847|r0:338"],
                   ["A0043004_0510035341_A_40_1_103", "A0043022_50052751"],
                   ["A0043001_0510034689_Layout1", "A0043018_0510040985_A-40.1-807"]],
    # portrait golden: ~103 x 166 m, 8 storeys; sibling floors only in distant storeys and mirrored
    "kth_triple": [["A0043035_50015847|r0:330"],
                   ["A0043034_50010535_PLAN2|r0:266"],
                   ["A0043004_0510035341_A_40_1_103", "A0043022_50052751"],
                   ["A0043001_0510034689_Layout1", "A0043018_0510040985_A-40.1-807"],
                   ["A0043018_0510040941_A-40.1-805|fx", "A0090002_0510032268_A_40_1_102"],
                   ["A0043004_0510035343_A_40_1_102|fx", "A0043022_50052752|fx"],
                   ["A0043001_0510034692_Layout1", "A0043018_0510040942_A-40.1-804|fx"],
                   ["A0043018_0510040946_A-40.1-806", "A0090002_0510032270_A_40_1_104|fx"]],
    # largest KTH: 12 storeys, all distinct clean landscape floors (quad extends triple with 4 more rows
    # of previously-unused buildings), mirrored to disguise reuse. ~103 x 250 m.
    "kth_quad": [["A0043035_50015847|r0:330"],
                 ["A0043034_50010535_PLAN2|r0:266"],
                 ["A0043004_0510035341_A_40_1_103", "A0043022_50052751"],
                 ["A0043001_0510034689_Layout1", "A0043018_0510040985_A-40.1-807"],
                 ["A0043018_0510040941_A-40.1-805|fx", "A0090002_0510032268_A_40_1_102"],
                 ["A0043004_0510035343_A_40_1_102|fx", "A0043022_50052752|fx"],
                 ["A0043001_0510034692_Layout1", "A0043018_0510040942_A-40.1-804|fx"],
                 ["A0043018_0510040946_A-40.1-806", "A0090002_0510032270_A_40_1_104|fx"],
                 ["A0050015_50041171", "A0043018_0510040943_A-40.1-803|fx"],
                 ["A0050015_50041172|fx", "A0050016_50041184"],
                 ["A0050015_50041173", "A0043020_0510032192_A_40_1_103|fx"],
                 ["A0094001_50055637|fx", "A0043038_0510045906_A_40_1_104"]],
}

# per-tile touch-ups from QC (tile-local pixel coords at 0.10 m): crop rows / fill wall patches
TILE_FIX = {
    "A0043035_50015847": dict(crop_rows=(0, 386)),           # drop the dashed bottom arcade strip
    "A0043035_50015848": dict(crop_rows=(0, 386)),
    # generous margins: QC coords are in the GTF frame, gt_tile's bbox crop shifts them a few px
    "A0043034_50010535_PLAN2": dict(patch=[(24, 46, 554, 570), (56, 74, 529, 551)]),  # y0,y1,x0,x1 wall fills
    "A0043012_0510035741_A_40_1_104": dict(despeckle=40),    # kill dangling door-jamb ticks
}


def _apply_fix(fid, t):
    fx = TILE_FIX.get(fid)
    if not fx:
        return t
    if "patch" in fx:
        for y0, y1, x0, x1 in fx["patch"]:
            t[y0:y1 + 1, x0:x1 + 1] = True
    if "despeckle" in fx:
        lbl, n = ndi.label(t)
        if n:
            sz = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
            small = np.flatnonzero(sz < fx["despeckle"]) + 1
            if len(small):
                t[np.isin(lbl, small)] = False
    if "crop_rows" in fx:
        r0, r1 = fx["crop_rows"]
        t = t[r0:r1].copy()
        t[0, :] = True; t[-1, :] = True                       # keep the cut edge sealed
    return t


def build_large(name, res):
    """Stack whole real floors (one or more chained per layer) into a building."""
    strips = []
    for layer in LARGE_SCENES[name]:
        tiles = []
        for spec in layer:
            fid, *mods = spec.split("|")
            t = _apply_fix(fid, gt_tile(fid, res))
            for m in mods:
                if m == "fx":
                    t = t[:, ::-1].copy()
                elif m == "fy":
                    t = t[::-1].copy()
                elif m.startswith("r"):
                    r0, r1 = m[1:].split(":")
                    t = t[int(r0):int(r1)].copy()
                    t[0, :] = True; t[-1, :] = True           # seal the cut edges
            tiles.append(t)
        strips.append(tiles[0] if len(tiles) == 1 else build_corridor(tiles, res))
    if len(strips) == 1:
        wall, links = strips[0], 0
    else:
        # RECTANGULAR building: normalize every storey to the narrowest one's width (crop from the right,
        # seal the cut edge; isolated sliced rooms refill via the connectivity pass below)
        w = min(s.shape[1] for s in strips)
        cut = []
        for s in strips:
            s = s[:, :w].copy()
            s[:, -1] = True
            cut.append(s)
        wall, links = stack_layers(cut, res)
    # solid 0.3 m border wall: cropped storeys must not leak half-open rooms to the map edge
    wall[:3, :] = True; wall[-3:, :] = True; wall[:, :3] = True; wall[:, -3:] = True
    wall = connect_components(wall, res)
    # connect_components may carve THROUGH the border wall (edge-hugging L-links): re-seal, then refill
    # whatever that re-isolates (only border-hugging dead-end nooks) so free space stays one component
    wall[:3, :] = True; wall[-3:, :] = True; wall[:, :3] = True; wall[:, -3:] = True
    lbl, n = ndi.label(~wall)
    if n > 1:
        sz = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        wall |= (lbl > 0) & (lbl != 1 + int(np.argmax(sz)))
    # scene-level despeckle: isolated black flecks left by crops and seam carves
    lbl, n = ndi.label(wall)
    if n:
        sz = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        small = np.flatnonzero(sz < 45) + 1
        if len(small):
            wall[np.isin(lbl, small)] = False
    if wall.shape[0] > wall.shape[1]:                        # benchmark maps are LANDSCAPE (long bottom edge)
        wall = np.rot90(wall).copy()
    free = ~wall
    info = dict(scene=name, layers=[l for l in LARGE_SCENES[name]], res_m=res,
                shape=list(wall.shape), width_m=round(wall.shape[1] * res, 1),
                height_m=round(wall.shape[0] * res, 1), free_area_m2=round(int(free.sum()) * res * res),
                vertical_links=links, largest_cc=round(largest_cc_fraction(wall), 4))
    return wall, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.10, help="metres per cell (finer = higher resolution)")
    ap.add_argument("--per", type=int, default=9, help="floors per corridor layer")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--layers", default="1,2,3", help="which buildings to (re)build, e.g. 1 for kth_single only")
    ap.add_argument("--mode", default="large", choices=["large", "collage"],
                    help="large = whole real floors stacked (default); collage = many-tile corridor strips")
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark", "kth"))
    a = ap.parse_args()
    names = {1: "kth_single", 2: "kth_double", 3: "kth_triple", 4: "kth_quad"}
    for n_layers in [int(x) for x in a.layers.split(",")]:
        name = names[n_layers]
        if a.mode == "large":
            wall, info = build_large(name, a.res)
        else:
            wall, info = build_building(n_layers, a.per, a.res, a.seed)
        save_building(wall, info, a.out, name)
        print(f"{name}: {info['width_m']}x{info['height_m']} m  {info['free_area_m2']} m2  "
              f"CC={info['largest_cc']}  links={info['vertical_links']}  -> {a.out}/{name}.{{npy,json,png}}")


if __name__ == "__main__":
    main()
