# SenseSwarm — Anonymous Project Page

Anonymous supplementary site for the double-blind submission
**"SenseSwarm: Swarm Exploration via Neural Structure Estimation with Shared World-Anticipation for Rapid Mapping."**

Live page: open `index.html`, or serve the repository with GitHub Pages.

## Contents

- `index.html` — project page: abstract, method, key results, and the full video gallery.
- `videos/` — all 108 exploration runs replayed from the logged campaign
  (12 floors × 9 coordination methods, 256 robots, seed 1), plus a 512-robot hero run on the
  twelve-story KTH composite. Named `<floor>__<method>_<N>_<seed>.mp4`.
  Purple is SenseSwarm; blue overlays are predicted structure; the faint outline in unknown
  space is the ground-truth floor plan, shown for orientation only and never visible to any method.
- `benchmark_maps/` — full-resolution renderings of the twelve benchmark floors
  (walls ink, free space white, 0.1 m per pixel) and `stats.txt` with per-floor dimensions and free area.
- `scripts/` — the deterministic benchmark construction scripts:
  `gen_hkust.py` (per-category BIM rasterization with door re-opening) and
  `gen_kth_corridor.py` (seeded story chaining, spine-aligned corridor fusion, 3 m inter-layer shafts).
- `images/` — the paper's figures.

## Benchmark reproduction

The occupancy grids are regenerated, not distributed: `gen_hkust.py` consumes the public SLABIM
BIM meshes (HuggingFace `BobH62/SLABIM`, `BIM.zip`) and `gen_kth_corridor.py` consumes the
KTH floorplan dataset drawings. Both scripts are deterministic and seeded; the exact selection
and stitching rules are documented in their headers and in the paper's experimental setup.

All content is anonymized for review.
