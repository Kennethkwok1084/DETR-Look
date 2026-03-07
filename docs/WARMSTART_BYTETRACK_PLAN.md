# Deformable DETR Warm-Start + ByteTrack (Minimal Tracking Data) Plan

## Goals (deliverables for Chapter 4-5)

- 2 qualitative collages (baseline vs warm-start, 2 columns each, 6-8 frames per sequence).
- 1 ablation table with IDSW / Frag / FPS (optional: IDF1 / HOTA).
- 1 system flowchart (Deformable DETR + warm-start + ByteTrack).

This plan assumes no self-training: use pretrained Deformable DETR and a minimal BDD100K tracking subset (3-5 sequences from val).

## 1) Minimal Tracking Data (BDD100K tracking val subset)

### 1.1 Download scope (smallest stable setup)

- Dataset: BDD100K tracking (val split).
- Select 3-5 sequences (each ~200-300 frames).
- Keep only a few classes for stability (recommended: `car`, `truck`, `bus`).

Notes:
- Detection dataset in `data/raw/bdd100k/` is single-frame only. It cannot produce IDSW/Frag.
- Tracking dataset provides per-frame object IDs. That is required for IDSW/Frag.

### 1.2 Suggested local layout

```
data/raw/bdd100k_tracking/
  images/val/<video_id>/*.jpg
  labels/val/<video_id>.json
  seq_list_min.json
```

### 1.3 Sequence selection (3-5 sequences)

Manual criteria:
- Dense traffic + occlusion (IDSW sensitive).
- Small objects like traffic lights/signs (Frag sensitive).

Optional helper script idea:
- Parse labels to find videos with the most frames and high object density.
- Pick top 5, then trim to 3-5 sequences.

### 1.4 Download commands and subset selector

Download commands (manual URLs required from BDD100K portal):

```bash
mkdir -p data/raw/bdd100k_tracking/raw

# Set these two URLs from https://bdd-data.berkeley.edu/portal.html#download
export BDD100K_TRACKING_IMAGES_URL="YOUR_IMAGES_ZIP_URL"
export BDD100K_TRACKING_LABELS_URL="YOUR_LABELS_ZIP_URL"

# Download
aria2c -c -x 8 -s 8 -d data/raw/bdd100k_tracking/raw "$BDD100K_TRACKING_IMAGES_URL"
aria2c -c -x 8 -s 8 -d data/raw/bdd100k_tracking/raw "$BDD100K_TRACKING_LABELS_URL"

# Unzip (adjust filenames as needed)
unzip -q data/raw/bdd100k_tracking/raw/*.zip -d data/raw/bdd100k_tracking/
```

Sequence list generator:

```bash
python tools/select_bdd100k_tracking_subset.py \
  --labels-dir data/raw/bdd100k_tracking/labels/val \
  --output data/raw/bdd100k_tracking/seq_list_min.json \
  --top-k 5 \
  --min-frames 200 \
  --classes car truck bus
```

## 2) Warm-Start in Deformable DETR (inference-only)

### 2.1 Cache definition (per sequence)

- `ref_points_prev`: top-K centers `(cx, cy)` in [0, 1].
- `scores_prev`: confidence list.
- (Optional) `wh_prev`: `(w, h)` in [0, 1].

Selection:
- Filter by `score > tau` (default `tau=0.5`).
- Keep top K (default `K=30` or `50`).

### 2.2 Code touchpoints

Files to modify (minimal):

- `models/deformable_detr_model.py`
  - add `warm_ref_points=None` to `forward(...)`.
  - pass `warm_ref_points` into the underlying model.
- `third_party/deformable_detr/models/deformable_detr.py`
  - update `forward(self, samples, warm_ref_points=None)` and pass to transformer.
- `third_party/deformable_detr/models/deformable_transformer.py`
  - update `forward(..., warm_ref_points=None)` and override initial reference points.

### 2.3 Warm-start insertion logic

In non-two-stage branch after:
```
reference_points = self.reference_points(query_embed).sigmoid()
```
Inject:
```
if warm_ref_points is not None:
    # warm_ref_points: [B, K, 2], normalized
    K = min(warm_ref_points.shape[1], reference_points.shape[1])
    reference_points[:, :K, :2] = warm_ref_points[:, :K, :2].clamp(eps, 1 - eps)
```

Rules:
- If cache invalid or empty -> skip warm-start (baseline fallback).
- Clamp to avoid 0/1 edge cases.
- Use only inference path (no training changes).

## 3) ByteTrack integration

### 3.1 Dependency strategy

Option A (vendoring):
- Add ByteTrack repo to `third_party/bytetrack/`.
- Wrap `BYTETracker` with a small adapter.

Wrapper:
- `utils/bytetrack_wrapper.py`

Option B (pip):
- Install ByteTrack or its tracker modules if available.

### 3.2 Tracker inputs / outputs

Input detections per frame:
- `x1, y1, x2, y2, score, cls`
- For ByteTrack: convert to `tlwh` or `tlbr` based on its API.

Output per frame:
- `x1, y1, x2, y2, track_id, cls, score`

Per-class tracking:
- Recommended to maintain a tracker per `cls` to reduce cross-class IDSW.

## 4) Evaluation (IDSW / Frag / FPS)

### 4.1 Ground truth conversion (BDD -> MOT format)

Create a converter:
- Input: BDD tracking label JSON (per video).
- Output: `gt/<video_id>.txt` in MOT format:
  ```
  frame,id,x,y,w,h,conf,class,vis
  ```
  where `conf=1`, `class` uses a mapping table, `vis=-1` if not available.

Recommended: filter to classes you track (e.g., vehicles only).

Script:
- `tools/bdd_tracking_to_mot.py`

### 4.2 Tracker output conversion

Create a converter:
- Input: ByteTrack results (per frame list).
- Output: `pred/<video_id>.txt` in MOT format.

Script:
- `tools/bytetrack_to_mot.py`

### 4.3 Metrics tool

Use one of:
- TrackEval (IDSW/Frag/IDF1/HOTA).
- motmetrics (IDSW/Frag/IDF1).

Keep evaluation config aligned with class filtering.

### 4.4 FPS

Measure two numbers:
- `FPS_model`: model inference only.
- `FPS_e2e`: detection + tracking + visualization (if enabled).

Report `FPS_e2e` in the table for conservative comparison.

## 5) Experiment pipeline (baseline vs warm-start)

### 5.1 Output layout

```
outputs/tracking/<run_name>/
  pred_mot/        # tracker results per video
  vis/             # per-frame visualization
  metrics.json     # IDSW / Frag / IDF1 / HOTA
  fps.json
```

### 5.2 Pipeline stages

1) Load frames from one sequence.
2) Run model inference:
   - baseline: no warm-start.
   - warm-start: use prev frame cache.
3) Update ByteTrack with detections.
4) Save results (MOT format + optional visualization).
5) Compute metrics against GT.

Main script:
- `tools/run_tracking_min.py`

### 5.3 Minimal run commands (baseline + warm-start)

Baseline:
```bash
python tools/run_tracking_min.py \
  --config configs/deformable_detr_baseline.yaml \
  --checkpoint /path/to/deformable_detr.pth \
  --images-root data/raw/bdd100k_tracking/images/val \
  --seq-list data/raw/bdd100k_tracking/seq_list_min.json \
  --output-dir outputs/tracking/baseline \
  --save-vis
```

Warm-start:
```bash
python tools/run_tracking_min.py \
  --config configs/deformable_detr_baseline.yaml \
  --checkpoint /path/to/deformable_detr.pth \
  --images-root data/raw/bdd100k_tracking/images/val \
  --seq-list data/raw/bdd100k_tracking/seq_list_min.json \
  --output-dir outputs/tracking/warmstart \
  --warm-start --warm-k 30 --warm-tau 0.5 \
  --save-vis
```

Metrics (IDSW/Frag/IDF1 + FPS):
```bash
python tools/bdd_tracking_to_mot.py \
  --labels-dir data/raw/bdd100k_tracking/labels/val \
  --output-dir outputs/tracking/gt_mot \
  --classes car truck bus

python tools/eval_motmetrics.py \
  --gt-dir outputs/tracking/gt_mot \
  --pred-dir outputs/tracking/baseline/pred_mot \
  --fps-json outputs/tracking/baseline/fps.json \
  --output outputs/tracking/baseline/metrics.json

python tools/eval_motmetrics.py \
  --gt-dir outputs/tracking/gt_mot \
  --pred-dir outputs/tracking/warmstart/pred_mot \
  --fps-json outputs/tracking/warmstart/fps.json \
  --output outputs/tracking/warmstart/metrics.json
```

Collages (2 sequences):
```bash
python tools/build_tracking_collage.py \
  --baseline-dir outputs/tracking/baseline/vis/<video_id> \
  --ours-dir outputs/tracking/warmstart/vis/<video_id> \
  --output outputs/tracking/collage_<video_id>.png \
  --max-frames 8
```

Ablation table:
```bash
python tools/build_ablation_table.py \
  --baseline outputs/tracking/baseline/metrics.json \
  --warm outputs/tracking/warmstart/metrics.json \
  --output outputs/tracking/ablation_table.png
```

Flowchart (Chapter 4):
```bash
python tools/build_warmstart_flowchart.py \
  --output outputs/figures/warmstart_bytetrack_flow.png
```

## 6) Figures and Table

### 6.1 Qualitative collages (2 images)

- Pick 2 sequences (high occlusion + small objects).
- Each collage: 6-8 frames, 2 columns (baseline vs warm-start).
- Draw boxes + track_id (same color per ID).

Script:
- `tools/build_tracking_collage.py`

### 6.2 Ablation table

Rows:
- Baseline (Deformable DETR + ByteTrack)
- Warm-start (Deformable DETR + Warm-start + ByteTrack)

Columns:
- IDSW (down)
- Frag (down)
- FPS (up)
- Optional: IDF1/HOTA

### 6.3 Flowchart (Chapter 4)

Blocks:
- Deformable DETR
- Warm-start cache (top-K reference points)
- ByteTrack
- Outputs (tracks + metrics)

## 7) Implementation checklist

1) Download BDD100K tracking val subset (3-5 sequences).
2) Add data layout under `data/raw/bdd100k_tracking/`.
3) Add warm-start parameter and hook in Deformable DETR.
4) Add ByteTrack wrapper and per-class trackers.
5) Build converters for GT and predictions (MOT format).
6) Run baseline vs warm-start on same sequences.
7) Generate collages + table + flowchart.

## 8) Risks and fallback

- If tracking labels are not available: IDSW/Frag cannot be computed.
- If classes are too sparse: restrict to vehicle classes only.
- If performance is unstable: reduce sequence length or K/tau.
