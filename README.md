# Open-Vocabulary Video Analytics

An end-to-end portfolio project for **text query → Grounding DINO → optional SAM2 → ByteTrack → temporal analytics**. Unlike a fixed COCO detector, it can localize a query such as `person wearing a red helmet` without retraining.

## What it demonstrates

| Capability | Implementation |
|---|---|
| Zero-shot, vision-language detection | Grounding DINO via Transformers |
| Instance masks | Optional SAM2, prompted by DINO boxes |
| Identity over time | ByteTrack via Supervision |
| Video insights | unique identities, dwell time, confidence, peak concurrency |
| Closed-set comparison | YOLO11 COCO baseline |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python scripts/make_demo_video.py
ovva --source assets/demo.mp4 --query 'person wearing a red helmet' --mock --output artifacts/mock-demo
```

The `--mock` run verifies the full video/ByteTrack/analytics path without a model download. It writes `annotated.mp4` and `summary.json`.

## Production inference

```bash
# Grounding DINO weights download automatically on first run
ovva --source input.mp4 --query 'person wearing a red helmet' --output artifacts/dino-run

# Closed-set baseline: produces fixed COCO labels such as `person`
ovva --source input.mp4 --query 'person wearing a red helmet' --baseline --output artifacts/yolo-baseline

# Optional SAM2 box-prompt masks
pip install -e '.[sam]'
ovva --source input.mp4 --query 'red hard hat' --segment \
  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt --sam-config configs/sam2.1/sam2.1_hiera_l.yaml

# Optional normalized region: x1,y1,x2,y2. Adds zone entries and per-track zone dwell time.
ovva --source input.mp4 --query 'person' --zone 0.10,0.10,0.90,0.90 --output artifacts/zone-run
```

## Results and evaluation protocol

Run the same source and query with DINO and the baseline, then compare the generated JSON. On attribute/compositional queries, Grounding DINO is expected to return a matching phrase while COCO YOLO can only produce one of its fixed 80 labels. Report these metrics on an annotated validation clip:

| Metric | Why it matters |
|---|---|
| query-level precision / recall | Does the box match the natural-language constraint? |
| IDF1 / HOTA | Is the same object kept under one identity? |
| mask IoU | Does SAM2 accurately delineate the detected instance? |
| dwell-time MAE | Are temporal business metrics reliable? |

### Verified Colab result

The open-vocabulary pipeline was run on a public person/bicycle/car video using a Colab T4 GPU, query `person`, and `--stride 2`.

| Metric | Result |
|---|---:|
| Processed frames | 324 |
| Effective sampling rate | 12 FPS |
| Unique person tracks | 8 |
| Peak concurrent tracks | 2 |
| Mean detection confidence | 0.727 |
| Longest observed track dwell | 6.0 s |

These are operational results from one unlabelled public clip, not a precision/recall benchmark. Use a labelled real video to report the evaluation metrics above.

## Colab

Open [notebooks/open_vocabulary_video_analytics_colab.ipynb](notebooks/open_vocabulary_video_analytics_colab.ipynb) in Google Colab to install the project, download a public sample video, run Grounding DINO + ByteTrack and the YOLO baseline, and display the results. The notebook uses the production-compatible Grounding DINO `threshold` API.

## Architecture

```text
natural-language query
        │
Grounding DINO (open vocabulary) ── optional SAM2 masks
        │
ByteTrack association across frames
        │
per-track dwell time, count, confidence → JSON + annotated MP4
```

## Design notes

- Models are lazily loaded, keeping CLI startup and test discovery fast.
- `frame_stride` trades temporal fidelity for throughput; analytics uses the effective sampling FPS.
- The baseline is intentionally not phrase-filtered: YOLO's labels are closed-set, which is the comparison being illustrated.
- An optional normalized zone adds entry count and dwell-time analytics for a region of interest.
- In production, pin model revisions, record hardware/latency, and evaluate phrase-level labels before using outputs in a safety-sensitive setting.
