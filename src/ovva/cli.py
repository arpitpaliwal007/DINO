from __future__ import annotations
import argparse, json
from .detectors import GroundingDINODetector, MockDetector, YOLOClosedSetDetector
from .pipeline import VideoAnalyticsPipeline
from .segmenters import SAM2Segmenter

def main() -> None:
    p = argparse.ArgumentParser(description="Open-vocabulary detection, segmentation, tracking, and analytics")
    p.add_argument("--source", required=True); p.add_argument("--query", required=True); p.add_argument("--output", default="artifacts/run")
    p.add_argument("--baseline", action="store_true"); p.add_argument("--mock", action="store_true"); p.add_argument("--segment", action="store_true")
    p.add_argument("--sam-checkpoint"); p.add_argument("--sam-config"); p.add_argument("--stride", type=int, default=1)
    p.add_argument("--zone", help="Normalized analytics zone: x1,y1,x2,y2 (each value 0-1)")
    args = p.parse_args()
    detector = MockDetector() if args.mock else (YOLOClosedSetDetector() if args.baseline else GroundingDINODetector())
    segmenter = None
    if args.segment:
        if not (args.sam_checkpoint and args.sam_config): p.error("--segment requires --sam-checkpoint and --sam-config")
        segmenter = SAM2Segmenter(args.sam_checkpoint, args.sam_config)
    zone = tuple(map(float, args.zone.split(","))) if args.zone else None
    if zone and (len(zone) != 4 or any(x < 0 or x > 1 for x in zone) or zone[0] >= zone[2] or zone[1] >= zone[3]):
        p.error("--zone must be x1,y1,x2,y2, normalized to 0-1 with x1<x2 and y1<y2")
    print(json.dumps(VideoAnalyticsPipeline(detector, segmenter, args.stride, zone).run(args.source, args.query, args.output), indent=2))

if __name__ == "__main__": main()
