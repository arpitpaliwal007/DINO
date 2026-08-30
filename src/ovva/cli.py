from __future__ import annotations
import argparse, json
from .detectors import GroundingDINODetector, MockDetector, YOLOClosedSetDetector
from .pipeline import VideoAnalyticsPipeline
from .segmenters import SAM2Segmenter

def main() -> None:
    p = argparse.ArgumentParser(description="Open-vocabulary detection, segmentation, tracking, and analytics")
    p.add_argument("--source", required=True); p.add_argument("--query", required=True); p.add_argument("--output", default="artifacts/run")
    p.add_argument("--baseline", action="store_true"); p.add_argument("--mock", action="store_true"); p.add_argument("--segment", action="store_true")
    p.add_argument("--sam-checkpoint"); p.add_argument("--sam-config"); p.add_argument("--stride", type=int, default=1); args = p.parse_args()
    detector = MockDetector() if args.mock else (YOLOClosedSetDetector() if args.baseline else GroundingDINODetector())
    segmenter = None
    if args.segment:
        if not (args.sam_checkpoint and args.sam_config): p.error("--segment requires --sam-checkpoint and --sam-config")
        segmenter = SAM2Segmenter(args.sam_checkpoint, args.sam_config)
    print(json.dumps(VideoAnalyticsPipeline(detector, segmenter, args.stride).run(args.source, args.query, args.output), indent=2))

if __name__ == "__main__": main()
