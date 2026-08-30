from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from .analytics import TemporalAnalytics
from .detectors import Detector
from .segmenters import SAM2Segmenter

class VideoAnalyticsPipeline:
    def __init__(self, detector: Detector, segmenter: SAM2Segmenter | None = None, frame_stride: int = 1) -> None:
        self.detector, self.segmenter, self.frame_stride = detector, segmenter, frame_stride
        self.tracker = sv.ByteTrack(track_activation_threshold=.25)
    def run(self, source: str | Path, query: str, output_dir: str | Path) -> dict:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened(): raise ValueError(f"Cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        analytics = TemporalAnalytics(fps=fps / self.frame_stride); boxes, labels = sv.BoxAnnotator(), sv.LabelAnnotator(); frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % self.frame_stride: frame_idx += 1; continue
            found = self.detector.detect(frame, query)
            if self.segmenter: found = self.segmenter.segment(frame, found)
            xyxy = np.asarray([d.xyxy for d in found], dtype=np.float32).reshape(-1, 4); confidence = np.asarray([d.confidence for d in found], dtype=np.float32)
            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=np.zeros(len(found), dtype=int))
            detections = self.tracker.update_with_detections(detections); ids = detections.tracker_id.tolist() if detections.tracker_id is not None else []
            analytics.observe(frame_idx, ids, detections.confidence.tolist())
            annotations = [f"#{tid} {d.label} {conf:.2f}" for tid, d, conf in zip(ids, found, detections.confidence)]
            writer.write(labels.annotate(boxes.annotate(frame.copy(), detections), detections, annotations)); frame_idx += 1
        cap.release(); writer.release()
        summary = analytics.summary() | {"query": query, "processed_frames": len(analytics.frame_counts), "fps": fps}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2)); return summary
