from __future__ import annotations
import numpy as np
from .types import Detection

class SAM2Segmenter:
    """SAM2 box-prompt segmentation. Installed only with `pip install .[sam]`."""
    def __init__(self, checkpoint: str, config: str, device: str = "cuda") -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("SAM2 is optional: install `pip install .[sam]` and provide checkpoint/config.") from exc
        self.predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))
    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        self.predictor.set_image(frame[:, :, ::-1]); output = []
        for detection in detections:
            masks, scores, _ = self.predictor.predict(box=np.asarray(detection.xyxy), multimask_output=False)
            output.append(Detection(detection.xyxy, detection.confidence, detection.label, masks[0].astype(bool) if len(scores) else None))
        return output
