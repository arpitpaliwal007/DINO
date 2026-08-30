from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from .types import Detection

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, query: str) -> list[Detection]: ...

class GroundingDINODetector(Detector):
    """Lazy Grounding DINO wrapper using the Hugging Face Transformers backend."""
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", box_threshold: float = .30, text_threshold: float = .25, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
    def detect(self, frame: np.ndarray, query: str) -> list[Detection]:
        import torch
        text = query.strip().lower().rstrip(".") + "."
        image = Image.fromarray(frame[:, :, ::-1])
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad(): outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=self.box_threshold, text_threshold=self.text_threshold, target_sizes=torch.tensor([image.size[::-1]], device=self.device))[0]
        return [Detection(tuple(map(float, box.tolist())), float(score), str(label)) for box, score, label in zip(result["boxes"], result["scores"], result["labels"])]

class YOLOClosedSetDetector(Detector):
    """COCO-only baseline. It deliberately cannot interpret arbitrary attributes."""
    def __init__(self, model: str = "yolo11n.pt", confidence: float = .25) -> None:
        from ultralytics import YOLO
        self.model, self.confidence = YOLO(model), confidence
    def detect(self, frame: np.ndarray, query: str) -> list[Detection]:
        result = self.model(frame, conf=self.confidence, verbose=False)[0]
        return [Detection(tuple(map(float, box.xyxy[0].tolist())), float(box.conf.item()), str(result.names[int(box.cls.item())])) for box in result.boxes]

class MockDetector(Detector):
    """Deterministic detector for smoke tests and demonstrations without model downloads."""
    def detect(self, frame: np.ndarray, query: str) -> list[Detection]:
        h, w = frame.shape[:2]
        return [Detection((w*.25, h*.20, w*.58, h*.87), .91, query)]
