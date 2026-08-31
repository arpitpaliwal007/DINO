from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class TrackState:
    first_frame: int
    last_frame: int
    frames: int = 0
    confidence_sum: float = 0.0

@dataclass
class TemporalAnalytics:
    fps: float
    tracks: dict[int, TrackState] = field(default_factory=dict)
    frame_counts: list[dict[str, int]] = field(default_factory=list)
    zone_frames: dict[int, int] = field(default_factory=dict)
    def observe(self, frame_index: int, track_ids: list[int], confidences: list[float],
                boxes: list[list[float]] | None = None,
                zone: tuple[float, float, float, float] | None = None) -> None:
        self.frame_counts.append({"frame": frame_index, "active_tracks": len(track_ids)})
        for track_id, confidence in zip(track_ids, confidences):
            state = self.tracks.setdefault(int(track_id), TrackState(frame_index, frame_index))
            state.last_frame = frame_index; state.frames += 1; state.confidence_sum += float(confidence)
        if boxes is not None and zone is not None:
            zx1, zy1, zx2, zy2 = zone
            for track_id, box in zip(track_ids, boxes):
                x1, y1, x2, y2 = box
                if zx1 <= (x1 + x2) / 2 <= zx2 and zy1 <= (y1 + y2) / 2 <= zy2:
                    self.zone_frames[int(track_id)] = self.zone_frames.get(int(track_id), 0) + 1
    def summary(self) -> dict:
        dwell = {str(k): round(v.frames / self.fps, 3) for k, v in self.tracks.items()}
        mean = sum((v.confidence_sum / v.frames for v in self.tracks.values()), 0) / max(len(self.tracks), 1)
        result = {"unique_tracks": len(self.tracks), "track_dwell_seconds": dwell, "mean_track_confidence": round(mean, 3), "peak_concurrent_tracks": max((x["active_tracks"] for x in self.frame_counts), default=0)}
        if self.zone_frames:
            result["zone_entries"] = len(self.zone_frames)
            result["zone_dwell_seconds"] = {str(k): round(v / self.fps, 3) for k, v in self.zone_frames.items()}
        return result
