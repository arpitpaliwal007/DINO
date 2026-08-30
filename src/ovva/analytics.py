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
    def observe(self, frame_index: int, track_ids: list[int], confidences: list[float]) -> None:
        self.frame_counts.append({"frame": frame_index, "active_tracks": len(track_ids)})
        for track_id, confidence in zip(track_ids, confidences):
            state = self.tracks.setdefault(int(track_id), TrackState(frame_index, frame_index))
            state.last_frame = frame_index; state.frames += 1; state.confidence_sum += float(confidence)
    def summary(self) -> dict:
        dwell = {str(k): round(v.frames / self.fps, 3) for k, v in self.tracks.items()}
        mean = sum((v.confidence_sum / v.frames for v in self.tracks.values()), 0) / max(len(self.tracks), 1)
        return {"unique_tracks": len(self.tracks), "track_dwell_seconds": dwell, "mean_track_confidence": round(mean, 3), "peak_concurrent_tracks": max((x["active_tracks"] for x in self.frame_counts), default=0)}
