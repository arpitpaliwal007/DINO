"""Creates a deterministic demo input so the pipeline can be smoke-tested."""
from pathlib import Path
import cv2, numpy as np
out = Path("assets/demo.mp4"); out.parent.mkdir(exist_ok=True)
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
for i in range(40):
    frame = np.full((360, 640, 3), 245, np.uint8); x = 140 + i * 6
    cv2.circle(frame, (x + 100, 75), 30, (0, 0, 220), -1); cv2.rectangle(frame, (x+65, 105), (x+135, 300), (80,80,80), -1)
    cv2.putText(frame, "synthetic: person wearing a red helmet", (18,335), cv2.FONT_HERSHEY_SIMPLEX, .6, (30,30,30), 1); writer.write(frame)
writer.release()
