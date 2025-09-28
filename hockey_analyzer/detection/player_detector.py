"""Player detection utilities for the hockey analyzer.

The implementation adopts the improved workflow validated in
``test_ice_player.py`` where detections are run on an ice-masked crop that is
optionally resized before invoking YOLO.  The module keeps the public interface
used throughout the project and remains compatible with the existing unit
tests that patch the underlying YOLO model.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


Detection = Dict[str, object]


class PlayerDetector:
    """Wrapper around a YOLO model with optional ice-aware preprocessing."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        inference_size: int = 960,
        crop_and_resize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.inference_size = inference_size
        self.crop_and_resize = crop_and_resize

        print(f"Loading YOLO model {model_name}...")
        self.model = YOLO(model_name)
        print("✅ YOLO model loaded")

    # ------------------------------------------------------------------
    def detect_players(
        self,
        frame: np.ndarray,
        ice_mask: Optional[np.ndarray] = None,
        crop_resize: Optional[bool] = None,
        target_size: Optional[int] = None,
    ) -> List[Detection]:
        """Detect players inside *frame*.

        Args:
            frame: Input BGR frame.
            ice_mask: Optional binary mask of the ice surface (255 = ice).
            crop_resize: Override for ``self.crop_and_resize``.
            target_size: Override for ``self.inference_size`` when cropping.
        """
        if frame is None:
            raise ValueError("Frame must not be None")

        if crop_resize is None:
            crop_resize = self.crop_and_resize
        if target_size is None:
            target_size = self.inference_size

        detection_frame = frame
        offset_x = offset_y = 0
        scale_x = scale_y = 1.0

        if ice_mask is not None:
            mask = self._ensure_binary_mask(ice_mask)
            if crop_resize and np.any(mask):
                x, y, w, h = cv2.boundingRect(mask)
                if w > 0 and h > 0:
                    offset_x, offset_y = x, y
                    mask_crop = mask[y : y + h, x : x + w]
                    detection_frame = frame[y : y + h, x : x + w]
                    detection_frame = cv2.bitwise_and(detection_frame, detection_frame, mask=mask_crop)

                    if target_size and target_size > 0:
                        if w >= h:
                            new_w = target_size
                            new_h = max(1, int(target_size * h / max(w, 1)))
                        else:
                            new_w = max(1, int(target_size * w / max(h, 1)))
                            new_h = target_size
                        detection_frame = cv2.resize(detection_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        scale_x = w / float(new_w)
                        scale_y = h / float(new_h)
                else:
                    detection_frame = cv2.bitwise_and(frame, frame, mask=mask)
            else:
                detection_frame = cv2.bitwise_and(frame, frame, mask=mask)

        results = self.model(
            detection_frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=[0],
            imgsz=target_size if target_size else None,
        )

        players: List[Detection] = []
        frame_h, frame_w = frame.shape[:2]

        for result in results:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for bbox_xyxy, confidence, cls_id in zip(boxes_xyxy, confidences, classes):
                if int(cls_id) != 0:
                    continue
                if float(confidence) < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = bbox_xyxy

                if ice_mask is not None and crop_resize:
                    x1 = x1 * scale_x + offset_x
                    x2 = x2 * scale_x + offset_x
                    y1 = y1 * scale_y + offset_y
                    y2 = y2 * scale_y + offset_y

                x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, frame_w, frame_h)

                width = max(0, int(round(x2 - x1)))
                height = max(0, int(round(y2 - y1)))
                center_x = int(round(x1 + width / 2))
                center_y = int(round(y1 + height / 2))

                players.append(
                    {
                        "position": (center_x, center_y),
                        "bbox": (int(round(x1)), int(round(y1)), width, height),
                        "confidence": float(confidence),
                    }
                )

        return players

    # ------------------------------------------------------------------
    def _ensure_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() not in (0, 255):
            mask = ((mask > 0).astype(np.uint8)) * 255
        return mask

    def _clamp_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float, float]:
        x1 = max(0.0, min(float(frame_w - 1), float(x1)))
        x2 = max(0.0, min(float(frame_w), float(x2)))
        y1 = max(0.0, min(float(frame_h - 1), float(y1)))
        y2 = max(0.0, min(float(frame_h), float(y2)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, object]:
        players = self.detect_players(frame)
        return {
            "player_count": len(players),
            "players": players,
            "timestamp": None,
        }


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    detector = PlayerDetector()
    print("Detector ready for testing!")
