"""Ice surface detection utilities for the hockey analyzer.

This module implements the improved ice detection pipeline that proved to be
reliable in ``test_ice_player.py``.  The implementation intentionally mirrors
that script so behaviour in the main application matches the successful test
harness while keeping the public interface used throughout the project.
"""

from __future__ import annotations

import cv2
import numpy as np


class IceDetector:
    """Detect the playable ice surface inside broadcast frames.

    The detector combines colour and texture cues, keeps only the largest
    connected component, constrains it with a convex hull boundary and finally
    cleans the mask with a couple of morphological operations.  The logic is a
    distilled version of the experimentation performed in ``test_ice_player.py``
    and provides a deterministic binary mask suitable for downstream modules.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.ice_mask: np.ndarray | None = None
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_ice_surface(self, frame: np.ndarray) -> np.ndarray:
        """Detect the rink ice inside *frame*.

        Args:
            frame: Input BGR frame.

        Returns:
            Binary ``np.uint8`` mask with 255 for ice pixels.
        """
        if frame is None:
            raise ValueError("Frame must not be None")

        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self._log("🧊 Ice detection (color + texture, no flood fill)")

            # 1) colour based mask (white / bright areas)
            ice_color = self.to_binary_mask(self.detect_white_surfaces(hsv, lab))
            self._log_pct("color", ice_color)

            # 2) keep only the largest connected component
            largest = self.find_largest_connected_component(ice_color)
            self._log_pct("largest white area", largest)

            # 3) texture mask (smooth regions) limited to the largest component
            ice_texture = self.to_binary_mask(self.detect_smooth_surfaces(gray))
            lines_within = cv2.bitwise_and(ice_texture, largest)
            self._log_pct("texture within largest", lines_within)

            # 4) convex hull boundary to trim boards / stands
            boundary = self.create_ice_boundary_mask(largest)
            boundary = self.to_binary_mask(boundary)
            self._log_pct("boundary", boundary)

            # 5) combine and clean
            combined = cv2.bitwise_and(cv2.bitwise_or(largest, lines_within), boundary)
            final = self.clean_ice_mask(self.to_binary_mask(combined))
            self._log_pct("final cleaned", final)

            self.ice_mask = final
            return final
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"❌ Ice surface detection failed: {exc}")
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            y1, y2 = int(h * 0.2), int(h * 0.8)
            x1, x2 = int(w * 0.1), int(w * 0.9)
            mask[y1:y2, x1:x2] = 255
            print("   Using conservative fallback mask")
            self.ice_mask = mask
            return mask

    def is_on_ice(self, position: tuple[int, int]) -> bool:
        """Return ``True`` when *position* lies inside the current ice mask."""
        if self.ice_mask is None:
            return True

        x, y = position
        if 0 <= x < self.ice_mask.shape[1] and 0 <= y < self.ice_mask.shape[0]:
            return bool(self.ice_mask[y, x])
        return False

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def detect_white_surfaces(self, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        l_channel = lab[:, :, 0]
        _, mask_lab = cv2.threshold(l_channel, 160, 255, cv2.THRESH_BINARY)

        return cv2.bitwise_or(mask_hsv, mask_lab)

    def detect_smooth_surfaces(self, gray: np.ndarray) -> np.ndarray:
        kernel = np.ones((15, 15), np.float32) / 225
        f = gray.astype(np.float32)
        mean = cv2.filter2D(f, -1, kernel)
        sqr_mean = cv2.filter2D(f * f, -1, kernel)
        texture = np.sqrt(np.maximum(sqr_mean - mean * mean, 0))
        thresh = np.percentile(texture, 30)
        return ((texture < thresh).astype(np.uint8) * 255)

    def find_largest_connected_component(self, binary_mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            self._log("   ⚠️  No connected components")
            return binary_mask

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest = (labels == largest_label).astype(np.uint8) * 255
        area = int(stats[largest_label, cv2.CC_STAT_AREA])
        self._log(f"   Kept largest component: {area} px, removed {num_labels - 2} others")
        return largest

    def create_ice_boundary_mask(self, largest_component: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._log("   ⚠️  No contours for boundary")
            return largest_component

        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        boundary = np.zeros_like(largest_component)
        cv2.fillPoly(boundary, [hull], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        boundary = cv2.dilate(boundary, kernel, iterations=1)
        return boundary

    def clean_ice_mask(self, mask: np.ndarray) -> np.ndarray:
        k5 = np.ones((5, 5), np.uint8)
        k15 = np.ones((15, 15), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k5)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k15)
        return cleaned

    def to_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() not in (0, 255):
            mask = ((mask > 0).astype(np.uint8)) * 255
        return mask

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _log_pct(self, label: str, mask: np.ndarray) -> None:
        if not self.verbose:
            return
        pct = 100.0 * float((mask > 0).sum()) / float(mask.size)
        print(f"   {label}: {pct:.1f}% of frame")


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    detector = IceDetector(verbose=True)
    print("Ice surface detector ready!")
