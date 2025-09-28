import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# IceDetector (no flood-fill)
# ----------------------------
class IceDetector:
    def __init__(self, verbose=True):
        self.ice_mask = None
        self.verbose = verbose

    def detect_ice_surface(self, frame):
        """Detect rink ice: color -> largest component -> refine by texture -> hull -> clean."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.verbose: print("🧊 Ice detection (color + texture, no flood fill)")

            # 1) color mask (white/bright)
            ice_color = self.to_binary_mask(self.detect_white_surfaces(hsv, lab))
            if self.verbose: self._pct("color", ice_color)

            # 2) keep only largest connected component (the rink)
            largest = self.find_largest_connected_component(ice_color)
            if self.verbose: self._pct("largest white area", largest)

            # 3) texture (smooth) → keep texture/lines only inside largest area
            ice_texture = self.to_binary_mask(self.detect_smooth_surfaces(gray))
            lines_within = cv2.bitwise_and(ice_texture, largest)
            if self.verbose: self._pct("texture within largest", lines_within)

            # 4) boundary via convex hull of largest area (trims boards/stands)
            boundary = self.create_ice_boundary_mask(largest)
            boundary = self.to_binary_mask(boundary)
            if self.verbose: self._pct("boundary", boundary)

            # 5) combine & clean
            combined = cv2.bitwise_and(cv2.bitwise_or(largest, lines_within), boundary)
            final = self.clean_ice_mask(self.to_binary_mask(combined))
            if self.verbose: self._pct("final cleaned", final)

            self.ice_mask = final
            return final

        except Exception as e:
            print(f"❌ Ice surface detection failed: {e}")
            # conservative fallback: center band
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            y1, y2 = int(h * 0.2), int(h * 0.8)
            x1, x2 = int(w * 0.1), int(w * 0.9)
            mask[y1:y2, x1:x2] = 255
            print("   Using conservative fallback mask")
            self.ice_mask = mask
            return mask

    def to_binary_mask(self, mask):
        """Ensure single-channel uint8 mask with values {0,255}."""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() not in (0, 255):
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def detect_white_surfaces(self, hsv, lab):
        # HSV: low saturation, high value
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        # LAB lightness
        l = lab[:, :, 0]
        _, mask_lab = cv2.threshold(l, 160, 255, cv2.THRESH_BINARY)

        return cv2.bitwise_or(mask_hsv, mask_lab)

    def detect_smooth_surfaces(self, gray):
        # texture via local stddev
        kernel = np.ones((15, 15), np.float32) / 225
        f = gray.astype(np.float32)
        mean = cv2.filter2D(f, -1, kernel)
        sqr_mean = cv2.filter2D(f * f, -1, kernel)
        texture = np.sqrt(np.maximum(sqr_mean - mean * mean, 0))
        thresh = np.percentile(texture, 30)
        return ((texture < thresh).astype(np.uint8) * 255)

    def find_largest_connected_component(self, binary_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            if self.verbose: print("   ⚠️  No connected components")
            return binary_mask
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest = (labels == largest_label).astype(np.uint8) * 255
        if self.verbose:
            area = int(stats[largest_label, cv2.CC_STAT_AREA])
            print(f"   Kept largest component: {area} px, removed {num_labels-2} others")
        return largest

    def create_ice_boundary_mask(self, largest_component):
        contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if self.verbose: print("   ⚠️  No contours for boundary")
            return largest_component
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        boundary = np.zeros_like(largest_component)
        cv2.fillPoly(boundary, [hull], 255)
        boundary = cv2.dilate(boundary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), 1)
        return boundary

    def clean_ice_mask(self, mask):
        k5 = np.ones((5, 5), np.uint8)
        k15 = np.ones((15, 15), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k5)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k15)
        return cleaned

    def _pct(self, label, mask):
        p = 100.0 * (mask > 0).sum() / mask.size
        print(f"   {label}: {p:.1f}% of frame")

# ----------------------------
# PlayerDetector (soft pre-mask + post-filter)
# ----------------------------
class PlayerDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.25, iou=0.45, device=None):
        print(f"Loading YOLO model {model_path}...")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        print("✅ YOLO model loaded")

    def detect_players(self, frame, ice_mask=None, crop_resize=False, target_size=640):
        """
        Detect players in a frame with optional ice masking and crop+resize

        Args:
            frame: Input BGR frame
            ice_mask: Optional binary mask of ice surface (255=ice, 0=not ice)
            crop_resize: If True, crop to ice bounding box and resize before detection
            target_size: YOLO input size for cropped detection

        Returns:
            List of player detections with original-frame coordinates
        """
        if ice_mask is None:
            detection_frame = frame
            offset_x, offset_y = 0, 0
        elif crop_resize:
            # Get bounding box of nonzero mask area
            x, y, w, h = cv2.boundingRect(ice_mask)
            offset_x, offset_y = x, y

            # Crop frame + mask
            detection_frame = frame[y:y+h, x:x+w]
            mask_crop = ice_mask[y:y+h, x:x+w]

            # Apply mask to crop (keeps only ice area visible)
            detection_frame = cv2.bitwise_and(detection_frame, detection_frame, mask=mask_crop)

            # Resize crop while preserving aspect ratio
            if w >= h:
                new_w, new_h = target_size, int(target_size * h / w)
                scale_x, scale_y = w / new_w, h / new_h
            else:
                new_w, new_h = int(target_size * w / h), target_size
                scale_x, scale_y = w / new_w, h / new_h

            detection_frame = cv2.resize(detection_frame, (new_w, new_h))

        else:
            # Use full-frame masking (slower, players smaller)
            detection_frame = cv2.bitwise_and(frame, frame, mask=ice_mask)
            offset_x, offset_y = 0, 0
            scale_x, scale_y = 1, 1

        # Run YOLO
        results = self.model.predict(
            detection_frame,
            imgsz=960,   # 👈 force bigger input size
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=[0]  # 👈 Only detect "person"
    )

        players = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Person class
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    # Map back to original coordinates
                    if ice_mask is not None and crop_resize:
                        x = x * scale_x + offset_x
                        y = y * scale_y + offset_y
                        w *= scale_x
                        h *= scale_y
                    elif ice_mask is not None:
                        # Already full-frame, no scale
                        pass
                    else:
                        # No mask, no scale
                        pass

                    if confidence > self.conf:
                        players.append({
                            'position': (int(x), int(y)),
                            'bbox': (int(x - w/2), int(y - h/2), int(w), int(h)),
                            'confidence': confidence
                        })
        return players

# ----------------------------
# Drawing / Utils
# ----------------------------
def draw_overlay(frame, mask=None, players=None, show_mask=False):
    out = frame.copy()
    if show_mask and mask is not None:
        # overlay semi-transparent mask (green)
        overlay = out.copy()
        g = np.zeros_like(out)
        g[:, :, 1] = 255
        g_mask = g.copy()
        g_mask[mask == 0] = 0
        cv2.addWeighted(g_mask, 0.25, overlay, 0.75, 0, overlay)
        out = overlay

    if players:
        for p in players:
            x, y, w, h = p["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(out, f"{p['confidence']:.2f}", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return out

def put_hud(img, fps, count, mode, mask_pct=None):
    text = f"FPS: {fps:.1f} | Players: {count} | Mode: {mode}"
    if mask_pct is not None:
        text += f" | Mask: {mask_pct:.1f}%"
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# ----------------------------
# Main Test Harness
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Test Ice Detection + Player Detection")
    ap.add_argument("--video", type=str, default="", help="Path to input video. Omit to use webcam 0.")
    ap.add_argument("--save", type=str, default="", help="Optional: path to save output video (e.g., out.mp4)")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model path")
    ap.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or 'mps'")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    ap.add_argument("--mode", type=str, default="pre_mask", choices=["pre_mask", "full"], help="Detection mode")
    ap.add_argument("--recompute_mask_every", type=int, default=120, help="Recompute ice mask every N frames (0 = only once)")
    ap.add_argument("--show_mask", action="store_true", help="Overlay the ice mask for visualization")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0 if args.video == "" else args.video)
    if not cap.isOpened():
        print("❌ Could not open video source.")
        return

    # Writer (lazy init when we know frame size)
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    ice = IceDetector(verbose=True)
    detector = PlayerDetector(model_path=args.model, device=args.device, conf=args.conf, iou=args.iou)

    ice_mask = None
    last_mask_frame_idx = -1
    frame_idx = 0
    t_last = time.time()
    fps = 0.0

    print("Press 'q' to quit, 'm' to toggle mask overlay, 'd' to toggle mode (pre_mask/full).")

    show_mask = args.show_mask
    mode = args.mode

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx == 0:
            # Initial ice mask
            ice_mask = ice.detect_ice_surface(frame)
            mask_pct = 100.0 * (ice_mask > 0).sum() / ice_mask.size
            print(f"Mask coverage: {mask_pct:.1f}%")

        # Recompute mask periodically if requested
        if args.recompute_mask_every > 0 and (frame_idx - last_mask_frame_idx >= args.recompute_mask_every):
            ice_mask = ice.detect_ice_surface(frame)
            last_mask_frame_idx = frame_idx
            mask_pct = 100.0 * (ice_mask > 0).sum() / ice_mask.size

        # Detect players
        t0 = time.time()
        players = detector.detect_players(frame, ice_mask=ice_mask, crop_resize=True)
        t1 = time.time()

        # FPS (EMA-ish)
        dt = max(1e-6, t1 - t0)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Draw
        vis = draw_overlay(frame, mask=ice_mask, players=players, show_mask=show_mask)
        vis = put_hud(vis, fps, len(players), mode, mask_pct=(100.0 * (ice_mask > 0).sum() / ice_mask.size))

        # Init writer if saving
        if args.save and writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(args.save, fourcc, max(15.0, min(60.0, fps if fps > 0 else 30.0)), (w, h))

        if writer is not None:
            writer.write(vis)

        cv2.imshow("Ice + Player Detection Test", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('d'):
            mode = "full" if mode == "pre_mask" else "pre_mask"

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
