import torch
import numpy as np
import cv2

class ColorIDMaskExtractor:
    """
    Extracts per-actor masks from color-coded pose reference using HSV hue-distance.
    Red=1, Blue=2, Green=3, Magenta=4. Handles touching bodies via seam filling.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_pose_reference": ("IMAGE",),
                "num_actors": ("INT", {"default": 2, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("actor1_mask", "actor2_mask", "actor3_mask", "actor4_mask")
    FUNCTION = "extract_masks"
    CATEGORY = "mask/compositing"

    TARGET_HUES = {1: 0, 2: 120, 3: 60, 4: 150}
    HUE_TOLERANCE = 15
    MIN_SATURATION = 30
    MIN_VALUE = 30

    def extract_masks(self, color_pose_reference, num_actors):
        img = (color_pose_reference[0].cpu().numpy() * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 0].astype(np.int16)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        valid = (S > self.MIN_SATURATION) & (V > self.MIN_VALUE)

        def hue_dist(h, target):
            d = np.abs(h - target)
            return np.minimum(d, 180 - d)

        raw_masks = []
        for i in range(1, 5):
            if i <= num_actors:
                outline = (valid & (hue_dist(H, self.TARGET_HUES[i]) <= self.HUE_TOLERANCE)).astype(np.uint8) * 255
                kernel = np.ones((5, 5), np.uint8)
                closed = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(closed)
                if contours:
                    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
                else:
                    filled = closed
                raw_masks.append(filled > 0)
            else:
                raw_masks.append(np.zeros(img.shape[:2], dtype=bool))

        # Seam filling for touching actors
        active_masks = [m for m in raw_masks[:num_actors] if m.any()]
        if len(active_masks) > 1:
            combined = np.zeros_like(active_masks[0])
            for m in active_masks: combined |= m
            
            dists = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) for m in raw_masks[:num_actors]]
            dist_stack = np.stack(dists, axis=0)
            nearest = np.argmin(dist_stack, axis=0)
            
            final_masks = []
            for i in range(num_actors):
                m = raw_masks[i].copy()
                seam_region = (nearest == i) & combined
                m = m | seam_region
                final_masks.append(m)
            raw_masks[:num_actors] = final_masks

        masks = []
        for i in range(4):
            if i < len(raw_masks):
                mask = torch.from_numpy(raw_masks[i].astype(np.float32))
            else:
                mask = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.float32)
            masks.append(mask.unsqueeze(0))
        return tuple(masks)

NODE_CLASS_MAPPINGS = {"ColorIDMaskExtractor": ColorIDMaskExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"ColorIDMaskExtractor": "Color ID Mask Extractor"}