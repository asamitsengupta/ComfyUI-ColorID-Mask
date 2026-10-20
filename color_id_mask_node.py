import torch
import numpy as np
import cv2

class ColorIDMaskExtractor:
    # Extracts per-actor masks from a color-coded pose reference using RGB colour-excess stroke detection
    # (works for outlines on a plain background AND for outlines drawn over a photo).
    # Red outline = Actor 1, Blue = Actor 2, Green = Actor 3, Magenta = Actor 4.
    # Where colors touch, each pixel goes to the actor with the nearest outline (no identity bleeding).

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

    # Stroke detection by RGB colour-EXCESS, not hue: outlines drawn over a PHOTO share hue with skin,
    # wood, and red objects, so the old hue mask + convex hull swallowed ~75% of the frame (doctor included).
    # Excess = how strongly the actor's colour dominates the other channels. Thresholds are per colour
    # because thin anti-aliased strokes over a photo are muted (blue strokes measured ~RGB(30,73,117)).
    # Red/blue are tuned on a real pose image; green/magenta are conservative defaults.
    EXCESS_MIN = {1: 90, 2: 28, 3: 40, 4: 40}
    MIN_BLOB = 30        # drop specks smaller than this (px)
    CLUSTER_GAP = 31     # strokes of one actor lie within this dilation of each other; farther blobs are noise

    @staticmethod
    def _excess(img, i):
        R, G, B = (img[..., c].astype(np.int16) for c in range(3))
        if i == 1: return R - np.maximum(G, B)      # red
        if i == 2: return B - np.maximum(R, G)      # blue
        if i == 3: return G - np.maximum(R, B)      # green
        return np.minimum(R, B) - G                 # magenta

    def extract_masks(self, color_pose_reference, num_actors):
        img = (color_pose_reference[0].cpu().numpy() * 255).astype(np.uint8)
        close_kernel = np.ones((15, 15), np.uint8)
        margin_kernel = np.ones((5, 5), np.uint8)

        strokes, raw_masks = [None] * 4, [None] * 4   # index 0..3 == actor 1..4
        for i in range(1, num_actors + 1):
            m = (self._excess(img, i) > self.EXCESS_MIN[i]).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.isin(lab, [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= self.MIN_BLOB])
            if keep.any():   # keep only the main stroke cluster (drops stray same-colour objects in the photo)
                d = cv2.dilate(keep.astype(np.uint8), np.ones((self.CLUSTER_GAP,) * 2, np.uint8))
                n2, lab2, _, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
                cnt = np.bincount(lab2[keep], minlength=n2); cnt[0] = 0
                keep &= lab2 == cnt.argmax()
            strokes[i - 1] = keep
            # convex hull of the (lightly closed) strokes: can't have interior holes however fragmented the line art is
            closed = cv2.morphologyEx(keep.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close_kernel, iterations=3,
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0)  # default border value smeared masks to the image edge
            ys, xs = np.nonzero(closed)
            filled = np.zeros_like(closed)
            if len(xs) > 0:
                cv2.fillConvexPoly(filled, cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32)), 255)
            raw_masks[i - 1] = filled > 0

        # Hulls of touching actors overlap: give each overlapping pixel to the actor whose OUTLINE is nearest.
        active = [i for i in range(4) if raw_masks[i] is not None]
        if len(active) > 1:
            dist = np.stack([cv2.distanceTransform((~strokes[i]).astype(np.uint8), cv2.DIST_L2, 3) for i in active])
            owner = np.argmin(dist, axis=0)
            cover = sum(raw_masks[i].astype(np.uint8) for i in active)
            for k, i in enumerate(active):
                raw_masks[i] = raw_masks[i] & ((cover == 1) | (owner == k))

        masks = []
        for i in range(1, 5):
            if raw_masks[i - 1] is not None:
                filled = cv2.dilate(raw_masks[i - 1].astype(np.uint8) * 255, margin_kernel, iterations=1)  # small margin only
                mask = torch.from_numpy((filled > 0).astype(np.float32))
            else:
                mask = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.float32)
            masks.append(mask.unsqueeze(0))
        return tuple(masks)


NODE_CLASS_MAPPINGS = {
    "ColorIDMaskExtractor": ColorIDMaskExtractor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorIDMaskExtractor": "Color ID Mask Extractor"
}
