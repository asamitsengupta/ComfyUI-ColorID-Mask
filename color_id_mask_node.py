import torch
import numpy as np
import cv2

class ColorIDMaskExtractor:
    # Extracts per-actor masks from a color-coded pose reference using hue-distance matching
    # (tolerant of anti-aliased/muted outline colors) instead of strict RGB thresholds.
    # Red outline = Actor 1, Blue = Actor 2, Green = Actor 3, Magenta = Actor 4.
    # Where colors touch = perfect mask boundary (no identity bleeding!)

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

    # OpenCV hue scale is 0-179. Targets: red=0, green=60, blue=120, magenta=150.
    TARGET_HUES = {1: 0, 2: 120, 3: 60, 4: 150}
    HUE_TOLERANCE = 14    # see v2 fix #1 above -- closest target-hue pair is 30 deg apart
    MIN_SATURATION = 25   # excludes near-gray background and black arrows/text/lineart
    MIN_VALUE = 30         # excludes near-black pixels

    def extract_masks(self, color_pose_reference, num_actors):
        img = (color_pose_reference[0].cpu().numpy() * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 0].astype(np.int16)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        valid = (S > self.MIN_SATURATION) & (V > self.MIN_VALUE)

        def hue_dist(h, target):
            d = np.abs(h - target)
            return np.minimum(d, 180 - d)

        # Contour-fill instead of radius dilation: a large dilate/close (as used previously) grows
        # outward by ~45-75px and bridges the gap between two DIFFERENT actor colors wherever their
        # outlines run close together (e.g. touching/embracing poses), merging both actors into one
        # blob. Filling the enclosed interior of each color's own (lightly-closed) contour instead
        # respects the actual outline shape and can't bleed across a neighboring color.
        close_kernel = np.ones((15, 15), np.uint8)   # v2 fix #2
        margin_kernel = np.ones((5, 5), np.uint8)

        raw_masks = [None, None, None, None]  # index 0..3 == actor 1..4
        for i in range(1, 5):
            if i <= num_actors:
                outline = (valid & (hue_dist(H, self.TARGET_HUES[i]) <= self.HUE_TOLERANCE)).astype(np.uint8) * 255
                closed = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, close_kernel, iterations=3)
                # v2 fix #4: convex hull instead of contour-fill -- dense internal linework (eyebrows,
                # hair strands, fold lines) can sit >21px from the main silhouette and never close into
                # one contour, leaving the true gap between them unfilled (background shows through the
                # "hole"). A hull of every outline pixel can't have an interior hole no matter how
                # fragmented the line art is; the only cost is mild over-fill in concave spots (armpit,
                # crossed knee) which is cosmetically harmless vs. background poking through clothing.
                ys, xs = np.nonzero(closed)
                filled = np.zeros_like(closed)
                if len(xs) > 0:
                    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
                    cv2.fillConvexPoly(filled, hull, 255)
                raw_masks[i - 1] = filled > 0

        # v2 fix #3: seam-fill any pixel left unclaimed by every actor, inside their combined
        # silhouette, by nearest-distance assignment -- this is exactly the touching/contact zone.
        active = [(i, m) for i, m in enumerate(raw_masks) if m is not None]
        if active:
            combined = np.zeros_like(active[0][1], dtype=bool)
            for _, m in active:
                combined |= m
            claimed = np.zeros_like(combined, dtype=bool)
            for _, m in active:
                claimed |= m
            unassigned = combined & ~claimed
            if unassigned.any():
                dists = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) for _, m in active]
                nearest = np.argmin(np.stack(dists), axis=0)
                for pos, (i, m) in enumerate(active):
                    raw_masks[i] = m | (unassigned & (nearest == pos))

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
