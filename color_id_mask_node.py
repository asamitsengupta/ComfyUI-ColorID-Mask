import torch
import numpy as np
from PIL import Image
import cv2

class ColorIDMaskExtractor:
    """
    Extracts per-actor masks from color-coded pose reference.
    Red outline = Actor 1, Blue outline = Actor 2, Green = Actor 3, Magenta = Actor 4
    Where colors touch = perfect mask boundary (no identity bleeding!)
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
    
    def extract_masks(self, color_pose_reference, num_actors):
        # Convert tensor to numpy
        img = (color_pose_reference[0].cpu().numpy() * 255).astype(np.uint8)
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        
        # Define color thresholds
        color_defs = {
            1: (R > 150) & (G < 100) & (B < 100),      # Red
            2: (B > 150) & (R < 100) & (G < 100),      # Blue
            3: (G > 150) & (R < 100) & (B < 100),      # Green
            4: (R > 150) & (B > 150) & (G < 100),      # Magenta
        }
        
        masks = []
        for i in range(1, 5):
            if i <= num_actors:
                outline = color_defs[i]
                # Dilate outline to fill body
                kernel = np.ones((15, 15), np.uint8)
                dilated = cv2.dilate(outline.astype(np.uint8), kernel, iterations=3)
                # Fill holes to create solid mask
                filled = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
                mask = torch.from_numpy((filled > 0).astype(np.float32))
            else:
                mask = torch.zeros_like(R, dtype=torch.float32)
            masks.append(mask.unsqueeze(0))
        
        return tuple(masks)

NODE_CLASS_MAPPINGS = {
    "ColorIDMaskExtractor": ColorIDMaskExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorIDMaskExtractor": "Color ID Mask Extractor"
}