# Color-ID Mask Generator for ComfyUI

**Solve Multi-Subject Identity Bleeding with 20 Lines of Code**

## The Problem

When generating images with multiple people touching (kissing, hugging, holding hands), current AI pipelines fail because:
- YOLO segmentation can't separate overlapping bodies
- Watershed algorithms pick wrong boundaries
- Identity bleeding occurs (faces blend together)

## The Solution

Instead of asking AI to guess where bodies end and begin, **you tell it explicitly with colors**:
- **Red outline** = Actor 1
- **Blue outline** = Actor 2  
- **Green outline** = Actor 3
- **Magenta outline** = Actor 4

Where the colors touch = perfect mask boundary. No guessing. 100% reliable.

## Installation

### Option 1: Automatic (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-ColorID-Mask.git
cd ComfyUI-ColorID-Mask
pip install -r requirements.txt