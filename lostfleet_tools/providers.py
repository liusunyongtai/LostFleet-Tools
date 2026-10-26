"""Optional future providers; the MVP never calls or fabricates AI results."""

from typing import Protocol

from PIL import Image


class SegmentationProvider(Protocol):
    def suggest_regions(self, image: Image.Image) -> list[dict]:
        """Return editable region proposals, using original image coordinates."""
        ...


class InpaintingProvider(Protocol):
    def complete_occlusion(self, image: Image.Image, missing_mask: Image.Image) -> Image.Image:
        """Return a separately reviewable generated image; never alter the source silently."""
        ...
