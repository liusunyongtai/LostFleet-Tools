"""Validate the cloud image environment using only stdlib and installed libraries."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image


class ImagePipelineTest(unittest.TestCase):
    def test_transparent_png(self):
        """Save and reopen a fully transparent 512x512 RGBA PNG."""
        image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        with tempfile.TemporaryDirectory(prefix="lostfleet-image-test-") as directory:
            path = Path(directory) / "transparent_512.png"
            image.save(path, format="PNG")
            self.assertTrue(path.is_file())
            with Image.open(path) as reopened:
                self.assertEqual(reopened.format, "PNG")
                self.assertEqual(reopened.size, (512, 512))
                self.assertEqual(reopened.mode, "RGBA")
                self.assertEqual(reopened.getbands(), ("R", "G", "B", "A"))
                self.assertEqual(reopened.getchannel("A").getextrema(), (0, 0))
                pixels = np.asarray(reopened)
                self.assertEqual(pixels.shape, (512, 512, 4))
                self.assertEqual(pixels.dtype, np.dtype("uint8"))
                self.assertEqual(int(np.count_nonzero(pixels[:, :, 3] == 0)), 512 * 512)
                self.assertTrue(np.array_equal(pixels, np.asarray(image)))

            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, (512, 512, 4))
            self.assertTrue(np.all(decoded[:, :, 3] == 0))
            rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
            self.assertTrue(np.array_equal(rgba, np.asarray(image)))
        print("Verified PNG: 512x512 RGBA; alpha range 0..0; 262144 transparent pixels")


if __name__ == "__main__":
    print(f"Python: {sys.version.split()[0]}")
    print(f"Pillow: {PIL.__version__}; OpenCV: {cv2.__version__}; NumPy: {np.__version__}")
    unittest.main(verbosity=2)
