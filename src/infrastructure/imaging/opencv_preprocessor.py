"""OpenCV-based image preprocessing pipeline.

This module contains `OpenCVPreprocessor`, a concrete implementation of
`IImagePreprocessor` that applies a sequence of common image-processing
operations to an input BGR image:

1. Convert to grayscale and normalize to [0, 1]
2. Optional log transform and/or contrast stretching
3. Optional denoising (Gaussian/median)
4. Optional background flattening via white top-hat
5. Optional unsharp masking (sharpening)
6. Optional foreground mask + largest-component crop with margin
7. Resize to a fixed square size
8. Compute simple sharpness/edge metrics and an optional quality flag
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from ...domain.entities import ProcessingConfig
from ...domain.services import IImagePreprocessor
from .kapur import kapur_entropy_threshold
from .quality import laplacian_variance, edge_density

# Map friendly names to OpenCV interpolation flags.
INTERP_MAP = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
}

# Type alias for bounding box: (x, y, w, h)
BBox = Tuple[int, int, int, int]


class OpenCVPreprocessor(IImagePreprocessor):
    def __init__(self, config: ProcessingConfig):
        self.cfg = config

    def preprocess(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Process a BGR image and return results.

        Returns a dict with keys:
            - 'processed': processed grayscale uint8 image
            - 'mask': Binary uint8 mask (255=fg), same size as 'processed'
            - 'bbox': Bounding box (x, y, w, h) before margin/resize
            - 'metrics': Dict with numeric quality metrics (and optional flag)
        """
        # 1) Convert to grayscale and normalize to [0, 1]
        gray_u8 = self._to_gray(img_bgr)
        temp_img = gray_u8.astype(np.float32) / 255.0

        # 2) Log transform and contrast stretching
        if self.cfg.log_transform:
            temp_img = self._apply_log_transform(temp_img, float(self.cfg.log_c))
        if self.cfg.contrast_stretch:
            temp_img = self._normalize01(temp_img)

        # 3) Denoise
        temp_img = self._denoise(temp_img)

        # 4) Background flattening
        if self.cfg.background_method == 'tophat':
            temp_img = self._flatten_background_tophat(temp_img)

        # 5) Sharpening
        if self.cfg.sharpen_method == 'unsharp':
            temp_img = self._unsharp_sharpen(temp_img)

        # Convert back to 8-bit for thresholding/IO
        img_u8 = (temp_img * 255).astype(np.uint8)

        # 6) Mask and crop (largest component + margin)
        mask_u8: Optional[np.ndarray] = None
        bbox: Optional[BBox] = None
        if self.cfg.mask_threshold != 'none':
            mask_u8, bbox = self._create_mask_and_bbox(img_u8)
            img_u8, mask_u8, bbox = self._crop_to_bbox(img_u8, mask_u8, bbox)

        # 7) Resize image (and mask if present)
        img_u8, mask_u8 = self._resize(img_u8, mask_u8)

        # 8) Quality metrics
        metrics = self._compute_quality_metrics(img_u8)

        return {
            'processed': img_u8,
            'mask': mask_u8,
            'bbox': bbox,
            'metrics': metrics,
        }

    @staticmethod
    def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR/BGRA/gray image to grayscale uint8 in [0, 255]."""
        if img_bgr.ndim == 2:
            g = img_bgr
        elif img_bgr.shape[2] == 4:
            g = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if g.dtype == np.float32 or g.dtype == np.float64:
            g = np.clip(g, 0, 1)
            g = (g * 255).astype(np.uint8)

        return g

    @staticmethod
    def _normalize01(x: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range (safe against tiny ranges)."""
        x_min = float(x.min())
        x_max = float(x.max())
        denom = max(1e-8, x_max - x_min)

        return (x - x_min) / denom

    def _apply_log_transform(self, img: np.ndarray, c: float) -> np.ndarray:
        """Apply log transform and renormalize to [0, 1]."""
        return self._normalize01(np.log1p(c * img))

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Denoise according to configuration (gaussian, median, or none)."""
        method = self.cfg.denoise_method

        if method == 'gaussian':
            k = max(3, int(self.cfg.denoise_ksize) | 1)  # force odd
            return cv2.GaussianBlur(img, (k, k), self.cfg.denoise_sigma)

        if method == 'median':
            k = max(3, int(self.cfg.denoise_ksize) | 1)
            den = cv2.medianBlur((img * 255).astype(np.uint8), k)
            return den.astype(np.float32) / 255.0

        return img

    def _flatten_background_tophat(self, img: np.ndarray) -> np.ndarray:
        """Flatten background using white top-hat (morph OPEN) and renormalize."""
        r = int(self.cfg.background_se_radius)
        k = 2 * r + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg_u8 = cv2.morphologyEx((img * 255).astype(np.uint8), cv2.MORPH_OPEN, se)
        flat = np.clip((img * 255).astype(np.float32) - bg_u8.astype(np.float32), 0, 255) / 255.0

        return self._normalize01(flat)

    def _unsharp_sharpen(self, img01: np.ndarray) -> np.ndarray:
        """Apply unsharp masking using configured radius and amount."""
        k = max(3, int(self.cfg.unsharp_radius) | 1)  # force odd
        blurred = cv2.GaussianBlur(img01, (k, k), 0)
        sharpened = img01 + self.cfg.unsharp_amount * (img01 - blurred)

        return np.clip(sharpened, 0.0, 1.0)

    def _create_mask_and_bbox(self, img_u8: np.ndarray) -> Tuple[np.ndarray, Optional[BBox]]:
        """Create a binary mask and (optionally) a bbox of the largest component."""
        th = self._threshold(img_u8)

        # Morphological closing to join lobes/holes
        r = max(1, int(self.cfg.closing_radius))
        k = 2 * r + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, se, iterations=1)

        # Keep largest connected component
        num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            idx = 1 + int(np.argmax(areas))
            comp = (labels == idx).astype(np.uint8) * 255
            x, y, w, h, _area = stats[idx, :5]
            bbox: BBox = (int(x), int(y), int(w), int(h))
            return comp, bbox
        else:
            return th, None

    def _threshold(self, img_u8: np.ndarray) -> np.ndarray:
        """Threshold according to configuration (Otsu/Kapur/any>0)."""
        method = self.cfg.mask_threshold

        if method == 'otsu':
            _, th = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th

        if method == 'kapur':
            t = kapur_entropy_threshold(img_u8)
            _, th = cv2.threshold(img_u8, int(t), 255, cv2.THRESH_BINARY)
            return th

        # Fallback: Any non-zero as foreground
        return (img_u8 > 0).astype(np.uint8) * 255

    def _crop_to_bbox(
        self,
        img_u8: np.ndarray,
        mask_u8: Optional[np.ndarray],
        bbox: Optional[BBox],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[BBox]]:
        """Crop image (and mask) to bbox with margin.

        If bbox is None but a mask exists, compute a tight bbox from the mask's
        non-zero pixels. The bbox in the return value is the pre-margin bbox,
        matching the original behavior.
        """
        if mask_u8 is None:
            return img_u8, None, bbox

        if bbox is None:
            ys, xs = np.where(mask_u8 > 0)
            if ys.size and xs.size:
                x, y = int(xs.min()), int(ys.min())
                w = int(xs.max() - xs.min() + 1)
                h = int(ys.max() - ys.min() + 1)
                bbox = (x, y, w, h)

        if bbox is not None:
            x, y, w, h = bbox
            m = int(self.cfg.crop_margin)
            x0 = max(0, x - m)
            y0 = max(0, y - m)
            x1 = min(img_u8.shape[1], x + w + m)
            y1 = min(img_u8.shape[0], y + h + m)
            img_u8 = img_u8[y0:y1, x0:x1]
            mask_u8 = mask_u8[y0:y1, x0:x1]

        return img_u8, mask_u8, bbox

    def _resize(
        self,
        img_u8: np.ndarray,
        mask_u8: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resize image (and mask) to the configured square size."""
        interp = INTERP_MAP.get(self.cfg.interpolation, cv2.INTER_CUBIC)
        size = int(self.cfg.resize_to)
        img_u8 = cv2.resize(img_u8, (size, size), interpolation=interp)
        if mask_u8 is not None:
            mask_u8 = cv2.resize(mask_u8, (size, size), interpolation=cv2.INTER_NEAREST)
        return img_u8, mask_u8

    def _compute_quality_metrics(self, img_u8: np.ndarray) -> Dict[str, Any]:
        """Compute simple image-quality metrics and optional quality gate flag."""
        metrics = {
            'laplacian_var': float(laplacian_variance(img_u8)),
            'edge_density': float(edge_density(img_u8)),
        }

        if self.cfg.quality_gate:
            metrics['low_quality'] = (
                (self.cfg.min_laplacian_var is not None and metrics['laplacian_var'] < self.cfg.min_laplacian_var)
                or
                (self.cfg.min_edge_density is not None and metrics['edge_density'] < self.cfg.min_edge_density)
            )

        return metrics
