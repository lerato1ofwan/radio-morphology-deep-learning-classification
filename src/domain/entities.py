from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class ImageSample:
    path: Path
    id: str

@dataclass
class ProcessingConfig:
    log_transform: bool = True
    log_c: float = 1.0
    contrast_stretch: bool = True

    denoise_method: str = "gaussian"  # "gaussian" | "median" | "none"
    denoise_sigma: float = 0.8
    denoise_ksize: int = 5  # must be odd

    background_method: str = "tophat"  # "tophat" | "none"
    background_se_radius: int = 24

    sharpen_method: str = "unsharp"  # "unsharp" | "none"
    unsharp_amount: float = 0.45
    unsharp_radius: int = 3  # kernel size for blur used in unsharp

    mask_threshold: str = "otsu"  # "otsu" | "kapur" | "none"
    closing_radius: int = 5
    crop_margin: int = 12  # pixels

    resize_to: int = 320  # final square size (pixels)
    interpolation: str = "bicubic"  # "bilinear" | "bicubic" | "nearest"

    save_mask: bool = True
    quality_gate: bool = True

    # thresholds for quality gate; set to None to skip
    min_edge_density: float = 0.001  # edges / pixel
    min_laplacian_var: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
