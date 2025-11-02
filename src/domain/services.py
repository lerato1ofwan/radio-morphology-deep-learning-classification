from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np


class IImagePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Return dict with keys:
            'processed' -> processed grayscale uint8 image
            'mask' -> binary uint8 mask (optional)
            'bbox' -> (x, y, w, h) used for crop
            'metrics' -> quality metrics dict
        """
        raise NotImplementedError


class IImageRepository(ABC):
    @abstractmethod
    def load_image(self, path: Path) -> 'np.ndarray':
        pass

    @abstractmethod
    def save_image(self, path: Path, image: 'np.ndarray') -> None:
        pass

    @abstractmethod
    def save_json(self, path: Path, data: dict) -> None:
        pass
