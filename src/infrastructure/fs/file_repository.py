from pathlib import Path
import json
import cv2
import numpy as np
from ...domain.services import IImageRepository


class FileSystemImageRepository(IImageRepository):
    def load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(str(path))
        return img

    def save_image(self, path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        cv2.imwrite(str(path), image)

    def save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
