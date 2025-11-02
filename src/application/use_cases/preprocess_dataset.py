from pathlib import Path
from typing import Iterable, List
from dataclasses import asdict
from ..application_logging import log
from ..utils import list_images
from ...domain.entities import ImageSample, ProcessingConfig
from ...domain.services import IImagePreprocessor, IImageRepository

def preprocess_dataset(
    input_dir: Path,
    output_subdir: str,
    repo: IImageRepository,
    preprocessor: IImagePreprocessor,
    config: ProcessingConfig,
    patterns: List[str] = None
) -> None:
    patterns = patterns or ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    samples: List[ImageSample] = [ImageSample(p, p.stem) for p in list_images(input_dir, patterns)]
    out_dir = input_dir / output_subdir
    masks_dir = out_dir / "masks"
    meta_dir = out_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.save_mask:
        masks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    log(f"Found {len(samples)} images under {input_dir}")
    for s in samples:
        try:
            img = repo.load_image(s.path)
            result = preprocessor.preprocess(img)
            processed = result["processed"]
            repo.save_image(out_dir / f"{s.id}.png", processed)

            if config.save_mask and result.get("mask") is not None:
                repo.save_image(masks_dir / f"{s.id}_mask.png", result["mask"])

            repo.save_json(meta_dir / f"{s.id}.json", {
                "source": str(s.path),
                "bbox": result.get("bbox"),
                "metrics": result.get("metrics"),
                "config": config.to_dict()
            })
        except Exception as e:
            log(f"ERROR processing {s.path}: {e}")
