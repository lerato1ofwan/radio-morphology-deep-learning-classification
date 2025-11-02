from pathlib import Path
import sys

# point to the project root
sys.path.append("/path/to/radio_preproc_clean_arch")

from src.domain.entities import ProcessingConfig
from src.infrastructure.fs.file_repository import FileSystemImageRepository
from src.infrastructure.imaging.opencv_preprocessor import OpenCVPreprocessor
from src.application.use_cases.preprocess_dataset import preprocess_dataset

cfg = ProcessingConfig()  # or set overrides, e.g., cfg.mask_threshold = "kapur"
repo = FileSystemImageRepository()
pre  = OpenCVPreprocessor(cfg)

def process_img(folder):
    preprocess_dataset(Path(folder), "_processed", repo, pre, cfg, patterns=["*.png"])