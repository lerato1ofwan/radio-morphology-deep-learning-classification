import argparse
from pathlib import Path
import yaml
from ...domain.entities import ProcessingConfig
from ...infrastructure.fs.file_repository import FileSystemImageRepository
from ...infrastructure.imaging.opencv_preprocessor import OpenCVPreprocessor
from ...application.use_cases.preprocess_dataset import preprocess_dataset
from ...application.application_logging import log

def parse_args():
    ap = argparse.ArgumentParser(description="Radio-source Image preprocessing" )
    ap.add_argument("input", type=str, help="Folder containing images (will recurse)")
    ap.add_argument("--output-subdir", type=str, default="_processed", help="Subfolder to write results under input dir")
    ap.add_argument("--config", type=str, default=None, help="YAML config overriding defaults")
    ap.add_argument("--patterns", type=str, nargs='*', default=None, help="Glob patterns, e.g., *.png *.jpg")

    return ap.parse_args()

def load_config(path: Path) -> ProcessingConfig:
    if path is None or not path.exists():
        return ProcessingConfig()

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    cfg = ProcessingConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg

def main():
    args = parse_args()
    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    cfg = load_config(Path(args.config)) if args.config else ProcessingConfig()
    log(f"Using config: {cfg.to_dict()}")
    repo = FileSystemImageRepository()
    pre = OpenCVPreprocessor(cfg)
    preprocess_dataset(input_dir, args.output_subdir, repo, pre, cfg, patterns=args.patterns)

if __name__ == "__main__":
    main()
