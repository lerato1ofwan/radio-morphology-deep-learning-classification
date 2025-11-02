from pathlib import Path
from typing import Iterable, List

def list_images(root: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(root.rglob(pat)))

    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)

    return unique
