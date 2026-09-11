from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}")
    return value


def load_default(name: str) -> dict[str, Any]:
    return load_yaml(project_root() / "configs" / f"{name}.yaml")
