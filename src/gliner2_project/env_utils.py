"""Environment reporting helpers for Colab smoke tests."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from typing import Any


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_environment_info() -> dict[str, Any]:
    """Collect runtime details without importing heavyweight packages unless needed."""

    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "gliner2": _package_version("gliner2"),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        info["gpu_memory_gb"] = (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        )
    except Exception as exc:  # pragma: no cover - depends on runtime packages.
        info["torch"] = f"unavailable: {exc}"
        info["cuda_available"] = False
        info["gpu_name"] = None
        info["gpu_memory_gb"] = None

    return info


def print_environment_info(info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Print and return environment details for notebook logs."""

    info = collect_environment_info() if info is None else info
    print(f"Python version: {info.get('python')}")
    print(f"Platform: {info.get('platform')}")
    print(f"Torch version: {info.get('torch')}")
    print(f"CUDA available: {info.get('cuda_available')}")
    print(f"GPU name: {info.get('gpu_name')}")
    print(f"Transformers version: {info.get('transformers')}")
    print(f"Datasets version: {info.get('datasets')}")
    print(f"GLiNER2 version: {info.get('gliner2')}")
    print(f"Current working directory: {info.get('cwd')}")
    return info
