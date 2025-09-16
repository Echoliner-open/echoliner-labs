"""EchoLiner open-source modular manufacturing toolkit."""

from __future__ import annotations

from importlib import metadata

__all__ = ["analytics", "robotics", "translation", "vision", "common", "__version__"]

try:
    __version__ = metadata.version("echoliner")
except metadata.PackageNotFoundError:  # pragma: no cover - local development
    __version__ = "0.2.0"
