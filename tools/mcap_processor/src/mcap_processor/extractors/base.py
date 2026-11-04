"""Base extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

from ..core import BagReader
from ..utils import PathLike, ensure_dir


class BaseExtractor(ABC):
    """Base class for topic data extractors."""

    def __init__(
        self,
        reader: BagReader,
        output_dir: PathLike,
        topics: Optional[Sequence[str]] = None,
    ):
        self.reader = reader
        self.output_dir = Path(output_dir)
        self.topics = list(topics) if topics else None

    def setup(self) -> Path:
        """Create output directory."""
        return ensure_dir(self.output_dir)

    @abstractmethod
    def extract(self, max_messages: Optional[int] = None) -> None:
        """Run extraction."""
        raise NotImplementedError
