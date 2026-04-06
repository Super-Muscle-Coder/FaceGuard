"""Entities for frame sanitization before fine-tune."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class SanitizedFrameRecord:
    source_path: str
    output_path: str
    image_type: str
    blur_score: float
    brightness: float
    has_face: bool
    face_width: int
    face_height: int
    quality_score: float


@dataclass
class FrameSanitizerReport:
    person_name: str
    total_input: int
    total_output: int
    removed_count: int
    per_type_input: Dict[str, int] = field(default_factory=dict)
    per_type_output: Dict[str, int] = field(default_factory=dict)
    records: List[SanitizedFrameRecord] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


__all__ = ["SanitizedFrameRecord", "FrameSanitizerReport"]

