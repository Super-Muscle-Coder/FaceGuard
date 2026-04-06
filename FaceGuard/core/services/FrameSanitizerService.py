"""Frame sanitizer service (Phase 3 in fine-tune research pipeline).

Responsibilities:
- Read extracted frames from data/temp/{person}/frames/{angle}
- Score/filter frames by blur/brightness/face quality
- Save sanitized frames to data/temp/{person}/sanitized_frames/{angle}
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from config.settings import (
    FRAME_SANITIZER_CONFIG,
    get_person_frame_dir,
    get_sanitized_type_dir,
    INFO,
    SUCCESS,
    WARNING,
)
from core.adapters.ModelAdapter import ModelAdapter
from core.entities.frame_sanitizer import FrameSanitizerReport, SanitizedFrameRecord

logger = logging.getLogger(__name__)


class FrameSanitizerService:
    def __init__(self, model_adapter: ModelAdapter, person_name: str):
        self.model_adapter = model_adapter
        self.person_name = person_name
        self.cfg = FRAME_SANITIZER_CONFIG

    @staticmethod
    def _laplacian_blur(img_bgr) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _brightness(img_bgr) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    def _thresholds_for_type(self, image_type: str) -> Tuple[float, Tuple[float, float]]:
        if image_type == "frontal":
            return (
                float(self.cfg["FRONTAL_BLUR_THRESHOLD"]),
                tuple(self.cfg["FRONTAL_BRIGHTNESS_RANGE"]),
            )
        if image_type == "horizontal":
            return (
                float(self.cfg["HORIZONTAL_BLUR_THRESHOLD"]),
                tuple(self.cfg["HORIZONTAL_BRIGHTNESS_RANGE"]),
            )
        return (
            float(self.cfg["VERTICAL_BLUR_THRESHOLD"]),
            tuple(self.cfg["VERTICAL_BRIGHTNESS_RANGE"]),
        )

    def _frame_quality(self, img_bgr, image_type: str) -> tuple[bool, dict]:
        blur = self._laplacian_blur(img_bgr)
        bright = self._brightness(img_bgr)
        blur_thr, bright_range = self._thresholds_for_type(image_type)

        has_face = True
        face_w = face_h = 0
        face_score = 1.0

        if self.cfg["ENABLE_FACE_CHECK"]:
            faces = self.model_adapter.detect_faces(img_bgr, threshold=0.5, return_dataclass=False)
            if not faces:
                has_face = False
                face_score = 0.0
            else:
                best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
                face_w = max(0, x2 - x1)
                face_h = max(0, y2 - y1)
                min_face = int(self.cfg["MIN_FACE_SIZE"])
                if face_w < min_face or face_h < min_face:
                    has_face = False
                    face_score = 0.2

        blur_ok = blur >= blur_thr
        bright_ok = bright_range[0] <= bright <= bright_range[1]
        face_ok = has_face

        blur_score = min(1.0, blur / max(1.0, blur_thr * 2.0))
        bright_score = 1.0 if bright_ok else 0.3
        quality = 0.45 * blur_score + 0.30 * bright_score + 0.25 * face_score

        keep = blur_ok and bright_ok and face_ok
        return keep, {
            "blur": blur,
            "brightness": bright,
            "has_face": has_face,
            "face_width": face_w,
            "face_height": face_h,
            "quality": float(max(0.0, min(1.0, quality))),
        }

    def _prepare_arcface_frame(self, img_bgr, image_type: str) -> tuple[bool, dict, object]:
        keep, info = self._frame_quality(img_bgr, image_type)

        if not keep:
            return False, info, None

        min_quality = float(self.cfg["MIN_QUALITY_SCORE"])
        if float(info["quality"]) < min_quality:
            return False, info, None

        output = None
        if self.cfg["ENABLE_ALIGN_FACE"]:
            faces = self.model_adapter.detect_faces(img_bgr, threshold=0.5, return_dataclass=False)
            if faces:
                best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                aligned = self.model_adapter.align_face(
                    img_bgr,
                    best.get("landmarks"),
                    target_size=tuple(self.cfg["ARCFACE_INPUT_SIZE"]),
                )
                if aligned is not None:
                    output = aligned

            if output is None:
                x1, y1, x2, y2 = [
                    0,
                    0,
                    img_bgr.shape[1],
                    img_bgr.shape[0],
                ]
                if self.cfg["ENABLE_FACE_CHECK"] and faces:
                    best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                    x1, y1, x2, y2 = [int(v) for v in best["bbox"]]

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                margin = float(self.cfg["FACE_MARGIN_RATIO"])
                mx = int(w * margin)
                my = int(h * margin)
                x1 = max(0, x1 - mx)
                y1 = max(0, y1 - my)
                x2 = min(img_bgr.shape[1], x2 + mx)
                y2 = min(img_bgr.shape[0], y2 + my)
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    return False, info, None
                output = cv2.resize(crop, tuple(self.cfg["ARCFACE_INPUT_SIZE"]))
        else:
            output = cv2.resize(img_bgr, tuple(self.cfg["ARCFACE_INPUT_SIZE"]))

        if output is None:
            return False, info, None

        info["output_width"] = int(output.shape[1])
        info["output_height"] = int(output.shape[0])
        return True, info, output

    def run(self) -> FrameSanitizerReport:
        logger.info(f"{INFO} Frame sanitizer start for: {self.person_name}")
        logger.info(
            f"{INFO} Config | input_size=%s align=%s margin=%.2f min_quality=%.2f keep_per_angle=[%d..%d]",
            self.cfg["ARCFACE_INPUT_SIZE"],
            self.cfg["ENABLE_ALIGN_FACE"],
            float(self.cfg["FACE_MARGIN_RATIO"]),
            float(self.cfg["MIN_QUALITY_SCORE"]),
            int(self.cfg["MIN_KEEP_PER_ANGLE"]),
            int(self.cfg["MAX_KEEP_PER_ANGLE"]),
        )

        input_root = get_person_frame_dir(self.person_name)
        if not input_root.exists():
            raise RuntimeError(f"Frame directory not found: {input_root}")

        per_type_input: Dict[str, int] = {}
        per_type_output: Dict[str, int] = {}
        records: List[SanitizedFrameRecord] = []

        total_input = 0
        total_output = 0

        for image_type in self.cfg["ANGLE_TYPES"]:
            src_dir = input_root / image_type
            out_dir = get_sanitized_type_dir(self.person_name, image_type)

            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            files = []
            if src_dir.exists():
                for ext in self.cfg["IMAGE_EXTS"]:
                    files.extend(src_dir.glob(f"*{ext}"))
            files = sorted(files)

            per_type_input[image_type] = len(files)
            total_input += len(files)

            accepted: List[tuple[Path, dict]] = []
            for p in files:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                keep, info, prepared = self._prepare_arcface_frame(img, image_type)
                if keep:
                    accepted.append((p, info, prepared))

            max_keep = int(self.cfg["MAX_KEEP_PER_ANGLE"])
            min_keep = int(self.cfg["MIN_KEEP_PER_ANGLE"])

            if len(accepted) > max_keep:
                accepted = sorted(accepted, key=lambda x: x[1]["quality"], reverse=True)[:max_keep]

            if len(accepted) < min_keep and files:
                rescored: List[tuple[Path, dict]] = []
                for p in files:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    _, info, prepared = self._prepare_arcface_frame(img, image_type)
                    if prepared is not None:
                        rescored.append((p, info, prepared))
                rescored = sorted(rescored, key=lambda x: x[1]["quality"], reverse=True)
                accepted = rescored[: min(min_keep, len(rescored))]

            for src, info, prepared in accepted:
                dst = out_dir / src.name
                ok = cv2.imwrite(
                    str(dst),
                    prepared,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg["OUTPUT_JPEG_QUALITY"])],
                )
                if not ok:
                    continue
                records.append(
                    SanitizedFrameRecord(
                        source_path=str(src),
                        output_path=str(dst),
                        image_type=image_type,
                        blur_score=float(info["blur"]),
                        brightness=float(info["brightness"]),
                        has_face=bool(info["has_face"]),
                        face_width=int(info["face_width"]),
                        face_height=int(info["face_height"]),
                        quality_score=float(info["quality"]),
                    )
                )

            per_type_output[image_type] = len(accepted)
            total_output += len(accepted)
            logger.info(
                f"{INFO} Sanitized {image_type}: %d -> %d",
                len(files),
                len(accepted),
            )

        report = FrameSanitizerReport(
            person_name=self.person_name,
            total_input=total_input,
            total_output=total_output,
            removed_count=max(0, total_input - total_output),
            per_type_input=per_type_input,
            per_type_output=per_type_output,
            records=records,
        )

        logger.info(
            f"{SUCCESS} Frame sanitizer completed: input=%d output=%d removed=%d",
            report.total_input,
            report.total_output,
            report.removed_count,
        )
        if report.total_output == 0:
            logger.warning(f"{WARNING} No sanitized frames kept for fine-tune")
        return report


__all__ = ["FrameSanitizerService"]
