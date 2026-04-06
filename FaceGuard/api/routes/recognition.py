"""Recognition API routes for ESP32CAM."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request

from config.settings import IOT_SERVICE_CONFIG

logger = logging.getLogger(__name__)

recognition_bp = Blueprint("recognition", __name__)


def _build_response(
    *,
    allowed: bool,
    identity: str,
    confidence: float,
    reason: str,
    camera_id: str,
    sequence_number: int | None = None,
    status_code: int = 200,
):
    payload = {
        "allowed": bool(allowed),
        "identity": identity,
        "confidence": float(max(confidence, 0.0)),
        "reason": reason,
        "camera_id": camera_id,
        "sequence_number": sequence_number,
        "timestamp": datetime.now().isoformat(),
    }

    # Backward-compatible fields for internal consumers
    payload["status"] = "allowed" if allowed else "denied"
    payload["message"] = reason

    return jsonify(payload), status_code


@recognition_bp.route("/recognize", methods=["POST"])
@recognition_bp.route("/api/v1/recognize", methods=["POST"])
def recognize():
    try:
        service = current_app.config["IOT_SERVICE"]
        req_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

        # API key check (ESP32 sends X-API-Key)
        api_key_header = IOT_SERVICE_CONFIG["API_KEY_HEADER"]
        expected_key = IOT_SERVICE_CONFIG["API_KEY"]
        provided_key = request.headers.get(api_key_header, "")
        if expected_key and provided_key != expected_key:
            logger.warning("recognize[%s]: invalid api key camera_id=%s", req_id, request.form.get("camera_id", request.args.get("camera_id", "unknown")))
            return _build_response(
                allowed=False,
                identity="Unknown",
                confidence=0.0,
                reason="Invalid API key",
                camera_id=request.form.get("camera_id", request.args.get("camera_id", "unknown")),
                status_code=403,
            )

        # Inputs from form-data
        camera_id = request.form.get("camera_id") or request.args.get("camera_id", "unknown")

        sequence_number = None
        raw_seq = request.form.get("sequence_number") or request.args.get("sequence_number")
        if raw_seq is not None:
            try:
                sequence_number = int(raw_seq)
            except Exception:
                sequence_number = None

        metadata = request.form.get("metadata")
        metadata_obj = None
        if metadata:
            try:
                metadata_obj = json.loads(metadata)
            except Exception:
                metadata_obj = {"raw": metadata}

        image_bytes = None
        payload_source = "none"
        content_type = request.headers.get("Content-Type", "")

        if "image" in request.files:
            image_bytes = request.files["image"].read()
            payload_source = "multipart:image"
        elif request.data:
            image_bytes = request.data
            payload_source = "raw"

        if not image_bytes:
            logger.warning(
                "recognize[%s]: empty payload camera_id=%s content_type=%s content_length=%s files=%s form_keys=%s",
                req_id,
                camera_id,
                content_type,
                request.content_length,
                list(request.files.keys()),
                list(request.form.keys()),
            )
            return _build_response(
                allowed=False,
                identity="Unknown",
                confidence=0.0,
                reason="No image payload found",
                camera_id=camera_id,
                sequence_number=sequence_number,
                status_code=400,
            )

        max_size = int(IOT_SERVICE_CONFIG["MAX_IMAGE_SIZE_MB"]) * 1024 * 1024
        if len(image_bytes) > max_size:
            logger.warning(
                "recognize[%s]: payload too large camera_id=%s bytes=%d max=%d",
                req_id,
                camera_id,
                len(image_bytes),
                max_size,
            )
            return _build_response(
                allowed=False,
                identity="Unknown",
                confidence=0.0,
                reason="Payload too large",
                camera_id=camera_id,
                sequence_number=sequence_number,
                status_code=413,
            )

        result = service.process_request(image_bytes=image_bytes, camera_id=camera_id)

        allowed = result.get("status") == "allowed"
        identity = result.get("identity", "Unknown")
        confidence = float(result.get("confidence", 0.0))
        reason = result.get("message", "Matched" if allowed else "Not matched")

        if metadata_obj is not None:
            logger.debug("metadata camera_id=%s sequence=%s data=%s", camera_id, sequence_number, metadata_obj)

        logger.info(
            "recognize[%s]: camera_id=%s seq=%s payload=%s bytes=%d allowed=%s identity=%s confidence=%.4f db_persons=%s",
            req_id,
            camera_id,
            sequence_number,
            payload_source,
            len(image_bytes),
            allowed,
            identity,
            confidence,
            result.get("database_persons"),
        )

        return _build_response(
            allowed=allowed,
            identity=identity,
            confidence=confidence,
            reason=reason,
            camera_id=camera_id,
            sequence_number=sequence_number,
            status_code=200 if allowed else 403,
        )

    except Exception as ex:
        logger.exception("Recognition failed: %s", ex)
        return _build_response(
            allowed=False,
            identity="Unknown",
            confidence=0.0,
            reason=str(ex),
            camera_id=request.form.get("camera_id", request.args.get("camera_id", "unknown")),
            status_code=500,
        )


__all__ = ["recognition_bp"]
