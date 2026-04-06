"""Flask API app factory for FaceGuard IoT runtime."""

from __future__ import annotations

import logging
from datetime import datetime

from flask import Flask, jsonify

from config.settings import IOT_SERVICE_CONFIG, IOT_API_CONFIG
from core.services.IoTService import IoTService
from api.routes import health_bp, recognition_bp

logger = logging.getLogger(__name__)


def create_iot_app(iot_service: IoTService | None = None) -> Flask:
    app = Flask(__name__)

    service = iot_service or IoTService()
    app.config["IOT_SERVICE"] = service

    app.register_blueprint(health_bp)
    app.register_blueprint(recognition_bp)

    @app.get("/")
    def _root():
        return {
            "service": "FaceGuardIoT",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "endpoints": [
                "/health",
                "/metrics",
                "/cameras",
                "/reload",
                "/debug/runtime",
                "/recognize",
                "/camera/<camera_id>/latest.jpg",
                "/camera/<camera_id>/stream.mjpg",
                "/camera/<camera_id>/viewer",
                f"{IOT_API_CONFIG['API_V1_PREFIX']}/recognize",
            ],
        }

    @app.get(IOT_API_CONFIG["API_V1_PREFIX"])
    def _api_v1_root():
        return jsonify(
            {
                "service": "FaceGuardIoT",
                "status": "running",
                "version": "v1",
                "recognize": f"{IOT_API_CONFIG['API_V1_PREFIX']}/recognize",
            }
        )

    return app


def run():
    app = create_iot_app()
    host = IOT_SERVICE_CONFIG["HOST"]
    port = int(IOT_SERVICE_CONFIG["PORT"])
    debug = bool(IOT_SERVICE_CONFIG["DEBUG"])
    logger.info("Starting FaceGuard IoT API on %s:%s debug=%s", host, port, debug)
    logger.info(
        "IoT API config: key_header=%s threshold=%.2f max_payload_mb=%d",
        IOT_SERVICE_CONFIG["API_KEY_HEADER"],
        float(IOT_SERVICE_CONFIG["DEFAULT_RECOGNITION_THRESHOLD"]),
        int(IOT_SERVICE_CONFIG["MAX_IMAGE_SIZE_MB"]),
    )
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()


__all__ = ["create_iot_app", "run"]
