"""
Health Check Routes.

Endpoints:
- GET /health - Basic health check
- GET /metrics - Service metrics
- GET /cameras - Camera statuses
"""

import logging
import time
from datetime import datetime
from flask import Blueprint, jsonify, Response, stream_with_context

logger = logging.getLogger(__name__)

# Create blueprint
health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check.
    
    Returns:
        200: Service is healthy
        500: Service is unhealthy
    """
    try:
        # Get service instance from app context
        from flask import current_app
        service = current_app.config['IOT_SERVICE']
        
        # Get database size
        database_persons = len(service.database)
        
        # Get uptime
        uptime = service.metrics.uptime
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database_persons": database_persons,
            "uptime_seconds": round(uptime, 2),
            "version": "1.0.0"
        }), 200
        
    except Exception as ex:
        logger.exception("Health check failed: %s", ex)
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(ex)
        }), 500


@health_bp.route('/debug/runtime', methods=['GET'])
def debug_runtime_state():
    """Dump runtime recognition state for diagnostics (DB/SQLite/head consistency)."""
    try:
        from flask import current_app
        service = current_app.config['IOT_SERVICE']
        state = service.get_debug_runtime_state()
        state["timestamp"] = datetime.now().isoformat()
        return jsonify(state), 200
    except Exception as ex:
        logger.exception("Failed to get debug runtime state: %s", ex)
        return jsonify({
            "error": "DebugRuntimeError",
            "message": str(ex),
            "timestamp": datetime.now().isoformat(),
        }), 500
  

@health_bp.route('/camera/<camera_id>/viewer', methods=['GET'])
def camera_viewer(camera_id: str):
    """Simple HTML viewer for ESP32 camera MJPEG stream."""
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>FaceGuard Camera Viewer - {camera_id}</title>
        <style>
          body {{ background:#020617; color:#e2e8f0; font-family:Arial,sans-serif; margin:0; padding:20px; }}
          h2 {{ margin:0 0 12px 0; }}
          img {{ max-width:100%; border:1px solid #334155; border-radius:10px; background:#0f172a; }}
          .hint {{ color:#94a3b8; margin-top:10px; }}
        </style>
      </head>
      <body>
        <h2>Camera: {camera_id}</h2>
        <img src=\"/camera/{camera_id}/stream.mjpg\" alt=\"stream\" />
        <div class=\"hint\">If stream is blank, ensure ESP32 is sending frames and camera_id matches exactly.</div>
      </body>
    </html>
    """
    return Response(html, mimetype='text/html; charset=utf-8')


@health_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get service metrics.
    
    Returns:
        200: Metrics data
    """
    try:
        from flask import current_app
        service = current_app.config['IOT_SERVICE']
        
        metrics = service.get_metrics()
        
        return jsonify(metrics.to_dict()), 200
        
    except Exception as ex:
        logger.exception("Failed to get metrics: %s", ex)
        return jsonify({
            "error": "MetricsError",
            "message": str(ex),
            "timestamp": datetime.now().isoformat()
        }), 500


@health_bp.route('/cameras', methods=['GET'])
def get_camera_statuses():
    """
    Get all camera statuses.
    
    Returns:
        200: Camera statuses
    """
    try:
        from flask import current_app
        service = current_app.config['IOT_SERVICE']
        
        statuses = service.get_camera_statuses()
        stale_sec = 120
        try:
            from config.settings import IOT_SERVICE_CONFIG
            stale_sec = int(IOT_SERVICE_CONFIG.get("CAMERA_STALE_SECONDS", 120))
        except Exception:
            pass
        
        # Convert to JSON-serializable format
        result = {}
        now = time.time()
        for camera_id, status in statuses.items():
            data = status.to_dict()
            try:
                age = max(0.0, now - datetime.fromisoformat(status.last_seen).timestamp())
            except Exception:
                age = -1.0
            data["is_stale"] = age >= stale_sec if age >= 0 else True
            data["last_seen_age_seconds"] = round(age, 2) if age >= 0 else None
            result[camera_id] = data
        
        return jsonify({
            "cameras": result,
            "total": len(result),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as ex:
        logger.exception("Failed to get camera statuses: %s", ex)
        return jsonify({
            "error": "StatusError",
            "message": str(ex),
            "timestamp": datetime.now().isoformat()
        }), 500


@health_bp.route('/camera/<camera_id>/stream.mjpg', methods=['GET'])
def stream_camera_frame(camera_id: str):
    """MJPEG stream of latest camera frames for live debugging."""
    try:
        from flask import current_app
        service = current_app.config['IOT_SERVICE']

        def gen():
            while True:
                jpg = service.get_last_frame_jpeg(camera_id)
                if jpg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.2)

        return Response(
            stream_with_context(gen()),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as ex:
        logger.exception("Failed to stream camera frame: %s", ex)
        return jsonify({
            "error": "StreamError",
            "message": str(ex),
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
        }), 500


@health_bp.route('/camera/<camera_id>/latest.jpg', methods=['GET'])
def latest_camera_frame(camera_id: str):
    """Return latest frame seen from a camera for debugging stream visibility."""
    try:
        from flask import current_app
        service = current_app.config['IOT_SERVICE']
        jpg = service.get_last_frame_jpeg(camera_id)
        if not jpg:
            return jsonify({
                "error": "NoFrame",
                "message": "No fresh frame available for this camera",
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
            }), 404

        return Response(jpg, mimetype='image/jpeg')
    except Exception as ex:
        logger.exception("Failed to get latest camera frame: %s", ex)
        return jsonify({
            "error": "FrameError",
            "message": str(ex),
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
        }), 500


@health_bp.route('/reload', methods=['POST'])
def reload_database():
    """
    Reload recognition database from MinIO.
    
    Returns:
        200: Database reloaded
        500: Reload failed
    """
    try:
        from flask import current_app, request
        service = current_app.config['IOT_SERVICE']
        
        # Check for force_refresh parameter
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        
        # Reload database
        service.reload_database(force_refresh=force_refresh)
        logger.info("health.reload: database reloaded force_refresh=%s persons=%d", force_refresh, len(service.database))
        
        return jsonify({
            "status": "success",
            "message": "Database reloaded",
            "persons": len(service.database),
            "force_refresh": force_refresh,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as ex:
        logger.exception("Failed to reload database: %s", ex)
        return jsonify({
            "error": "ReloadError",
            "message": str(ex),
            "timestamp": datetime.now().isoformat()
        }), 500


__all__ = ["health_bp"]
