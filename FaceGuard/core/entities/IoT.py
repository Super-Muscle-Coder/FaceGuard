from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class ServiceMetrics:
    started_at: float
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0

    @property
    def uptime(self) -> float:
        import time
        return max(0.0, time.time() - self.started_at)

    def update_latency(self, latency_ms: float):
        self.total_requests += 1
        if self.total_requests == 1:
            self.avg_latency_ms = latency_ms
        else:
            alpha = 0.2
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms

    def to_dict(self) -> Dict:
        return {
            "uptime_seconds": round(self.uptime, 2),
            "total_requests": self.total_requests,
            "success_requests": self.success_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


@dataclass
class CameraStatus:
    camera_id: str
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_identity: str = "Unknown"
    last_confidence: float = 0.0
    last_result: str = "denied"
    requests: int = 0

    def to_dict(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "last_seen": self.last_seen,
            "last_identity": self.last_identity,
            "last_confidence": self.last_confidence,
            "last_result": self.last_result,
            "requests": self.requests,
        }


@dataclass
class IoTRecognitionResult:
    status: str
    identity: str
    confidence: float
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "identity": self.identity,
            "confidence": self.confidence,
            "message": self.message,
            "timestamp": self.timestamp,
        }


__all__ = ["ServiceMetrics", "CameraStatus", "IoTRecognitionResult"]
