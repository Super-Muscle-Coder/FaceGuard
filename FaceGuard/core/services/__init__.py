"""Unified exports for FaceGuard services (lazy import)."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    "DataCollectionService": (".DataCollectionService", "DataCollectionService"),
    "collect_videos": (".DataCollectionService", "collect_videos"),
    "VideoQualityService": (".VideoQualityService", "VideoQualityService"),
    "FrameExtractionService": (".FrameExtractionService", "FrameExtractionService"),
    "EmbeddingService": (".EmbeddingService", "EmbeddingService"),
    "SanitizerService": (".SanitizerService", "SanitizerService"),
    "RecognitionService": (".RecognitionService", "RecognitionService"),
    "SimpleFaceRecognizer": (".RecognitionService", "SimpleFaceRecognizer"),
    "ThresholdTuner": (".RecognitionService", "ThresholdTuner"),
    "QualityGate": (".RecognitionService", "QualityGate"),
    "FineTuneService": (".FineTuneService", "FineTuneService"),
    "FrameSanitizerService": (".FrameSanitizerService", "FrameSanitizerService"),
    "MasterTraningService": (".MasterTraningService", "MasterTraningService"),
    "PackagingService": (".PackagingService", "PackagingService"),
    "launch_packaging_gui": (".PackagingService", "launch_packaging_gui"),
    "IoTService": (".IoTService", "IoTService"),
    "MasterWorkflowService": (".MasterWorkflowService", "MasterWorkflowService"),
    "PipelineError": (".MasterWorkflowService", "PipelineError"),
    "run_pipeline_for_person": (".MasterWorkflowService", "run_pipeline_for_person"),
}


__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_name, attr_name = _EXPORT_MAP[name]
    mod = import_module(mod_name, __name__)
    value = getattr(mod, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
