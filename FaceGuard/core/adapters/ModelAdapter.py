"""
Model Adapter - Wrapper cho ONNX Models.

Cung cấp unified interface cho:
- SCRFD (face detector)
- ArcFace (face recognizer)

===================  LỊCH SỬ REFACTOR  ====================
Refactored 17-01-2026:
- Mục đích: Chuẩn hóa code style với các adapter khác
- Logic: Không thay đổi (wrapper cho ONNX runtime)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from config.settings import EMBEDDING_CONFIG
from core.entities import DetectedFace

logger = logging.getLogger(__name__)


class ModelAdapter:
    """
    Wrapper cho ONNX models - SCRFD và ArcFace.
    
    Architecture:
    - SCRFD (detector): detect_faces() → bboxes + landmarks + scores
    - ArcFace (recognizer): extract_embedding() → 512-dim vector
    - Alignment: align_face() → 112x112 normalized face
    
    Usage:
        adapter = ModelAdapter(scrfd_path, arcface_path)
        faces = adapter.detect_faces(img, threshold=0.5)
        for face in faces:
            aligned = adapter.align_face(img, face["landmarks"])
            embedding = adapter.extract_embedding(aligned)
    
    Note:
    - Tất cả ngưỡng lấy từ EMBEDDING_CONFIG
    - Support CPU và GPU (via ONNX providers)
    - Phù hợp cho Face_Embedding và Product_Packager
    """

    def __init__(
        self,
        scrfd_path: Path,
        arcface_path: Path,
        det_input_size: Tuple[int, int] = (640, 640),
        rec_input_size: Tuple[int, int] = (112, 112),
        providers: Optional[List[str]] = None,
    ):
        """
        Khởi tạo ModelAdapter.
        
        Args:
            scrfd_path: Đường dẫn SCRFD model (.onnx)
            arcface_path: Đường dẫn ArcFace model (.onnx)
            det_input_size: Input size cho detector
            rec_input_size: Input size cho recognizer
            providers: ONNX providers (CPU/GPU)
        """
        providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_input_size
        self.rec_size = rec_input_size

        # Load detector
        self.det_session = ort.InferenceSession(str(scrfd_path), providers=providers)
        self.det_input = self.det_session.get_inputs()[0].name
        self.det_outputs = [o.name for o in self.det_session.get_outputs()]

        # Load recognizer
        self.rec_session = ort.InferenceSession(str(arcface_path), providers=providers)
        self.rec_input = self.rec_session.get_inputs()[0].name

        logger.info(
            "ModelAdapter initialized: SCRFD=%s, ArcFace=%s",
            scrfd_path.name, arcface_path.name
        )

    # ---------------- FACE DETECTION ----------------
    def detect_faces(
        self,
        img_bgr: np.ndarray,
        threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        return_dataclass: bool = False,
    ) -> List[Dict]:
        """
        Detect faces trong ảnh.
        
        Args:
            img_bgr: Ảnh BGR format (OpenCV)
            threshold: Detection threshold (mặc định từ config)
            nms_threshold: NMS threshold (mặc định từ config)
            return_dataclass: Return DetectedFace dataclass thay vì dict
            
        Returns:
            List of detected faces:
            - Dict mode: {"bbox": [x1, y1, x2, y2], "score": float, "landmarks": np.ndarray}
            - Dataclass mode: DetectedFace(bbox, confidence)
        """
        # Lấy ngưỡng từ config nếu không được cung cấp
        threshold = (
            threshold if threshold is not None
            else EMBEDDING_CONFIG["DETECTION_THRESHOLD"]
        )
        nms_threshold = (
            nms_threshold if nms_threshold is not None
            else EMBEDDING_CONFIG["NMS_THRESHOLD"]
        )

        # Chuẩn bị input (resize + letterbox)
        h, w = img_bgr.shape[:2]
        scale = min(self.det_size[0] / w, self.det_size[1] / h)
        resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        det_img = np.zeros((self.det_size[1], self.det_size[0], 3), dtype=np.uint8)
        det_img[: resized.shape[0], : resized.shape[1], :] = resized

        # Normalize
        blob = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        # Inference
        outputs = self.det_session.run(self.det_outputs, {self.det_input: blob})
        
        # Parse outputs (multi-scale anchors)
        faces: List[Dict] = []
        scores_list = []
        bboxes_list = []
        kpss_list = []

        feat_strides = [8, 16, 32]
        fmc = 3  # số feature maps per stride
        num_anchors = 2
        anchor_centers = {}

        # Generate anchor centers cho từng stride
        for stride in feat_strides:
            feat_h = self.det_size[1] // stride
            feat_w = self.det_size[0] // stride
            centers = np.stack(
                np.mgrid[:feat_h, :feat_w][::-1], axis=-1
            ).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                centers = np.stack([centers] * num_anchors, axis=1).reshape((-1, 2))
            anchor_centers[stride] = centers

        # Decode predictions
        for i, stride in enumerate(feat_strides):
            scores = outputs[i]
            bbox_preds = outputs[i + fmc] * stride
            kps_preds = outputs[i + fmc * 2] * stride

            centers = anchor_centers[stride]
            bboxes = self._distance2bbox(centers, bbox_preds)
            kpss = self._distance2kps(centers, kps_preds)
            
            # Scale về kích thước gốc
            bboxes /= scale
            kpss /= scale

            scores_list.append(scores)
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        # Merge multi-scale predictions
        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)

        # Sort by score
        order = scores.argsort()[::-1]
        scores = scores[order]
        bboxes = bboxes[order]
        kpss = kpss[order]

        # Threshold filtering
        keep = np.where(scores > threshold)[0]
        scores = scores[keep]
        bboxes = bboxes[keep]
        kpss = kpss[keep]

        # Non-maximum suppression
        keep = self._nms(bboxes, scores, nms_threshold)
        
        # Format outputs
        for idx in keep:
            x1, y1, x2, y2 = bboxes[idx].astype(int)
            bbox = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
            score = float(scores[idx])
            landmarks = kpss[idx].astype(np.float32)
            
            if return_dataclass:
                faces.append(
                    DetectedFace(
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        confidence=score,
                    )
                )
            else:
                faces.append({
                    "bbox": bbox,
                    "score": score,
                    "landmarks": landmarks
                })
        
        return faces

    # ---------------- ALIGN & EMBEDDING ----------------
    @staticmethod
    def align_face(
        img: np.ndarray,
        landmarks: np.ndarray,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """
        Affine alignment face theo landmarks.
        
        Args:
            img: Ảnh gốc (BGR)
            landmarks: 5 facial landmarks (5x2 array)
            target_size: Kích thước output
            
        Returns:
            Aligned face (112x112) hoặc None nếu thất bại
            
        Note:
            Landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        if landmarks is None or len(landmarks) != 5:
            return None
        
        # Template landmarks (ArcFace standard)
        src = np.array(
            [
                [38.2946, 51.6963],  # left eye
                [73.5318, 51.5014],  # right eye
                [56.0252, 71.7366],  # nose
                [41.5493, 92.3655],  # left mouth
                [70.7299, 92.2041],  # right mouth
            ],
            dtype=np.float32,
        )
        dst = landmarks.astype(np.float32)
        
        # Estimate affine transform
        tform = cv2.estimateAffinePartial2D(dst, src)[0]
        if tform is None:
            return None
        
        # Warp image
        return cv2.warpAffine(img, tform, target_size, borderValue=0.0)

    def extract_embedding(self, aligned_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding từ aligned face.
        
        Args:
            aligned_bgr: Aligned face (112x112 BGR)
            
        Returns:
            512-dim normalized embedding hoặc None nếu thất bại
        """
        # Resize nếu cần
        if aligned_bgr.shape[:2] != self.rec_size:
            aligned_bgr = cv2.resize(aligned_bgr, self.rec_size)
        
        # Preprocess (BGR → RGB, normalize)
        face = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1))[None, ...].astype(np.float32)
        face = (face - 127.5) / 127.5
        
        # Inference
        emb = self.rec_session.run(None, {self.rec_input: face})[0].flatten()
        
        # L2 normalization
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        return emb

    # ---------------- HELPERS ----------------
    @staticmethod
    def _distance2bbox(points, distance):
        """Convert distance predictions to bboxes (SCRFD format)."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points, distance):
        """Convert distance predictions to keypoints (SCRFD format)."""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1).reshape((-1, 5, 2))

    @staticmethod
    def _nms(boxes, scores, nms_threshold):
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU với các boxes còn lại
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Giữ lại boxes có IoU <= threshold
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]

        return keep


__all__ = ["ModelAdapter"]