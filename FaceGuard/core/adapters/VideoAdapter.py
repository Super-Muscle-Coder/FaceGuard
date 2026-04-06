"""
Video Adapter - Wrapper cho OpenCV Operations.

Cung cấp unified interface cho:
- Camera operations (open/close/record)
- Video file operations (read/write/probe)
- Metadata extraction

===================  LỊCH SỬ REFACTOR  ====================
Refactored 17-01-2026:
- Loại bỏ: RAW_VIDEO_DIRS import (đã migrate sang storage keys)
- Giữ lại: TEMP_VIDEO_DIR (local temp), configs
- Mục đích: Clean imports, tương thích với config mới
"""

import os
import cv2
import time
import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Iterable, List

from config.settings import (
    DATA_COLLECTION_CONFIG,
    VIDEO_QUALITY_CONFIG,
)
from core.entities import (
    DataVideoInfo as CollectionVideoInfo,
    VideoState,
    VideoType,
    ExtractFrameVideoInfo as ExtractVideoInfo
)

logger = logging.getLogger(__name__)


class VideoAdapter:
    """
    Wrapper tiện ích cho camera và video files.
    
    Architecture:
    - Mở/đóng camera theo config DATA_COLLECTION_CONFIG
    - Đọc frame theo generator pattern
    - Ghi clip (15s mặc định) với overlay countdown
    - Probe metadata thành dataclass đã định nghĩa trong core.entities
    
    Note:
    - Adapter này chỉ xử lý IO video/camera (không liên quan storage keys)
    - Sử dụng thuần OpenCV, không phụ thuộc storage backends
    - Videos được lưu tạm trong TEMP_VIDEO_DIR, sau đó di chuyển qua StorageAdapter
    """
        
    def __init__(self, camera_id: int = DATA_COLLECTION_CONFIG["CAMERA_ID"]):
        """
        Khởi tạo VideoAdapter.
        
        Args:
            camera_id: Camera index (mặc định từ config)
        """
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None

    # ---------------- CAMERA ----------------
    def open_camera(
        self,
        width: int = DATA_COLLECTION_CONFIG["VIDEO_RESOLUTION"][0],
        height: int = DATA_COLLECTION_CONFIG["VIDEO_RESOLUTION"][1],
        fps: int = DATA_COLLECTION_CONFIG["VIDEO_FPS"],
    ) -> bool:
        """
        Mở camera với cấu hình resolution và FPS.
        
        Args:
            width: Video width
            height: Video height
            fps: Frames per second
            
        Returns:
            True nếu mở camera thành công
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error("Không thể mở camera %s", self.camera_id)
                self.cap = None
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            ret, _ = self.cap.read()
            if not ret:
                logger.error("Camera mở được nhưng không đọc được frame")
                self.close_camera()
                return False

            logger.info("Camera đã mở: %sx%s @ %.1f fps", width, height, fps)
            return True
        except Exception as ex:
            logger.exception("open_camera thất bại: %s", ex)
            self.close_camera()
            return False

    def close_camera(self) -> None:
        """Đóng camera và giải phóng tài nguyên."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # ✅ FIX: Don't call destroyAllWindows() - may conflict with GUI
        # Only destroy if not using external GUI
        # cv2.destroyAllWindows()

    def close(self) -> None:
        """
        Alias cho close_camera() để API thân thiện hơn.

        Usage:
            adapter.close()  # Thay vì adapter.close_camera()
        """
        self.close_camera()

    def is_open(self) -> bool:
        """Kiểm tra camera có đang mở không."""
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Optional[Tuple[bool, Optional[any]]]:
        """
        Đọc một frame từ camera.
        
        Returns:
            (True, frame) nếu thành công, (False, None) nếu thất bại, None nếu camera chưa mở
        """
        if not self.is_open():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    # ---------------- FILE IO ----------------
    @staticmethod
    def get_video_info(path: str) -> Optional[dict]:
        """
        Lấy metadata của video file.
        
        Args:
            path: Đường dẫn video file
            
        Returns:
            Dict chứa fps, total_frames, width, height, duration, path
        """
        if not os.path.exists(path):
            return None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0
        cap.release()
        return {
            "fps": fps,
            "total_frames": total,
            "width": width,
            "height": height,
            "duration": duration,
            "path": str(path),
        }

    @staticmethod
    def iter_video_frames(
        path: str, stride: int = 1
    ) -> Generator[Tuple[int, any], None, None]:
        """
        Generator để iterate qua frames của video.
        
        Args:
            path: Đường dẫn video file
            stride: Bước nhảy frame (1 = mọi frame, 2 = skip 1 frame, ...)
            
        Yields:
            (frame_index, frame)
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error("Không thể mở video: %s", path)
            return
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
        cap.release()

    @staticmethod
    def write_video(
        frames: Iterable[any],
        output_path: str,
        fps: float,
        resolution: Tuple[int, int],
        codec: str = DATA_COLLECTION_CONFIG["VIDEO_CODEC"],
    ) -> bool:
        """
        Ghi frames thành video file.
        
        Args:
            frames: Iterable of frames
            output_path: Đường dẫn output
            fps: Frames per second
            resolution: (width, height)
            codec: FourCC codec code
            
        Returns:
            True nếu ghi thành công
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
            if not writer.isOpened():
                logger.error("Không thể tạo VideoWriter: %s", output_path)
                return False
            count = 0
            for frame in frames:
                writer.write(frame)
                count += 1
            writer.release()
            logger.info("Đã lưu video %s (%d frames)", output_path, count)
            return True
        except Exception as ex:
            logger.exception("write_video thất bại: %s", ex)
            return False

    # ---------------- RECORDING (CAMERA -> FILE) ----------------
    def record_clip(
        self,
        output_path: str,
        duration_sec: int = DATA_COLLECTION_CONFIG["RECORDING_DURATION"],
        show_preview: bool = False,
        overlay_countdown: bool = False,
        fps: int = DATA_COLLECTION_CONFIG["VIDEO_FPS"],
        codec: str = DATA_COLLECTION_CONFIG["VIDEO_CODEC"],
    ) -> Tuple[bool, float, Tuple[int, int], float]:
        """
        Quay video clip từ camera.
        
        Args:
            output_path: Đường dẫn lưu video
            duration_sec: Thời lượng quay (giây)
            show_preview: Hiển thị preview trong khi quay
            overlay_countdown: Hiển thị countdown trên video
            fps: Frames per second
            codec: FourCC codec code
            
        Returns:
            (success, actual_duration, resolution, fps)
        """
        if not self.is_open():
            logger.error("Camera chưa mở")
            return False, 0.0, (0, 0), 0.0

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error("Không thể tạo writer: %s", output_path)
            return False, 0.0, (0, 0), 0.0

        start = time.time()
        frames = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                elapsed = time.time() - start
                remaining = max(0, duration_sec - elapsed)
                if overlay_countdown:
                    self._draw_countdown(frame, remaining, duration_sec)
                writer.write(frame)
                frames += 1

                if show_preview:
                    cv2.imshow("Recording", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC để dừng
                        break
                if elapsed >= duration_sec:
                    break
        finally:
            writer.release()
            if show_preview:
                cv2.destroyWindow("Recording")

        actual = time.time() - start
        logger.info(
            "Đã quay %s: %.2fs, %d frames, %sx%s",
            output_path, actual, frames, width, height
        )
        return True, actual, (width, height), float(fps)

    @staticmethod
    def _draw_countdown(frame, remaining: float, total: int) -> None:
        """Vẽ thanh progress và countdown lên frame."""
        h, w = frame.shape[:2]
        bar_h = 20
        progress = 1.0 - (remaining / total)
        
        # Progress bar background
        cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 80), -1)
        
        # Progress bar fill
        cv2.rectangle(frame, (0, 0), (int(w * progress), bar_h), (0, 200, 0), -1)
        
        # Countdown text
        txt = f"{int(remaining)+1}s"
        cv2.putText(
            frame, txt, (w // 2 - 30, bar_h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )

    # ---------------- SNAPSHOT ----------------
    @staticmethod
    def save_frame(frame, output_path: str) -> bool:
        """
        Lưu một frame thành ảnh.
        
        Args:
            frame: Frame cần lưu
            output_path: Đường dẫn output
            
        Returns:
            True nếu lưu thành công
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(output_path, frame)
        if ok:
            logger.info("Đã lưu frame: %s", output_path)
        else:
            logger.error("Lưu frame thất bại: %s", output_path)
        return ok

    # ---------------- METADATA HELPERS ----------------
    @staticmethod
    def probe_collection_info(
        path: Path, video_type: VideoType
    ) -> Optional[CollectionVideoInfo]:
        """
        Probe video file và tạo CollectionVideoInfo dataclass.
        
        Args:
            path: Đường dẫn video file
            video_type: Loại video (frontal/profile/...)
            
        Returns:
            CollectionVideoInfo hoặc None nếu probe thất bại
        """
        info = VideoAdapter.get_video_info(str(path))
        if not info:
            return None
        file_size = path.stat().st_size if path.exists() else 0
        return CollectionVideoInfo(
            video_type=video_type.value,
            temp_path=str(path),
            final_path=None,
            duration=info["duration"],
            resolution=(info["width"], info["height"]),
            fps=info["fps"],
            file_size=file_size,
            recorded_at=path.stat().st_mtime_ns.__str__(),
            state=VideoState.REVIEWING,
        )

    @staticmethod
    def probe_extract_info(path: Path) -> Optional[ExtractVideoInfo]:
        """
        Probe video file và tạo ExtractVideoInfo dataclass.
        
        Args:
            path: Đường dẫn video file
            
        Returns:
            ExtractVideoInfo hoặc None nếu probe thất bại
        """
        info = VideoAdapter.get_video_info(str(path))
        if not info:
            return None
        return ExtractVideoInfo(
            path=str(path),
            fps=info["fps"],
            total_frames=info["total_frames"],
            duration=info["duration"],
            width=info["width"],
            height=info["height"],
        )

    @staticmethod
    def validate_basic(path: Path) -> Tuple[bool, str]:
        """
        Validate cơ bản video file (duration, resolution).
        
        Args:
            path: Đường dẫn video file
            
        Returns:
            (is_valid, message)
        """
        info = VideoAdapter.get_video_info(str(path))
        if not info:
            return False, "Không thể mở video"

        if info["duration"] < VIDEO_QUALITY_CONFIG["MIN_DURATION"]:
            return False, f"Video quá ngắn ({info['duration']:.1f}s)"
        if info["duration"] > VIDEO_QUALITY_CONFIG["MAX_DURATION"]:
            return False, f"Video quá dài ({info['duration']:.1f}s)"
        
        min_w, min_h = VIDEO_QUALITY_CONFIG["MIN_RESOLUTION"]
        if info["width"] < min_w or info["height"] < min_h:
            return False, f"Resolution quá thấp ({info['width']}x{info['height']})"
        
        return True, "OK"

    @staticmethod
    def temp_video_path(person: str, video_type: VideoType) -> Path:
        """
        Utility tạo temp path theo naming convention.

        Refactored 08-03-2026:
        - Use get_video_dir() helper instead of TEMP_VIDEO_DIR
        - Production-ready folder structure

        Args:
            person: Tên người
            video_type: Loại video

        Returns:
            Path to temp video file
        """
        from config.settings import get_video_dir

        video_dir = get_video_dir(person, video_type.value)

        # Find next available filename
        counter = 1
        while True:
            name = f"video_{counter:03d}.mp4"
            video_path = video_dir / name
            if not video_path.exists():
                return video_path
            counter += 1


__all__ = ["VideoAdapter"]