"""FaceGuard application entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from core.services import MasterWorkflowService


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FaceGuard training workflow runner")
    parser.add_argument("--person", type=str, help="Tên người dùng cần train")
    parser.add_argument("--target-frames", type=int, default=100, help="Số frame mục tiêu cho mỗi video")
    parser.add_argument("--auto-reject-threshold", type=float, default=0.6, help="Ngưỡng auto-reject của phase thu thập")
    parser.add_argument("--non-interactive", action="store_true", help="Chạy không cần xác nhận tay ở quality gate")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = _build_parser()
    args = parser.parse_args(argv)

    person_name = args.person.strip() if args.person else ""
    if not person_name:
        person_name = input("Nhập tên người dùng cần training: ").strip()

    if not person_name:
        logger.error("Person name không được để trống")
        return 1

    workflow = MasterWorkflowService(auto_cleanup=True)
    success = workflow.run_complete_pipeline(
        person_name=person_name,
        auto_reject_threshold=args.auto_reject_threshold,
        target_frames=args.target_frames,
        interactive=not args.non_interactive,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
