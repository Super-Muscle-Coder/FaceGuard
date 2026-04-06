"""Entry point for fine-tune research pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from core.services.MasterTraningService import MasterTraningService


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FaceGuard fine-tune research pipeline runner")
    parser.add_argument("--person", type=str, help="Tên người dùng cần fine-tune")
    parser.add_argument("--target-frames", type=int, default=100, help="Số frame mục tiêu cho mỗi video")
    parser.add_argument("--auto-reject-threshold", type=float, default=0.6, help="Ngưỡng auto-reject của phase thu thập")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = _build_parser()
    args = parser.parse_args(argv)

    person_name = args.person.strip() if args.person else ""
    if not person_name:
        person_name = input("Nhập tên người dùng cần fine-tune: ").strip()

    if not person_name:
        logger.error("Person name không được để trống")
        return 1

    workflow = MasterTraningService()

    # Reuse mode: if sanitized data already exists in data/temp/{person}/sanitized_frames
    if workflow.has_reusable_sanitized_data(person_name):
        logger.info("Found reusable sanitized data for %s -> skip phase 1/2/3", person_name)
        success = workflow.run_finetune_only_from_existing(person_name)
    else:
        success = workflow.run_complete_pipeline(
            person_name=person_name,
            auto_reject_threshold=args.auto_reject_threshold,
            target_frames=args.target_frames,
        )

    if not success:
        return 1

    # User confirmation: deploy or keep temp for further tuning
    try:
        decision = input("Upload to MinIO/SQLite and finalize (Y/N)? ").strip().lower()
    except Exception:
        decision = "n"

    while decision not in ("y", "yes", "n", "no"):
        try:
            decision = input("Please enter Y or N: ").strip().lower()
        except Exception:
            decision = "n"

    if decision in ("y", "yes"):
        workflow.deploy_last_finetune(cleanup_temp=True)
        logger.info("Deployment completed: MinIO + SQLite updated, replay buffer saved, temp cleaned")
    else:
        logger.info("Deployment skipped by user. Temp data retained for next fine-tune run.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
