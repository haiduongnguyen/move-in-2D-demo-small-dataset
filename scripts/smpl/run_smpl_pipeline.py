#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.smpl.pipeline import (  # noqa: E402
    DEFAULT_4D_HUMANS_ROOT,
    build_smpl_dataset,
    check_setup,
    convert_raw_outputs,
    prepare_test_run,
    render_previews,
    run_4dhumans_extract,
    run_test_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMPL data branch pipeline for Penn Action.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup-check", help="Check whether 4D-Humans and SMPL files are available.")
    setup.add_argument("--fourd-humans-root", default=str(DEFAULT_4D_HUMANS_ROOT))

    prepare = sub.add_parser("prepare-test", help="Create a 3-5 video SMPL test run manifest.")
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--num-videos", type=int, default=5)
    prepare.add_argument("--video-ids", nargs="*", default=None)

    extract = sub.add_parser("extract-test", help="Run 4D-Humans on selected test videos.")
    extract.add_argument("--run-name", required=True)
    extract.add_argument("--fourd-humans-root", default=str(DEFAULT_4D_HUMANS_ROOT))
    extract.add_argument("--python-bin", default=sys.executable)
    extract.add_argument("--extra-arg", action="append", default=[])
    extract.add_argument("--dry-run", action="store_true")

    convert = sub.add_parser("convert", help="Convert raw 4D-Humans outputs into project SMPL npz files.")
    convert.add_argument("--run-name", required=True)

    build = sub.add_parser("build-dataset", help="Build sample-level 64-frame SMPL dataset manifest.")
    build.add_argument("--run-name", required=True)
    build.add_argument("--output-run-name", default=None)

    preview = sub.add_parser("render-previews", help="Render quick SMPL/2D comparison previews.")
    preview.add_argument("--dataset-run-name", required=True)
    preview.add_argument("--max-samples", type=int, default=8)

    test = sub.add_parser("test", help="Run prepare -> extract -> convert -> build -> preview for a small test.")
    test.add_argument("--run-name", required=True)
    test.add_argument("--num-videos", type=int, default=5)
    test.add_argument("--video-ids", nargs="*", default=None)
    test.add_argument("--fourd-humans-root", default=str(DEFAULT_4D_HUMANS_ROOT))
    test.add_argument("--python-bin", default=sys.executable)
    test.add_argument("--extra-arg", action="append", default=[])
    test.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "setup-check":
        result = check_setup(fourd_humans_root=args.fourd_humans_root)
        print(json.dumps(result.report, indent=2, sort_keys=True))
        if not result.ok:
            raise SystemExit(2)
    elif args.command == "prepare-test":
        path = prepare_test_run(args.run_name, num_videos=args.num_videos, video_ids=args.video_ids)
        print(path)
    elif args.command == "extract-test":
        path = run_4dhumans_extract(
            run_name=args.run_name,
            fourd_humans_root=args.fourd_humans_root,
            python_bin=args.python_bin,
            extra_args=args.extra_arg,
            dry_run=args.dry_run,
        )
        print(path)
    elif args.command == "convert":
        print(convert_raw_outputs(args.run_name))
    elif args.command == "build-dataset":
        print(build_smpl_dataset(args.run_name, output_run_name=args.output_run_name))
    elif args.command == "render-previews":
        print(render_previews(args.dataset_run_name, max_samples=args.max_samples))
    elif args.command == "test":
        result = run_test_pipeline(
            run_name=args.run_name,
            num_videos=args.num_videos,
            video_ids=args.video_ids,
            fourd_humans_root=args.fourd_humans_root,
            python_bin=args.python_bin,
            extra_args=args.extra_arg,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

