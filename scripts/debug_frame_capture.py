#!/usr/bin/env python3
"""Watch /tmp/samowl for completed inference frames and copy debug images.

Detects a new frame by polling hotspots.json mtime. Each time it changes,
copies mask_*.png, boundary.png, masked_depth.png to debug_dir/frame_NNNN/.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="/tmp/samowl")
    p.add_argument("--debug-dir", required=True)
    p.add_argument("--hotspots", default="hotspots.json")
    p.add_argument("--poll-interval", type=float, default=0.25)
    return p.parse_args()


def collect_debug_files(work_dir: Path) -> list[Path]:
    candidates = []
    for pattern in ("mask*.png", "boundary*.png", "masked_depth*.png"):
        candidates.extend(work_dir.glob(pattern))
    # Also grab the hotspots JSON snapshot
    for f in work_dir.glob("hotspots.json"):
        candidates.append(f)
    return candidates


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    hotspot_path = work_dir / args.hotspots
    last_mtime = None
    frame_count = 0

    print(f"[capture] watching {hotspot_path}", flush=True)
    print(f"[capture] saving frames to {debug_dir}", flush=True)

    try:
        while True:
            try:
                mtime = hotspot_path.stat().st_mtime
            except FileNotFoundError:
                time.sleep(args.poll_interval)
                continue

            if mtime != last_mtime:
                last_mtime = mtime
                frame_count += 1
                frame_dir = debug_dir / f"frame_{frame_count:04d}"
                frame_dir.mkdir(exist_ok=True)

                files = collect_debug_files(work_dir)
                for src in files:
                    try:
                        shutil.copy2(src, frame_dir / src.name)
                    except Exception as e:
                        print(f"[capture] warn: could not copy {src}: {e}", flush=True)

                print(f"[capture] frame {frame_count:04d} — saved {len(files)} files to {frame_dir}", flush=True)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(f"[capture] stopped — captured {frame_count} frames", flush=True)


if __name__ == "__main__":
    main()
