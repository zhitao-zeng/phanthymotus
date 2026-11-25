"""Seed verified hand-gesture TensorRT engines and pin pose config to them."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from utils import model_downloader as md


def seed(family: str, config_path: str) -> dict[str, str]:
    md.MODELS_ROOT = "/opt"
    paths = md.ensure_hand_detector_models(f"/opt/vision-hand-gesture/{family}", family=family)
    pose_path = paths["yolo26n-pose.engine"]
    hand_path = paths["yolov10n_hands_640.engine"]
    gest_path = paths["yolov10n_gestures_640.engine"]

    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["plugins"]["pose"].update(
        model=pose_path,
        hands=True,
        hand_backend="detector",
        hand_det_engine_path=hand_path,
        gesture_det_engine_path=gest_path,
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jp-version", required=True)
    parser.add_argument("--config", default="/work/config.yaml")
    args = parser.parse_args()
    family = "jp61" if args.jp_version == "61" else "jp511"
    seeded = seed(family, args.config)
    print("[build] pre-seeded hand-gesture profile", family, sorted(seeded))
