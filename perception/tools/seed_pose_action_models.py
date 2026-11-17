"""Seed matching engines and pin the image's pose profile to those assets."""
from pathlib import Path
import argparse
import yaml

from utils import model_downloader as md


def seed(family, config_path):
    md.MODELS_ROOT = '/opt'
    pose = md.ensure_pose_model('/opt/vision-pose/'+family, family=family)
    action = md.ensure_action_model('/opt/vision-action/'+family, family=family)
    segment = (md.ensure_segment_action_model('/opt/vision-action-segment/'+family, family=family)
               if family in md.SEGMENT_ACTION_MODEL_BUNDLES else {})
    if segment:
        path = Path(config_path)
        config = yaml.safe_load(path.read_text())
        config['plugins']['pose'].update(
            activity_mode='segment', model=segment['yolo26s-pose-batch4.engine'],
            activity_idle_flush_s=0.0)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return pose, action, segment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jp-version', required=True)
    parser.add_argument('--config', default='/work/config.yaml')
    args = parser.parse_args()
    family = 'jp61' if args.jp_version == '61' else 'jp511'
    paths = seed(family, args.config)
    print('[build] pre-seeded pose profile', family, [list(p) for p in paths])
