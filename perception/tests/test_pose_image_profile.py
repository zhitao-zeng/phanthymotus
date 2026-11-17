"""Image defaults match the seeded profile without relying on launcher env."""
import yaml
import pytest

from tools import seed_pose_action_models as seed


@pytest.mark.parametrize('family', ['jp61', 'jp511'])
def test_only_seeded_segment_profile_changes_image_config(tmp_path, monkeypatch, family):
    config = {'plugins': {'pose': {'enabled': True, 'activity_mode': 'rolling'},
                          'asr': {'enabled': False}}}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(seed.md, 'ensure_pose_model', lambda *a, **k: {'pose.engine': '/old-pose'})
    monkeypatch.setattr(seed.md, 'ensure_action_model', lambda *a, **k: {'action.engine': '/old-action'})
    monkeypatch.setattr(seed.md, 'ensure_segment_action_model', lambda *a, **k: {
        'yolo26s-pose-batch4.engine': '/opt/segment/pose.engine'})
    monkeypatch.setattr(seed.md, 'MODELS_ROOT', seed.md.MODELS_ROOT)
    seed.seed(family, path)
    result = yaml.safe_load(path.read_text())
    assert result['plugins']['asr'] == {'enabled': False}
    assert result['plugins']['pose']['enabled'] is True
    if family == 'jp61':
        assert result['plugins']['pose']['activity_mode'] == 'segment'
        assert result['plugins']['pose']['model'] == '/opt/segment/pose.engine'
    else:
        assert result == config
