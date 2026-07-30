import json
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from fnmatch import fnmatchcase
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_ROOT = REPO_ROOT / "perception"
BASE_REVISION = "241b72d8128a1bcf472754d71a7ccc0db76a3907"
ONE_MIB = 1024 * 1024
GIT_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
FORBIDDEN_MODEL_SUFFIXES = (
    ".pth.tar",
    ".pth",
    ".pt",
    ".onnx",
    ".engine",
    ".plan",
    ".safetensors",
    ".mnn",
)


def _git(repo: Path, *args: str | bytes) -> bytes:
    command = [b"git"]
    command.extend(
        value if isinstance(value, bytes) else os.fsencode(value)
        for value in args
    )
    result = subprocess.run(
        command,
        cwd=os.fsencode(repo),
        check=True,
        capture_output=True,
    )
    return result.stdout


def _nul_paths(repo: Path, *args: str | bytes) -> tuple[bytes, ...]:
    return tuple(path for path in _git(repo, *args).split(b"\0") if path)


def _object_id_from_tree(
    repo: Path,
    revision: str,
    path: bytes,
) -> bytes | None:
    records = _git(repo, "ls-tree", "-z", revision, "--", path).split(b"\0")
    for record in records:
        if not record:
            continue
        metadata, actual_path = record.split(b"\t", 1)
        _, object_type, object_id = metadata.split(b" ", 2)
        if actual_path == path and object_type == b"blob":
            return object_id
    return None


def _object_id_from_index(repo: Path, path: bytes) -> bytes | None:
    records = _git(repo, "ls-files", "--stage", "-z", "--", path).split(b"\0")
    for record in records:
        if not record:
            continue
        metadata, actual_path = record.split(b"\t", 1)
        _, object_id, stage = metadata.split(b" ", 2)
        if (
            actual_path == path
            and stage == b"0"
            and object_id.strip(b"0")
        ):
            return object_id
    return None


def _git_blob(repo: Path, object_id: bytes) -> bytes:
    size = int(_git(repo, "cat-file", "-s", object_id))
    if size >= ONE_MIB:
        return b"\0" * ONE_MIB
    return _git(repo, "cat-file", "blob", object_id)


def _worktree_blob(repo: Path, path: bytes) -> bytes:
    absolute = os.fsencode(repo) + b"/" + path
    if os.path.islink(absolute):
        return os.readlink(absolute)
    size = os.lstat(absolute).st_size
    if size >= ONE_MIB:
        return b"\0" * ONE_MIB
    with open(absolute, "rb") as handle:
        return handle.read()


def _branch_blob_candidates(
    repo: Path,
    base_revision: str,
) -> tuple[tuple[str, bytes, bytes], ...]:
    candidates: list[tuple[str, bytes, bytes]] = []

    head_paths = _nul_paths(
        repo,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        "--diff-filter=AM",
        f"{base_revision}..HEAD",
    )
    for path in head_paths:
        object_id = _object_id_from_tree(repo, "HEAD", path)
        if object_id is not None:
            candidates.append(("head", path, _git_blob(repo, object_id)))

    index_paths = _nul_paths(
        repo,
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--no-renames",
        "--diff-filter=AM",
        "HEAD",
    )
    for path in index_paths:
        object_id = _object_id_from_index(repo, path)
        if object_id is not None:
            candidates.append(("index", path, _git_blob(repo, object_id)))

    worktree_paths = _nul_paths(
        repo,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        "--diff-filter=AM",
    )
    for path in worktree_paths:
        candidates.append(("worktree", path, _worktree_blob(repo, path)))

    untracked_paths = _nul_paths(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    for path in untracked_paths:
        candidates.append(("untracked", path, _worktree_blob(repo, path)))
    return tuple(candidates)


def _branch_blob_violations(
    candidates: tuple[tuple[str, bytes, bytes], ...],
) -> tuple[tuple[str, str, bytes], ...]:
    violations: list[tuple[str, str, bytes]] = []
    forbidden = tuple(os.fsencode(value) for value in FORBIDDEN_MODEL_SUFFIXES)
    for source, path, blob in candidates:
        normalized_path = path.lower()
        if normalized_path.endswith(forbidden):
            violations.append(("forbidden_model_artifact", source, path))
        if len(blob) >= ONE_MIB:
            violations.append(("file_too_large", source, path))
        if blob.startswith(GIT_LFS_PREFIX):
            violations.append(("git_lfs_pointer", source, path))
    return tuple(violations)


def _is_dockerfile_path(path: bytes) -> bool:
    name = path.rsplit(b"/", 1)[-1].lower()
    return name == b"dockerfile" or name.startswith(b"dockerfile.")


def _docker_instruction_sources(arguments: str) -> tuple[str, ...]:
    stripped = arguments.strip()
    json_start = stripped.find("[")
    if json_start >= 0:
        option_text = stripped[:json_start].strip()
        if option_text:
            shlex.split(option_text, comments=True, posix=True)
        try:
            values = json.loads(stripped[json_start:])
        except json.JSONDecodeError:
            return ()
        if (
            not isinstance(values, list)
            or len(values) < 2
            or any(not isinstance(value, str) for value in values)
        ):
            return ()
        return tuple(values[:-1])

    try:
        tokens = shlex.split(stripped, comments=True, posix=True)
    except ValueError:
        return ()
    index = 0
    options_with_separate_values = {
        "--chown",
        "--chmod",
        "--exclude",
        "--from",
    }
    while index < len(tokens) and tokens[index].startswith("--"):
        option = tokens[index].lower()
        if "=" not in option and option in options_with_separate_values:
            index += 2
        else:
            index += 1
    values = tokens[index:]
    if len(values) < 2:
        return ()
    return tuple(values[:-1])


def _source_may_include_obstacle_models(source: str) -> bool:
    if re.match(r"^[a-z][a-z0-9+.-]*://", source, flags=re.IGNORECASE):
        return False
    normalized = source.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/").rstrip("/")
    if "$" in normalized:
        return True
    if normalized in {
        "",
        ".",
        "*",
        "**",
        "perception",
        "perception/*",
        "perception/**",
        "perception/models",
        "models",
        "models/obstacle-distance",
    }:
        return True
    if normalized.startswith("perception/models/"):
        return True
    if normalized.startswith("models/obstacle-distance/"):
        return True
    if any(
        fnmatchcase(candidate, normalized)
        for candidate in (
            "perception/models",
            "perception/models/obstacle-distance",
            "perception/models/obstacle-distance/model.engine",
            "models/obstacle-distance",
            "models/obstacle-distance/model.engine",
        )
    ):
        return True
    wildcard_prefix = re.split(r"[*?[]", normalized, maxsplit=1)[0].rstrip("/")
    return wildcard_prefix in {
        "",
        "perception",
        "perception/models",
        "models",
        "models/obstacle-distance",
    }


def _dockerfile_model_sources(source: str) -> tuple[str, ...]:
    normalized = re.sub(r"\\[ \t]*\r?\n", " ", source)
    dangerous: list[str] = []
    for line in normalized.splitlines():
        match = re.match(
            r"^\s*(?:copy|add)\s+(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        for candidate in _docker_instruction_sources(match.group(1)):
            if _source_may_include_obstacle_models(candidate):
                dangerous.append(candidate)
    return tuple(dangerous)


def _dockerfile_candidates(repo: Path) -> tuple[tuple[bytes, bytes], ...]:
    tracked = _nul_paths(repo, "ls-files", "-z")
    modified = set(
        _nul_paths(
            repo,
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=AM",
        )
    )
    untracked = _nul_paths(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    candidates: list[tuple[bytes, bytes]] = []
    for path in tracked:
        if not _is_dockerfile_path(path):
            continue
        if path in modified:
            blob = _worktree_blob(repo, path)
        else:
            object_id = _object_id_from_index(repo, path)
            if object_id is None:
                continue
            blob = _git_blob(repo, object_id)
        candidates.append((path, blob))
    for path in untracked:
        if _is_dockerfile_path(path):
            candidates.append((path, _worktree_blob(repo, path)))
    return tuple(candidates)


class ObstacleDistancePackagingTest(unittest.TestCase):
    def test_default_config_keeps_complete_model_handoff_disabled(self):
        config = (PERCEPTION_ROOT / "config.yaml").read_text(encoding="utf-8")
        expected = """  obstacle_distance:
    enabled: false
    mode: model
    backend_factory: ""
    scene_mode: metadata
    fixed_scene: ""
    depth_backend: lifelong_nk
    segmentation_backend: yolo26n_seg
    depth_model_dir: /models/obstacle-distance/lifelong-nk
    segmentation_model_dir: /models/obstacle-distance/yolo26n-seg
    decision_threshold_m: 1.0
    fallback_distance_m: 3.0
    soft_timeout_s: 2.5
    min_interval_ms: 0
    indoor:
      roi: [0, 300, 213, 426]
      min_depth_m: 0.3
      max_depth_m: 10.0
      percentile: 1.0
      min_valid_pixels: 64
    vehicle:
      allowed_classes: [person, car, truck, bus, motorcycle, bicycle]
      min_confidence: 0.25
      percentile: 1.0
      min_depth_m: 0.3
      max_depth_m: 80.0
      camera_to_bumper_offset_m: 1.0
      allow_approximate_geometry: false
      calibration: {}
"""
        self.assertIn(expected, config)

    def test_bundle_registers_obstacle_distance_plugin(self):
        source = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "from plugins.obstacle_distance import ObstacleDistancePlugin",
            source,
        )
        self.assertIn(
            'plugins_cfg.get("obstacle_distance", {}).get("enabled", False)',
            source,
        )

    def test_readme_links_formal_handoff_document(self):
        readme = (PERCEPTION_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "[障碍物距离模型交接](docs/obstacle_distance.md)",
            readme,
        )

    def test_handoff_document_covers_runtime_and_model_contracts(self):
        document = (
            PERCEPTION_ROOT / "docs" / "obstacle_distance.md"
        ).read_text(encoding="utf-8")
        required = (
            "Lifelong-MonoDepth",
            "NK.pth.tar",
            "NYU",
            "index 0",
            "KITTI",
            "index 1",
            "YOLO26n-seg",
            "diagnostic_constant",
            "model",
            "class DepthBackend(Protocol):",
            "domain: SceneDomain",
            "deadline_monotonic: float",
            ") -> DepthPrediction:",
            "class InstanceSegmentationBackend(Protocol):",
            ") -> Sequence[InstanceMask]:",
            "module.path:function_name",
            "tuple[DepthBackend, InstanceSegmentationBackend]",
            "[right, down, forward]",
            "camera_to_ego",
            "row-major",
            "bumper_xy",
            "missing_calibration",
            "approximate_geometry",
            '"action": "start"',
            '"action": "stop"',
            '"action": "info"',
            '"action": "config"',
            "{input_topic}/obstacle_distance",
            "data/json",
            "scene_mode",
            "metadata",
            "fixed_scene",
            "deadline",
            "publish",
            "evaluate_obstacle_distance.py",
            "image_path,scene,gt_distance_m",
            "RMSE",
            "O(n log n)",
            "/models/obstacle-distance/lifelong-nk",
            "/models/obstacle-distance/yolo26n-seg",
            "24.9M",
            "30 MB",
            "INT8",
            ".pth.tar",
            ".safetensors",
            "Git",
            "test_ocr_packaging.py",
            "test_ocr_model_downloader.py",
            "plugins.ocr_preprocess",
            "Python 3.14",
            "3 failures",
            "1 error",
            "codex-primary-runtime",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, document)
        self.assertNotIn("/Users/4paradigm", document)
        self.assertIn("${PYTHON}", document)
        self.assertIn("队列最多保留一帧", document)
        self.assertIn("另有一帧", document)

    def test_branch_blob_collector_scopes_and_reads_every_git_state(self):
        collector = globals().get("_branch_blob_candidates")
        self.assertIsNotNone(
            collector,
            "branch blob collector must inspect Git objects and worktree deltas",
        )
        if collector is None:
            return

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args: str) -> bytes:
                return subprocess.run(
                    ["git", *args],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                ).stdout

            git("init", "-q")
            git("config", "user.name", "Packaging Test")
            git("config", "user.email", "packaging@example.test")
            (repo / "legacy.onnx").write_bytes(b"legacy model")
            git("add", "legacy.onnx")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD").strip().decode("ascii")

            (repo / "branch.txt").write_bytes(b"head version")
            (repo / "sparse.txt").write_bytes(b"sparse blob")
            os.symlink("missing-target", repo / "branch-link")
            git("add", "branch.txt", "sparse.txt", "branch-link")
            git("commit", "-qm", "branch")
            git("update-index", "--skip-worktree", "sparse.txt")
            (repo / "sparse.txt").unlink()

            (repo / "branch.txt").write_bytes(b"worktree version")
            (repo / "staged.plan").write_bytes(b"staged engine")
            git("add", "staged.plan")
            object_id = subprocess.run(
                [b"git", b"hash-object", b"-w", b"--stdin"],
                cwd=os.fsencode(repo),
                input=b"non-utf8 index blob",
                check=True,
                capture_output=True,
            ).stdout.strip()
            subprocess.run(
                [
                    b"git",
                    b"update-index",
                    b"--add",
                    b"--cacheinfo",
                    b"100644," + object_id + b",index-\xff.pt",
                ],
                cwd=os.fsencode(repo),
                check=True,
                capture_output=True,
            )
            (repo / "untracked.pt").write_bytes(b"untracked weight")

            candidates = collector(repo, base)
            by_source_and_path = {
                (source, path): blob
                for source, path, blob in candidates
            }
            all_paths = {path for _, path, _ in candidates}

            self.assertNotIn(b"legacy.onnx", all_paths)
            self.assertEqual(
                by_source_and_path[("head", b"branch-link")],
                b"missing-target",
            )
            self.assertEqual(
                by_source_and_path[("head", b"sparse.txt")],
                b"sparse blob",
            )
            self.assertEqual(
                by_source_and_path[("index", b"staged.plan")],
                b"staged engine",
            )
            self.assertEqual(
                by_source_and_path[("worktree", b"branch.txt")],
                b"worktree version",
            )
            self.assertEqual(
                by_source_and_path[("index", b"index-\xff.pt")],
                b"non-utf8 index blob",
            )
            self.assertEqual(
                by_source_and_path[("untracked", b"untracked.pt")],
                b"untracked weight",
            )

    def test_branch_blob_policy_rejects_only_changed_blob_violations(self):
        violations = globals().get("_branch_blob_violations")
        self.assertIsNotNone(
            violations,
            "branch blob policy must validate size, suffix and LFS content",
        )
        if violations is None:
            return

        candidates = (
            ("head", b"docs/ok.md", b"ok"),
            ("index", b"models/NK.pth.tar", b"weight"),
            ("worktree", b"large.bin", b"x" * ONE_MIB),
            (
                "untracked",
                b"pointer.bin",
                b"version https://git-lfs.github.com/spec/v1\n",
            ),
        )
        found = violations(candidates)
        self.assertEqual(
            {kind for kind, _, _ in found},
            {"forbidden_model_artifact", "file_too_large", "git_lfs_pointer"},
        )

    def test_dockerfile_parser_rejects_broad_and_model_copy_sources(self):
        detector = globals().get("_dockerfile_model_sources")
        path_check = globals().get("_is_dockerfile_path")
        self.assertIsNotNone(detector, "Docker COPY/ADD parser is required")
        self.assertIsNotNone(path_check, "Dockerfile path matcher is required")
        if detector is None or path_check is None:
            return

        self.assertTrue(path_check(b"nested/DockerFile"))
        self.assertTrue(path_check(b"perception/dockerfile.JETSON"))
        self.assertFalse(path_check(b"nested/not-a-dockerfile"))

        dangerous = (
            "COPY perception /work",
            "ADD . /context",
            "CoPy --chown=1000:1000 \\\n  perception/models \\\n  /models",
            'COPY ["perception/models/obstacle-distance", "/models"]',
            'ADD --chown=robot:robot ["models/obstacle-distance", "/models"]',
            "COPY * /work",
            "COPY perception/model* /work",
        )
        for source in dangerous:
            with self.subTest(source=source):
                self.assertTrue(detector(source))

        safe = (
            "COPY perception/plugins/ /work/plugins/",
            'COPY ["perception/config.yaml", "/work/config.yaml"]',
            "COPY --from=builder /usr/local/lib/ /usr/local/lib/",
            "ADD https://example.test/metadata.json /tmp/metadata.json",
        )
        for source in safe:
            with self.subTest(source=source):
                self.assertFalse(detector(source))

    def test_branch_delta_blobs_are_small_plain_source_files(self):
        violations = _branch_blob_violations(
            _branch_blob_candidates(REPO_ROOT, BASE_REVISION)
        )
        rendered = [
            f"{kind}: {source}: {os.fsdecode(path)}"
            for kind, source, path in violations
        ]
        self.assertEqual([], rendered)

    def test_dockerfiles_do_not_copy_sources_containing_obstacle_models(self):
        dockerfiles = _dockerfile_candidates(REPO_ROOT)
        self.assertTrue(dockerfiles)
        for path, blob in dockerfiles:
            source = blob.decode("utf-8", errors="replace")
            with self.subTest(path=os.fsdecode(path)):
                self.assertEqual((), _dockerfile_model_sources(source))


if __name__ == "__main__":
    unittest.main()
