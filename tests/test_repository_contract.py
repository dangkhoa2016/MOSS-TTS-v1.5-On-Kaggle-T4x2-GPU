import subprocess
from pathlib import Path

import pytest

from moss_t4x2.audit import (
    FORBIDDEN_TRACKED_SUFFIXES,
    forbidden_tracked_files,
    is_forbidden,
    tracked_files,
)

ROOT = Path(__file__).resolve().parents[1]


def test_required_release_files_exist():
    for name in [
        "README.md",
        "README.vi.md",
        "PROJECT-CONTRACT.md",
        "PROJECT-CONTRACT.vi.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "LICENSE-NOTES.md",
        "NOTICE.txt",
        "Makefile",
        "requirements/kaggle.txt",
        "references/upstream.lock",
        "references/checkpoint-sha256.json",
        "docs/architecture.md",
        "docs/reproducibility.md",
        "docs/qualification-matrix.md",
        "docs/README.md",
    ]:
        assert (ROOT / name).is_file(), name


def test_forbidden_suffix_set_is_canonical():
    assert FORBIDDEN_TRACKED_SUFFIXES == (".bin", ".gguf", ".onnx", ".pt", ".pth", ".safetensors")


@pytest.mark.parametrize(
    "path",
    ["weights/model.safetensors", "a/b.GGUF", "x.onnx", "y.pt", "z.pth", "w.BIN"],
)
def test_is_forbidden_detects_weight_artifacts(path):
    assert is_forbidden(path)


@pytest.mark.parametrize("path", ["README.md", "moss_t4x2/audit.py", "evidence/x.json", "results/.gitkeep"])
def test_is_forbidden_allows_source_files(path):
    assert not is_forbidden(path)


def test_forbidden_tracked_files_is_sorted_and_filtered():
    assert forbidden_tracked_files(["b.bin", "a.md", "A.SAFETENSORS"]) == ["A.SAFETENSORS", "b.bin"]


def test_git_index_does_not_track_model_weight_formats():
    assert forbidden_tracked_files(tracked_files(ROOT)) == []


def test_makefile_audit_delegates_to_the_canonical_audit_module():
    text = (ROOT / "Makefile").read_text()
    assert "python -m moss_t4x2.audit" in text
    for suffix in FORBIDDEN_TRACKED_SUFFIXES:
        assert suffix not in text, f"{suffix} duplicated in Makefile instead of reusing the audit module"


def test_workflow_audit_delegates_to_the_canonical_audit_module():
    text = (ROOT / ".github/workflows/repository-audit.yml").read_text()
    assert "python -m moss_t4x2.audit" in text
    for suffix in FORBIDDEN_TRACKED_SUFFIXES:
        assert suffix not in text, f"{suffix} duplicated in workflow instead of reusing the audit module"


def test_audit_module_reports_clean_repository():
    result = subprocess.run(
        ["python3", "-m", "moss_t4x2.audit"], cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    )
    assert result.returncode == 0, result.stderr
    assert "MODEL_WEIGHTS_TRACKED=NO" in result.stdout


def test_workflow_pip_cache_uses_explicit_dependency_path():
    text = (ROOT / ".github/workflows/repository-audit.yml").read_text()
    assert "cache-dependency-path: requirements/kaggle.txt" in text


def test_dependabot_pip_watches_the_requirements_directory():
    text = (ROOT / ".github/dependabot.yml").read_text()
    assert 'package-ecosystem: "pip"' in text
    assert 'directory: "/requirements"' in text


def test_gitignore_blocks_every_forbidden_suffix():
    ignored = (ROOT / ".gitignore").read_text().lower()
    for suffix in FORBIDDEN_TRACKED_SUFFIXES:
        assert f"*{suffix}" in ignored, suffix