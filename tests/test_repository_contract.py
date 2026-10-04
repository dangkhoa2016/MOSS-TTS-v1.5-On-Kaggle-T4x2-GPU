from pathlib import Path
import subprocess


def test_required_release_files_exist():
    for name in ['README.md','README.vi.md','PROJECT-CONTRACT.md','CHANGELOG.md','CITATION.cff','LICENSE','LICENSE-NOTES.md','NOTICE.txt','Makefile','requirements/kaggle.txt','references/upstream.lock','docs/architecture.md','docs/reproducibility.md','docs/qualification-matrix.md']:
        assert Path(name).is_file(), name


def test_git_index_does_not_track_model_weight_formats():
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    forbidden=('.safetensors','.gguf','.pth','.pt','.onnx')
    assert not [p for p in tracked if p.lower().endswith(forbidden)]
