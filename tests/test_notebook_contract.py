"""Semantic contract tests for the production notebook.

These tests parse the notebook JSON and inspect code structure, so a notebook that
hard-codes qualification booleans or pins a mutable ref cannot pass.
"""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-production-demo.ipynb"
GATE_KEYS = (
    "public_source_pin",
    "gpu_t4x2",
    "model_identity_and_safetensors",
    "declared_dtype_bf16",
    "checkpoint_dtype_verification",
    "gguf",
    "quantization",
    "model_patch",
    "architecture_modified",
)


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


@pytest.fixture(scope="module")
def code_cells(notebook) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def markdown(notebook) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")


@pytest.fixture(scope="module")
def all_text(notebook) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def _cell_containing(code_cells: list[str], needle: str) -> str:
    matches = [cell for cell in code_cells if needle in cell]
    assert matches, f"no code cell contains {needle!r}"
    return matches[0]


def test_notebook_is_valid_nbformat(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert len(notebook["cells"]) >= 10
    assert all(cell.get("id") for cell in notebook["cells"])


def test_notebook_code_cells_are_syntactically_valid(code_cells):
    for cell in code_cells:
        ast.parse(cell)


def test_source_is_pinned_to_an_immutable_ref(code_cells):
    cell = _cell_containing(code_cells, "MOSS_TTS_SOURCE_REF")
    assert '--no-checkout' in cell
    assert '"fetch", "--depth", "1"' in cell
    assert '"checkout", "--detach", "FETCH_HEAD"' in cell
    assert "rev-parse" in cell
    assert "PUBLIC_SOURCE_HEAD=" in cell
    assert "PUBLIC_SOURCE_PIN=PASS" in cell


def test_mutable_default_ref_is_rejected(code_cells):
    cell = _cell_containing(code_cells, "MOSS_TTS_SOURCE_REF")
    assert 'os.environ.get("MOSS_TTS_SOURCE_REF", "main")' not in cell
    assert 'os.environ.get("MOSS_TTS_SOURCE_REF", "")' in cell
    assert 'COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")' in cell
    assert "RELEASE_TAG = re.compile(" in cell
    assert "raise SystemExit" in cell
    assert "RESOLVED_HEAD != SOURCE_REF" in cell


def test_dependency_baseline_is_bootstrapped_and_verified(code_cells):
    cell = _cell_containing(code_cells, "requirements")
    assert '"pip", "install", "-r"' in cell
    assert "requirements/kaggle.txt" in cell or '"kaggle.txt"' in cell
    assert "DEPENDENCY_BASELINE=PASS" in cell
    assert "specifier.contains" in cell
    assert "assert not violations" in cell


def test_notebook_runs_preflight_inventory_and_generation(code_cells):
    joined = "\n".join(code_cells)
    assert "scripts/preflight.py" in joined
    assert "scripts/inventory_model.py" in joined
    assert "scripts/generate.py" in joined
    assert '"--report"' in joined
    assert "check=True" in joined


def test_reports_are_parsed_from_files_not_declared(code_cells):
    joined = "\n".join(code_cells)
    assert "json.loads((REPORTS" in joined
    assert "preflight.json" in joined
    assert "inventory.json" in joined
    assert "report_path.read_text()" in joined


def test_scorecard_is_derived_not_literal(code_cells):
    cell = _cell_containing(code_cells, "scorecard = {")
    tree = ast.parse(cell)
    scorecard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "scorecard" for t in node.targets)
    )
    entries = {key.value: value for key, value in zip(scorecard.value.keys, scorecard.value.values)}
    assert set(GATE_KEYS) <= set(entries)
    for key in GATE_KEYS:
        value = entries[key]
        assert not isinstance(value, ast.Constant), f"{key} is a hard-coded literal"
        assert not (isinstance(value, ast.Dict) and not value.keys), f"{key} is an empty literal"


def test_no_hardcoded_qualification_literals(all_text):
    for literal in (
        '"original_safetensors": True',
        '"bf16": True',
        '"gpu_t4x2": True',
        '"gguf": False',
        '"model_patch": False',
        '"quantization": "none"',
    ):
        assert literal not in all_text, literal


def test_assertions_precede_the_final_pass_marker(code_cells):
    cell = _cell_containing(code_cells, "KAGGLE_PRODUCTION_DEMO=PASS")
    lines = cell.splitlines()
    pass_index = next(i for i, line in enumerate(lines) if "KAGGLE_PRODUCTION_DEMO=PASS" in line)
    assert_lines = [i for i, line in enumerate(lines) if line.strip().startswith("assert ")]
    assert len(assert_lines) >= 8, "scorecard must assert every technical gate"
    assert max(assert_lines) < pass_index, "final PASS marker must be printed after all assertions"


def test_every_marker_print_is_preceded_by_its_assertion(code_cells):
    expectations = {
        "QUANTIZATION=": "quantization",
        "GGUF=NO": "gguf",
        "MODEL_PATCH=NONE": "model_patch",
        "MODEL_IDENTITY_AND_SAFETENSORS=PASS": "declared_dtype",
        "GPU_T4X2=PASS": "gpu_count",
        "PUBLIC_SOURCE_PIN=PASS": "RESOLVED_HEAD",
        "DEPENDENCY_BASELINE=PASS": "violations",
    }
    for marker, gate in expectations.items():
        cell = _cell_containing(code_cells, marker)
        lines = cell.splitlines()
        marker_index = next(i for i, line in enumerate(lines) if marker in line)
        gate_asserts = [
            i
            for i, line in enumerate(lines)
            if line.strip().startswith("assert ") and gate in line
        ]
        assert gate_asserts, f"no assertion guards {marker} (expected a gate named {gate})"
        assert max(gate_asserts) < marker_index, f"{marker} printed before its assertion"


def test_human_listening_review_is_a_separate_gate(all_text):
    assert "HUMAN_LISTENING_REVIEW=PENDING" in all_text
    assert "human_listening_review" in all_text


def test_reviewer_facing_cases_are_rendered_inline(all_text):
    for needle in (
        "GPU T4 x2",
        "IPython.display",
        "Audio(",
        "IPA",
        "Vietnamese",
        "code-switch",
    ):
        assert needle in all_text, needle


def test_notebook_does_not_claim_a_published_release(markdown):
    assert "not an official OpenMOSS release" in markdown
    assert "Publication is **not** complete" in markdown


def test_marker_names_are_conservative(all_text):
    assert "MODEL_IDENTITY_AND_SAFETENSORS=PASS" in all_text
    assert "MODEL_ORIGINAL_SAFETENSORS=PASS" not in all_text