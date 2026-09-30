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


def test_release_notebook_defaults_to_immutable_v1_tag_and_allows_sha_override(code_cells):
    cell = _cell_containing(code_cells, "MOSS_TTS_SOURCE_REF")
    assert 'DEFAULT_SOURCE_REF = "v1.0.0"' in cell
    assert 'os.environ.get("MOSS_TTS_SOURCE_REF", DEFAULT_SOURCE_REF)' in cell
    assert 'COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")' in cell
    assert r'RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+$")' in cell
    assert "COMMIT_SHA.fullmatch(SOURCE_REF)" in cell
    assert "RELEASE_TAG.fullmatch(SOURCE_REF)" in cell
    assert "mutable refs such as 'main' are not accepted" in cell
    assert "RESOLVED_HEAD != SOURCE_REF" in cell
    assert "FETCH_HEAD^{commit}" in cell


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


def test_preflight_failure_surfaces_captured_diagnostics_before_raise(code_cells):
    cell = _cell_containing(code_cells, "scripts/preflight.py")
    assert "check=True" not in cell
    assert "print(preflight_run.stdout" in cell
    assert "print(preflight_run.stderr" in cell
    assert "preflight_run.check_returncode()" in cell
    assert cell.index("print(preflight_run.stdout") < cell.index("preflight_run.check_returncode()")
    assert cell.index("print(preflight_run.stderr") < cell.index("preflight_run.check_returncode()")


def test_inventory_failure_surfaces_diagnostics_and_attachment_hint_before_raise(code_cells):
    cell = _cell_containing(code_cells, "scripts/inventory_model.py")
    assert "check=True" not in cell
    assert "print(inventory_run.stdout" in cell
    assert "print(inventory_run.stderr" in cell
    assert "MODEL_ATTACHMENT_HINT" in cell
    assert "/kaggle/input/models" in cell
    assert "inventory_run.check_returncode()" in cell
    assert cell.index("print(inventory_run.stdout") < cell.index("inventory_run.check_returncode()")
    assert cell.index("print(inventory_run.stderr") < cell.index("inventory_run.check_returncode()")
    assert cell.index("MODEL_ATTACHMENT_HINT") < cell.index("inventory_run.check_returncode()")


def test_repo_scripts_receive_repo_root_on_pythonpath(code_cells):
    joined = "\n".join(code_cells)
    assert "SUBPROCESS_ENV = os.environ.copy()" in joined
    assert 'SUBPROCESS_ENV["PYTHONPATH"]' in joined
    assert "str(ROOT)" in joined
    for script in ("scripts/preflight.py", "scripts/inventory_model.py", "scripts/generate.py"):
        cell = _cell_containing(code_cells, script)
        assert "env=SUBPROCESS_ENV" in cell, f"{script} must receive repo-root PYTHONPATH"


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


def test_notebook_keeps_project_independence_and_post_release_public_flow(markdown):
    assert "not an official OpenMOSS release" in markdown
    assert "OFFICIAL_KAGGLE_ACCOUNT_ACCEPTANCE=PENDING" not in markdown
    assert "post-release" in markdown.lower() or "sau khi v1.0.0" in markdown.lower()


def test_marker_names_are_conservative(all_text):
    assert "MODEL_IDENTITY_AND_SAFETENSORS=PASS" in all_text
    assert "MODEL_ORIGINAL_SAFETENSORS=PASS" not in all_text

SHOWCASE_CASES = {
    "vi_narrative",
    "vi_longform",
    "en_narrative",
    "en_technical",
    "mix_vi_en",
    "mix_en_vi",
    "ipa",
}


def test_every_code_cell_has_explanatory_markdown_immediately_before(notebook):
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        assert index > 0, f"code cell {cell.get('id')} cannot be the first cell"
        previous = cells[index - 1]
        assert previous["cell_type"] == "markdown", (
            f"code cell {cell.get('id')} must have an explanatory markdown cell immediately before it"
        )
        explanation = "".join(previous["source"]).strip()
        assert len(explanation) >= 40, f"markdown before {cell.get('id')} is too thin to guide a reviewer"


def test_showcase_has_two_vietnamese_two_english_two_codeswitch_and_ipa(all_text):
    for case_id in SHOWCASE_CASES:
        assert case_id in all_text, case_id
    assert all_text.count('language="Vietnamese"') >= 3
    assert all_text.count('language="English"') >= 3
    assert "Vietnamese-only" in all_text
    assert "English-only" in all_text
    assert "code-switch" in all_text


def test_each_reviewer_showcase_call_has_its_own_guidance_markdown(notebook):
    cells = notebook["cells"]
    showcase_calls = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "run_showcase_case(" in source and "def run_showcase_case" not in source:
            showcase_calls.append((index, cell))
    assert len(showcase_calls) == 7
    for index, cell in showcase_calls:
        guidance = "".join(cells[index - 1]["source"])
        assert "What to listen for / Điểm cần nghe" in guidance, cell.get("id")
        assert "Sample text / Nội dung mẫu" in guidance, cell.get("id")


def test_showcase_helper_prints_reviewer_facing_runtime_metrics(code_cells):
    helper = _cell_containing(code_cells, "def run_showcase_case")
    for metric in (
        "audio_duration_seconds",
        "generation_rtf",
        "decode_rtf",
        "gpu0_peak_alloc_gib",
        "gpu1_peak_alloc_gib",
    ):
        assert metric in helper, metric
    assert "Audio(" in helper


def test_final_scorecard_requires_every_showcase_case(code_cells):
    cell = _cell_containing(code_cells, "scorecard = {")
    for case_id in SHOWCASE_CASES:
        assert case_id in cell, case_id

def test_every_executable_cell_guidance_separates_english_and_vietnamese(notebook):
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        guidance = "".join(cells[index - 1]["source"])
        assert "### English" in guidance, f"missing English section before {cell.get('id')}"
        assert "### Tiếng Việt" in guidance, f"missing Vietnamese section before {cell.get('id')}"
        assert guidance.index("### English") < guidance.index("### Tiếng Việt"), (
            f"English section must precede Vietnamese section before {cell.get('id')}"
        )


def test_notebook_overview_has_visually_separate_language_blocks(notebook):
    overview = "".join(notebook["cells"][0]["source"])
    assert "## English" in overview
    assert "## Tiếng Việt" in overview
    assert overview.index("## English") < overview.index("## Tiếng Việt")


def test_each_reviewer_sample_has_structured_bilingual_review_guidance(notebook):
    cells = notebook["cells"]
    required_sections = (
        "### English",
        "### Tiếng Việt",
        "### Sample text / Nội dung mẫu",
        "### What to listen for / Điểm cần nghe",
        "### Expected evidence / Kết quả cần thấy",
    )
    sample_guidance = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "run_showcase_case(" in source and "def run_showcase_case" not in source:
            sample_guidance.append("".join(cells[index - 1]["source"]))
    assert len(sample_guidance) == 7
    for guidance in sample_guidance:
        for section in required_sections:
            assert section in guidance, section


def test_every_executable_step_states_expected_evidence_bilingually(notebook):
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        guidance = "".join(cells[index - 1]["source"])
        assert "Expected evidence / Kết quả cần thấy" in guidance, cell.get("id")

def test_checkpoint_sha256_recompute_is_optional_and_disabled_by_default(code_cells):
    joined = "\n".join(code_cells)
    assert "VERIFY_CHECKPOINT_SHA256 = False" in joined
    cell = _cell_containing(code_cells, "scripts/inventory_model.py")
    assert "VERIFY_CHECKPOINT_SHA256" in cell
    assert '"--skip-sha256"' in cell
    assert "SHA256_RECOMPUTE=SKIPPED" in cell
    assert "SHA256_REFERENCE=PREQUALIFIED" in cell
    assert "SHA256_RECOMPUTE=PASS" in cell
    assert "SHA256_REFERENCE=PASS" in cell

def test_technical_samples_use_ipa_control_for_bf16(notebook):
    cells = notebook["cells"]
    expected_ipa = "/biː ɛf sɪkˈstiːn/"
    for case_id in ("vi_longform", "en_technical"):
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if f'"{case_id}"' not in source:
                continue
            assert expected_ipa in source, case_id
            assert "BF16" not in source, f"{case_id} must not synthesize raw BF16"
            guidance = "".join(cells[index - 1]["source"])
            assert "BF16" in guidance, case_id
            assert expected_ipa in guidance, case_id
            assert "pronunciation control" in guidance.lower(), case_id
            break
        else:
            raise AssertionError(f"missing showcase case: {case_id}")


def test_showcase_generation_streams_live_progress_instead_of_capturing_silently(code_cells):
    cell = _cell_containing(code_cells, "def run_showcase_case")
    assert "capture_output=True" not in cell
    assert "subprocess.Popen" in cell
    assert "stdout=subprocess.PIPE" in cell
    assert "stderr=subprocess.STDOUT" in cell
    assert "for line in completed.stdout" in cell or "for line in process.stdout" in cell
    assert "flush=True" in cell


def test_generate_script_emits_stage_progress_markers():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "generate.py").read_text()
    for marker in (
        "STAGE=PROCESSOR_SETUP START",
        "STAGE=PROCESSOR_SETUP DONE",
        "STAGE=MODEL_LOAD START",
        "STAGE=MODEL_LOAD DONE",
        "STAGE=GENERATE START",
        "STAGE=GENERATE DONE",
        "STAGE=DECODE START",
        "STAGE=DECODE DONE",
    ):
        assert marker in text
    assert "flush=True" in text
