from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOC_STEMS = [
    "README",
    "architecture",
    "benchmark-methodology",
    "pronunciation-control",
    "qualification-matrix",
    "reproducibility",
    "engineering-overview",
    "evidence-index",
    "development-history",
    "troubleshooting",
]


def test_readme_has_badges_language_switch_and_pre_release_state():
    text = (ROOT / "README.md").read_text()
    for needle in [
        "Repository Audit",
        "License-MIT",
        "OpenMOSS",
        "Kaggle",
        "T4",
        "BF16",
        "Language / Ngôn ngữ",
        "README.vi.md",
    ]:
        assert needle in text, needle
    assert "Release-v1.0.0-blue" not in text, "pre-release repository must not imply a published release"
    assert "release candidate" in text.lower()
    assert "not an official OpenMOSS release" in text
    assert "Publication is gated" in text or "publication is gated" in text


def test_vietnamese_readme_mirrors_the_english_claims():
    english = (ROOT / "README.md").read_text()
    vietnamese = (ROOT / "README.vi.md").read_text()
    assert "release candidate" in vietnamese.lower()
    assert "Release-v1.0.0-blue" not in vietnamese
    assert "OpenMOSS" in vietnamese
    assert len(vietnamese.splitlines()) >= len(english.splitlines()) * 0.9


def test_publication_state_is_consistent_across_release_documents():
    notes = (ROOT / "RELEASE_NOTES_v1.0.0.md").read_text()
    notes_vi = (ROOT / "RELEASE_NOTES_v1.0.0.vi.md").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    changelog_vi = (ROOT / "CHANGELOG.vi.md").read_text()
    assert "gated" in notes.lower()
    assert "release candidate" in notes.lower()
    assert "bị chặn" in notes_vi.lower()
    assert "release candidate" in notes_vi.lower()
    assert "not been tagged" in changelog.lower()
    assert "chưa được tag" in changelog_vi.lower()


def test_github_governance_set_matches_reference_quality():
    required = [
        ".github/CODEOWNERS",
        ".github/CODE_OF_CONDUCT.md",
        ".github/CODE_OF_CONDUCT.vi.md",
        ".github/CONTRIBUTING.md",
        ".github/CONTRIBUTING.vi.md",
        ".github/SECURITY.md",
        ".github/SECURITY.vi.md",
        ".github/SUPPORT.md",
        ".github/SUPPORT.vi.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/PULL_REQUEST_TEMPLATE.vi.md",
        ".github/dependabot.yml",
        ".github/workflows/repository-audit.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/bug_report.vi.md",
        ".github/ISSUE_TEMPLATE/documentation.md",
        ".github/ISSUE_TEMPLATE/documentation.vi.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/ISSUE_TEMPLATE/feature_request.vi.md",
        ".github/ISSUE_TEMPLATE/question.md",
        ".github/ISSUE_TEMPLATE/question.vi.md",
    ]
    missing = [p for p in required if not (ROOT / p).is_file()]
    assert not missing, missing


def test_repo_mit_license_and_third_party_license_boundary():
    license_text = (ROOT / "LICENSE").read_text()
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>" in license_text
    third = (ROOT / "THIRD_PARTY_LICENSES/OPENMOSS-APACHE-2.0").read_text()
    assert "Apache License" in third and "Version 2.0" in third
    notes = (ROOT / "LICENSE-NOTES.md").read_text()
    for needle in [
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        "MOSS Audio Tokenizer",
        "Apache License 2.0",
        "not stored in this Git repository",
    ]:
        assert needle in notes


def test_public_docs_are_bilingual_pairs_with_switchers():
    for stem in DOC_STEMS:
        english = ROOT / "docs" / f"{stem}.md"
        vietnamese = ROOT / "docs" / f"{stem}.vi.md"
        assert english.is_file(), english
        assert vietnamese.is_file(), vietnamese
        assert "Language / Ngôn ngữ" in english.read_text(), english
        assert "Language / Ngôn ngữ" in vietnamese.read_text(), vietnamese


def test_docs_index_links_every_public_document():
    for name, suffix in (("README.md", ".md"), ("README.vi.md", ".vi.md")):
        text = (ROOT / "docs" / name).read_text()
        for stem in DOC_STEMS:
            if stem == "README":
                continue
            assert f"]({stem}{suffix})" in text, f"docs/{name} does not link {stem}{suffix}"


def test_vietnamese_project_contract_has_full_parity():
    english = (ROOT / "PROJECT-CONTRACT.md").read_text()
    vietnamese = (ROOT / "PROJECT-CONTRACT.vi.md").read_text()
    for heading in ("## Baseline", "## Target topology", "## Hard rules"):
        assert heading in english
    for heading in ("## Baseline", "## Target topology", "## Hard rules"):
        assert heading in vietnamese, heading
    assert "934d6826b084c46a0d033402174d5f8ac4ed2519" in vietnamese
    assert "dangkhoa2016/openmoss-team-moss-tts-v1-5" in vietnamese
    assert "24 kHz" in vietnamese or "24 kHz" in english
    assert "GGUF" in vietnamese
    assert "INT8/INT4/AWQ/GPTQ" in vietnamese


def test_development_history_links_traceable_evidence():
    for name, matrix in (
        ("development-history.md", "qualification-matrix.md"),
        ("development-history.vi.md", "qualification-matrix.vi.md"),
    ):
        text = (ROOT / "docs" / name).read_text()
        assert "evidence/prepublication-canonical-smoke.json" in text, name
        assert "notebooks/kaggle-production-demo.ipynb" in text, name
        assert matrix in text, name
        assert "FP16" in text, name
        assert "934d6826b084c46a0d033402174d5f8ac4ed2519" in text, name


def test_benchmark_methodology_matches_emitted_metrics():
    for name in ("benchmark-methodology.md", "benchmark-methodology.vi.md"):
        text = (ROOT / "docs" / name).read_text()
        for metric in (
            "processor_setup_seconds",
            "model_load_seconds",
            "generate_seconds",
            "decode_seconds",
            "audio_duration_seconds",
            "generated_steps",
            "generation_rtf",
            "decode_rtf",
            "total_seconds",
        ):
            assert metric in text, f"{name} does not document {metric}"
        assert "cold" in text.lower(), name
        assert "steady-state" in text.lower(), name


def test_evidence_boundary_wording_is_conservative():
    for name in ("evidence/README.md", "evidence/README.vi.md"):
        text = (ROOT / name).read_text()
        assert "prepublication-canonical-smoke.json" in text
        assert "not the final" in text.lower() or "không phải" in text.lower()
    english = (ROOT / "evidence/README.md").read_text()
    assert "proves" not in english.lower()
    assert "demonstrates" in english.lower()


def test_reproducibility_requires_an_immutable_source_pin():
    for name in ("reproducibility.md", "reproducibility.vi.md"):
        text = (ROOT / "docs" / name).read_text()
        assert "MOSS_TTS_SOURCE_REF" in text, name
        assert "40" in text, name
        assert "main" in text.lower(), name
        assert "Restart Session" in text, name


def test_notebook_guidance_requires_the_source_pin():
    for name in ("README.md", "README.vi.md"):
        text = (ROOT / "notebooks" / name).read_text()
        assert "MOSS_TTS_SOURCE_REF" in text, name
        assert "Restart Session" in text, name


def test_public_markers_are_conservative_across_the_repository():
    readme = (ROOT / "README.md").read_text()
    vietnamese = (ROOT / "README.vi.md").read_text()
    matrix = (ROOT / "docs/qualification-matrix.md").read_text()
    evidence_index = (ROOT / "docs/evidence-index.md").read_text()
    for text in (readme, vietnamese, matrix, evidence_index):
        assert "MODEL_ORIGINAL_SAFETENSORS" not in text
    assert "MODEL_IDENTITY_AND_SAFETENSORS" in readme
    assert "MODEL_IDENTITY_AND_SAFETENSORS" in evidence_index


def test_sha256_reference_manifest_is_not_claimed_as_authority_yet():
    text = (ROOT / "references/checkpoint-sha256.json").read_text()
    assert '"confirmation": "pending"' in text
    assert '"expected_sha256": {}' in text
    readme = (ROOT / "README.md").read_text()
    assert "UNVERIFIED" in readme