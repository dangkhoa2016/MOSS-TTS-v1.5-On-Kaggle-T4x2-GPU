from pathlib import Path
import json

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


def test_readme_is_engineering_first_project_landing_page():
    english = (ROOT / "README.md").read_text()
    vietnamese = (ROOT / "README.vi.md").read_text()

    for text in (english, vietnamese):
        for needle in [
            "Repository Audit", "License-MIT", "OpenMOSS", "Kaggle", "T4", "BF16",
            "Language / Ngôn ngữ", "Release-v1.0.0-blue",
            "HUMAN_LISTENING_REVIEW=PASS",
            "not an official OpenMOSS release" if text is english else "không phải release chính thức của OpenMOSS",
        ]:
            assert needle in text, needle
        assert "POST_RELEASE_PUBLIC_KAGGLE_RUN=PLANNED" not in text
        assert "only remaining item" not in text.lower()
        assert "Pre--publication" not in text

    expected_en = [
        "## Why this project exists",
        "## What this repository provides",
        "## Architecture",
        "## Quick start on Kaggle",
        "## Engineering design",
        "## Qualified results",
        "## Verification and reproducibility",
        "## Scope and non-goals",
        "## Documentation",
        "## Repository layout",
        "## Release",
        "## Licensing and attribution",
    ]
    expected_vi = [
        "## Vì sao dự án này tồn tại",
        "## Repository này cung cấp gì",
        "## Kiến trúc",
        "## Chạy nhanh trên Kaggle",
        "## Thiết kế engineering",
        "## Kết quả đã qualification",
        "## Verification và khả năng tái lập",
        "## Phạm vi và những gì dự án không tuyên bố",
        "## Tài liệu",
        "## Cấu trúc repository",
        "## Release",
        "## Giấy phép và attribution",
    ]
    assert [line for line in english.splitlines() if line.startswith("## ")] == expected_en
    assert [line for line in vietnamese.splitlines() if line.startswith("## ")] == expected_vi

    for forbidden in (
        "## Release status",
        "## Evidence and release authority",
        "## Pronunciation findings",
    ):
        assert forbidden not in english
    for forbidden in (
        "## Trạng thái release",
        "## Evidence và release authority",
        "## Kết quả review phát âm",
    ):
        assert forbidden not in vietnamese

    assert english.index("## Quick start on Kaggle") < english.index("## Release")
    assert vietnamese.index("## Chạy nhanh trên Kaggle") < vietnamese.index("## Release")
    assert "flowchart LR" in english
    assert "flowchart LR" in vietnamese
    assert "notebooks/kaggle-production-demo.ipynb" in english
    assert "notebooks/kaggle-production-demo.ipynb" in vietnamese


def test_publication_state_is_consistent_across_release_documents():
    notes = (ROOT / "RELEASE_NOTES_v1.0.0.md").read_text()
    notes_vi = (ROOT / "RELEASE_NOTES_v1.0.0.vi.md").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    changelog_vi = (ROOT / "CHANGELOG.vi.md").read_text()
    required_urls = (
        "https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU",
        "https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/notebooks/kaggle-production-demo.ipynb",
        "https://www.kaggle.com/models/dangkhoa2016/openmoss-team-moss-tts-v1-5",
        "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5",
        "https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/releases/tag/v1.0.0",
    )
    for text in (notes, notes_vi):
        lowered = text.lower()
        assert "OFFICIAL_KAGGLE_ACCOUNT_ACCEPTANCE=PENDING" not in text
        assert "HUMAN_LISTENING_REVIEW=PASS" in text
        assert "POST_RELEASE_PUBLIC_KAGGLE_RUN=PLANNED" not in text
        assert "post-release follow-up" not in lowered
        assert "bước sau release" not in lowered
        assert "release candidate" not in lowered
        for url in required_urls:
            assert url in text, url

    expected_en = [
        "## Release summary",
        "## Engineering qualification",
        "## Qualification results",
        "## Release evidence",
        "## Run on Kaggle",
        "## Links",
    ]
    expected_vi = [
        "## Tóm tắt release",
        "## Engineering qualification",
        "## Kết quả qualification",
        "## Release evidence",
        "## Chạy trên Kaggle",
        "## Liên kết",
    ]
    assert "`MOSS` and `BF16` pronunciation were reviewed explicitly" not in notes
    assert "Phát âm `MOSS` và `BF16` được review riêng" not in notes_vi
    for text, topology_word in ((notes, "two"), (notes_vi, "hai")):
        for needle in (topology_word, "BF16", "HUMAN_LISTENING_REVIEW=PASS"):
            assert needle.lower() in text.lower(), needle
    assert "has **not been tagged**" not in changelog
    assert "still pending confirmation" not in changelog
    assert "**chưa được tag**" not in changelog_vi
    assert "vẫn chờ xác nhận" not in changelog_vi
    assert [line for line in notes.splitlines() if line.startswith("## ")] == expected_en
    assert [line for line in notes_vi.splitlines() if line.startswith("## ")] == expected_vi
    assert len(notes.splitlines()) <= 110
    assert len(notes_vi.splitlines()) <= 110
    for text in (notes, notes_vi):
        assert "## Release authority" not in text
        assert "Repository Audit" not in text
    assert "1.0.0" in changelog
    assert "1.0.0" in changelog_vi


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


def test_sha256_reference_manifest_is_confirmed_authority():
    text = (ROOT / "references/checkpoint-sha256.json").read_text()
    assert '"confirmation": "confirmed"' in text
    assert '"expected_sha256": {}' not in text

    evidence_index = (ROOT / "docs/evidence-index.md").read_text()
    evidence_index_vi = (ROOT / "docs/evidence-index.vi.md").read_text()
    assert "SHA256_REFERENCE=PASS" in evidence_index
    assert "SHA256_REFERENCE=PASS" in evidence_index_vi

    for name in ("README.md", "README.vi.md"):
        readme = (ROOT / name).read_text()
        assert "SHA256_REFERENCE=PASS" not in readme, name
        assert "UNVERIFIED" not in readme, name


def test_candidate_listening_and_executed_evidence_support_release_with_public_run_post_release():
    english_record = ROOT / "evidence" / "v1.0.0-human-listening-review.md"
    vietnamese_record = ROOT / "evidence" / "v1.0.0-human-listening-review.vi.md"
    for record in (english_record, vietnamese_record):
        assert record.is_file(), record.name
        text = record.read_text()
        assert "f43518ea4b866b42a75c7b600ab146d1b9005bd8" in text
        assert "HUMAN_LISTENING_REVIEW=PASS" in text
        for case_id in ("vi_narrative", "vi_longform", "en_narrative", "en_technical", "mix_vi_en", "mix_en_vi", "ipa"):
            assert case_id in text, f"{record.name}: {case_id}"

    manifest = ROOT / "evidence" / "moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text())
    assert data["asset_filename"] == "moss-tts-v1-5-kaggle-t4x2-candidate-evidence.ipynb"
    assert data["source_ref"] == "f43518ea4b866b42a75c7b600ab146d1b9005bd8"
    assert data["sha256"] == "ba9fff08dcf46207d69bc3e6d6a366622d800d6afeaf2b4c418ca8b599f7a7bf"
    assert data["size_bytes"] == 5793026
    assert data["release_asset"] is True
    assert data["tracked_in_git"] is False

    for name in ("README.md", "README.vi.md", "RELEASE_NOTES_v1.0.0.md", "RELEASE_NOTES_v1.0.0.vi.md"):
        text = (ROOT / name).read_text()
        assert "HUMAN_LISTENING_REVIEW=PASS" in text, name
        assert "OFFICIAL_KAGGLE_ACCOUNT_ACCEPTANCE=PENDING" not in text, name

    for name in ("RELEASE_NOTES_v1.0.0.md", "RELEASE_NOTES_v1.0.0.vi.md"):
        text = (ROOT / name).read_text()
        assert "POST_RELEASE_PUBLIC_KAGGLE_RUN=PLANNED" not in text, name

    matrix = (ROOT / "docs" / "qualification-matrix.md").read_text()
    evidence_index = (ROOT / "docs" / "evidence-index.md").read_text()
    assert "HUMAN_LISTENING_REVIEW=PASS" in matrix
    assert "OFFICIAL_KAGGLE_ACCOUNT_ACCEPTANCE=PENDING" not in matrix
    assert "OFFICIAL_KAGGLE_ACCOUNT_ACCEPTANCE=PENDING" not in evidence_index
    assert "moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json" in evidence_index


def test_public_notebook_guidance_defaults_to_v1_0_0_and_keeps_sha_override():
    for name in ("README.md", "README.vi.md"):
        text = (ROOT / "notebooks" / name).read_text()
        assert "MOSS_TTS_SOURCE_REF" in text, name
        assert "v1.0.0" in text, name
        assert "main" in text.lower(), name
    for name in ("reproducibility.md", "reproducibility.vi.md"):
        text = (ROOT / "docs" / name).read_text()
        assert "MOSS_TTS_SOURCE_REF" in text, name
        assert "v1.0.0" in text, name
        assert "main" in text.lower(), name
