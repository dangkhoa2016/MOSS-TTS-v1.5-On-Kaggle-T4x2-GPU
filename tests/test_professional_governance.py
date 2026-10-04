from pathlib import Path
import subprocess

ROOT = Path('.')


def test_readme_has_professional_badges_and_language_switch():
    text=(ROOT/'README.md').read_text()
    for needle in ['Repository Audit','License-MIT','OpenMOSS','Kaggle','T4','BF16','Release','Language / Ngôn ngữ','README.vi.md']:
        assert needle in text, needle


def test_github_governance_set_matches_reference_quality():
    required = [
        '.github/CODEOWNERS',
        '.github/CODE_OF_CONDUCT.md','.github/CODE_OF_CONDUCT.vi.md',
        '.github/CONTRIBUTING.md','.github/CONTRIBUTING.vi.md',
        '.github/SECURITY.md','.github/SECURITY.vi.md',
        '.github/SUPPORT.md','.github/SUPPORT.vi.md',
        '.github/PULL_REQUEST_TEMPLATE.md','.github/PULL_REQUEST_TEMPLATE.vi.md',
        '.github/dependabot.yml','.github/workflows/repository-audit.yml',
        '.github/ISSUE_TEMPLATE/config.yml',
        '.github/ISSUE_TEMPLATE/bug_report.md','.github/ISSUE_TEMPLATE/bug_report.vi.md',
        '.github/ISSUE_TEMPLATE/documentation.md','.github/ISSUE_TEMPLATE/documentation.vi.md',
        '.github/ISSUE_TEMPLATE/feature_request.md','.github/ISSUE_TEMPLATE/feature_request.vi.md',
        '.github/ISSUE_TEMPLATE/question.md','.github/ISSUE_TEMPLATE/question.vi.md',
    ]
    missing=[p for p in required if not (ROOT/p).is_file()]
    assert not missing, missing


def test_repo_mit_license_and_third_party_license_boundary():
    license_text=(ROOT/'LICENSE').read_text()
    assert 'MIT License' in license_text
    assert 'Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>' in license_text
    third=(ROOT/'THIRD_PARTY_LICENSES/OPENMOSS-APACHE-2.0').read_text()
    assert 'Apache License' in third and 'Version 2.0' in third
    notes=(ROOT/'LICENSE-NOTES.md').read_text()
    for needle in ['OpenMOSS-Team/MOSS-TTS-v1.5','MOSS Audio Tokenizer','Apache License 2.0','not stored in this Git repository']:
        assert needle in notes


def test_public_docs_are_bilingual_pairs_with_switchers():
    english = [
        'README','architecture','benchmark-methodology','pronunciation-control',
        'qualification-matrix','reproducibility','engineering-overview',
        'evidence-index','development-history','troubleshooting',
    ]
    for stem in english:
        en=ROOT/'docs'/f'{stem}.md'; vi=ROOT/'docs'/f'{stem}.vi.md'
        assert en.is_file(), en
        assert vi.is_file(), vi
        assert 'Language / Ngôn ngữ' in en.read_text(), en
        assert 'Language / Ngôn ngữ' in vi.read_text(), vi


def test_version_and_release_metadata_exist():
    assert (ROOT/'VERSION').read_text().strip() == '1.0.0'
    assert (ROOT/'RELEASE_NOTES_v1.0.0.md').is_file()
