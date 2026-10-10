from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = (
    ROOT
    / "causal_phase_tracking"
    / "causal_cecHT"
    / "vendor"
    / "eikeosmers_cecht"
)
PHASE_PATH = VENDOR_DIR / "phase.py"
LICENSE_PATH = VENDOR_DIR / "LICENSE"
PROVENANCE_PATH = VENDOR_DIR / "UPSTREAM_PROVENANCE.json"

REPOSITORY = "https://github.com/eikeosmers/cecHT"
COMMIT = "e079c3115f202a4272cf64143bf33ddddc247778"
COMMIT_TREE_SHA1 = "740ff402a2814995c48e000ab9a4023a9d0aa769"
EXPECTED_VENDOR_FILES = {"LICENSE", "UPSTREAM_PROVENANCE.json", "phase.py"}
EXPECTED_UPSTREAM_FILES = {
    "phase.py": {
        "sha256": "49b85cfd75f382aa9b12e00fca4cbfd903bf4daad0101ccba42b162f2af27bea",
        "git_blob_sha1": "977692f8124d11499d09c8bedb5a6e4ab004b780",
        "size_bytes": 11673,
    },
    "LICENSE": {
        "sha256": "fdd2688b462490149d03d26ce94343d7720dba2d28f270abe331fe09bc7ede8f",
        "git_blob_sha1": "7968deffe55510f82c6b68e4cd69ac250a1105c9",
        "size_bytes": 1520,
    },
}


def load_provenance() -> dict:
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def test_approved_vendor_snapshot_exists() -> None:
    assert PHASE_PATH.is_file()
    assert LICENSE_PATH.is_file()
    assert PROVENANCE_PATH.is_file()


def test_provenance_freezes_source_license_and_environment() -> None:
    provenance = load_provenance()

    assert provenance["schema_version"] == "1.0"
    assert provenance["repository"] == REPOSITORY
    assert provenance["commit"] == COMMIT
    assert provenance["commit_tree_sha1"] == COMMIT_TREE_SHA1
    assert provenance["retrieval_date"] == "2026-08-06"
    assert provenance["license"] == {
        "spdx_identifier": "BSD-3-Clause",
        "file": "LICENSE",
    }
    assert provenance["paper"] == {
        "title": "Optimal Calibration of the Endpoint-corrected Hilbert Transform",
        "authors": ["Eike Osmers", "Dorothea Kolossa"],
        "arxiv": "2601.13962v2",
        "url": "https://arxiv.org/abs/2601.13962",
    }
    assert provenance["canonical_test_environment"] == {
        "python": "3.13.9",
        "numpy": "2.3.5",
        "scipy": "1.16.3",
    }
    assert provenance["verification_dependencies"] == {
        "jsonschema": "4.25.0",
        "pytest": "8.4.2",
    }
    assert provenance["local_modifications"] is False


def test_running_environment_is_the_approved_canonical_environment() -> None:
    provenance = load_provenance()
    expected = provenance["canonical_test_environment"]
    dependencies = provenance["verification_dependencies"]

    assert platform.python_version() == expected["python"]
    assert np.__version__ == expected["numpy"]
    assert scipy.__version__ == expected["scipy"]
    assert version("pytest") == dependencies["pytest"]
    assert version("jsonschema") == dependencies["jsonschema"]


def test_vendor_scope_is_the_approved_minimal_snapshot() -> None:
    assert {path.name for path in VENDOR_DIR.iterdir()} == EXPECTED_VENDOR_FILES

    provenance = load_provenance()
    assert provenance["runtime_files"] == ["phase.py"]
    assert set(provenance["files"]) == {"LICENSE", "phase.py"}


def test_vendored_bytes_match_recorded_hashes_and_pinned_raw_urls() -> None:
    provenance = load_provenance()

    for name, path in {"phase.py": PHASE_PATH, "LICENSE": LICENSE_PATH}.items():
        content = path.read_bytes()
        recorded = provenance["files"][name]
        assert recorded["raw_url"] == (
            f"https://raw.githubusercontent.com/eikeosmers/cecHT/{COMMIT}/{name}"
        )
        assert recorded["sha256"] == EXPECTED_UPSTREAM_FILES[name]["sha256"]
        assert recorded["git_blob_sha1"] == EXPECTED_UPSTREAM_FILES[name][
            "git_blob_sha1"
        ]
        assert recorded["size_bytes"] == EXPECTED_UPSTREAM_FILES[name][
            "size_bytes"
        ]
        assert hashlib.sha256(content).hexdigest() == recorded["sha256"]
        assert git_blob_sha1(content) == recorded["git_blob_sha1"]
        assert len(content) == recorded["size_bytes"]


def test_license_is_complete_bsd_3_clause_with_upstream_attribution() -> None:
    license_text = LICENSE_PATH.read_text(encoding="utf-8")

    assert "BSD 3-Clause License" in license_text
    assert "Copyright (c) 2019, nbara" in license_text
    assert "Copyright (c) 2026, eosmers" in license_text
    assert "Redistribution and use in source and binary forms" in license_text
    assert "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS" in license_text


def test_provenance_records_the_endpoint_index_evidence_boundary() -> None:
    provenance = load_provenance()

    assert provenance["endpoint_index_evidence"] == {
        "accepted_endpoint_index": -1,
        "readme_online_example_index": 0,
        "readme_conflict_recorded": True,
        "simulation_url": (
            f"https://github.com/eikeosmers/cecHT/blob/{COMMIT}/"
            "simulations/simple.py"
        ),
        "window_loop_url": (
            f"https://github.com/eikeosmers/cecHT/blob/{COMMIT}/utils.py"
        ),
    }
