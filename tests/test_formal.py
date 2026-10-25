"""The Lean developments: sources against the recorded builds."""
import build_lean
from conftest import artifact


def test_core_build_record_certifies_the_current_code():
    rec = artifact("lean_build.json")
    assert rec["status"] == "clean" and rec["warnings"] == 0
    assert rec["code_sha256"] == build_lean.code_sha256(False)


def test_bridge_build_record_certifies_the_current_code():
    rec = artifact("lean_bridge_build.json")
    assert rec["status"] == "clean" and rec["warnings"] == 0
    assert rec["code_sha256"] == build_lean.code_sha256(True)


def test_no_proof_is_left_open():
    assert build_lean.sorry_free()


def test_axioms_of_every_exported_result_are_pinned():
    assert build_lean.pinned_results() == artifact("lean_build.json")["pinned_results"]
