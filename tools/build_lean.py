"""Build the Lean 4 developments and record the result.

formal/ is a Lean 4 project without dependencies (CertgenFormal); its
Boundary.lean pins the axioms of every exported result with #guard_msgs, so
the build fails if a proof is incomplete or its axioms change. formal/bridge/
is a separate project that instantiates the development over the real numbers
of Mathlib.

    python tools/build_lean.py                      # core build (needs lake)
    python tools/build_lean.py --bridge-dir DIR     # also the Mathlib bridge

DIR is a working copy of formal/ whose bridge/.lake holds Mathlib v4.15.0 and
its dependencies at the revisions of bridge/lake-manifest.json, prepared
once by `lake build` (or `lake exe cache get`) in DIR/bridge; the current
sources are copied into it before every build. The core is built in a
temporary copy, so formal/ never receives build products. The toolchain is
the one elan resolves for formal/lean-toolchain.

Writes results/lean_build.json and, with --bridge-dir,
results/lean_bridge_build.json. Each record carries the number of Lean
warnings in the sources of the project and `code_sha256`, a digest of the
sources it built: the Lean files without comments (the expected messages of
#guard_msgs are kept, since the build checks them), the lake files and the
toolchain. tests/test_formal.py compares the digest with the current
sources, so a record certifies the current code for as long as only
comments change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORMAL = os.path.join(ROOT, "formal")
CORE_OUT = os.path.join(ROOT, "results", "lean_build.json")
BRIDGE_OUT = os.path.join(ROOT, "results", "lean_bridge_build.json")
MATHLIB = "v4.15.0"
CORE_ENTRIES = ["CertgenFormal", "CertgenFormal.lean", "lakefile.toml",
                "lake-manifest.json", "lean-toolchain"]
BRIDGE_FILES = ["Bridge.lean", "BridgeKantorovich.lean", "lakefile.toml",
                "lake-manifest.json", "lean-toolchain"]
PROJECT_FILES = ("lakefile.toml", "lake-manifest.json", "lean-toolchain")
# a Lean diagnostic: "warning: <file>.lean:<line>:<column>: ..."
_LEAN_WARNING = re.compile(r"^warning: (\S+?\.lean):\d+:\d+:", re.M)


_GUARD_DOC = re.compile(r"/--\s*(?:info|warning|error):")


def strip_lean_comments(text: str) -> str:
    """Remove -- line comments and nested /- -/ block comments.

    The expected messages of #guard_msgs (`/-- info: ... -/`) are checked by
    the build and are kept.
    """
    out, depth, i, n = [], 0, 0, len(text)
    while i < n:
        if depth == 0 and _GUARD_DOC.match(text, i):
            j = text.index("-/", i) + 2
            out.append(text[i:j])
            i = j
        elif text.startswith("/-", i):
            depth += 1
            i += 2
        elif depth > 0 and text.startswith("-/", i):
            depth -= 1
            i += 2
        elif depth == 0 and text.startswith("--", i):
            while i < n and text[i] != "\n":
                i += 1
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _strip_toml_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0].rstrip() for line in text.splitlines())


def source_files(include_bridge: bool) -> list:
    files = []
    for dp, _, fs in os.walk(FORMAL):
        if ".lake" in dp:
            continue
        for f in fs:
            if f.endswith(".lean") or f in PROJECT_FILES:
                rel = os.path.relpath(os.path.join(dp, f), FORMAL).replace(os.sep, "/")
                if rel.startswith("bridge") and not include_bridge:
                    continue
                files.append(rel)
    return sorted(files)


def code_sha256(include_bridge: bool) -> str:
    """sha256 over the relative paths and the sources without comments."""
    h = hashlib.sha256()
    for rel in source_files(include_bridge):
        with open(os.path.join(FORMAL, rel), encoding="utf-8") as fh:
            text = fh.read()
        if rel.endswith(".lean"):
            text = strip_lean_comments(text)
        elif rel.endswith(".toml"):
            text = _strip_toml_comments(text)
        h.update(rel.encode())
        h.update("\n".join(l.rstrip() for l in text.splitlines()
                           if l.strip()).encode())
    return h.hexdigest()


def lean_warnings(log: str) -> int:
    """Number of Lean warnings in the sources of this project.

    Messages of lake itself and warnings in dependencies (.lake/packages)
    are not counted.
    """
    return sum(1 for f in _LEAN_WARNING.findall(log)
               if ".lake" not in f.replace("\\", "/").split("/"))


def pinned_results() -> int:
    """Number of exported results whose axioms Boundary.lean pins."""
    with open(os.path.join(FORMAL, "CertgenFormal", "Boundary.lean"),
              encoding="utf-8") as fh:
        return fh.read().count("#guard_msgs in")


def sorry_free() -> bool:
    for rel in source_files(True):
        if rel.endswith(".lean"):
            with open(os.path.join(FORMAL, rel), encoding="utf-8") as fh:
                if "sorry" in strip_lean_comments(fh.read()):
                    return False
    return True


def _lake():
    """The lake executable of the pinned toolchain, and its version.

    elan resolves the toolchain of formal/lean-toolchain; its own proxy is
    bypassed so that messages of elan are not mistaken for build output.
    """
    elan = shutil.which("elan") or next(
        (p for p in (os.path.expanduser(os.path.join("~", ".elan", "bin", n))
                     for n in ("elan.exe", "elan")) if os.path.exists(p)), None)
    lake = None
    if elan:
        try:
            w = subprocess.run([elan, "which", "lake"], cwd=FORMAL,
                               capture_output=True, encoding="utf-8",
                               errors="replace", timeout=120)
            cand = w.stdout.strip().splitlines()[-1] if w.stdout.strip() else ""
            lake = cand if cand and os.path.exists(cand) else None
        except (OSError, subprocess.SubprocessError):
            lake = None
    lake = lake or shutil.which("lake")
    if lake is None:
        return None, ""
    try:
        v = subprocess.run([lake, "--version"], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=120)
        return lake, v.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, ""


def build_core(lake: str, version: str) -> dict:
    tmp = tempfile.mkdtemp(prefix="certgen_lean_")
    try:
        work = os.path.join(tmp, "formal")
        os.makedirs(work)
        for entry in CORE_ENTRIES:
            src = os.path.join(FORMAL, entry)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(work, entry),
                                ignore=shutil.ignore_patterns(".lake"))
            else:
                shutil.copy(src, os.path.join(work, entry))
        p = subprocess.run([lake, "build"], cwd=work, capture_output=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        log = p.stdout + p.stderr
        ok = p.returncode == 0
        if not ok:
            print(log[-3000:])
        return {"status": "clean" if ok else "failed",
                "warnings": lean_warnings(log), "lake": version}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_bridge(lake: str, version: str, bridge_dir: str) -> dict:
    bd = os.path.abspath(bridge_dir)
    if not os.path.isdir(os.path.join(bd, "bridge", ".lake", "packages", "mathlib")):
        raise SystemExit(f"{bd}: bridge/.lake/packages/mathlib not found")
    for entry in CORE_ENTRIES:
        src, dst = os.path.join(FORMAL, entry), os.path.join(bd, entry)
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
    for f in BRIDGE_FILES:
        shutil.copy(os.path.join(FORMAL, "bridge", f), os.path.join(bd, "bridge", f))
    p = subprocess.run([lake, "build"], cwd=os.path.join(bd, "bridge"),
                       capture_output=True, encoding="utf-8", errors="replace",
                       timeout=7200)
    log = p.stdout + p.stderr
    ok = p.returncode == 0
    if not ok:
        print(log[-3000:])
    return {"status": "clean" if ok else "failed",
            "warnings": lean_warnings(log), "lake": version,
            "mathlib": MATHLIB, "modules": ["Bridge", "BridgeKantorovich"],
            "code_sha256": code_sha256(True)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-dir", default=None,
                    help="copy of formal/ with Mathlib in bridge/.lake")
    a = ap.parse_args(argv)
    lake, version = _lake()
    if lake is None:
        print("no Lean toolchain (lake) found; install elan and Lean 4.15.0")
        return 2
    rec = build_core(lake, version)
    rec.update({"pinned_results": pinned_results(), "sorry_free": sorry_free(),
                "main_modules": len([f for f in os.listdir(
                    os.path.join(FORMAL, "CertgenFormal")) if f.endswith(".lean")]),
                "code_sha256": code_sha256(False)})
    with open(CORE_OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=2)
        fh.write("\n")
    print(f"core: {rec['status']}, {rec['warnings']} warnings, "
          f"{rec['pinned_results']} pinned results")
    ok = rec["status"] == "clean" and rec["warnings"] == 0
    if a.bridge_dir:
        br = build_bridge(lake, version, a.bridge_dir)
        with open(BRIDGE_OUT, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(br, fh, indent=2)
            fh.write("\n")
        print(f"bridge: {br['status']}, {br['warnings']} warnings")
        ok = ok and br["status"] == "clean" and br["warnings"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
