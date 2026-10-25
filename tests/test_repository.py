"""Integrity of the repository: code, experiments, artifacts, reproduction."""
import ast
import io
import os
import re

import pytest

from conftest import ROOT, artifact

CODE_DIRS = ("certgen", "experiments", "tools", "tests")


def _python_files():
    for d in CODE_DIRS:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith(".py"):
                yield os.path.join(d, f)


def test_no_undefined_names_or_unused_imports():
    pyflakes = pytest.importorskip("pyflakes.api")
    from pyflakes.reporter import Reporter
    problems = []
    for rel in _python_files():
        buf = io.StringIO()
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            pyflakes.check(fh.read(), rel, Reporter(buf, buf))
        problems += buf.getvalue().splitlines()
    assert problems == []


def test_every_experiment_states_what_it_writes():
    for f in sorted(os.listdir(os.path.join(ROOT, "experiments"))):
        if f.endswith(".py") and not f.startswith("_"):
            with open(os.path.join(ROOT, "experiments", f), encoding="utf-8") as fh:
                doc = ast.get_docstring(ast.parse(fh.read())) or ""
            assert re.search(r"results/[a-z0-9_]+\.json", doc), f


def _generators():
    out = {}
    for d in ("experiments", "tools"):
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith(".py"):
                with open(os.path.join(ROOT, d, f), encoding="utf-8") as fh:
                    src = fh.read()
                for name in re.findall(r"([a-z0-9_]+\.json)", src):
                    out.setdefault(name, []).append(f"{d}/{f}")
    return out


def test_every_artifact_has_a_generator():
    gens = _generators()
    orphans = [f for f in os.listdir(os.path.join(ROOT, "results"))
               if f.endswith(".json") and f not in gens]
    assert orphans == []


def test_experiments_are_listed_in_their_readme():
    with open(os.path.join(ROOT, "experiments", "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    missing = [f for f in os.listdir(os.path.join(ROOT, "experiments"))
               if f.endswith(".py") and not f.startswith("_") and f not in readme]
    assert missing == []


def test_reproduction_log_reports_no_failure():
    log = artifact("reproduction_log.json")["summary"]
    assert not log["failed"]
    assert not log["artifacts_differing"]


def test_readme_usage_example_runs():
    """The Python example of README.md runs as written on a synthetic table."""
    import certgen as cg
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    ns = {"X_real": cg.data.gaussian_copula(4000, d=6, seed=1),
          "X_generated": cg.data.gaussian_copula(2000, d=6, seed=2)}
    exec(compile(code, "README.md", "exec"), ns)


def test_package_metadata_is_complete():
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        meta = tomllib.load(fh)["project"]
    assert meta["license"] == "MIT" and meta["authors"]
    assert os.path.exists(os.path.join(ROOT, "LICENSE"))
    assert os.path.exists(os.path.join(ROOT, "CITATION.cff"))


def test_project_is_self_contained():
    """No file refers to material outside the project or to a local machine."""
    outside = re.compile("|".join([
        r"\bHist" + r"ory/",                  # an archive beside the project
        r"\.\./\.\./",                        # paths above the project root
        r"[A-Za-z]:[\/]+Users[\/]",         # Windows home directories
        r"/home/[a-z]", r"/Users/[A-Za-z]",   # Unix home directories
    ]))
    this = os.path.abspath(__file__)
    found = []
    for dp, dn, fs in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in (".git", ".lake", "__pycache__", ".venv",
                                            ".pytest_cache") and not d.endswith(".egg-info")]
        for f in fs:
            p = os.path.join(dp, f)
            if p == this or not f.endswith((".py", ".md", ".tex", ".bib", ".toml",
                                            ".cff", ".lean", ".json", ".txt")):
                continue
            with open(p, encoding="utf-8", errors="replace") as fh:
                if outside.search(fh.read()):
                    found.append(os.path.relpath(p, ROOT))
    assert found == []
