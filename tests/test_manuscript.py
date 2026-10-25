"""The manuscript against the artifacts: numbers, citations, figures, PDF."""
import hashlib
import os
import re
import shutil
import subprocess

import pytest

import manuscript_numbers as mn
import paper_numbers
from conftest import ROOT, artifact

PAPER = os.path.join(ROOT, "paper")


@pytest.fixture(scope="module")
def tex():
    with open(os.path.join(PAPER, "main.tex"), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def numbers():
    return paper_numbers.all_numbers()


def test_every_computed_number_appears_in_the_manuscript(tex, numbers):
    assert mn.missing_values(tex, numbers) == []


def test_every_number_in_the_manuscript_has_a_source(tex, numbers):
    assert mn.unaccounted_literals(tex, numbers) == []


def test_source_is_ascii(tex):
    assert all(ord(c) < 128 for c in tex)


def test_software_version_matches_the_package(tex):
    import certgen
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        pyproject = re.search(r'^version = "([^"]+)"', fh.read(), re.M).group(1)
    stated = re.search(r"\(version (\d+\.\d+\.\d+)\)", tex).group(1)
    assert stated == pyproject == certgen.__version__


def _cited_keys(tex):
    keys = set()
    for m in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        keys.update(k.strip() for k in m.split(","))
    return keys


def test_citations_and_bibliography_agree(tex):
    import verify_bibliography
    with open(os.path.join(PAPER, "references.bib"), encoding="utf-8") as fh:
        entries = {e["key"] for e in verify_bibliography.parse_bib(fh.read())}
    cited = _cited_keys(mn.strip_comments(tex))
    assert cited - entries == set(), "cited but not in references.bib"
    assert entries - cited == set(), "in references.bib but never cited"


def test_every_reference_is_verified():
    import verify_bibliography
    reg = artifact("bibliography_registry.json")["summary"]
    assert not reg["mismatch"] and not reg["unresolved"]
    assert reg["verified"] + reg["manual"] == reg["entries"]
    assert not reg["consensus_disagree"]
    with open(verify_bibliography.BIB, encoding="utf-8") as fh:
        assert verify_bibliography.offline(verify_bibliography.parse_bib(fh.read())) == []


def test_figures_are_drawn_from_the_current_artifacts(tex):
    manifest = artifact("figures_manifest.json")["figures"]
    included = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", tex))
    assert set(manifest) == included
    for fig, meta in manifest.items():
        assert os.path.exists(os.path.join(PAPER, "figures", fig))
        for name, digest in meta["inputs"].items():
            with open(os.path.join(ROOT, "results", name), "rb") as fh:
                assert hashlib.sha256(fh.read()).hexdigest() == digest, (fig, name)


def test_section_structure(tex):
    sections = re.findall(r"\\section\{([^}]*)\}", mn.body(tex))
    assert sections == ["Introduction", "Related Work", "The Resolution Floor",
                        "Designing an Evaluation", "Reporting an Evaluation",
                        "Limits of Latent-Generative Certificates",
                        "A Certificate Without a Lipschitz Constant",
                        "Empirical Study on Tabular Benchmarks",
                        "Implementation and Verification", "Discussion",
                        "Conclusion", "Proofs", "Experimental Details"]


def test_abstract_length(tex):
    abstract = tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")]
    words = re.sub(r"\\[a-zA-Z]+|[{}$%\\]", " ", abstract).split()
    assert len(words) <= 250


def test_no_project_internal_vocabulary(tex):
    """The manuscript describes results, not the history of the project."""
    body = mn.body(tex)
    for pattern in (r"earlier version", r"previous(?:ly)? (?:version|release)",
                    r"retract", r"\bPhase[- ]\d", r"\bR\d{1,2}\b", r"\bB-\d{2}\b",
                    r"\bsweep\b", r"test suite", r"this project"):
        assert not re.search(pattern, body, re.I), pattern


def _tool(name):
    return shutil.which(name)


@pytest.mark.skipif(not _tool("pdffonts"), reason="pdffonts not installed")
def test_pdf_embeds_every_font_and_has_no_type3():
    out = subprocess.run(["pdffonts", os.path.join(PAPER, "main.pdf")],
                         capture_output=True, encoding="utf-8", errors="replace").stdout
    rows = [r.split() for r in out.strip().splitlines()[2:] if r.strip()]
    assert rows
    assert not [r[0] for r in rows if "Type 3" in " ".join(r)]
    assert not [r[0] for r in rows if len(r) >= 7 and r[-4] == "no"]


@pytest.mark.skipif(not _tool("pdftotext"), reason="pdftotext not installed")
def test_pdf_numbering_of_formal_statements():
    text = subprocess.run(["pdftotext", "-enc", "UTF-8", os.path.join(PAPER, "main.pdf"), "-"],
                          capture_output=True, encoding="utf-8", errors="replace").stdout
    assert all(f"Proposition {k} (" in text for k in (1, 2, 3, 4))
    assert "Lemma 1 (" in text and "Proposition 5" not in text


def test_pdf_is_newer_than_its_sources():
    pdf = os.path.getmtime(os.path.join(PAPER, "main.pdf"))
    for f in ("main.tex", "references.bib"):
        assert os.path.getmtime(os.path.join(PAPER, f)) <= pdf + 1, f
