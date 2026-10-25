"""Verify every entry of paper/references.bib against bibliographic indexes.

Each entry is looked up in a primary index:

    Crossref         by its DOI when the entry carries one;
    arXiv            by its identifier (or the DataCite record of the same
                     DOI 10.48550/arXiv.<id> when the arXiv API is busy);
    publisher pages  citation_* metadata of the venue page (PAGE_META);
    Crossref search  otherwise, by title and first author.

For each entry the registry holds the fetched record and five comparisons
with the entry: surnames in order, first initials, title similarity, year,
and volume, first page and DOI where both sides state them. The verdict is
`verified` only if all of them hold. Every verified entry is also looked up
in a second, independent index, OpenAlex, which must agree on title, first
author and year. Entries no index covers (old journals, a corporate author)
are `manual`, with the source they were checked against.

    python tools/verify_bibliography.py             # network; writes the registry
    python tools/verify_bibliography.py --offline   # compare the .bib with the registry
    python tools/verify_bibliography.py --log       # write docs/bibliography.md

Writes results/bibliography_registry.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIB = os.path.join(ROOT, "paper", "references.bib")
OUT = os.path.join(ROOT, "results", "bibliography_registry.json")
LOG = os.path.join(ROOT, "docs", "bibliography.md")
UA = {"User-Agent": "certgen-bibliography-verifier/3.3"}
try:
    import ssl as _ssl
    import certifi as _certifi
    _CTX = _ssl.create_default_context(cafile=_certifi.where())
except ImportError:  # pragma: no cover
    _CTX = None

# conference papers whose authoritative record is their arXiv preprint
ARXIV_HINTS = {"dziugaite2017": "1703.11008", "block2022": "2106.04018",
               "szegedy2014": "1312.6199", "borisov2023": "2210.06280",
               "bounliphone2016": "1511.04581", "bouthillier2021": "2103.03098",
               "cuturi2013": "1306.0895", "fazlyab2019": "1906.04893",
               "heusel2017": "1706.08500", "kynkanniemi2019": "1904.06991",
               "lopezpaz2017": "1610.06545", "pope2021": "2104.08894",
               "sajjadi2018": "1806.00035", "xu2019": "1907.00503",
               "kingma2014": "1312.6114", "kingma2015": "1412.6980",
               "paszke2019": "1912.01703", "salimans2016": "1606.03498"}

# entries no index covers adequately, with the source they were checked against
MANUAL = {
    "birkhoff1946": "Revista Serie A, Universidad Nacional de Tucuman 5 "
                    "(1946) 147-151; not indexed by arXiv or Crossref; the "
                    "standard citation of Birkhoff's theorem",
    "mathlib2020": "corporate author; Crossref DOI 10.1145/3372885.3373824 "
                   "(CPP 2020, pp. 367-381), title 'The lean mathematical "
                   "library', author 'The mathlib Community'",
    "westlake1981": "Biometrics 37(3):589-594, 1981; Crossref indexes the "
                    "exchange under DOI 10.2307/2530573 with Kirkwood and "
                    "Westlake as joint authors and the title 'Bioequivalence "
                    "Testing -- A Need to Rethink'; the cited form is "
                    "Westlake's response",
    "villani2009": "book (Springer, Grundlehren 338); Crossref DOI "
                   "10.1007/978-3-540-71050-9 carries the title 'Optimal "
                   "Transport' without the subtitle 'Old and New'",
    "kantorovich1942": "Doklady Akademii Nauk SSSR 37 (1942) 199-201; zbMATH "
                       "Zbl 0061.09705; Crossref indexes the English "
                       "translation, J. Math. Sci. 133(4):1381-1382 (2006), "
                       "DOI 10.1007/s10958-006-0049-2",
    "hanley1983": "JAMA 249(13):1743-1745, 1983, DOI "
                  "10.1001/jama.1983.03330370053031; Crossref lists only the "
                  "first author and the first page; the article is by Hanley "
                  "and Lippman-Hand",
    "kelly2026": "the citation the repository requests for several of its "
                 "data sets together, https://archive.ics.uci.edu/citation",
    "vanschoren2014": "SIGKDD Explorations 15(2):49-60, 2014; Crossref DOI "
                      "10.1145/2641190.2641198 carries the title 'OpenML' "
                      "without the subtitle 'networked science in machine "
                      "learning'",
    "alimoglu1997": "ICDAR 1997 (Ulm, August 1997), vol. 2, pp. 637-640; "
                    "Crossref DOI 10.1109/ICDAR.1997.620583 agrees on "
                    "authors, title and pages but gives no year, and "
                    "OpenAlex gives 2002",
}

# arXiv versions of publisher-page entries, for the second-index lookup only
OPENALEX_ARXIV = {"vanbreugel2023": "2302.12580", "virmaux2018": "1805.10965"}

# publisher landing pages carrying citation_* metadata
PAGE_META = {
    "gretton2012": "https://jmlr.org/papers/v13/gretton12a.html",
    "schrab2023": "https://jmlr.org/papers/v24/21-1289.html",
    "pedregosa2011": "https://jmlr.org/papers/v12/pedregosa11a.html",
    "levina2004": "https://papers.nips.cc/paper/2577-maximum-likelihood-estimation-of-intrinsic-dimension",
    "vanbreugel2023": "https://proceedings.mlr.press/v206/breugel23a.html",
    # the arXiv listing orders the authors Scaman, Virmaux; the NeurIPS
    # proceedings, the cited venue, order them Virmaux, Scaman
    "virmaux2018": "https://papers.nips.cc/paper/7640-lipschitz-regularity-of-deep-neural-networks-analysis-and-efficient-estimation",
}

LATEX_ACCENTS = [(r"\'", ""), (r"\"", ""), (r"\`", ""), (r"\^", ""),
                 (r"\v", ""), (r"\u", ""),
                 (r"\~", ""), ("{\\ss}", "ss"), ("\\ss", "ss"), ("{\\i}", "i"),
                 ("\\i", "i"), ("{", ""), ("}", ""), ("~", " "), ("---", "-"),
                 ("--", "-")]


def detex(s: str) -> str:
    for a, b in LATEX_ACCENTS:
        s = s.replace(a, b)
    s = re.sub(r"\\emph", "", s)
    return " ".join(s.split())


def fold(s: str) -> str:
    s = s.replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fields(body: str) -> dict:
    """Fields of one BibTeX entry body (values in braces, nested braces kept)."""
    out, i = {}, 0
    while True:
        m = re.compile(r"\s*,?\s*([A-Za-z]+)\s*=\s*\{").match(body, i)
        if not m:
            break
        depth, j = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(body[j], 0)
            j += 1
        out[m.group(1).lower()] = " ".join(body[m.end():j - 1].split())
        i = j
    return out


def _authors(field: str):
    authors, etal = [], False
    for name in re.split(r"\s+and\s+", field):
        name = name.strip()
        if name == "others":
            etal = True
            continue
        if name.startswith("{") and name.endswith("}"):
            authors.append({"surname": detex(name), "initials": ""})
            continue
        surname, _, given = name.partition(",")
        initials = "".join(re.findall(r"([A-Z])", detex(given)))
        authors.append({"surname": detex(surname.strip()), "initials": initials})
    return authors, etal


def parse_bib(text: str):
    """Entries of a .bib file in the form the comparisons use."""
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),", text):
        start = m.end()
        depth, j = 1, start
        while depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        f = _fields(text[start:j - 1])
        authors, etal = _authors(f.get("author", ""))
        arx = re.search(r"arXiv:(\d{4}\.\d{4,5})", f.get("howpublished", "")
                        + " " + f.get("eprint", ""))
        out.append({"key": m.group(2).strip(), "type": m.group(1).lower(),
                    "year": f.get("year", ""), "authors": authors, "et_al": etal,
                    "title": detex(f.get("title", "")),
                    "venue": detex(f.get("journal") or f.get("booktitle")
                                   or f.get("publisher") or f.get("howpublished", "")),
                    "volume": f.get("volume", ""), "pages": f.get("pages", ""),
                    "doi": f.get("doi", ""),
                    "arxiv": arx.group(1) if arx else None})
    return out


def _get(url, timeout=40, tries=4):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as ex:  # transient network failures are retried
            last = ex
            time.sleep(2.0 * (k + 1))
    raise last


def q_datacite(aid):
    """The arXiv record through its DataCite DOI 10.48550/arXiv.<id>, the
    same metadata arXiv registers, served independently of the arXiv API."""
    a = json.loads(_get(f"https://api.datacite.org/dois/10.48550/arXiv.{aid}"))["data"]["attributes"]
    auth = [{"family": c.get("familyName") or c.get("name", "").split(",")[0],
             "given": c.get("givenName", "")} for c in a.get("creators", [])]
    return {"source": "arXiv-DataCite", "id": aid, "title": " ".join(a["titles"][0]["title"].split()),
            "authors": auth, "year": str(a.get("publicationYear", "")), "venue": "arXiv"}


def q_arxiv(aid):
    try:
        txt = _get(f"https://export.arxiv.org/api/query?id_list={aid}", tries=1)
    except Exception:  # busy or down: the DataCite record of the same DOI
        return q_datacite(aid)
    e = txt.split("<entry>")[1]
    title = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
    names = re.findall(r"<name>(.*?)</name>", e)
    year = re.search(r"<published>(\d{4})", e).group(1)
    auth = []
    for n in names:
        toks = n.split()
        auth.append({"family": toks[-1], "given": " ".join(toks[:-1])})
    return {"source": "arXiv", "id": aid, "title": title, "authors": auth,
            "year": year, "venue": "arXiv"}


def q_meta(url):
    import html as _html
    txt = _get(url)

    def metas(name):
        return [_html.unescape(m) for m in re.findall(
            r'<meta\s+name="%s"\s+content="([^"]*)"\s*/?>' % name, txt)]
    auth = []
    for a in metas("citation_author"):
        if "," in a:
            fam, giv = [x.strip() for x in a.split(",", 1)]
        else:
            tk = a.split()
            fam, giv = tk[-1], " ".join(tk[:-1])
        auth.append({"family": fam, "given": giv})
    year = (metas("citation_publication_date") or metas("citation_date")
            or metas("citation_year") or [""])[0][:4]
    return {"source": "publisher-page", "id": url,
            "title": (metas("citation_title") or [""])[0], "authors": auth,
            "year": year, "venue": (metas("citation_journal_title")
                                    or metas("citation_conference_title") or [""])[0]}


def q_arxiv_title(title):
    q = urllib.parse.quote('ti:"%s"' % re.sub(r"[^A-Za-z0-9 -]", " ", title))
    txt = _get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=5")
    out = []
    for e in txt.split("<entry>")[1:]:
        t = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
        names = re.findall(r"<name>(.*?)</name>", e)
        aid = re.search(r"<id>http://arxiv.org/abs/([^<v]+)", e)
        auth = [{"family": n.split()[-1], "given": " ".join(n.split()[:-1])}
                for n in names]
        out.append({"source": "arXiv-title", "id": aid.group(1) if aid else "",
                    "title": t, "authors": auth,
                    "year": re.search(r"<published>(\d{4})", e).group(1),
                    "venue": "arXiv"})
    return out


def q_crossref(title, first):
    url = ("https://api.crossref.org/works?rows=8&query.bibliographic="
           + urllib.parse.quote(f"{title} {first}"))
    items = json.loads(_get(url))["message"]["items"]
    out = []
    for it in items:
        auth = [{"family": a.get("family", ""), "given": a.get("given", "")}
                for a in it.get("author", [])]
        y = (it.get("issued", {}).get("date-parts", [[None]])[0][0]
             or it.get("published-print", {}).get("date-parts", [[None]])[0][0])
        out.append({"source": "Crossref", "id": it.get("DOI"),
                    "title": " ".join((it.get("title") or [""])[0].split()),
                    "authors": auth, "year": str(y) if y else "",
                    "venue": (it.get("container-title") or [""])[0],
                    "volume": it.get("volume", ""), "issue": it.get("issue", ""),
                    "page": it.get("page", "")})
    return out


def q_doi(doi):
    m = json.loads(_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    auth = [{"family": a.get("family", a.get("name", "")), "given": a.get("given", "")}
            for a in m.get("author", [])]
    y = (m.get("issued", {}).get("date-parts", [[None]])[0][0]
         or m.get("published-print", {}).get("date-parts", [[None]])[0][0])
    title = " ".join((m.get("title") or [""])[0].split())
    return {"source": "Crossref-DOI", "id": m.get("DOI"),
            "title": re.sub(r"^\d+\.\s+", "", title),   # chapter numbers
            "authors": auth, "year": str(y) if y else "",
            "venue": (m.get("container-title") or [""])[0],
            "volume": m.get("volume", ""), "issue": m.get("issue", ""),
            "page": m.get("page", "")}


def q_openalex(e, rec):
    """The independent second index. Looked up by the primary record's DOI
    when there is one, else by title; returns the consensus verdict."""
    doi = rec.get("id") if rec and rec.get("source", "").startswith("Crossref") else None
    # arXiv records (and publisher-page entries with a known arXiv version) are
    # looked up by their DataCite DOI: exact, and spared the search endpoint
    aid = (rec.get("id") if rec and rec.get("source", "").startswith("arXiv")
           else OPENALEX_ARXIV.get(e["key"]))
    if not doi and aid:
        doi = f"10.48550/arxiv.{aid}"
    if doi:
        url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi.lower())
        cands = [json.loads(_get(url))]
    else:
        url = ("https://api.openalex.org/works?per-page=5&search="
               + urllib.parse.quote(e["title"]))
        cands = json.loads(_get(url)).get("results", [])
    best, bs = None, 0.0
    for w in cands:
        sc = difflib.SequenceMatcher(None, fold(w.get("title") or ""), fold(e["title"])).ratio()
        if sc > bs:
            best, bs = w, sc
    if best is None or bs < 0.90:
        return {"source": "OpenAlex", "agrees": False, "reason": "not indexed under this title"}
    names = [fold((a.get("author") or {}).get("display_name", ""))
             for a in best.get("authorships", [])[:3]]
    first_ok = any(fold(e["authors"][0]["surname"]) in n for n in names) if e["authors"] else False
    y = best.get("publication_year")
    year_ok = bool(y) and abs(int(e["year"]) - int(y)) <= 3
    return {"source": "OpenAlex", "id": best.get("id"), "title_similarity": round(bs, 3),
            "first_author_ok": first_ok, "year": y, "year_ok": year_ok,
            "agrees": bool(first_ok and year_ok)}


def q_dblp(title):
    url = ("https://dblp.org/search/publ/api?format=json&h=8&q="
           + urllib.parse.quote(title))
    hits = json.loads(_get(url)).get("result", {}).get("hits", {}).get("hit", [])
    out = []
    for h in hits:
        info = h["info"]
        a = info.get("authors", {}).get("author", [])
        a = a if isinstance(a, list) else [a]
        auth = []
        for x in a:
            nm = re.sub(r"\s\d{4}$", "", x["text"] if isinstance(x, dict) else x)
            tk = nm.split()
            auth.append({"family": tk[-1], "given": " ".join(tk[:-1])})
        out.append({"source": "DBLP", "id": info.get("key"),
                    "title": info.get("title", "").rstrip("."), "authors": auth,
                    "year": str(info.get("year", "")), "venue": info.get("venue", "")})
    return out


def best_record(e, cands):
    """Highest title similarity among candidates whose first author's surname
    matches the entry's; None if no candidate reaches 0.90."""
    best, bs = None, 0.0
    for r in cands:
        if not r["authors"] or not e["authors"]:
            continue
        if not _surname_match(e["authors"][0]["surname"], r["authors"][0]["family"]) \
                and fold(e["authors"][0]["surname"]) not in fold(
                    r["authors"][0]["given"] + r["authors"][0]["family"]):
            continue
        sc = difflib.SequenceMatcher(None, fold(r["title"]), fold(e["title"])).ratio()
        if sc > bs:
            best, bs = r, sc
    return (best if bs >= 0.90 else None), best, bs


def _surname_match(tex_sur, fam):
    a, b = fold(tex_sur), fold(fam)
    return a == b or a.endswith(b) or b.endswith(a)


def compare(e, rec):
    """Comparisons of an entry with its fetched record."""
    ta, fa = e["authors"], rec["authors"]
    n = len(ta)
    sur_ok = (len(fa) >= n and all(_surname_match(ta[i]["surname"], fa[i]["family"])
                                   for i in range(n))
              and (e["et_al"] or len(fa) == n))
    # multi-word surnames ("van Breugel", "del Barrio") may be split differently
    if not sur_ok and len(fa) == n:
        sur_ok = all(fold(ta[i]["surname"]) in fold(fa[i]["given"] + fa[i]["family"])
                     for i in range(n))
    init_ok, bad_init = True, []
    for i in range(min(n, len(fa))):
        g = re.sub(r"[^A-Za-z -]", "", fa[i]["given"])
        g_init = "".join(w[0].upper() for w in re.split(r"[ -]+", g) if w)
        t_init = ta[i]["initials"]
        if t_init and g_init and t_init[0] != g_init[0]:
            init_ok = False
            bad_init.append(f"{ta[i]['surname']}: entry {t_init} vs index {fa[i]['given']}")
    tsim = difflib.SequenceMatcher(None, fold(e["title"]), fold(rec["title"])).ratio()
    title_ok = tsim >= 0.90
    year_ok = (not rec["year"]) or e["year"] == rec["year"]
    year_note = ""
    if (not year_ok and rec["year"] and rec["source"].startswith("arXiv")
            and 0 < int(e["year"]) - int(rec["year"]) <= 3):
        year_ok, year_note = True, (f"cited {e['year']} (venue) vs arXiv preprint "
                                    f"{rec['year']}")
    if not year_ok and rec["year"] and abs(int(e["year"]) - int(rec["year"])) == 1:
        year_ok, year_note = True, (f"year {e['year']} vs index {rec['year']} "
                                    "(online and print years differ)")
    notes = []
    if rec.get("volume") and e.get("volume") and str(rec["volume"]) != e["volume"]:
        notes.append(f"volume: index {rec['volume']}")
    if rec.get("page") and e.get("pages"):
        if str(rec["page"]).split("-")[0] != e["pages"].split("-")[0]:
            notes.append(f"pages: index {rec['page']}")
    if e.get("doi") and rec.get("source", "").startswith("Crossref") \
            and e["doi"].lower() != str(rec.get("id", "")).lower():
        notes.append(f"doi: index {rec.get('id')}")
    nums_ok = not notes
    return {"surnames_ok": bool(sur_ok), "initials_ok": init_ok,
            "numbers_ok": nums_ok, "numbers_note": "; ".join(notes),
            "initial_mismatches": bad_init, "title_similarity": round(tsim, 3),
            "title_ok": title_ok, "year_ok": year_ok, "year_note": year_note,
            "verdict": "verified" if (sur_ok and init_ok and title_ok and year_ok
                                      and nums_ok) else "MISMATCH"}


def lookup(e):
    """The primary record of an entry, or (None, error, near miss)."""
    if e["doi"]:
        return q_doi(e["doi"]), None, None
    if e["key"] in PAGE_META:
        return q_meta(PAGE_META[e["key"]]), None, None
    if e["arxiv"] or e["key"] in ARXIV_HINTS:
        return q_arxiv(e["arxiv"] or ARXIV_HINTS[e["key"]]), None, None
    cands, err = [], None
    for fn, arg in ((q_crossref, (e["title"], e["authors"][0]["surname"])),
                    (q_arxiv_title, (e["title"],))):
        try:
            cands += fn(*arg)
        except Exception as ex:  # recorded in the registry
            err = f"{fn.__name__}: {type(ex).__name__}: {ex}"
    rec, cand, sc = best_record(e, cands)
    return rec, err, ((cand, sc) if rec is None and cand is not None and sc >= 0.5 else None)


def build(entries):
    out = []
    for e in entries:
        if e["key"] in MANUAL:
            out.append({"key": e["key"], "verdict": "manual",
                        "reason": MANUAL[e["key"]], "entry": e})
            continue
        try:
            rec, err, near = lookup(e)
        except Exception as ex:  # recorded in the registry
            rec, err, near = None, f"{type(ex).__name__}: {ex}", None
        if rec is None and near is not None:
            cand, sc = near
            cmp_ = compare(e, cand)
            cmp_["verdict"] = "MISMATCH"
            cmp_["near_miss"] = (f"first author matches, title similarity {sc:.2f}; "
                                 f"index title '{cand['title']}'")
            out.append({"key": e["key"], "verdict": "MISMATCH", "record": cand,
                        "comparison": cmp_, "entry": e})
            continue
        if rec is None:
            out.append({"key": e["key"], "verdict": "UNRESOLVED", "error": err,
                        "entry": e})
            continue
        cmp_ = compare(e, rec)
        try:
            cons = q_openalex(e, rec)
        except Exception as ex:  # recorded in the registry
            cons = {"source": "OpenAlex", "agrees": False,
                    "reason": f"{type(ex).__name__}: {ex}"}
        out.append({"key": e["key"], "verdict": cmp_["verdict"], "record": rec,
                    "comparison": cmp_, "consensus": cons, "entry": e})
        print(f"{cmp_['verdict']:<9} {e['key']:<20} {rec['source']:<15} "
              f"sim={cmp_['title_similarity']:.2f} {cmp_['numbers_note']}", flush=True)
        time.sleep(0.3)
    return out


def summary(res):
    return {"entries": len(res),
            "verified": sum(r["verdict"] == "verified" for r in res),
            "manual": sum(r["verdict"] == "manual" for r in res),
            "mismatch": [r["key"] for r in res if r["verdict"] == "MISMATCH"],
            "unresolved": [r["key"] for r in res if r["verdict"] == "UNRESOLVED"],
            "consensus_agree": sum(bool(r.get("consensus", {}).get("agrees")) for r in res),
            "consensus_disagree": {r["key"]: r.get("consensus", {}).get("reason", "differs")
                                   for r in res if r["verdict"] == "verified"
                                   and not r.get("consensus", {}).get("agrees")}}


def offline(entries) -> list:
    """Entries that no longer match their verified record."""
    with open(OUT, encoding="utf-8") as fh:
        reg = {r["key"]: r for r in json.load(fh)["entries"]}
    bad = []
    for e in entries:
        r = reg.get(e["key"])
        if r is None:
            bad.append((e["key"], "not in the registry"))
        elif r["verdict"] == "manual":
            continue
        elif r["verdict"] != "verified" or compare(e, r["record"])["verdict"] != "verified":
            bad.append((e["key"], "does not match its verified record"))
    if set(reg) - {e["key"] for e in entries}:
        bad.append(("registry", f"entries not in the .bib: {sorted(set(reg) - {e['key'] for e in entries})}"))
    return bad


def write_log():
    """docs/bibliography.md: one row per reference with its verification."""
    with open(OUT, encoding="utf-8") as fh:
        reg = json.load(fh)
    s = reg["summary"]
    lines = [
        "# Bibliographic verification",
        "",
        "Generated by `tools/verify_bibliography.py --log` from "
        "`results/bibliography_registry.json`.",
        "",
        f"{s['entries']} references: {s['verified']} verified against a primary "
        f"index and confirmed by OpenAlex ({s['consensus_agree']} agreements), "
        f"{s['manual']} checked by hand against the source named below. "
        f"Registry date: {s['date']}.",
        "",
        "| key | first author, year | identifier | primary index | second index |",
        "|---|---|---|---|---|",
    ]
    for r in reg["entries"]:
        e = r["entry"]
        first = e["authors"][0]["surname"] if e["authors"] else ""
        rec = r.get("record") or {}
        ident = rec.get("id") or (f"arXiv:{e['arxiv']}" if e.get("arxiv") else "")
        if r["verdict"] == "manual":
            prim, cons = "manual: " + r["reason"].split(";")[0], "-"
        else:
            prim = f"{rec.get('source', '')} ({r['verdict']})"
            cons = "agrees" if r.get("consensus", {}).get("agrees") else "differs"
        lines.append(f"| `{r['key']}` | {first}, {e['year']} | {ident} | {prim} | {cons} |")
    with open(LOG, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(LOG, ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--retry-consensus", action="store_true",
                    help="repeat only the second-index lookups that failed")
    ap.add_argument("--only", default=None,
                    help="comma-separated keys to verify again; the other "
                         "records of the registry are kept")
    a = ap.parse_args(argv)
    with open(BIB, encoding="utf-8") as fh:
        entries = parse_bib(fh.read())
    if a.log:
        write_log()
        return 0
    if a.only:
        keys = set(a.only.split(","))
        with open(OUT, encoding="utf-8") as fh:
            reg = json.load(fh)
        fresh = {r["key"]: r for r in build([e for e in entries if e["key"] in keys])}
        in_bib = {e["key"] for e in entries}
        reg["entries"] = sorted(
            [fresh.get(r["key"], r) for r in reg["entries"] if r["key"] in in_bib]
            + [r for k, r in fresh.items()
               if k not in {x["key"] for x in reg["entries"]}],
            key=lambda r: r["key"])
        reg["summary"] = summary(reg["entries"])
        reg["summary"]["date"] = _dt.date.today().isoformat()
        with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(reg, fh, indent=1)
            fh.write("\n")
        print(json.dumps(reg["summary"], indent=1))
        return 0 if not (reg["summary"]["mismatch"] or reg["summary"]["unresolved"]) else 1
    if a.retry_consensus:
        with open(OUT, encoding="utf-8") as fh:
            reg = json.load(fh)
        by_key = {e["key"]: e for e in entries}
        for r in reg["entries"]:
            c = r.get("consensus", {})
            if r["verdict"] == "verified" and not c.get("agrees") and "Error" in c.get("reason", ""):
                time.sleep(5)
                try:
                    r["consensus"] = q_openalex(by_key[r["key"]], r["record"])
                except Exception as ex:  # recorded in the registry
                    r["consensus"] = {"source": "OpenAlex", "agrees": False,
                                      "reason": f"{type(ex).__name__}: {ex}"}
                print(f"{r['key']:<20} agrees={r['consensus'].get('agrees')}", flush=True)
        date = reg["summary"]["date"]
        reg["summary"] = summary(reg["entries"])
        reg["summary"]["date"] = date
        with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(reg, fh, indent=1)
            fh.write("\n")
        print(json.dumps(reg["summary"], indent=1))
        return 0
    if a.offline:
        bad = offline(entries)
        print("all entries match their verified records" if not bad else bad)
        return 0 if not bad else 1
    res = build(entries)
    summ = summary(res)
    summ["date"] = _dt.date.today().isoformat()
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"_definition": ("verification of every entry of "
                                   "paper/references.bib against a primary "
                                   "bibliographic index and OpenAlex"),
                   "summary": summ, "entries": res}, fh, indent=1)
        fh.write("\n")
    print(json.dumps(summ, indent=1))
    return 0 if not (summ["mismatch"] or summ["unresolved"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
