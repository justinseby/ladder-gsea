"""
pubmed.py — PubMed retrieval and journal quality filtering.

Single public entry point:

    papers = fetch_papers(genes=["TP53", "FLT3"], config=config)

Returns a list of paper dicts, sorted high-quality journals first,
then by number of query genes mentioned in the abstract/full text.

Each paper dict contains:
    title, authors, journal, year, pmid, pmcid,
    issn, eissn, abstract, full_text,
    genes_mentioned, gene_count, is_top_journal
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import requests

if TYPE_CHECKING:
    from ladder.config import LADDERConfig

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# ── Journal filter — loaded once from bundled CSV ─────────────────────────────

def _load_journal_filter() -> Tuple[Set[str], Set[str]]:
    """
    Load the bundled Clarivate JIF CSV and return two sets:
        (hq_issn_set, hq_eissn_set)
    Both sets contain normalised ISSN strings (no hyphens, uppercase).
    Returns empty sets if the CSV is missing — journal filtering is
    silently disabled, nothing crashes.
    """
    try:
        from importlib.resources import files
        import pandas as pd

        csv_path = files("ladder.data").joinpath("journals_filtered_JIF_ge_4.csv")
        df = pd.read_csv(str(csv_path))
        df.columns = [c.strip() for c in df.columns]

        issn_set: Set[str]  = set()
        eissn_set: Set[str] = set()

        for _, row in df.iterrows():
            issn  = _norm_issn(row.get("ISSN",  ""))
            eissn = _norm_issn(row.get("eISSN", ""))
            if issn:  issn_set.add(issn)
            if eissn: eissn_set.add(eissn)

        return issn_set, eissn_set

    except Exception:
        return set(), set()


# Load once at module import — not on every function call
_HQ_ISSN_SET, _HQ_EISSN_SET = _load_journal_filter()


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_papers(genes: List[str], config) -> List[Dict]:
    """
    Fetch PubMed papers for a gene set and return them ranked.

    Steps:
        1. Build a PubMed query from gene names + config.pubmed_context
        2. eSearch → get up to 300 PMIDs sorted by relevance
        3. eFetch → parse abstracts, journal metadata, ISSNs
        4. For open-access papers: fetch PMC full text, re-score gene count
        5. Sort: high-quality journals first, then by gene_count descending
        6. Return top config.max_papers results
    """
    query = _build_query(genes, config.pubmed_context)
    pmids = _esearch(query, config)

    if not pmids:
        return []

    root   = _efetch(pmids, config)
    papers = _parse_articles(root, genes)

    # PMC full-text enrichment
    for paper in papers:
        if paper.get("pmcid"):
            time.sleep(0.35)
            ft = _fetch_pmc_fulltext(paper["pmcid"], config)
            if ft:
                paper["full_text"] = ft[:3000] + ("…" if len(ft) > 3000 else "")
                corpus = paper["abstract"] + " " + ft
                paper["genes_mentioned"] = _find_genes(genes, corpus)
                paper["gene_count"]      = len(paper["genes_mentioned"])

    # Sort: HQ journals first, then by gene count descending
    papers.sort(key=lambda p: (0 if p["is_top_journal"] else 1, -p["gene_count"]))

    return papers[: config.max_papers]


# ── Query builder ─────────────────────────────────────────────────────────────

def _build_query(genes: List[str], pubmed_context: str) -> str:
    """Build a PubMed boolean query combining gene names and context terms."""
    gene_terms = " OR ".join(f'"{g}"[Title/Abstract]' for g in genes)

    ctx_parts = [
        re.sub(r"[()]+", "", c).strip()
        for c in pubmed_context.split(",")
        if re.sub(r"[()]+", "", c).strip()
    ]
    ctx_terms = " OR ".join(f'"{c}"[Title/Abstract]' for c in ctx_parts)

    today = date.today().strftime("%Y/%m/%d")

    return (
        f"({gene_terms}) AND ({ctx_terms})"
        f' AND ("2015/01/01"[PDAT] : "{today}"[PDAT])'
    )


# ── NCBI helpers ──────────────────────────────────────────────────────────────

def _ncbi_params(config, extra: Dict) -> Dict:
    """Attach NCBI credentials to a params dict if available."""
    p = {**extra}
    if config.ncbi_email:
        p["email"] = config.ncbi_email
        p["tool"]  = "LADDER_GeneSetAnnotator"
    if config.ncbi_api_key:
        p["api_key"] = config.ncbi_api_key
    return p


def _esearch(query: str, config) -> List[str]:
    """Run eSearch and return a list of PMIDs."""
    try:
        time.sleep(0.34)
        r = requests.post(
            f"{NCBI_BASE}/esearch.fcgi",
            data=_ncbi_params(config, {
                "db":      "pubmed",
                "term":    query,
                "retmax":  300,
                "retmode": "json",
                "sort":    "relevance",
            }),
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def _efetch(pmids: List[str], config) -> Optional[ET.Element]:
    """Fetch full PubMed records for a list of PMIDs. Returns XML root."""
    try:
        time.sleep(0.34)
        r = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params=_ncbi_params(config, {
                "db":      "pubmed",
                "id":      ",".join(pmids[:300]),
                "rettype": "abstract",
                "retmode": "xml",
            }),
            timeout=60,
        )
        r.raise_for_status()
        return ET.fromstring(r.content)
    except Exception:
        return None


def _fetch_pmc_fulltext(pmcid: str, config) -> Optional[str]:
    """
    Fetch full-text XML from PubMed Central for a given PMCID.
    Returns concatenated body paragraphs, or None if unavailable.
    """
    try:
        r = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params=_ncbi_params(config, {
                "db":      "pmc",
                "id":      pmcid,
                "rettype": "full",
                "retmode": "xml",
            }),
            timeout=30,
        )
        if not r.ok:
            return None

        root = ET.fromstring(r.content)
        body = root.find(".//body")
        if body is None:
            return None

        paragraphs = []
        for elem in body.iter():
            if elem.tag in {"p", "title", "sec"}:
                txt = "".join(elem.itertext()).strip()
                if txt:
                    paragraphs.append(txt)

        full_text = "\n\n".join(paragraphs)
        return full_text if full_text else None

    except Exception:
        return None


# ── XML parser ────────────────────────────────────────────────────────────────

def _parse_articles(root: Optional[ET.Element], genes: List[str]) -> List[Dict]:
    """Parse a PubMed eFetch XML root into a list of paper dicts."""
    if root is None:
        return []

    seen:   set       = set()
    papers: List[Dict] = []

    for art in root.findall(".//PubmedArticle"):
        try:
            pm   = art.find(".//PMID")
            pmid = pm.text.strip() if pm is not None and pm.text else ""
            if pmid in seen:
                continue
            seen.add(pmid)

            # Abstract — skip papers with no abstract
            abs_parts = art.findall(".//AbstractText")
            abstract  = " ".join("".join(p.itertext()) for p in abs_parts).strip()
            if not abstract:
                continue

            te    = art.find(".//ArticleTitle")
            title = "".join(te.itertext()).strip() if te is not None else "No title"

            je      = art.find(".//Journal/Title")
            journal = je.text.strip() if je is not None and je.text else "Unknown"

            ye   = art.find(".//PubDate/Year")
            year = ye.text if ye is not None else "Unknown"

            # Authors (first 6 then "et al.")
            auth_els = art.findall(".//Author")
            names = []
            for a in auth_els[:6]:
                ln = a.findtext("LastName", "")
                fn = a.findtext("ForeName", "")
                if ln:
                    names.append(f"{ln} {fn}".strip())
            authors = ", ".join(names) + (" et al." if len(auth_els) > 6 else "")

            # ISSNs
            issn, eissn = "", ""
            for issn_el in art.findall(".//Journal/ISSN"):
                if issn_el.get("IssnType") == "Print":
                    issn  = issn_el.text or ""
                elif issn_el.get("IssnType") == "Electronic":
                    eissn = issn_el.text or ""

            # PMCID (for open-access full-text fetch)
            pmcid = ""
            for id_el in art.findall(".//ArticleIdList/ArticleId"):
                if id_el.get("IdType") == "pmc":
                    pmcid = id_el.text or ""

            mentioned = _find_genes(genes, abstract)

            paper = {
                "title":           title,
                "authors":         authors,
                "journal":         journal,
                "issn":            issn,
                "eissn":           eissn,
                "year":            year,
                "pmid":            pmid,
                "pmcid":           pmcid,
                "abstract":        abstract[:700] + ("…" if len(abstract) > 700 else ""),
                "full_text":       "",
                "genes_mentioned": mentioned,
                "gene_count":      len(mentioned),
                "is_top_journal":  _is_high_quality(issn, eissn),
            }
            papers.append(paper)

        except Exception:
            continue

    return papers


# ── Utilities ─────────────────────────────────────────────────────────────────

def _find_genes(genes: List[str], text: str) -> List[str]:
    """Return the subset of genes found as whole words in text."""
    return [
        g for g in genes
        if re.search(rf"\b{re.escape(g)}\b", text, re.IGNORECASE)
    ]


def _is_high_quality(issn: str, eissn: str) -> bool:
    """Return True if either ISSN matches the bundled JIF ≥ 4 list."""
    if _norm_issn(issn)  in _HQ_ISSN_SET  - {""}: return True
    if _norm_issn(eissn) in _HQ_EISSN_SET - {""}: return True
    return False


def _norm_issn(val) -> str:
    """Normalise an ISSN to uppercase, no hyphens."""
    if not val or (isinstance(val, float)):
        return ""
    return str(val).strip().replace("-", "").upper()