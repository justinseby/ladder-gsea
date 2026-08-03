"""
annotator.py — LADDERAnnotator, the main public class.

Wires annotation.py, pubmed.py, and validation.py into two clean methods:

    Single gene set:
        result = annotator.run(genes=["TP53", "FLT3", "DNMT3A"])

    Multiple gene sets (communities / batch):
        results = annotator.run_batch(
            communities={"1": ["TP53", "MDM2"], "2": ["FLT3", "KIT"]}
        )

Both return LADDERResult (or a list of them) — a single dataclass
that carries every field from annotation + validation in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ladder.config     import LADDERConfig
from ladder.annotation import annotate, AnnotationResult
from ladder.pubmed     import fetch_papers
from ladder.validation import validate, ValidationResult


# ── Combined result type ──────────────────────────────────────────────────────

@dataclass
class LADDERResult:
    """
    Complete output for one gene set — annotation + pubmed + validation
    combined into a single object.

    Annotation fields
    -----------------
    genes                      : Input gene list
    enrichment_pathways        : Enrichr pathways (empty if gseapy not installed)
    proc_with / conf_with      : With-enrichment annotation (pre-validation)
    proc_without / conf_without: Without-enrichment annotation (pre-validation)
    analysis_with              : Detailed analysis text — with-enrichment arm
    analysis_without           : Detailed analysis text — without-enrichment arm
    reasoning_with             : Pathway reasoning — with-enrichment arm
    reasoning_without          : Pathway reasoning — without-enrichment arm
    contributing_genes_with    : Contributing genes — with-enrichment arm
    contributing_genes_without : Contributing genes — without-enrichment arm

    Validation fields
    -----------------
    conf_with_after            : Updated confidence — with-enrichment arm
    conf_without_after         : Updated confidence — without-enrichment arm
    final_process              : Literature-validated winning process name
    final_conf                 : Validated confidence score
    conflict                   : True if contradictory evidence found
    conflict_desc              : Description of the conflict (or "No conflicts detected")
    citations                  : Supporting citation strings
    validation_text            : Full validation summary

    PubMed fields
    -------------
    papers                     : List of paper dicts from PubMed
    papers_total               : Total papers retrieved
    papers_hq                  : Papers from high-quality journals (JIF ≥ 4)
    papers_fulltext            : Papers with PMC full text retrieved

    Debug fields
    ------------
    raw_annotation             : Full LLM response from annotation step
    raw_validation             : Full LLM response from validation step
    gene_set_id                : Label / community ID (set by run_batch)
    """

    # Annotation
    genes:                      List[str]
    enrichment_pathways:        List[str]
    proc_with:                  str
    conf_with:                  float
    proc_without:               str
    conf_without:               float
    analysis_with:              str
    analysis_without:           str
    reasoning_with:             str
    reasoning_without:          str
    contributing_genes_with:    str
    contributing_genes_without: str

    # Validation
    conf_with_after:            float
    conf_without_after:         float
    final_process:              str
    final_conf:                 float
    conflict:                   bool
    conflict_desc:              str
    citations:                  List[str]
    validation_text:            str

    # PubMed
    papers:                     List[Dict]
    papers_total:               int
    papers_hq:                  int
    papers_fulltext:            int

    # Debug / traceability
    raw_annotation:             str
    raw_validation:             str
    gene_set_id:                str = "1"


# ── Main class ────────────────────────────────────────────────────────────────

class LADDERAnnotator:
    """
    Main entry point for the LADDER pipeline.

    Usage
    -----
        from ladder import LADDERAnnotator, LADDERConfig

        config = LADDERConfig(
            llm_provider   = "openai",
            llm_api_key    = "sk-...",
            pubmed_context = "Acute Myeloid Leukemia, AML",
            ncbi_email     = "you@email.com",
        )

        annotator = LADDERAnnotator(config)

        # Single gene set
        result = annotator.run(genes=["TP53", "FLT3", "DNMT3A", "MDM2"])

        # Multiple gene sets
        results = annotator.run_batch({
            "community_1": ["TP53", "MDM2", "CDKN1A"],
            "community_2": ["FLT3", "KIT",  "PDGFRA"],
        })
    """

    def __init__(self, config: LADDERConfig) -> None:
        self.config = config

    # ── Single gene set ───────────────────────────────────────────────────────

    def run(
        self,
        genes:       List[str],
        gene_set_id: str = "1",
        verbose:     bool = True,
    ) -> LADDERResult:
        """
        Run the full LADDER pipeline on a single gene set.

        Parameters
        ----------
        genes       : List of gene symbols, e.g. ["TP53", "FLT3", "DNMT3A"]
        gene_set_id : Optional label for this gene set (used in run_batch)
        verbose     : Print progress to stdout (default True)

        Returns
        -------
        LADDERResult
        """
        if not genes:
            raise ValueError("genes list cannot be empty.")

        _log(verbose, f"[{gene_set_id}] Step 1/3 — Annotation ({len(genes)} genes)…")
        ann: AnnotationResult = annotate(genes, self.config)

        _log(verbose, f"[{gene_set_id}] Step 2/3 — PubMed retrieval…")
        papers = fetch_papers(genes, self.config)
        _log(
            verbose,
            f"[{gene_set_id}]            {len(papers)} papers · "
            f"{sum(1 for p in papers if p['is_top_journal'])} HQ · "
            f"{sum(1 for p in papers if p.get('full_text'))} full text",
        )

        _log(verbose, f"[{gene_set_id}] Step 3/3 — Validation…")
        val: ValidationResult = validate(ann, papers, self.config)

        _log(
            verbose,
            f"[{gene_set_id}] ✔ Done  —  {val.final_process}  "
            f"(conf {val.final_conf:.2f})"
            + ("  ⚠ conflict" if val.conflict else ""),
        )

        return _merge(ann, val, papers, gene_set_id)

    # ── Batch / communities ───────────────────────────────────────────────────

    def run_batch(
        self,
        communities: Dict[str, List[str]],
        verbose:     bool = True,
    ) -> List[LADDERResult]:
        """
        Run the full LADDER pipeline on multiple gene sets.

        Parameters
        ----------
        communities : Dict mapping a label → list of gene symbols
                      e.g. {"community_1": ["TP53", "MDM2"], ...}
        verbose     : Print progress to stdout (default True)

        Returns
        -------
        List[LADDERResult] — one per community, in input order
        """
        if not communities:
            raise ValueError("communities dict cannot be empty.")

        total   = len(communities)
        results = []

        for idx, (gene_set_id, genes) in enumerate(communities.items(), 1):
            _log(verbose, f"\n── Gene set {idx}/{total}  [{gene_set_id}] ──")
            result = self.run(genes, gene_set_id=gene_set_id, verbose=verbose)
            results.append(result)

        _log(verbose, f"\n✔ Batch complete — {total} gene sets annotated.")
        return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge(
    ann:        AnnotationResult,
    val:        ValidationResult,
    papers:     List[Dict],
    gene_set_id: str,
) -> LADDERResult:
    """Flatten AnnotationResult + ValidationResult into one LADDERResult."""
    return LADDERResult(
        # Annotation
        genes                      = ann.genes,
        enrichment_pathways        = ann.enrichment_pathways,
        proc_with                  = ann.proc_with,
        conf_with                  = ann.conf_with,
        proc_without               = ann.proc_without,
        conf_without               = ann.conf_without,
        analysis_with              = ann.analysis_with,
        analysis_without           = ann.analysis_without,
        reasoning_with             = ann.reasoning_with,
        reasoning_without          = ann.reasoning_without,
        contributing_genes_with    = ann.contributing_genes_with,
        contributing_genes_without = ann.contributing_genes_without,
        # Validation
        conf_with_after            = val.conf_with_after,
        conf_without_after         = val.conf_without_after,
        final_process              = val.final_process,
        final_conf                 = val.final_conf,
        conflict                   = val.conflict,
        conflict_desc              = val.conflict_desc,
        citations                  = val.citations,
        validation_text            = val.validation_text,
        # PubMed
        papers                     = papers,
        papers_total               = len(papers),
        papers_hq                  = sum(1 for p in papers if p["is_top_journal"]),
        papers_fulltext            = sum(1 for p in papers if p.get("full_text")),
        # Debug
        raw_annotation             = ann.raw_response,
        raw_validation             = val.raw_response,
        gene_set_id                = gene_set_id,
    )


def _log(verbose: bool, msg: str) -> None:
    """Print progress message if verbose is True."""
    if verbose:
        print(msg)