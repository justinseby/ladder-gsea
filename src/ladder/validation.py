"""
validation.py — literature validation of annotation results.

Single public entry point:

    result = validate(annotation=ann_result, papers=papers, config=config)

Takes the AnnotationResult from annotation.py and the paper list from
pubmed.py, runs a second LLM call that reads the abstracts and updates
both confidence scores, selects the better-supported process, and flags
any conflicting evidence across the retrieved papers.

Returns a ValidationResult dataclass.
If no papers are provided, returns a ValidationResult with scores
carried over from the annotation unchanged and a note explaining why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from ladder._llm import call_llm

if TYPE_CHECKING:
    from ladder.config import LADDERConfig
    from ladder.annotation import AnnotationResult


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Structured output from one validation call.

    Fields
    ------
    conf_with_before      : Confidence of with-enrichment arm before validation
    conf_with_after       : Confidence of with-enrichment arm after validation
    conf_without_before   : Confidence of without-enrichment arm before validation
    conf_without_after    : Confidence of without-enrichment arm after validation

    final_process         : Literature-validated winning process name
    final_conf            : Confidence of the validated winner

    conflict              : True if papers contain contradictory evidence
    conflict_desc         : Human-readable description of the conflict (or "No conflicts detected")

    citations             : Supporting citation strings for the final process
    validation_text       : Full validation summary from the LLM
    papers_used           : Number of papers passed to the LLM

    raw_response          : Full LLM response string (for debugging)
    """

    conf_with_before:    float
    conf_with_after:     float
    conf_without_before: float
    conf_without_after:  float

    final_process:       str
    final_conf:          float

    conflict:            bool
    conflict_desc:       str

    citations:           List[str]
    validation_text:     str
    papers_used:         int

    raw_response:        str


# ── Public entry point ────────────────────────────────────────────────────────

def validate(
    annotation: "AnnotationResult",
    papers:     List[Dict],
    config,
) -> ValidationResult:
    """
    Validate an AnnotationResult against a list of PubMed papers.

    Steps:
        1. If no papers, return ValidationResult carrying annotation
           scores forward unchanged
        2. Build validation prompt from annotation fields + paper abstracts
        3. Call LLM once via call_llm()
        4. Parse structured fields from the response
        5. Return ValidationResult
    """
    # Step 1 — no papers: pass-through with explanation
    if not papers:
        final = (
            annotation.proc_with
            if annotation.conf_with >= annotation.conf_without
            else annotation.proc_without
        )
        return ValidationResult(
            conf_with_before    = annotation.conf_with,
            conf_with_after     = annotation.conf_with,
            conf_without_before = annotation.conf_without,
            conf_without_after  = annotation.conf_without,
            final_process       = final,
            final_conf          = max(annotation.conf_with, annotation.conf_without),
            conflict            = False,
            conflict_desc       = "No papers available — scores carried forward unchanged.",
            citations           = [],
            validation_text     = "No PubMed papers were retrieved for this gene set. "
                                  "Confidence scores reflect LLM annotation only.",
            papers_used         = 0,
            raw_response        = "",
        )

    # Step 2 — build prompt
    prompt = _build_prompt(annotation, papers)

    # Step 3 — LLM call
    raw = call_llm(
        config,
        system_prompt=config.system_prompt,
        user_prompt=prompt,
        max_tokens=config.max_tokens_val,
    )

    # Step 4 — parse
    parsed = _parse_response(raw, annotation)

    # Step 5 — return
    return ValidationResult(
        conf_with_before    = annotation.conf_with,
        conf_with_after     = parsed["conf_with_after"],
        conf_without_before = annotation.conf_without,
        conf_without_after  = parsed["conf_without_after"],
        final_process       = parsed["final_process"],
        final_conf          = parsed["final_conf"],
        conflict            = parsed["conflict"],
        conflict_desc       = parsed["conflict_desc"],
        citations           = parsed["citations"],
        validation_text     = parsed["validation_text"],
        papers_used         = len(papers),
        raw_response        = raw,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(annotation: "AnnotationResult", papers: List[Dict]) -> str:
    """Build the validation prompt from annotation fields and paper abstracts."""

    genes_str = ", ".join(annotation.genes)

    papers_section = ""
    for i, p in enumerate(papers, 1):
        papers_section += (
            f"PAPER {i}:\n"
            f"Title: {p['title']}\n"
            f"Authors: {p['authors']}\n"
            f"Journal: {p['journal']} ({p['year']})\n"
            f"Genes Mentioned: {', '.join(p['genes_mentioned'])}\n"
            f"Abstract: {p['abstract']}\n\n"
        )

    return f"""You are a scientific expert in genomics and bioinformatics tasked with \
validating gene set analysis results using ONLY the provided literature.

GENES: {genes_str}

ORIGINAL ANALYSIS WITH ENRICHMENT:
Process Name: {annotation.proc_with}
Original Confidence Score: {annotation.conf_with}
Analysis: {annotation.analysis_with[:1000]}...

ORIGINAL ANALYSIS WITHOUT ENRICHMENT:
Process Name: {annotation.proc_without}
Original Confidence Score: {annotation.conf_without}
Analysis: {annotation.analysis_without[:1000]}...

PROVIDED LITERATURE (USE ONLY THESE STUDIES):
{papers_section}

VALIDATION TASK:
Based STRICTLY on the provided literature above, evaluate both analyses and provide \
updated confidence scores.

**ABSOLUTE REQUIREMENT FOR PAPER CITATIONS:**
Throughout your entire response, whenever you reference any paper by number, you MUST \
immediately provide the complete paper information in this exact format:
"Paper X: 'Full Paper Title', Complete Author List"
NEVER use abbreviated citations like "Paper X: Author et al."
ALWAYS include the full title in quotes and complete author names.

CRITICAL REQUIREMENTS:
1. Use ONLY the provided papers — do not add external knowledge
2. For each process, check if the genes are supported by the literature
3. Provide updated confidence scores based on evidence strength
4. Select the better-supported process as the final choice
5. The final selected process MUST be either:
   (a) exactly one of the two original process names, OR
   (b) "Neither process" ONLY if both updated confidence scores are ≤ 0.05
6. Final Confidence = updated confidence of the selected process

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

VALIDATION OF ENRICHMENT ANALYSIS:
Evidence Assessment: [Detailed assessment based strictly on provided papers]
Original Confidence: {annotation.conf_with}
Updated Confidence: [Your revised score 0.00-1.00 based on literature evidence]
Supporting Papers: [List paper numbers that support this process]

VALIDATION OF DIRECT ANALYSIS:
Evidence Assessment: [Detailed assessment based strictly on provided papers]
Original Confidence: {annotation.conf_without}
Updated Confidence: [Your revised score 0.00-1.00 based on literature evidence]
Supporting Papers: [List paper numbers that support this process]

FINAL PROCESS SELECTION:
Selected Process: [Choose the better-supported process name]
Final Confidence: [The updated confidence score for your selected process]
Selection Reasoning: [Explain why this process has stronger literature support]

CONFLICT ANALYSIS:
CONFLICTING_EVIDENCE_FOUND: [TRUE/FALSE]
CONFLICT_DESCRIPTION: [Brief description of any conflicts found, or "No conflicts detected"]

SUPPORTING CITATIONS:
[List citations as: "Title, Authors" for papers supporting the final process]

VALIDATION ANALYSIS TEXT:
[Comprehensive summary — reasoning for confidence adjustments and evidence from the \
provided studies that led to the final process selection.
MANDATORY: When mentioning paper numbers include COMPLETE citation with FULL TITLES \
AND AUTHORS.]
"""


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str, annotation: "AnnotationResult") -> Dict:
    """
    Parse all structured fields from the validation LLM response.
    Every field falls back gracefully if the pattern is not matched —
    annotation scores are used as fallbacks so nothing is ever None.
    """
    out: Dict = {}

    # Updated confidence — with-enrichment arm
    m = re.search(
        r"VALIDATION OF ENRICHMENT ANALYSIS:.*?Updated Confidence:\s*([\d.]+)",
        raw, re.DOTALL,
    )
    out["conf_with_after"] = float(m.group(1)) if m else annotation.conf_with

    # Updated confidence — without-enrichment arm
    m = re.search(
        r"VALIDATION OF DIRECT ANALYSIS:.*?Updated Confidence:\s*([\d.]+)",
        raw, re.DOTALL,
    )
    out["conf_without_after"] = float(m.group(1)) if m else annotation.conf_without

    # Final selected process
    m = re.search(r"Selected Process:\s*(.+?)(?=\n|$)", raw)
    out["final_process"] = m.group(1).strip() if m else annotation.final_process

    # Final confidence
    m = re.search(r"Final Confidence:\s*([\d.]+)", raw)
    out["final_conf"] = float(m.group(1)) if m else annotation.final_conf

    # Conflict flag
    m = re.search(r"CONFLICTING_EVIDENCE_FOUND:\s*(TRUE|FALSE)", raw, re.IGNORECASE)
    out["conflict"] = (m.group(1).upper() == "TRUE") if m else False

    # Conflict description
    m = re.search(
        r"CONFLICT_DESCRIPTION:\s*(.+?)(?=\n\n|SUPPORTING CITATIONS:|$)",
        raw, re.DOTALL,
    )
    out["conflict_desc"] = m.group(1).strip() if m else "No conflicts detected"

    # Supporting citations
    m = re.search(
        r"SUPPORTING CITATIONS:\s*(.+?)(?=\n\n|VALIDATION ANALYSIS TEXT:|$)",
        raw, re.DOTALL,
    )
    if m:
        out["citations"] = [
            line.strip()
            for line in m.group(1).strip().split("\n")
            if line.strip()
        ]
    else:
        out["citations"] = []

    # Validation summary text
    m = re.search(r"VALIDATION ANALYSIS TEXT:\s*(.+?)$", raw, re.DOTALL)
    out["validation_text"] = m.group(1).strip() if m else raw

    return out