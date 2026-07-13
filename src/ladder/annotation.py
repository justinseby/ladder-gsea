"""
annotation.py — dual LLM annotation for a gene set.

Single public entry point:

    result = annotate(genes=["TP53", "FLT3", "DNMT3A"], config=config)

Returns an AnnotationResult dataclass containing both the
with-enrichment and without-enrichment annotations, plus
the raw LLM response for debugging.

Enrichment via gseapy is optional — if gseapy is not installed
the with-enrichment path is skipped gracefully and
PROCESS WITH ENRICHMENT is set to "Unknown Pathway" (confidence 0.00),
exactly as the prompt instructs the LLM to do when no pathways exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from ladder._llm import call_llm

if TYPE_CHECKING:
    from ladder.config import LADDERConfig


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class AnnotationResult:
    """
    Structured output from one annotation call.

    Fields
    ------
    genes                    : The input gene list
    enrichment_pathways      : Pathways from Enrichr (empty if gseapy absent)

    proc_with                : Process name — with-enrichment arm
    conf_with                : Confidence score — with-enrichment arm (0–1)
    analysis_with            : Detailed analysis text — with-enrichment arm
    reasoning_with           : Pathway reasoning — with-enrichment arm
    contributing_genes_with  : Contributing genes — with-enrichment arm

    proc_without             : Process name — without-enrichment arm
    conf_without             : Confidence score — without-enrichment arm (0–1)
    analysis_without         : Detailed analysis text — without-enrichment arm
    reasoning_without        : Pathway reasoning — without-enrichment arm
    contributing_genes_without : Contributing genes — without-enrichment arm

    final_process            : Pre-validation winner (higher confidence arm)
    final_conf               : Confidence of the pre-validation winner

    raw_response             : Full LLM response string (for debugging)
    """

    genes:                      List[str]
    enrichment_pathways:        List[str]

    proc_with:                  str
    conf_with:                  float
    analysis_with:              str
    reasoning_with:             str
    contributing_genes_with:    str

    proc_without:               str
    conf_without:               float
    analysis_without:           str
    reasoning_without:          str
    contributing_genes_without: str

    final_process:              str
    final_conf:                 float

    raw_response:               str


# ── Public entry point ────────────────────────────────────────────────────────

def annotate(genes: List[str], config) -> AnnotationResult:
    """
    Run dual LLM annotation on a gene set.

    Steps:
        1. Attempt enrichment via gseapy (Reactome + KEGG + GO BP)
           — silently skipped if gseapy is not installed
        2. Build unified prompt covering both analysis arms
        3. Call LLM once via call_llm()
        4. Parse structured fields from the response
        5. Pick pre-validation winner (higher confidence arm)
        6. Return AnnotationResult
    """
    # Step 1 — enrichment (optional)
    enrichment_pathways = _run_enrichment(genes)

    # Step 2 — build prompt
    prompt = _build_prompt(genes, enrichment_pathways)

    # Step 3 — LLM call
    raw = call_llm(
        config,
        system_prompt=config.system_prompt,
        user_prompt=prompt,
        max_tokens=config.max_tokens_ann,
    )

    # Step 4 — parse
    proc_with,    conf_with    = _extract_process(raw, "WITH")
    proc_without, conf_without = _extract_process(raw, "WITHOUT")

    analysis_with    = _extract_section(raw, "ANALYSIS TEXT WITH ENRICHMENT")
    analysis_without = _extract_section(raw, "ANALYSIS TEXT WITHOUT ENRICHMENT")
    reasoning_with   = _extract_section(raw, "PATHWAY REASONING WITH ENRICHMENT")
    reasoning_without= _extract_section(raw, "PATHWAY REASONING WITHOUT ENRICHMENT")
    genes_with       = _extract_section(raw, "CONTRIBUTING GENES WITH ENRICHMENT")
    genes_without    = _extract_section(raw, "CONTRIBUTING GENES WITHOUT ENRICHMENT")

    # Step 5 — pre-validation winner
    if conf_with >= conf_without:
        final_process = proc_with
        final_conf    = conf_with
    else:
        final_process = proc_without
        final_conf    = conf_without

    return AnnotationResult(
        genes                      = genes,
        enrichment_pathways        = enrichment_pathways,
        proc_with                  = proc_with,
        conf_with                  = conf_with,
        analysis_with              = analysis_with,
        reasoning_with             = reasoning_with,
        contributing_genes_with    = genes_with,
        proc_without               = proc_without,
        conf_without               = conf_without,
        analysis_without           = analysis_without,
        reasoning_without          = reasoning_without,
        contributing_genes_without = genes_without,
        final_process              = final_process,
        final_conf                 = final_conf,
        raw_response               = raw,
    )


# ── Enrichment ────────────────────────────────────────────────────────────────

def _run_enrichment(genes: List[str]) -> List[str]:
    """
    Run Enrichr across GO BP, Reactome, and KEGG.
    Returns a deduplicated list of significant pathway terms (adj p < 0.01).
    Returns an empty list — without raising — if:
        - gseapy is not installed
        - the gene list is too small
        - the Enrichr API is unreachable
    """
    try:
        import gseapy as gp
    except ImportError:
        return []   # gseapy optional — caller handles empty list

    try:
        import time
        databases = [
            "GO_Biological_Process_2021",
            "Reactome_2022",
            "KEGG_2021_Human",
        ]
        pathways: List[str] = []
        for db in databases:
            time.sleep(1.2)   # polite gap between Enrichr calls
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=[db],
                organism="human",
                outdir=None,
                no_plot=True,
                cutoff=0.01,
            )
            df = enr.results
            if not df.empty:
                df = df.sort_values("Adjusted P-value")
                pathways += list(df.head(5)["Term"])

        return list(set(pathways))

    except Exception:
        return []


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(genes: List[str], enrichment_pathways: List[str]) -> str:
    """
    Build the unified annotation prompt covering both analysis arms.
    If enrichment_pathways is empty the LLM is instructed to set
    PROCESS WITH ENRICHMENT to 'Unknown Pathway' with confidence 0.00.
    """
    genes_str = ", ".join(genes)

    enrichment_section = ""
    if enrichment_pathways:
        enrichment_section = "\nEnrichment Analysis Results:\n" + \
            "\n".join(f"- {p}" for p in enrichment_pathways)

    enrichment_instruction = (
        "Using the enrichment pathways above (if any), provide an analysis of the gene set"
    )

    return f"""You are a bioinformatics expert conducting pathway analysis of gene sets. \
For the gene set: [{genes_str}]{enrichment_section}

Please perform analysis in clearly labeled sections:

===== SECTION 1: ANALYSIS WITH ENRICHMENT =====
{enrichment_instruction}:
1. Propose a concise, descriptive name for the biological process
2. Assign a confidence score (0.00-1.00) for this process
3. Provide explicit reasoning for choosing this pathway/process
4. List the specific genes from the gene set that contribute to the process
5. IMPORTANT: A minimum of TWO genes from the gene set MUST be associated with a common \
biological process to annotate it as a valid pathway
6. If there isn't sufficient evidence or fewer than two genes are associated with a common \
pathway, label as "Unknown Pathway" with a low confidence score
7. If no enrichment pathways are provided, you MUST set PROCESS WITH ENRICHMENT to \
"Unknown Pathway." and assign confidence to 0.00

Output format for this section:
PROCESS WITH ENRICHMENT: [Process Name] ([Confidence Score])

PATHWAY REASONING WITH ENRICHMENT:
[Your explicit reasoning for choosing this pathway/process]

CONTRIBUTING GENES WITH ENRICHMENT:
[Comma-separated list of contributing genes]

ANALYSIS TEXT WITH ENRICHMENT:
[Detailed analysis text explaining the biological significance and mechanisms of this \
process, without citations]

===== SECTION 2: ANALYSIS WITHOUT ENRICHMENT =====
Based SOLELY on your knowledge of these genes, without considering the enrichment results:
1. Propose a concise, descriptive name for the biological process
2. Assign a confidence score (0.00-1.00) for this process
3. List the specific genes from the gene set that are involved in the process
4. IMPORTANT: A minimum of TWO genes from the gene set MUST be associated with a common \
biological process to annotate it as a valid pathway
5. If there isn't sufficient evidence or fewer than two genes share a common biological \
function, label as "Unknown Pathway" with a low confidence score

Output format for this section:
PROCESS WITHOUT ENRICHMENT: [Process Name] ([Confidence Score])

CONTRIBUTING GENES WITHOUT ENRICHMENT:
[Comma-separated list of contributing genes]

PATHWAY REASONING WITHOUT ENRICHMENT:
[Your explicit reasoning for choosing this pathway/process]

ANALYSIS TEXT WITHOUT ENRICHMENT:
[Detailed analysis text explaining the biological significance and mechanisms of this \
process, without citations]

===== SECTION 3: FINAL PROCESS SELECTION =====
Compare both analyses and provide:
1. Which process (with or without enrichment) has higher confidence and why
2. Final reasoning for the chosen process

Output format for this section:
FINAL PROCESS REASONING:
[Detailed explanation of why you chose the final process, comparing both analyses and \
explaining which one provides stronger evidence]

Analytical Guidelines:
- Be concise and avoid unnecessary words
- Be factual without editorializing
- Be specific, avoiding overly general statements
- Avoid listing individual protein facts
- Group proteins by similar functions
- Discuss their interplay, synergistic or antagonistic effects
- Focus on functional integration within the system
- Include at least 2-3 specific paper citations when discussing established pathway relationships

Confidence Score Instructions:
- Assign a score from 0.00 to 1.00
- 0.00 indicates lowest confidence
- 1.00 reflects highest confidence
- Base the score on the proportion of genes participating in the identified process"""


# ── Response parsers ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip markdown bold/italic formatting from LLM output."""
    if not text or not isinstance(text, str):
        return text or ""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*",     r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\w)", "",    text)
    return text.strip()


def _extract_process(text: str, prefix: str) -> Tuple[str, float]:
    """
    Extract process name and confidence score from a WITH or WITHOUT arm.
    Tries multiple regex patterns to handle LLM formatting variation.
    Falls back to (name, 0.0) if the confidence is unparseable.
    """
    patterns = [
        rf"PROCESS {prefix} ENRICHMENT:\s*([^(]+?)\s*\(([0-9.]+)\)",
        rf"PROCESS {prefix} ENRICHMENT:\s*([^(]*?)\s*\(\s*([0-9.]+)\s*\)",
        rf"PROCESS {prefix} ENRICHMENT:\s*(.*?)\s*\(\s*Confidence Score:\s*([0-9.]+)\s*\)",
        rf"PROCESS {prefix} ENRICHMENT:\s*(.*?)\s*\*\*Confidence Score:\s*([0-9.]+)\*\*",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            name = _clean(m.group(1).strip())
            try:
                conf = float(m.group(2))
                if name and conf >= 0:
                    return name, conf
            except ValueError:
                continue

    # Fallback — extract name only, confidence defaults to 0.0
    m = re.search(rf"PROCESS {prefix} ENRICHMENT:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    if m:
        name = _clean(re.sub(r"\s*\([^)]*$", "", m.group(1)).strip())
        return name, 0.0

    return f"Unknown Process {prefix} Enrichment", 0.0


def _extract_section(text: str, section_name: str) -> str:
    """
    Extract a named section from the LLM response.
    Tries progressively looser patterns before giving up.
    """
    patterns = [
        rf"{re.escape(section_name)}:\s*(.*?)(?=\n\n[\w\s]+:|===|$)",
        rf"{re.escape(section_name)}:\s*(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
        rf"{re.escape(section_name)}:\s*(.*?)(?=\n\*\*|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            result = m.group(1).strip()
            if result:
                return _clean(result)
    return f"{section_name} not found"