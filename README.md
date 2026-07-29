# LADDER · `ladder-gsea`

**Literature-Assisted Dual-annotation and Documentation & Evidence-based Reasoning**

A Python package for LLM-powered gene set annotation and literature validation, designed for computational biology workflows in cancer genomics and multi-omics research.
You can install the package from PyPI: https://pypi.org/project/ladder-gsea/

---

## Overview

LADDER annotates gene sets by running a **dual- LLM annotation** — one is using pathway enrichment results (GO, Reactome, KEGG via Enrichr), one using only the model's prior knowledge — then retrieves relevant PubMed literature and runs a second LLM call to **validate and adjudicate** between the two annotations. The result is a single confidence-scored, literature-supported pathway annotation per gene set.

The pipeline has three stages:

```
Gene set → [1] Dual Annotation (LLM + Enrichr)
         → [2] PubMed Retrieval (abstracts + PMC full text)
         → [3] Literature Validation (LLM)
         → LADDERResult
```

---

## Installation

```bash
pip install ladder-gsea
```

---

## Quick Start

```python
from ladder import LADDERAnnotator, LADDERConfig

config = LADDERConfig(
    llm_provider   = "openai",          # "openai" | "deepseek" | "anthropic"
    llm_api_key    = "sk-...",
    pubmed_context = "Acute Myeloid Leukemia, AML",
    ncbi_email     = "you@gmail.com",
)

annotator = LADDERAnnotator(config)

result = annotator.run(genes=["TP53", "FLT3", "DNMT3A", "MDM2", "PIM1"])

print(result.final_process)   # e.g. "Transcriptional Regulation and Apoptosis in AML"
print(result.final_conf)      # e.g. 0.87
print(result.conflict)        # True/False — conflicting evidence in the literature
```

---

## Batch Annotation

Annotate multiple gene sets (e.g. network communities) in one call:

```python
results = annotator.run_batch({
    "community_1": ["TP53", "MDM2", "CDKN1A", "BAX"],
    "community_2": ["FLT3", "KIT", "PDGFRA", "PDGFD"],
    "community_3": ["HOXA9", "HOXA5", "PBX3", "MEIS1"],
})

for r in results:
    print(f"{r.gene_set_id}: {r.final_process} (conf={r.final_conf:.2f})")
```

---

## Configuration

All options are set in `LADDERConfig`:

```python
config = LADDERConfig(
    # Required
    llm_provider   = "openai",                    # LLM backend
    llm_api_key    = "sk-...",                    # API key for chosen provider
    pubmed_context = "Acute Myeloid Leukemia, AML, leukemia",  # Disease/context terms
    ncbi_email     = "you@institution.edu",       # Required by NCBI E-utilities policy

    # Optional
    llm_model      = "gpt-4o",                   # Defaults: gpt-4o / deepseek-chat / claude-sonnet-4-20250514
    ncbi_api_key   = "...",                       # Free NCBI key — raises rate limit from 3 to 10 req/s
    max_papers     = 50,                          # Max PubMed papers per gene set
    max_tokens_ann = 4000,                        # Token budget for annotation call
    max_tokens_val = 8000,                        # Token budget for validation call
    system_prompt  = "You are a bioinformatics expert focused on AML...",  # Override default
)
```

**Getting an NCBI API key** (free, recommended): [https://www.ncbi.nlm.nih.gov/account/](https://www.ncbi.nlm.nih.gov/account/)

---

## Result Fields

`LADDERResult` contains the full output of all three pipeline stages:

### Annotation

| Field                                                    | Description                                           |
| -------------------------------------------------------- | ----------------------------------------------------- |
| `genes`                                                  | Input gene list                                       |
| `enrichment_pathways`                                    | Enrichr pathways used (empty if gseapy not installed) |
| `proc_with` / `conf_with`                                | Annotation from the enrichment-guided arm             |
| `proc_without` / `conf_without`                          | Annotation from the knowledge-only arm                |
| `analysis_with` / `analysis_without`                     | Detailed LLM analysis text for each arm               |
| `reasoning_with` / `reasoning_without`                   | Pathway reasoning for each arm                        |
| `contributing_genes_with` / `contributing_genes_without` | Key genes per arm                                     |

### Validation

| Field                                    | Description                                              |
| ---------------------------------------- | -------------------------------------------------------- |
| `final_process`                          | Literature-validated winning process name                |
| `final_conf`                             | Final confidence score (0–1)                             |
| `conf_with_after` / `conf_without_after` | Updated confidence scores after literature review        |
| `conflict`                               | `True` if contradictory evidence was found across papers |
| `conflict_desc`                          | Description of the conflict (or "No conflicts detected") |
| `citations`                              | Supporting citation strings from the validation step     |
| `validation_text`                        | Full LLM validation summary                              |

### PubMed

| Field             | Description                                |
| ----------------- | ------------------------------------------ |
| `papers`          | List of retrieved paper dicts              |
| `papers_total`    | Total papers retrieved                     |
| `papers_hq`       | Papers from high-impact journals (JIF ≥ 4) |
| `papers_fulltext` | Papers with PMC full text retrieved        |

Each paper dict includes: `title`, `authors`, `journal`, `year`, `pmid`, `pmcid`, `abstract`, `full_text`, `genes_mentioned`, `gene_count`, `is_top_journal`.

---

## Journal Quality Filtering

LADDER ships with a curated list of journals with Impact Factor ≥ 4 (sourced from Clarivate). Papers from these journals are ranked first during PubMed retrieval. This list is bundled at `ladder/data/journals_filtered_JIF_ge_4.csv` and loaded automatically — no setup needed.

---

## Requirements

- Python ≥ 3.9
- `requests >= 2.28`
- `pandas >= 1.5`
- An API key for at least one supported LLM provider
- An NCBI account email (required by NCBI E-utilities policy)

---

## Citation

If you use LADDER in your research, please cite:

> _LADDER: Literature-Assisted Dual-annotation and Documentation & Evidence-based Reasoning for gene set annotation_ (manuscript in preparation)

---

## License

MIT
