"""
LADDERConfig — everything the user configures, in one place.

Usage:
    from ladder import LADDERConfig

    config = LADDERConfig(
        llm_provider   = "openai",
        llm_api_key    = "sk-...",
        system_prompt  = "You are a bioinformatics expert in AML.",
        pubmed_context = "Acute Myeloid Leukemia, AML, leukemia",
        ncbi_email     = "you@email.com",
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# Default prompts — user can override entirely in LADDERConfig
_DEFAULT_SYSTEM_PROMPT = (
    "You are a bioinformatics expert. "
    "Provide detailed pathway analysis with scientific literature support. "
    "Always cite specific papers when discussing pathway relationships."
)


@dataclass
class LADDERConfig:
    """
    Single configuration object for the LADDER pipeline.

    Required fields
    ---------------
    llm_provider   : "openai" | "deepseek" | "anthropic"
    llm_api_key    : API key for the chosen provider
    pubmed_context : Comma-separated disease/context terms added to every
                     PubMed query alongside gene names.
                     e.g. "Acute Myeloid Leukemia, AML, leukemia"
    ncbi_email     : Your email — required by NCBI for all E-utilities calls.

    Optional fields
    ---------------
    llm_model      : Model string. Defaults are set per provider if omitted.
                     e.g. "gpt-4o", "deepseek-chat", "claude-sonnet-4-20250514"
    system_prompt  : Full system prompt sent to the LLM. Override to change
                     disease context, persona, or standing instructions.
    ncbi_api_key   : NCBI API key (free). Without it NCBI rate-limits to 3
                     requests/sec. With it: 10/sec. Get one at
                     https://www.ncbi.nlm.nih.gov/account/
    max_papers     : Max papers to retrieve per gene set. Default 50.
    max_tokens_ann : Max tokens for the annotation LLM call. Default 4000.
    max_tokens_val : Max tokens for the validation LLM call. Default 8000.
    """

    # ── Required ──────────────────────────────────────────────────────────────
    llm_provider:   str   # "openai" | "deepseek" | "anthropic"
    llm_api_key:    str
    pubmed_context: str   # e.g. "Acute Myeloid Leukemia, AML, leukemia"
    ncbi_email:     str

    # ── Optional with sensible defaults ───────────────────────────────────────
    llm_model:      Optional[str] = None   # resolved per provider if None
    system_prompt:  str = field(default=_DEFAULT_SYSTEM_PROMPT)
    ncbi_api_key:   Optional[str] = None
    max_papers:     int = 50
    max_tokens_ann: int = 4000
    max_tokens_val: int = 8000

    # ── Post-init validation ──────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Validate inputs and resolve default model strings."""

        valid_providers = {"openai", "deepseek", "anthropic"}
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"llm_provider must be one of {valid_providers}, "
                f"got '{self.llm_provider}'"
            )

        if not self.llm_api_key:
            raise ValueError("llm_api_key cannot be empty.")

        if not self.ncbi_email:
            raise ValueError(
                "ncbi_email is required by NCBI E-utilities policy. "
                "Use your real email — NCBI uses it only to contact you "
                "if your script causes problems."
            )

        if not self.pubmed_context:
            raise ValueError(
                "pubmed_context cannot be empty. "
                "Provide at least one disease or context term, "
                "e.g. 'Acute Myeloid Leukemia, AML'."
            )

        # Resolve default model per provider
        if self.llm_model is None:
            _defaults = {
                "openai":    "gpt-4o",
                "deepseek":  "deepseek-chat",
                "anthropic": "claude-sonnet-4-20250514",
            }
            self.llm_model = _defaults[self.llm_provider]

        if self.max_papers < 1:
            raise ValueError("max_papers must be at least 1.")