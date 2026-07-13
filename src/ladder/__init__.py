"""
LADDER · ladder-gsea
Literature-Assisted Dual-annotation and Documentation & Evidence-based Reasoning
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ladder-gsea")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from ladder.config    import LADDERConfig
from ladder.annotator import LADDERAnnotator, LADDERResult

__all__ = [
    "LADDERConfig",
    "LADDERAnnotator",
    "LADDERResult",
    "__version__",
]