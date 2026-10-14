"""Natural-language questions over a pharmaceutical sales table, with guardrails."""

__version__ = "0.2.0"

from .pipeline import Answer, Explanation, execute, explain  # noqa: F401

__all__ = ["Answer", "Explanation", "execute", "explain", "__version__"]
