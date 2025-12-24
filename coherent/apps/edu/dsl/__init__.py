"""DSL helpers for MathLang Edu edition."""

from coherent.core.parser import Parser as CoreParser


class EduParser(CoreParser):
    """Thin wrapper for future Edu-specific DSL extensions."""


__all__ = ["EduParser"]
