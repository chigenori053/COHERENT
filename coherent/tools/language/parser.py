import re
from typing import List, Optional, Tuple
from coherent.tools.language.models import (
    SemanticIR, IntentType, MathDomain, Goal, GoalType, 
    InputItem, InputItemType, Constraints, LanguageMeta
)

class SemanticParser:
    """
    Parses natural language input into Semantic Intermediate Representation (SIR).
    Combines rule-based and (future) LLM-based approaches.
    """
    
    def __init__(self):
        pass

    def parse(self, text: str) -> SemanticIR:
        """
        Main entry point for parsing natural language.
        """
        # L1: Surface Processing
        clean_text = self._l1_surface_processing(text)
        
        # L2: Semantic Extraction (Rule-based first)
        sir = self._l2_rule_based_parsing(clean_text)
        
        if sir:
            return sir
            
        # Fallback to default/unknown if rules fail (Future: LLM fallback)
        return self._create_fallback_sir(clean_text)

    def _l1_surface_processing(self, text: str) -> str:
        """
        Normalizes text, removes extra whitespace, handles basic symbols.
        """
        text = text.strip()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def _l2_rule_based_parsing(self, text: str) -> Optional[SemanticIR]:
        """
        Attempts to parse using regex patterns and keywords.
        """
        lower_text = text.lower()
        
        # Pattern 1: Explicit "Solve [expression]"
        # Matches: "solve x + 2 = 4", "solve for x: 2x = 10"
        if "solve" in lower_text or "calculate" in lower_text:
            expression = self._extract_expression(text, ["solve", "calculate", "for", ":"])
            if expression:
                return SemanticIR(
                    task=IntentType.SOLVE,
                    math_domain=self._infer_domain(expression),
                    goal=Goal(type=GoalType.FINAL_VALUE),
                    inputs=[InputItem(type=InputItemType.EQUATION, value=expression)],
                    language_meta=LanguageMeta(original_language="en") # Todo: Detect language
                )

        # Pattern 2: Explicit "Verify [expression]"
        if "verify" in lower_text or "check" in lower_text:
            expression = self._extract_expression(text, ["verify", "check", "that", ":"])
            if expression:
                return SemanticIR(
                    task=IntentType.VERIFY,
                    math_domain=self._infer_domain(expression), # Often logic or unknown
                    goal=Goal(type=GoalType.PROOF),
                    inputs=[InputItem(type=InputItemType.EQUATION, value=expression)],
                    language_meta=LanguageMeta(original_language="en")
                )

        # Pattern 3: Simple Math Expression Only "2 + 2"
        # If it looks like math, default to solving/simplifying
        if self._is_math_expression(text):
             return SemanticIR(
                task=IntentType.SOLVE,
                math_domain=MathDomain.ARITHMETIC,
                goal=Goal(type=GoalType.FINAL_VALUE),
                inputs=[InputItem(type=InputItemType.EXPRESSION, value=text)],
                language_meta=LanguageMeta(original_language="en")
            )

        return None

    def _extract_expression(self, text: str, keywords: List[str]) -> Optional[str]:
        """
        Simple heuristic to strip keywords and return the rest as expression.
        """
        # Very naive implementation for Phase 1
        lower_text = text.lower()
        
        # Sort keywords by length desc to remove longest first
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        cleaned = lower_text
        for kw in sorted_keywords:
            cleaned = cleaned.replace(kw, "")
            
        return cleaned.strip()

    def _infer_domain(self, expression: str) -> MathDomain:
        """
        Infers math domain based on symbols.
        """
        if any(c in expression for c in ['x', 'y', 'z', '=']):
            return MathDomain.ALGEBRA
        if any(c in expression for c in ['d/dx', 'integral', 'limit']):
            return MathDomain.CALCULUS
        return MathDomain.ARITHMETIC

    def _is_math_expression(self, text: str) -> bool:
        """
        Checks if text is primarily digits and math symbols.
        """
        return bool(re.match(r'^[\d\s\+\-\*\/\^\(\)\.=a-zA-Z]+$', text)) and any(c.isdigit() for c in text)

    def _create_fallback_sir(self, text: str) -> SemanticIR:
        """
        Creates a generic SIR when parsing fails.
        """
        return SemanticIR(
            task=IntentType.EXPLAIN, # Default to explanation if we don't understand
            math_domain=MathDomain.UNKNOWN,
            inputs=[InputItem(type=InputItemType.TEXT, value=text)],
            language_meta=LanguageMeta(original_language="en", detected_intent_confidence=0.0)
        )
