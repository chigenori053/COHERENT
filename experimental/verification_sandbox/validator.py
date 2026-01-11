"""
Validator for Verification Sandbox
Checks strict structural compliance of projected code against Language Spec.
"""

from .languages import LanguageSpec
import re

class Validator:
    def validate(self, code: str, spec: LanguageSpec) -> dict:
        missing = []
        
        # 1. Keyword check
        for kw in spec.keywords:
            # Simple substring check (robustness can be improved with regex boundary)
            if kw not in code:
                missing.append(kw)
        
        # 2. Structural Integrity (Indent calc, Bracket matching) - Simplified
        # Just check if Block Start/End exists if defined
        if spec.syntax['block_start'] and spec.syntax['block_start'] not in code:
             if spec.syntax['block_start'] not in missing: missing.append(spec.syntax['block_start'])
        
        score = 1.0 - (len(missing) / len(spec.keywords)) if spec.keywords else 1.0
        
        return {
            'language': spec.id,
            'structure_match': max(0.0, score),
            'missing_elements': missing,
            'result': 'PASS' if not missing else 'FAIL'
        }
