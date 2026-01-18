"""
Transform Intent Parser (TIP) - MVP

Analyzes user input to deterministically extract "what to convert" (source_text)
and "how to convert it" (intent & parameters).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
from typing import List, Optional, Dict, Any

class TransformType(Enum):
    TRANSLATION = "TRANSLATION"
    SUMMARIZATION = "SUMMARIZATION"       # MVP implementation: Definition only
    FORMAT_CONVERSION = "FORMAT_CONVERSION" # MVP implementation: Definition only
    UNKNOWN = "UNKNOWN"

@dataclass
class TransformIntent:
    transform_type: TransformType
    source_text: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    needs_escalation: bool = False
    escalation_reason: str = ""

class TransformIntentParser:
    """
    Parses raw user input into a structured TransformIntent.
    
    Philosophy:
    - Deterministic: Same input always yields same output.
    - Shallow: No deep meaning understanding.
    - Quality-Agnostic: "Translate garbage" -> "Garbage" (Executor handles quality).
    """

    def __init__(self):
        # Patterns for INTENT recognition
        self.translation_patterns = [
            r"translate", r"to (japanese|english|french)", 
            r"翻訳", r"語で", r"訳して"
        ]
        
        # Patterns for PARAMETER extraction
        self.lang_map = {
            "english": "en", "英語": "en",
            "japanese": "ja", "日本語": "ja",
            "french": "fr", "フランス語": "fr",
            "german": "de", "ドイツ語": "de",
            "italian": "it", "イタリア語": "it",
            "spanish": "es", "スペイン語": "es",
            "chinese": "zh", "中国語": "zh",
            "korean": "ko", "韓国語": "ko",
        }
        
        # "N languages" priority list (Top ~15 by utility/population)
        self.language_priority_list = [
            "en", "zh", "es", "fr", "de", 
            "it", "pt", "ru", "ko", "ja", 
            "ar", "hi", "tr", "nl", "sv"
        ]

    def parse(self, raw_input: str) -> TransformIntent:
        """
        Main parsing method.
        """
        if not raw_input:
             return TransformIntent(
                transform_type=TransformType.UNKNOWN,
                source_text="",
                needs_escalation=True,
                escalation_reason="EMPTY_INPUT"
            )

        # 1. Classify Type
        t_type = self._classify_type(raw_input)
        
        # 2. Extract Source Text
        source_text = self._extract_source_text(raw_input)
        
        # 3. Extract Parameters (if TRANSLATION)
        params = {}
        if t_type == TransformType.TRANSLATION:
            params = self._analyze_translation_params(raw_input)
            
        # 4. Check for Escalation (Validation)
        needs_escalation = False
        escalation_reason = ""
        
        if t_type == TransformType.UNKNOWN:
            pass 
        
        if not source_text and t_type != TransformType.UNKNOWN:
             needs_escalation = True
             escalation_reason = "SOURCE_TEXT_NOT_FOUND"

        return TransformIntent(
            transform_type=t_type,
            source_text=source_text,
            parameters=params,
            confidence=1.0 if not needs_escalation else 0.0,
            needs_escalation=needs_escalation,
            escalation_reason=escalation_reason
        )

    def _classify_type(self, text: str) -> TransformType:
        """
        Classifies the intent type based on keywords.
        """
        text_lower = text.lower()
        
        # Check Translation
        for p in self.translation_patterns:
            if re.search(p, text_lower):
                return TransformType.TRANSLATION
                
        # Future: Summarization, Format Conversion
        
        return TransformType.UNKNOWN

    def _extract_source_text(self, text: str) -> str:
        """
        Extracts the text to be transformed based on priority rules.
        """
        
        # --- Priority 1: Explicit Quotes ---
        jp_quote_match = re.search(r"「(.+?)」", text, re.DOTALL)
        if jp_quote_match:
            return jp_quote_match.group(1).strip()
            
        en_quote_match = re.search(r"[\"“](.+?)[\"”]", text, re.DOTALL)
        if en_quote_match:
             return en_quote_match.group(1).strip()

        # --- Priority 2: Preceding Block ---
        paragraphs = re.split(r"\n\s*\n", text.strip())
        if len(paragraphs) >= 2:
            return "\n\n".join(paragraphs[:-1]).strip()
            
        lines = text.strip().splitlines()
        if len(lines) >= 2:
            last_line = lines[-1]
            if self._classify_type(last_line) != TransformType.UNKNOWN:
                return "\n".join(lines[:-1]).strip()

        # --- Priority 3: Full Text (Fallback) ---
        return "" 

    def _analyze_translation_params(self, text: str) -> Dict[str, Any]:
        """
        Extracts translation parameters (target languages).
        Supports "N languages" (Nか国語) to select top N from priority list.
        """
        text_lower = text.lower()
        targets = []
        
        # 1. Quantity check "N languages", "Nか国語"
        # Match number before "か国語" or "languages"
        # Regex for Japanese: (\d+)か国語
        # Regex for English: (\d+)\s*languages
        
        n_match = re.search(r"(\d+)(?:か国語| languages?)", text_lower)
        if n_match:
            try:
                n = int(n_match.group(1))
                # Clamp to max available
                n = min(n, len(self.language_priority_list))
                return {"target_languages": self.language_priority_list[:n]}
            except ValueError:
                pass

        # 2. specific language check
        for lang_name, lang_code in self.lang_map.items():
            if lang_name in text_lower:
                if lang_code not in targets:
                    targets.append(lang_code)
                    
        # Verify if targets are found
        if not targets:
            targets = ["en"]
            
        return {"target_languages": targets}
