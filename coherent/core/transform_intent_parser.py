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
        
        # "5 languages" preset
        self.preset_5_langs = ["en", "it", "zh", "es", "ko"]

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
            # We don't escalate here implicitly; Executor Selector might escalate.
            # But per spec checks: "transform_type が特定できない場合 -> Executor Selector で Escalation"
            pass 
        
        if not source_text and t_type != TransformType.UNKNOWN:
             # If we determined it IS a transform task, but can't find source -> Escalation
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
        
        Priority 1: Explicit Quotes (「...」, "...")
        Priority 2: Preceding Block (Separated by newline from the command)
        Priority 3: Full Text (Fallback)
        """
        
        # --- Priority 1: Explicit Quotes ---
        # Japanese brackets
        jp_quote_match = re.search(r"「(.+?)」", text, re.DOTALL)
        if jp_quote_match:
            return jp_quote_match.group(1).strip()
            
        # English double quotes (naive)
        # Note: This might capture "word" inside a sentence. 
        # For MVP, we'll assume if quotes exist, that's the target.
        # To be safer, maybe we require it to be reasonably long or the main part?
        # Specification says: “...”
        en_quote_match = re.search(r"[\"“](.+?)[\"”]", text, re.DOTALL)
        if en_quote_match:
             return en_quote_match.group(1).strip()

        # --- Priority 2: Preceding Block ---
        # Look for the command block (last line or lines?). 
        # If there is a clear separation (newlines), the top part is source.
        
        # Simple heuristic: Split by newlines. 
        # If multiple blocks, assume the last block is the command and the rest is source.
        # "Command-like" check on the last block?
        
        # Let's try splitting by double newline first (paragraph break)
        paragraphs = re.split(r"\n\s*\n", text.strip())
        if len(paragraphs) >= 2:
            # Assume last paragraph is command, everything before is source
            # But we must verify the last paragraph actually looks like a command?
            # For TIP, we assume TaskGate already said "this is valid".
            # So just take everything except the last paragraph.
            return "\n\n".join(paragraphs[:-1]).strip()
            
        # If simple single newlines?
        lines = text.strip().splitlines()
        if len(lines) >= 2:
            # If the last line contains the transform command pattern, 
            # treat lines[:-1] as source.
            last_line = lines[-1]
            if self._classify_type(last_line) != TransformType.UNKNOWN:
                return "\n".join(lines[:-1]).strip()

        # --- Priority 3: Full Text (Fallback) ---
        # This is risky because it includes the command itself ("Translate 'Hello'").
        # If we couldn't separate, we might return empty to force escalation, 
        # OR return full text if the implementation allows filtering later.
        # Spec says: "Priority 3: Full text (Fallback) -> Escalation candidate"
        
        # However, if the text is JUST "Translate this", returning "Translate this" as source is wrong.
        # If the text is "I love cars", and implicit intent? The spec implies explicit triggers.
        
        # Let's return "" to trigger SOURCE_TEXT_NOT_FOUND if we can't cleanly separate,
        # UNLESS the whole text IS the source (but then where is the command?).
        # Wait, Task Gate passes "raw_input". 
        # If input is just "Translate this", there is no source.
        # If input is "Hello", treating it as source requires implicit translation detection (not implemented here).
        
        # Spec 5.1 says "Escalation Candidate". 
        # So returning valid-looking string might be dangerous if it includes the command.
        
        # For MVP, let's try to remove the command phrase if found in the single line?
        # Or just return None/Empty to be safe and escalate.
        
        return "" 

    def _analyze_translation_params(self, text: str) -> Dict[str, Any]:
        """
        Extracts translation parameters (target languages).
        """
        text_lower = text.lower()
        targets = []
        
        # 1. strict quantity check "5 languages", "5か国語"
        if "5か国語" in text or "5 languages" in text_lower:
            return {"target_languages": self.preset_5_langs}
            
        # 2. specific language check
        for lang_name, lang_code in self.lang_map.items():
            if lang_name in text_lower:
                if lang_code not in targets:
                    targets.append(lang_code)
                    
        # Verify if targets are found
        if not targets:
            # Default to English if not specified (Safety)
            targets = ["en"]
            
        return {"target_languages": targets}
