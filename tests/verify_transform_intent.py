"""
Verification Script for Transform Intent Parser (TIP)
"""

import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util

# Dynamic import to bypass 'coherent' package init dependencies (like torch/numpy)
spec = importlib.util.spec_from_file_location(
    "transform_intent_parser", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../coherent/core/transform_intent_parser.py"))
)
module = importlib.util.module_from_spec(spec)
sys.modules["transform_intent_parser"] = module
spec.loader.exec_module(module)

from transform_intent_parser import TransformIntentParser, TransformType

def run_tests():
    parser = TransformIntentParser()
    
    print("=== Running Transform Intent Parser Verification ===\n")

    # --- Case 1: Japanese Quotes (Explicit) ---
    input1 = "「私は、スーパーカーを運転するのが好きだ」\n上記の文章を5か国語に翻訳して"
    print(f"Test Case 1: Japanese Quotes\nInput:\n{input1}")
    result1 = parser.parse(input1)
    print(f"Result: Type={result1.transform_type.name}, Source='{result1.source_text}', Params={result1.parameters}")
    
    assert result1.transform_type == TransformType.TRANSLATION
    assert result1.source_text == "私は、スーパーカーを運転するのが好きだ"
    assert result1.parameters.get("target_languages") == ["en", "it", "zh", "es", "ko"]
    print("✅ PASS\n")


    # --- Case 2: Block Separation (Priority 2) ---
    input2 = """
Assuming that the logic is correct.

Translate the above to Japanese.
"""
    print(f"Test Case 2: Block Separation\nInput:\n{input2}")
    result2 = parser.parse(input2)
    print(f"Result: Type={result2.transform_type.name}, Source='{result2.source_text}', Params={result2.parameters}")

    assert result2.transform_type == TransformType.TRANSLATION
    assert "Assuming that" in result2.source_text
    assert result2.parameters.get("target_languages") == ["ja"]
    print("✅ PASS\n")


    # --- Case 3: No Source Text (Escalation) ---
    input3 = "5か国語に翻訳して"
    print(f"Test Case 3: No Source Text\nInput: {input3}")
    result3 = parser.parse(input3)
    print(f"Result: Type={result3.transform_type.name}, Source='{result3.source_text}', Escalation={result3.needs_escalation}, Reason={result3.escalation_reason}")
    
    assert result3.transform_type == TransformType.TRANSLATION
    assert result3.needs_escalation == True
    assert result3.escalation_reason == "SOURCE_TEXT_NOT_FOUND"
    print("✅ PASS\n")


    # --- Case 4: English Quotes ---
    input4 = 'Translate "Hello World" to Spanish'
    print(f"Test Case 4: English Quotes\nInput: {input4}")
    result4 = parser.parse(input4)
    print(f"Result: Type={result4.transform_type.name}, Source='{result4.source_text}', Params={result4.parameters}")
    
    assert result4.transform_type == TransformType.TRANSLATION
    assert result4.source_text == "Hello World"
    assert result4.parameters.get("target_languages") == ["es"]
    print("✅ PASS\n")

    # --- Case 5: Default Language ---
    input5 = "「テスト」を翻訳して"
    print(f"Test Case 5: Default Language\nInput: {input5}")
    result5 = parser.parse(input5)
    print(f"Result: Params={result5.parameters}")
    
    assert result5.parameters.get("target_languages") == ["en"]
    print("✅ PASS\n")

    print("\n🎉 ALL TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
