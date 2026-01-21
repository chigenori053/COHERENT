from typing import Dict, List, Any, Optional

class LanguageEngine:
    """
    Phase D Language Engine (Verbalization Engine).
    Responsible for rendering semantic_blocks into Natural Language (Japanese).
    Strictly follows Phase D-1 (Level 0) specifications.
    Target: L0 (Fixed Template).
    """

    def __init__(self):
        # 6.2 Content -> Action Mapping
        self.action_map = {
            "COOLING_ACTION": "冷却処理を実行する",
            "GO_TO_OFFICE": "出社する",
            "POWER_SAVE_MODE": "省電力モードに切り替える",
            "SEND": "送信する",
            "ACTION_UNKNOWN": "指定された操作を行う",
            "MATCH_STATUS": "試合を中止する", # Inferred from example B-1 "試合を中止する"
            "UNKNOWN": "指定された操作を行う"
        }

        # 10 Certainty Rules
        self.certainty_suffix = {
            "CERTAIN": "",
            "UNCERTAIN": "可能性があります",
            "UNKNOWN": "不明です"
        }

        # Subject Display Map (Simple fallback if not raw)
        # In L0 we might use the raw subject string if available, 
        # but the spec examples show specific Japanese words.
        # Ideally the Input Object should have canonical subject names or we map them.
        # Based on Spec 11 Examples:
        # TEMPERATURE -> 温度
        # WEATHER -> 雨 (Special case in example? "雨の場合") -> Actually Condition Value mapping handling needed.
        # Let's assume the Semantic Block 'subject' is the key.
        self.subject_map = {
            "TEMPERATURE": "温度",
            "BATTERY": "バッテリー",
            "WEATHER": "天候", # Or special handling
            "DAY_OF_WEEK": "", # Special handling usually
            "HOLIDAY": "祝日", 
            "LOCATION": "場所",
            "CONDITION": "指定された条件" # Example C-1
        }
        
        # Value map for display
        self.value_map = {
            "HIGH": "高い",
            "LOW": "低い",
            "RAIN": "雨", # Example B-1 "雨の場合" -> Subject:WEATHER, Value:RAIN?
            "INDOOR": "屋内",
            "MONDAY": "月曜日",
            "UNKNOWN": "不明",
            "20%": "20%" # handling raw values?
        }

    def render(self, input_object: Dict[str, Any]) -> str:
        """
        Render the input object into a Japanese string.
        """
        output_lines = []

        semantic_blocks = input_object.get("semantic_blocks", [])
        missing_info = input_object.get("missing_information", [])

        # Filter blocks by role
        # L0 Policy: 1 semantic_block -> 1 Sentence (mostly).
        # But INTENT is the main sentence, Conditions are generated within it.
        # EXCEPTION is independent.

        # Find INTENT block
        intent_block = next((b for b in semantic_blocks if b["role"] == "INTENT"), None)
        
        # Find EXCEPTION blocks
        exception_blocks = [b for b in semantic_blocks if b["role"] == "EXCEPTION"]

        # Main Sentence Generation
        if intent_block:
            main_sentence = self._render_intent(intent_block)
            output_lines.append(main_sentence)

        # Exception Sentences
        for exc_block in exception_blocks:
            exc_sentence = self._render_exception(exc_block)
            output_lines.append(exc_sentence)

        # Missing Information
        if missing_info:
            output_lines.append("※ 以下の情報が不足しています：")
            for info in missing_info:
                desc = info.get("description", "")
                output_lines.append(f"・{desc}") # Spec example uses '・' or '-'? Spec 9.1 says '-' but Example C-1 uses '・'. using '・' as per Example C-1.

        return "\n".join(output_lines)

    def _render_intent(self, block: Dict[str, Any]) -> str:
        content = block.get("content", "ACTION_UNKNOWN")
        conditions = block.get("conditions", [])
        certainty = block.get("certainty", "CERTAIN")

        # 6.1 Basic Template: [条件] の場合、[行為] を実行する。
        # Action part
        action_text = self.action_map.get(content, self.action_map["ACTION_UNKNOWN"])
        
        # If certainty is not CERTAIN, modify end of action?
        # Spec 10: Suffix to sentence.
        # "実行する" + "可能性があります" -> "実行する可能性があります"
        # "送信する" + "不明です" -> "送信するか不明です" ? 
        # Spec examples just say "文末に...を付与".
        # Let's append simply for L0. 
        # But Japanese conjugation might be needed for perfect naturalness, 
        # however L0 is "Fixed Template". 
        # "省電力モードに切り替える" + "可能性があります" -> "省電力モードに切り替える可能性があります" (OK)
        suffix = self.certainty_suffix.get(certainty, "")

        # Condition part
        cond_text = ""
        if conditions:
            # Join multiple conditions? Spec doesn't strictly say, but implies list.
            # Assuming AND if multiple? L0 assumes simple single structure usually.
            cond_parts = []
            for cond in conditions:
                cond_parts.append(self._render_condition(cond))
            cond_text = " かつ ".join(cond_parts) # Simple join
        
        # Just in case condition text is empty/special
        full_sentence = ""
        if cond_text:
            full_sentence = f"{cond_text}、{action_text}"
        else:
            # If no condition, just action? Spec 6.1 says "[条件] の場合、[行為]..."
            # If no condition, maybe just "[行為]"?
            # Example C-1 has "指定された条件が満たされた場合" even if specific condition is missing?
            # No, C-1 Input likely HAS a condition block saying "SPECIFIED_CONDITION" or similar.
            full_sentence = f"{action_text}"
            
        if suffix:
            # Remove last '。' if present to append? No, Action map doesn't have '。'
            # But standard Japanese sentence ending.
            full_sentence += suffix
        
        return full_sentence + "。"

    def _render_exception(self, block: Dict[str, Any]) -> str:
        # 8.1 Rule: "ただし、[例外条件] の場合はこの限りではない。"
        conditions = block.get("conditions", [])
        
        cond_text = ""
        if conditions:
            # Usually strict single condition for exception in examples
            cond = conditions[0]
            # Special Rule 8.1 Note: value = UNKNOWN -> "祝日の場合" (Subject+の場合)
            if cond.get("value") == "UNKNOWN":
                 subj = self._resolve_subject(cond.get("subject"))
                 cond_text = f"{subj}の場合"
            else:
                 # Standard condition render
                 # But Exception usually implies "Polarity True" for the exception condition itself 
                 # (e.g. "If Indoor (True), then Exception").
                 # The 'Polarity' in the semantic block might reflect the LOGICAL effect on the main action (False),
                 # but here we render the condition itself.
                 # Example: "ただし、屋内の場合は..."
                 # Input condition likely: Subject:LOCATION, Value:INDOOR.
                 cond_text = self._render_condition(cond)
        
        return f"ただし、{cond_text}はこの限りではない。"

    def _render_condition(self, cond: Dict[str, Any]) -> str:
        # 7.1 [subject] が [value] の場合
        # 7.2 polarity = false -> [subject] が [value] でない場合

        subject = cond.get("subject")
        predicate = cond.get("predicate", "EQ")
        value = cond.get("value")
        polarity = cond.get("polarity", True)

        subj_text = self._resolve_subject(subject)
        val_text = self._resolve_value(value, subject) # Pass subject for context if needed

        # 7.1 Handling
        # Standard: "{subj_text}が{val_text}の場合"
        # But Example 1: "温度が高い場合" (Not "温度が高いの場合")
        # Example 2: "雨の場合" (Subject:WEATHER, Val:RAIN -> "天候が雨の場合"?? No, likely just "雨の場合")
        # Example 5: "バッテリーが 20% 未満の場合"
        
        # Heuristics for L0 naturalness based on Predicate
        
        phrase = ""
        
        # Numeric predicates
        if predicate in ["LT", "GT", "LE", "GE"]:
            # "バッテリー(Subj) が 20%(Val) 未満(Pred) の場合"
            op_text = ""
            if predicate == "LT": op_text = "未満"
            if predicate == "LE": op_text = "以下"
            if predicate == "GT": op_text = "超える"
            if predicate == "GE": op_text = "以上"
            
            phrase = f"{subj_text}が {val_text} {op_text}の場合"
            
        elif predicate == "EQ":
            # Direct mapping handling
            # If value is Adjective-like (High/Low/Hot), construct "SubjectがValue"
            # If value is Noun-like (Rain/Monday), construct "Valueの場合" (omit Subject?)
            #  -> Example B-1: "雨の場合" (Subject likely WEATHER).
            #  -> Example B-3: "月曜日の場合" (Subject likely DAY).
            #  -> Example A-1: "温度が高い場合" (Subject TEMPERATURE, Value HIGH).
            
            if value == "HIGH":
                 phrase = f"{subj_text}が高いの場合" # Fix: "高い" ends with 'i', no 'no'.
                 # Simple fix: just "{subj_text}が{val_text}場合"
                 phrase = f"{subj_text}が高い場合"
            elif value == "LOW":
                 phrase = f"{subj_text}が低い場合"
            elif value in ["RAIN", "SNOW", "MONDAY", "TUESDAY", "HOLIDAY", "INDOOR", "OUTDOOR"]:
                 # Omit subject for these obvious ones in L0? Or Spec 7.1 says "[subject] が [value] の場合".
                 # BUT Example B-1 says "雨の場合". It DEVIATES from 7.1 strictly?
                 # Spec 7.1 Example: "温度が高い場合" (Fits S+V).
                 # Spec 11 Ex B-1: "雨の場合".
                 # Let's handle "Obvious Subject" suppression or "Value IS the text".
                 phrase = f"{val_text}の場合"
            elif subject == "CONDITION":
                 # Example C-1 "指定された条件が満たされた場合"
                 if value == "FULFILLED" or value == "MET":
                      phrase = "指定された条件が満たされた場合"
                 else:
                      phrase = "指定された条件の場合"
            else:
                 # Default 7.1
                 phrase = f"{subj_text}が{val_text}の場合"

        else:
             phrase = f"{subj_text}が{val_text}の場合"

        # Polarity Handling
        if not polarity:
             # 7.2 "でない場合"
             # Replace "の場合" with "でない場合"
             if phrase.endswith("の場合"):
                 phrase = phrase[:-3] + "でない場合"
             else:
                 phrase += "でない場合" # Fallback

        return phrase

    def _resolve_subject(self, code: str) -> str:
        return self.subject_map.get(code, code)

    def _resolve_value(self, code: Any, subject: str = "") -> str:
        # If numeric, return as string
        if isinstance(code, int) or isinstance(code, float):
             # 20 -> "20%"? The unit might be lost? 
             # Spec 11 Ex N-1: "20%". Input probably has "20%".
             return str(code)
        
        # Try map
        return self.value_map.get(code, str(code))
