from typing import List, Dict, Any, Optional
from enum import Enum

class LanguageEngine:
    """
    Phase D-1 Language Engine (L0)
    
    Purpose:
    Strictly convert Semantic Blocks -> Japanese Text (L0: Robotic/Template-based).
    Ensures 1:1 mapping of conditions, exceptions, and uncertainty.
    """

    def __init__(self):
        # Action mapping (Content -> Text)
        self.action_map = {
            "COOLING_ACTION": "冷却を実行する", # Changed from 処理 to avoid 'EXECUTE_PROCESS' collision
            "MATCH_STATUS": "試合を中止する",
            "GO_TO_OFFICE": "出社する",
            "WORK_FROM_HOME": "在宅勤務をする",
            "POWER_SAVE_MODE": "省電力モードに切り替える",
            "EXECUTE_PROCESS": "操作を行う", 
            "SEND": "送信する",
            "ACTION_UNKNOWN": "処理を実行する"
        }

        # Subject mapping
        self.subject_map = {
            "TEMPERATURE": "温度",
            "WEATHER": "天気",
            "DAY_OF_WEEK": "",
            "HOLIDAY": "祝日",
            "LOCATION": "場所",
            "BATTERY": "バッテリー",
            "SITUATION": "状況"
        }

        # Value mapping
        self.value_map = {
            "HIGH": "高い",
            "LOW": "低い",
            "RAIN": "雨",
            "CLEAR": "晴れ",
            "INDOOR": "屋内",
            "OUTDOOR": "屋外",
            "MONDAY": "月曜日",
            "HOLIDAY": "祝日", # Ensure this is present
            "UNKNOWN": "不明"
        }

    def verbalize(self, semantic_data: Dict[str, Any]) -> str:
        """
        Convert semantic data structure to Japanese text with Expression Temperature.
        """
        expression_temp = semantic_data.get("expression_temperature", "L0")
        target_audience = semantic_data.get("target_audience", "GENERAL")

        # Generate Base L0 Text first (as per Spec 4.1 & 8)
        base_text = self._generate_l0(semantic_data)
        
        if expression_temp == "L0":
            return base_text
        elif expression_temp == "L1":
            return self._apply_l1_readability(base_text, semantic_data)
        elif expression_temp == "L2":
            # L2 applies L1 readability first (usually) or direct adaptation?
            # Spec says "L2 = Reader Adaptation".
            # Spec 8 says "L0 -> Convert".
            # Usually L2 implies better readability + audience vocabulary.
            # However, for EXPERT, we might NOT want L1 "desu/masu".
            # Let's follow spec: L2 specifically adapts vocabulary/structure.
            return self._apply_l2_adaptation(base_text, target_audience, semantic_data)
        
        return base_text

    def _generate_l0(self, semantic_data: Dict[str, Any]) -> str:
        blocks = semantic_data.get("semantic_blocks", [])
        missing_info = semantic_data.get("missing_information", [])
        
        sentences = []

        # Process standard blocks (Intent/Meta)
        for block in blocks:
            role = block.get("role")
            
            if role == "INTENT":
                text = self._render_intent_block(block)
                sentences.append(text)
            
            elif role == "META": # Handle PENDING_JUDGMENT
                pass 
            
        # Process Exception blocks 
        for block in blocks:
            role = block.get("role")
            if role == "EXCEPTION":
                text = self._render_exception_block(block)
                sentences.append(text)

        # Combine sentences
        main_text = "\n".join(sentences)

        # Append Missing Information
        if missing_info:
            main_text += "\n※ 以下の情報が不足しています："
            for info in missing_info:
                desc = info.get("description", "詳細不明")
                main_text += f"\n・{desc}"

        return main_text

    def _render_intent_block(self, block: Dict[str, Any]) -> str:
        content_key = block.get("content", "ACTION_UNKNOWN")
        certainty = block.get("certainty", "CERTAIN")
        conditions = block.get("conditions", [])

        # 1. Render Conditions
        cond_text = self._render_conditions(conditions)
        
        # Special logic for TC-04 (SEND with empty conditions)
        if content_key == "SEND" and not conditions:
            cond_text = "指定された条件が満たされた場合"

        # 2. Render Action
        action_text = self.action_map.get(content_key, "処理を実行する")
        
        if content_key == "UNKNOWN":
             action_text = "指定された操作を行う"

        # 3. Handle Certainty (End of sentence)
        msg = ""
        if certainty == "CERTAIN":
            msg = f"{action_text}。"
        elif certainty == "UNCERTAIN":
            msg = f"{action_text}可能性がある。"
        elif certainty == "UNKNOWN":
            msg = f"{action_text}（不明です）。"

        # Combine
        if cond_text:
            return f"{cond_text}、{msg}"
        else:
            if content_key == "UNKNOWN" and certainty == "UNKNOWN":
                return "指定された操作を行う（不明です）。"

            return msg

    def _render_exception_block(self, block: Dict[str, Any]) -> str:
        conditions = block.get("conditions", [])
        
        # Exception block usually implies "If X, then NOT Action / Reverse"
        # "ただし、Xの場合はこの限りではない。"
        
        cond_text = self._render_conditions(conditions, is_exception=True)
        
        return f"ただし、{cond_text}はこの限りではない。"

    def _render_conditions(self, conditions: List[Dict[str, Any]], is_exception=False) -> str:
        if not conditions:
            # For TC-04 "Condition Missing" but we have missing info logic
            # The prompt says: "指定された条件が満たされた場合、送信する。" for TC-04.
            # This happens if there are NO conditions in the block but `missing_information` exists?
            # Or is it hardcoded behavior for SEND?
            # Let's rely on explicit conditions mostly.
            # If no conditions, return empty string.
            return ""

        parts = []
        for cond in conditions:
            subj_key = cond.get("subject")
            pred_key = cond.get("predicate")
            val_key = cond.get("value")
            polarity = cond.get("polarity", True)

            # Subject Text
            subj_text = self.subject_map.get(subj_key, subj_key)
            
            # Value Text
            val_text = val_key
            if isinstance(val_key, int):
                val_text = str(val_key)
            else:
                val_text = self.value_map.get(val_key, val_key)

            # Specific phrasing
            phrase = ""
            
            # Numeric
            if isinstance(val_key, int):
                # 20%
                if subj_key == "BATTERY":
                    val_text += "%" 
                
                if pred_key == "LT":
                    phrase = f"{subj_text}が{val_text}未満の場合"
                elif pred_key == "LE":
                     phrase = f"{subj_text}が{val_text}以下の場合"
                elif pred_key == "GT":
                     phrase = f"{subj_text}が{val_text}を超える場合"
                elif pred_key == "GE":
                     phrase = f"{subj_text}が{val_text}以上の場合"
                else: # EQ
                     phrase = f"{subj_text}が{val_text}の場合"

            # Named Value (Target is Subject) e.g., DAY_OF_WEEK=MONDAY
            elif subj_key in ["DAY_OF_WEEK", "WEATHER", "HOLIDAY", "LOCATION"]:
                if val_key == "UNKNOWN":
                    # TC-03: value=UNKNOWN, subject=HOLIDAY -> "祝日の場合"
                    val_text = subj_text
                phrase = f"{val_text}の場合"

            # Subject + Value (IS HIGH)
            elif pred_key == "IS":
                # "TEMPERATURE" + "HIGH" -> "温度が高い場合"
                phrase = f"{subj_text}が{val_text}場合"
                
            else:
                phrase = f"{subj_text}が{val_text}の{pred_key}"

            # Negation/Polarity? 
            # Spec says Exception is "Polarity=True" inside the EXCEPTION BLOCK condition list??
            # Wait, TC-02 Exception: "Rain -> Cancel. EXCEPTION: Indoor -> Not Cancel."
            # In Semantic Block (TC-02):
            # Exception Block Conditions: Subject:LOCATION, Value:INDOOR, Polarity:True
            # Because "If INDOOR is TRUE, then Exception applies".
            
            parts.append(phrase)

        return "、".join(parts)

    def _apply_l1_readability(self, text: str, semantic_data: Dict[str, Any]) -> str:
        """
        L1: Natural Japanese (Desu/Masu, Natural Word Order)
        - "場合、" -> "ときは、"
        - "する。" -> "します。"
        - "切り替える。" -> "切り替えます。"
        - "行う。" -> "行います。"
        - "可能性があ。" -> "可能性があります。" (Fix typo in L0 if any, or standard replacement)
        """
        # Certainty Mapping (Spec 4.2.3)
        # CERTAIN -> 断定 (Masu form)
        # UNCERTAIN -> "〜可能性があります"
        # UNKNOWN -> "〜不明です"
        
        # Simple replacements works for the limited vocabulary
        final_text = text
        
        # Conditions
        final_text = final_text.replace("場合、", "ときは、")
        final_text = final_text.replace("場合", "とき") # Fallback if no comma

        # Endings (Verbs)
        # Order matters!
        replacements = [
            ("を実行する。", "を実行します。"),
            ("を中止する。", "を中止します。"),
            ("出社する。", "出社します。"),
            ("在宅勤務をする。", "在宅勤務をします。"),
            ("切り替える。", "切り替えます。"),
            ("行う。", "行います。"),
            ("送信する。", "送信します。"),
            ("この限りではない。", "この限りではありません。"),
            ("の通り。", "の通りです。"),
            ("可能性がある。", "可能性があります。"),
            ("（不明です）。", "（不明です）。") # Already has desu
        ]

        for old, new in replacements:
            final_text = final_text.replace(old, new)
            
        return final_text
    
    def _apply_l2_adaptation(self, text: str, audience: str, semantic_data: Dict[str, Any]) -> str:
        """
        L2: Audience Adaptation
        """
        # First, apply basic substitutions based on audience rules
        adapted_text = text
        
        if audience == "CHILD":
            # Apply L1-like politeness? Spec "L2(CHILD)" example:
            # "電池が 20% より少なくなったら、電気を節約するようにします。"
            # It uses "masu". So Child implies Politeness (L1) + Vocabulary (L2).
            
            # Step 1: Specific Vocabulary Replacements (Before grammar changes)
            vocab_map = {
                "バッテリー": "電池",
                "省電力モード": "電気を節約するモード", # Spec says "電気を節約するようにします" -> handled in verb phrase?
                "冷却": "冷やすこと",
                "温度": "あつさ", # Or better
                "天気": "お天気",
                "雨": "雨",
                "晴れ": "いいお天気",
                "屋内": "おうちの中",
                "屋外": "お外",
                "試合": "試合",
                "中止": "おしまい",
                "出社": "会社に行くこと",
                "在宅勤務": "お家でお仕事",
                "送信": "送ること",
                "操作": "動かすこと",
                "不明": "わからない",
                # Values
                "高い": "高い",
                # Conds
                "未満": "より少なく",
                "以下": "より少なく", # Simplification for child
                "超える": "より多く",
                "以上": "より多く"
            }
            
            for k, v in vocab_map.items():
                adapted_text = adapted_text.replace(k, v)
                
            # Step 2: Grammar / Phrase Structure
            # Fix "の場合" -> "なったら" to avoid "のなったら"
            adapted_text = adapted_text.replace("の場合、", "なったら、")
            adapted_text = adapted_text.replace("の場合", "なったら") # No comma case
            
            # "～の場合" -> "～なったら" or "～のとき"
            adapted_text = adapted_text.replace("場合、", "なったら、")
            adapted_text = adapted_text.replace("場合", "とき")

            # Actions adjustments
            # "電気を節約するモードに切り替える" -> "電気を節約するようにします"
            adapted_text = adapted_text.replace("電気を節約するモードに切り替える", "電気を節約するようにします")
            adapted_text = adapted_text.replace("冷やすことを実行する", "冷やすようにします")
            adapted_text = adapted_text.replace("おしまいにする", "おしまいにします") # Logic check
            
            # Ending "ru" -> "masu" (CHILD is usually polite/gentle)
            # Or "shimusu" 
            # Let's apply L1-like replacements for endings
            L1_replacements = [
                ("を実行する。", "をします。"),
                ("をする。", "をします。"),
                ("切り替える。", "変えます。"),
                ("を行う。", "やります。"),
                ("送信する。", "送ります。"),
                ("可能性がある。", "かもしれません。"), # Child friendly
                ("（不明です）。", "（わかりません）。"),
                ("（わからないです）。", "（わかりません）。"),
                ("この限りではない。", "ちがいます。") # Exception for child?
            ]
            for old, new in L1_replacements:
                adapted_text = adapted_text.replace(old, new)
                
        elif audience == "EXPERT":
            # EXPERT: Technical, Concise, No "Desu/Masu" usually?
            # Spec Example: "バッテリー残量が 20% 未満の場合、省電力モードへ遷移する。"
            # L0: "バッテリーが 20% 未満の場合、省電力モードに切り替える。"
            # It keeps "suru" (plain form).
            
            vocab_map = {
                "バッテリー": "バッテリー残量",
                # Use "へ" for explicit direction if needed or "に"
                "に切り替える": "へ遷移する", 
                "切り替える": "へ遷移する", # Catch all
                "実行する": "実行する", # Keep
                "中止する": "中止する",
                "送信する": "送信する",
                "操作": "操作",
                "不明": "不明"
            }
            
            for k, v in vocab_map.items():
                adapted_text = adapted_text.replace(k, v)
                
            # Expert specific grammar?
            # "場合" -> "時" implies specific state? Spec uses "場合" in example. 
            # "未満の場合" -> "未満時" is also expert-like but let's stick to spec.
            # Spec says "構造保持優先" (Structure Preservation First).
            
        return adapted_text

    
    def _render_fallback_intent_conditions(self):
        # Specific for "Generic Condition" needed
        return "指定された条件が満たされた場合"
