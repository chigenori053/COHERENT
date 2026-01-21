import json
import os
import sys

# Bypass coherent package init
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.getcwd(), 'coherent', 'core', 'experimental'))

try:
    import language_capability as lc
    LanguageCapabilityVerifier = lc.LanguageCapabilityVerifier
    SemanticRole = lc.SemanticRole
except ImportError:
    sys.path.append(os.path.join(current_dir, '..', 'coherent', 'core', 'experimental'))
    import language_capability as lc
    LanguageCapabilityVerifier = lc.LanguageCapabilityVerifier
    SemanticRole = lc.SemanticRole

def run_verification():
    print("Starting Phase C.5 Structural Health Verification...")
    verifier = LanguageCapabilityVerifier()
    
    test_cases = [
        # --- Category A: Meaning Identity / Normalization ---
        # "温度が高いときはファンを回してください" -> Subject:TEMPERATURE, Value:HIGH, Polarity:True
        {"id": "A-1", "text": "温度が高いときはファンを回してください。", 
         "expect_logic": {"subject": "TEMPERATURE", "value": "HIGH", "polarity": True}},
        
        # --- Category B: Condition / Exception Normalization ---
        # "雨が降ったら...ただし屋内なら除く" -> 
        #   Cond1: WEATHER=RAIN (True)
        #   Cond2: LOCATION=INDOOR (False) [Exception]
        {"id": "B-1", "text": "雨が降ったら試合は中止。ただし屋内なら除く。", 
         "expect_logic_list": [
             {"subject": "WEATHER", "value": "RAIN", "polarity": True},
             {"subject": "LOCATION", "value": "INDOOR", "polarity": False}
         ]},
         
        {"id": "B-3", "text": "月曜日は出社。ただし祝日の場合は在宅。",
         "expect_logic_list": [
             {"subject": "DAY_OF_WEEK", "value": "MONDAY", "polarity": True},
             {"subject": "HOLIDAY", "value": "UNKNOWN", "polarity": False} # Or value: HOLIDAY? Mapped simply?
         ]},
         
        # --- Category C: Typed Missing Info ---
        {"id": "C-1", "text": "終わったら送ってください。", 
         "expect_missing_type": "EVENT"},
         
        {"id": "C-2", "text": "いつもの手順で進めて。", 
         "expect_missing_type": "CRITERIA"},
         
        # --- Value Normalization (Numeric) ---
        {"id": "N-1", "text": "バッテリーが20%未満なら省電力にする。",
         "expect_logic": {"subject": "BATTERY", "predicate": "LT", "value": 20}}
    ]
    
    results = []
    
    for case in test_cases:
        res = verifier.process(case["text"])
        passed = True
        reason = "OK"
        
        # Validate Logic Check
        if "expect_logic" in case:
            found = False
            target = case["expect_logic"]
            # Look in all blocks
            for block in res.get("semantic_blocks", []):
                for cond in block.get("conditions", []):
                    # Check match
                    match = True
                    for k, v in target.items():
                        if cond.get(k) != v:
                            match = False
                            break
                    if match: found = True
            
            if not found:
                passed = False
                reason = f"Expected logic {target} not found in output."

        # Validate Logic List (for Exceptions)
        if "expect_logic_list" in case:
            # Flatten all conditions conditions found
            all_conds = []
            for block in res.get("semantic_blocks", []):
                all_conds.extend(block.get("conditions", []))
            
            for target in case["expect_logic_list"]:
                found = False
                for cond in all_conds:
                    match = True
                    for k, v in target.items():
                        if cond.get(k) != v:
                            match = False
                            break
                    if match: found = True
                if not found:
                    passed = False
                    reason = f"Expected logic {target} not found."
                    break

        # Validate Missing Type
        if "expect_missing_type" in case:
            found = False
            for m in res.get("missing_information", []):
                if m.get("type") == case["expect_missing_type"]:
                    found = True
            if not found:
                passed = False
                reason = f"Missing info type {case['expect_missing_type']} not found."

        results.append({
            "id": case["id"],
            "text": case["text"],
            "passed": passed,
            "reason": reason,
            "output": res
        })
        
    # Save Report
    report_data = {
        "summary": "Phase C.5 Structural Verification",
        "results": results
    }
    
    os.makedirs("/Users/chigenori/development/COHERENT_ver2.0/report", exist_ok=True)
    with open("/Users/chigenori/development/COHERENT_ver2.0/report/PHASE_C5_LOG.json", "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    with open("/Users/chigenori/development/COHERENT_ver2.0/report/PHASE_C5_REPORT.md", "w") as f:
        f.write("# Phase C.5 Structural Health Verification Report\n\n")
        pass_count = sum(1 for r in results if r["passed"])
        total = len(results)
        f.write(f"## Summary\n- Total: {total}\n- Passed: {pass_count}\n- Pass Rate: {pass_count/total*100:.1f}%\n\n")
        f.write("## Details\n| ID | Text | Passed | Reason |\n|---|---|---|---|\n")
        for r in results:
            clean_text = r['text'].replace('\n', ' ')
            f.write(f"| {r['id']} | {clean_text} | {'✅' if r['passed'] else '❌'} | {r['reason']} |\n")

    print(f"Verification Complete. {pass_count}/{total} passed.")

if __name__ == "__main__":
    run_verification()
