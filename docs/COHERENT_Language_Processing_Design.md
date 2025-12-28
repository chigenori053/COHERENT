# COHERENT Language Processing Capability Design Specification
# COHERENT 言語処理機能設計仕様書

## 1. Purpose (目的)
This document defines the architecture and design principles for implementing controllable, testable, and learnable language processing capabilities within the COHERENT system.
本書は、COHERENTシステム内において、制御可能（Controllable）、テスト可能（Testable）、かつ学習可能（Learnable）な言語処理機能を実装するためのアーキテクチャおよび設計原則を定義します。

The goal is to transform natural language inputs into executable semantic structures without relying on black-box reasoning models.
その目標は、ブラックボックスな推論モデルに依存せず、自然言語入力を実行可能な意味的構造（Semantic Structures）に変換することです。

## 2. Non-Goals (非目標)
- General conversational AI
  - 一般的な対話型AI
- Free-form text generation for reasoning
  - 推論のための自由形式のテキスト生成
- LLM-only problem solving
  - LLMのみによる問題解決

## 3. Design Principles (設計原則)
- All language input must be converted into structured semantics.
  - 全ての言語入力は、構造化された意味表現（Structured Semantics）に変換されなければならない。
- Reasoning authority remains in CoreRuntime and ReasoningAgent.
  - 推論の主体（Authority）は、CoreRuntimeおよびReasoningAgentに保持される。
- Language understanding must be learnable and reusable.
  - 言語理解は、学習可能かつ再利用可能でなければならない。

## 4. Architecture Overview (アーキテクチャ概要)
Natural Language → Semantic Parser → Semantic IR → CoreRuntime → Optical Memory
自然言語 → 意味解析器 (Semantic Parser) → 意味的中間表現 (Semantic IR) → CoreRuntime → 光ホログラフィックメモリ (Optical Memory)

## 5. Language Capability Layers (言語機能レイヤー)
### L1 Surface Processing (表層処理)
Normalization, symbol completion, multilingual handling.
正規化、記号補完、多言語対応。

### L2 Semantic Parsing (意味解析)
Extract intent, domain, and mathematical objects.
意図（Intent）、ドメイン、数学的オブジェクトの抽出。

### L3 Intent Classification (意図分類)
Enumerated intent outputs with confidence scores.
列挙された意図の出力と信頼度スコア。

### L4 Semantic Intermediate Representation (SIR) (意味的中間表現)
Defines task, domain, goals, inputs, constraints, ambiguity, and metadata.
タスク、ドメイン、ゴール、入力、制約、曖昧性、およびメタデータを定義。

## 6. Semantic IR Schema (Semantic IR スキーマ)
```json
{
  "task": "solve | verify | hint | explain",
  "math_domain": "arithmetic | algebra | calculus | linear_algebra",
  "goal": { "type": "final_value | transformation | proof" },
  "inputs": [{ "type": "expression", "value": "(x - y)^2" }],
  "constraints": { "symbolic_only": true },
  "explanation_level": 0,
  "language_meta": { "original_language": "ja", "ambiguity": 0.23 }
}
```

## 7. Semantic Parser Design (意味解析器の設計)
Hybrid rule-based and LLM-based parser.
ルールベースとLLMベースのハイブリッドパーサ。
LLM output must be strict JSON with no reasoning.
LLMの出力は、推論を含まない厳密なJSONでなければならない。

## 8. CoreRuntime Integration (CoreRuntimeとの統合)
Intent-based routing to solving, verification, hint, or explanation engines.
意図（Intent）に基づき、解決（Solving）、検証（Verification）、ヒント（Hint）、または説明（Explanation）エンジンへのルーティングを行う。

## 9. Optical Memory Integration (光メモリとの統合)
Language semantics stored as reusable experience units enabling recall-first reasoning.
言語の意味内容は再利用可能な経験単位（Experience Units）として保存され、想起優先（Recall-First）の推論を可能にする。

## 10. Ambiguity Handling (曖昧性の処理)
Ambiguity scores inform DecisionEngine to accept, review, or request clarification.
曖昧性スコアにより、DecisionEngineは受け入れ、レビュー、または明確化の要求を判断する。

## 11. Testing Strategy (テスト戦略)
- Paraphrase equivalence
  - 言い換えの等価性
- Ambiguous instruction handling
  - 曖昧な指示の処理
- Student-like malformed input
  - 生徒のような形式不備な入力の処理

## 12. Roadmap (ロードマップ)
Phase 1: IR definition and intent routing.
フェーズ1: IRの定義と意図ルーティング。
Phase 2: Optical memory integration.
フェーズ2: 光メモリの統合。
Phase 3: Ambiguity-aware educational UX.
フェーズ3: 曖昧性を考慮した教育的UX。

## 13. Conclusion (結論)
This design enables COHERENT to become an executable semantic intelligence system rather than a conversational model.
本設計により、COHERENTは単なる対話モデルではなく、実行可能な意味的知能システムとなることが可能になる。
