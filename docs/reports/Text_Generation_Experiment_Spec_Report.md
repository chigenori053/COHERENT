# 文字生成および構造化実験仕様レポート

## 1. 概要
本レポートは、**Coherent Cognitive Architecture**におけるホログラフィックメモリ (Dynamic Holographic Memory, DHM) を用いた、シンボル生成および構造保持に関する一連の実験仕様をまとめたものである。
これらの実験の共通目的は、**「シンボルそのものを記憶せず、属性 (Attributes) や構造 (Structure) の合成によって、未知または既知のシンボル・系列を動的に生成・想起できるか」**を検証することにある。

すべての実験において以下の**共通制約 (Hard Constraints)** が適用される：
*   **シンボル非保存**: 生成対象となるシンボル（数字、文字、ローマ数字など）のホログラムそのものを MemorySpace に保存してはならない。保存されるのは「属性」や「構造」のホログラムのみである。
*   **学習なし (Training-free)**: 勾配降下法やバックプロパゲーションによる学習を行わない。
*   **決定論的プロセス**: 拡散モデル (DDIM) 等を使用する場合でも、決定論的な動作を保証し、再現性を確保する。

---

## 2. 数字生成実験 (Digit Generation)

### 2.1 目的
抽象的な属性ホログラムの合成のみから、10進数の数字 {0, 1, ..., 9} を正しく想起・生成できることを実証する。

### 2.2 対象空間
*   **シンボル集合**: DIGITS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

### 2.3 属性システム
数字を以下の意味的・形状的属性に分解して表現する。
*   **TYPE**: {digit}
*   **PARITY (偶奇)**: {even, odd}
*   **MAGNITUDE (大きさ)**: {low, mid, high}
*   **PRIME_STATUS (素数性)**: {prime, composite, neither}
*   **LOOP_COUNT (閉路数)**: {0loop, 1loop, 2loop}
*   **STROKE_CLASS (形状)**: {straight, curved, mixed}

### 2.4 クエリ構築と評価
*   **クエリ**: $H_{query} = \text{Normalize}(\odot \text{Encode}(Attribute_i))$
*   **評価**: $H_{query}$ と属性合成により生成された候補 $H_{candidate}$ との共鳴度 (Resonance) を計算し、Top-1 正解率で評価する。
*   **成功基準**: Top-1 Accuracy 100% (10/10)

---

## 3. ローマ数字生成実験 (Roman Numeral Generation)

### 3.1 目的
1から99までの整数に対応する正規のローマ数字表記（例: 4 -> IV, 49 -> XLIX）を、構造化された属性から生成する。単純な属性合成だけでなく、順序や減算則などの**構造的属性**の検証を含む。

### 3.2 対象空間
*   **整数範囲**: 1 - 99
*   **使用シンボル**: I, V, X, L, C

### 3.3 属性システム (構造的)
*   **値分解**: HAS_ONES, HAS_TENS, ONES_MAG_{LOW, SUB, MID}, TENS_MAG_{LOW, SUB, MID}
*   **構造制御**:
    *   ORDER_TENS_FIRST (10の位を先に)
    *   USE_SUBTRACTIVE (減算則の使用、例: IV, IX)
    *   REPEAT_ALLOWED (同一シンボルの繰り返し許可)

### 3.4 評価
*   **成功基準**: 1-99 全ての整数において、正解のローマ数字系列が高い共鳴度でTop-1として想起されること (Accuracy >= 95%)。

---

## 4. 平仮名生成実験 (Hiragana Generation)

### 4.1 目的
日本語の平仮名を、その構成要素である**音韻属性 (Phonological Attributes)** から生成する。文字の形状ではなく、音韻構造（母音・子音）に基づいてシンボルを特定できるかを検証する。

### 4.2 フェーズ構成
1.  **Phase 1**: 母音のみ (あいうえお)
2.  **Phase 2**: カ行追加 (かきくけこ)
3.  **Phase 3**: 全清音 (46文字)

### 4.3 属性システム
*   **SCRIPT**: {hiragana}
*   **VOWEL (母音)**: {a, i, u, e, o, n} (「ん」を含む)
*   **CONSONANT (子音)**: {none, k, s, t, n, h, m, y, r, w}
*   **VOICE (清濁)**: {voiceless} (現在は清音のみ対象)
*   **SPECIAL_MARK**: {none, nasal}

### 4.4 マッピング例
*   `あ` = VOWEL:a, CONSONANT:none
*   `か` = VOWEL:a, CONSONANT:k

### 4.5 成功基準
*   Phase 3 (全清音) において 46/46 の完全正解。

---

## 5. Stage 2 構造実験 (Structure Experiment)

### 5.1 目的 (Hypothesis)
DHMが**異種シンボル (Heterogeneous Symbols)** の混合系列において、その順序構造を保持し、かつ共鳴デコーディングによって正しく構造を復元できるかを検証する。
仮説：
1.  H1: DHMは異種シンボルの順序構造を保持・抽出できる。
2.  H2: 共鳴デコーディングは全体構造 (Global Structure) を優先する。

### 5.2 シンボル定義
アルファベット、数字、記号の混合セット (計42種)。
*   **LETTERS**: A-Z
*   **DIGITS**: 0-9
*   **SYMBOLS**: +, -, *, /, =, #

### 5.3 構造生成とバインディング
*   **系列長 (L)**: 2, 3, 4
*   **パターン**: [Letter, Digit], [Digit, Letter], [Letter, Symbol] など多様な組み合わせ。
*   **動的バインディング (Dynamic Binding)**:
    *   累積的アダマール積と正規化を用いる。
    *   $H_{temp} = H(a_1)$
    *   $H_{next} = \text{Normalize}(H_{temp} * H(a_k))$ (for k=2..L)

### 5.4 評価指標
デコーディング候補集合には、正解(Exact)、順序入れ替え、カテゴリ違い、ランダムなどの撹乱候補を含める。
*   **Exact Match Rate (EM)**: 完全一致率 (目標 >= 95%)
*   **Category Confusion Rate (CCR)**: カテゴリ（文字・数字・記号）の誤認率。
*   **Structural Confusion Rate (SCR)**: 順序構造の誤認率 (目標 ~ 0)。
*   **Margin Stability**: 正解と2位候補との共鳴度の差。

---

## 参照ファイル
*   `experimental/digit_generation/spec/digit_experiment_spec.md`
*   `experimental/roman_generation/spec/roman_experiment_spec.md`
*   `experimental/hiragana_generation/spec/hiragana_experiment_spec.md`
*   `experimental/stage2_structure/spec/stage2_spec.md`
