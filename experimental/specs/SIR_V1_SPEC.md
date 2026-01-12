# SIR v1.0 ― 公式スキーマ定義

## 0. 設計原則（仕様拘束）
1.  **表層非依存 (Surface Independent)**: 言語・記法・順序を捨象する。
2.  **構造保持 (Structure Preserving)**: 意味＝関係の集合として保持する。
3.  **固定次元に射影可能 (Projectable)**: 固定次元ベクトル（DHM入力）へ変換可能であること。
4.  **可逆性は不要**: 生成（Decoding）は後段の役割とする。

## 1. SIR 全体構造 (JSON Schema)
```json
{
  "sir_version": "1.0",
  "modality": "natural_language | math | code",
  "semantic_core": {
    "entities": [],
    "relations": [],
    "operations": [],
    "constraints": []
  },
  "structure_signature": {
    "graph_hash": "HexDigest",
    "depth": 0,
    "branching_factor": 0.0
  },
  "abstraction_level": 0.0,
  "confidence": 0.0
}
```

## 2. Semantic Core（意味核）

### 2.1 Entities（概念ノード）
*   **正規化**: NL「x is greater than y」, Math `x > y`, Code `if x > y:` を同一構造にする。

```json
{
  "id": "E1",
  "type": "concept | variable | constant | function",
  "label": "abstract_label",
  "attributes": {
    "quantifier": "forall | exists | none",
    "domain": "number | boolean | sequence | abstract",
    "role": "subject | object | operand | iterator"
  }
}
```

### 2.2 Relations（意味関係）
*   **順序**: 意味に寄与しない場合はソートして正規化。言語差を消滅させる。

```json
{
  "id": "R1",
  "type": "comparison | causal | hierarchical | dataflow",
  "from": "E1",
  "to": "E2",
  "polarity": "positive | negative | neutral"
}
```

### 2.3 Operations（操作・演算）
*   **commutative (可換性)**: `true` の場合、`operands` はソートされる（例: `3+x` == `x+3`）。
*    loop正規化: `for`, `while`, `Σ` は `loop` に統一。

```json
{
  "id": "O1",
  "operator": "add | multiply | compare | assign | loop | map",
  "operands": ["E1", "E2"],
  "properties": {
    "commutative": true,
    "associative": true,
    "side_effect": false
  }
}
```

### 2.4 Constraints（制約・条件）
```json
{
  "id": "C1",
  "type": "logical | numerical | boundary",
  "expression": "normalized_constraint",
  "scope": ["E1", "E2"]
}
```

## 3. Structure Signature（意味構造署名）

### 3.1 Graph Hash
意味グラフを順序不変・名前不変でハッシュ化する。

$$
GraphHash = H\left(\sum_i \phi(E_i) + \sum_j \psi(R_j) + \sum_k \omega(O_k)\right)
$$

*   **入力**: 名前・表記（Labelの文字面）はハッシュ計算に含めない。構造特徴のみを使う。

### 3.2 Metrics
*   **Depth**: 意味依存の最大ネスト深度。
*   **Branching**: 概念結合密度。

## 4. Metrics
*   **Abstraction Level**: $A = 1 - \frac{\text{surface\_features}}{\text{total\_features}}$
*   **Confidence**: $C = \min(1, \frac{\text{validated\_relations}}{\text{total\_relations}})$

## 5. SIR ベクトル射影 (Vector Projection)
DHMへの入力となるベクトル $s$ を生成する。

$$
s = \left[ \sum_{e \in E} f_e(e), \sum_{r \in R} f_r(r), \sum_{o \in O} f_o(o), \sum_{c \in C} f_c(c) \right] \in \mathbb{R}^d
$$

## 6. モダリティ別マッピング保証
| 入力 | SIRで保持されるもの |
| :--- | :--- |
| **日本語** | 命題・関係 |
| **英語** | 命題・関係 |
| **数式** | 演算構造 |
| **Code** | 制御・データフロー |

SIRレベルでモダリティの差分は完全に捨象される。

## 7. 検証成功条件
1.  **同一意味 → 同一 graph_hash**: 異なる表現でも意味が同じならハッシュが一致すること。
2.  **意味近接 → DHM 位相近接**: 似た意味はDHM空間上で近くなること。
3.  **表記差 → SIRで消滅**: 表記揺れがSIRに残らないこと。
