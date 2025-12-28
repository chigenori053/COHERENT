# 数学カリキュラム網羅テスト項目 (中学～高校)

Coherentシステムの計算・検証能力を確認するための、学習指導要領に基づいたテスト項目リストです。

## 中学校 (Junior High School)

### 中1: 正負の数・文字式・一次方程式
1.  **正負の計算**: `Verify -5 + 8 - (-3) = 6`
2.  **四則演算**: `Verify 3 * (4 - 2) / 6 = 1`
3.  **文字式**: `Simplify 2(3x - 5) + 4x`
4.  **一次方程式**: `Solve 3x + 5 = 14`

### 中2: 連立方程式・一次関数
5.  **連立方程式 (代入法)**: `Solve System: y = 2x, x + y = 3`
6.  **連立方程式 (加減法)**: `Solve System: 2x + 3y = 8, 2x + y = 4`
7.  **一次関数**: `Solve 2x - 4 = 0` (x切片)

### 中3: 展開・因数分解・平方根・二次方程式
8.  **多項式の展開**: `Expand (x + 3)(x - 3)`
9.  **因数分解**: `Factor x^2 - 4x + 4`
10. **平方根**: `Calculate sqrt(18) + sqrt(2)`
11. **二次方程式 (解の公式/因数分解)**: `Solve x^2 - 5x + 6 = 0`

---

## 高校 (High School)

### 数学I: 数と式・二次関数・図形と計量
12. **タスキ掛け因数分解**: `Factor 2x^2 + 5x + 3`
13. **二次関数の頂点**: `CompleteSquare x^2 - 6x + 2`
14. **三角比 (基本)**: `Calculate sin(pi/6) + cos(pi/3)`
15. **余弦定理**: `Calculate c^2 = a^2 + b^2 - 2ab cos(C) where a=3, b=4, C=pi/3`

### 数学II: 式と証明・複素数・図形と方程式・指数対数・微分積分(整式)
16. **複素数の計算**: `Expand (1 + 2i)(3 - i)`
17. **円の方程式**: `Verify x^2 + y^2 = 25 pass point (3, 4)`
18. **指数関数**: `Solve 2^x = 16`
19. **対数関数**: `Calculate log2(8) + log2(4)`
20. **微分 (導関数)**: `Differentiate x^3 - 3x^2 + 2x`
21. **不定積分**: `Integrate 3x^2 dx`
22. **定積分**: `Calculate Integrate 3x^2 from 0 to 2`

### 数学III: 極限・微分積分(様々な関数)・複素数平面
23. **極限**: `Limit sin(x)/x as x -> 0`
24. **合成関数の微分**: `Differentiate sin(x^2)`
25. **部分積分**: `Integrate x * e^x dx`
26. **複素数平面 (極形式)**: `Convert 1 + i to polar` (※システムが対応していれば)

### 数学B/C: 数列・ベクトル
27. **等差数列の和**: `Calculate Sum k from k=1 to n`
28. **ベクトルの内積**: `DotProduct [1, 2] [3, 4]`

---

## 実行推奨コマンド例
Streamlit UIに入力してテストしてください。

- **因数分解テスト**: `Factor x^2 - 9`
- **微分テスト**: `Differentiate x^3`
- **積分テスト**: `Integrate 2x`
- **方程式テスト**: `Solve x^2 - 4x - 5 = 0`
