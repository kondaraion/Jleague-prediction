# Jリーグ試合予想システム - テストガイド

このドキュメントでは、Jリーグ試合予想システムのテストスイートについて説明します。

## テスト構成

### テストファイル構成
```
tests/
├── __init__.py              # テストパッケージ初期化
├── test_data_processing.py  # データ処理機能のテスト
├── test_machine_learning.py # 機械学習機能のテスト
├── test_web_scraping.py     # Webスクレイピング機能のテスト
└── test_utils.py            # ユーティリティ機能のテスト

conftest.py                  # pytest設定とフィクスチャ
pytest.ini                  # pytest設定ファイル
```

## テスト実行方法

### 基本的なテスト実行

```bash
# 全テストの実行
pytest

# 詳細出力付きでテスト実行
pytest -v

# カバレッジレポート付きでテスト実行
pytest --cov=. --cov-report=html

# 特定のテストファイルのみ実行
pytest tests/test_data_processing.py

# 特定のテストクラス/関数のみ実行
pytest tests/test_machine_learning.py::TestMachineLearning::test_train_prediction_model_basic
```

### マーカーを使用したテスト実行

```bash
# 単体テストのみ実行
pytest -m unit

# 統合テストのみ実行
pytest -m integration

# Web関連テストのみ実行
pytest -m web

# 時間のかかるテストを除外
pytest -m "not slow"
```

## テストカテゴリ

### 1. データ処理テスト (`test_data_processing.py`)

**テスト対象機能：**
- 特徴量作成 (`create_features`)
- 試合データ統計 (`get_match_data_stats`)
- チーム勝率計算 (`calculate_team_win_rate`)
- ゴール統計計算 (`calculate_team_goals`)
- 対戦成績計算 (`calculate_head_to_head`)
- 最近の調子計算 (`calculate_recent_form`)

**主要テストケース：**
- 正常なデータでの計算結果検証
- 空データ・無効データの処理
- エッジケース（チーム名不一致など）の処理

### 2. 機械学習テスト (`test_machine_learning.py`)

**テスト対象機能：**
- モデル訓練 (`train_prediction_model`)
- 試合予測 (`predict_match`)
- モデル評価 (`evaluate_current_model`)
- 特徴量重要度抽出
- クロスバリデーション

**主要テストケース：**
- モデル訓練の成功/失敗
- 予測結果の妥当性検証
- モデル永続化（保存・読み込み）
- 評価メトリクスの計算
- エラーハンドリング

### 3. Webスクレイピングテスト (`test_web_scraping.py`)

**テスト対象機能：**
- Jリーグデータスクレイピング (`scrape_jleague_data`)
- HTMLパース機能
- データ検証・重複除去
- レート制限遵守
- エラーハンドリング

**主要テストケース：**
- 正常なスクレイピング処理
- ネットワークエラー時の処理
- 無効なHTML処理
- タイムアウト処理
- 大量データ処理のパフォーマンス

### 4. ユーティリティテスト (`test_utils.py`)

**テスト対象機能：**
- 特徴量名翻訳 (`translate_feature_name`)
- チーム詳細取得 (`get_team_details`)
- CSVダウンロードリンク作成
- システム診断 (`system_diagnostics`)
- データエクスポート機能

**主要テストケース：**
- 翻訳機能の正確性
- システムヘルスチェック
- データ検証機能
- 設定管理機能

## フィクスチャ

### 主要フィクスチャ (`conftest.py`)

- `sample_match_data`: テスト用の試合データ
- `sample_team_stats`: チーム統計のサンプル
- `temp_data_dir`: 一時データディレクトリ
- `mock_match_csv`: モックCSVファイル
- `trained_model_mock`: 訓練済みモデルのモック
- `mock_requests_get`: Web リクエストのモック

## テスト実行環境のセットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. テスト用の環境変数（必要に応じて）

```bash
export JLEAGUE_TEST_MODE=1
export JLEAGUE_DATA_DIR=./test_data
```

## カバレッジレポート

テストカバレッジレポートはHTMLとターミナル出力の両方で利用可能です：

```bash
# カバレッジレポート生成
pytest --cov=. --cov-report=html --cov-report=term-missing

# HTMLレポートの確認
open htmlcov/index.html
```

## 継続的インテグレーション

GitHub Actions や他のCI/CDツールでの自動テスト実行設定例：

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## テスト作成のガイドライン

### 1. テスト命名規則
- テストファイル: `test_*.py`
- テストクラス: `Test*`
- テスト関数: `test_*`

### 2. テスト構造
```python
def test_function_name_scenario(self, fixtures):
    # Arrange: テストデータの準備
    
    # Act: テスト対象の実行
    
    # Assert: 結果の検証
```

### 3. モックの使用
- 外部依存（ファイルI/O、Web API）は必ずモック
- 時間に依存する処理もモック
- 予測可能なテスト結果の確保

### 4. エラーケースのテスト
- 正常系だけでなく異常系も必ずテスト
- エッジケースの考慮
- 適切なエラーハンドリングの検証

## トラブルシューティング

### よくある問題と解決方法

1. **インポートエラー**
   ```bash
   # Pythonパスの確認
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **テンポラリファイルの残存**
   ```bash
   # テスト後のクリーンアップ
   pytest --tb=short --clean-tmpdir
   ```

3. **モックが効かない**
   - パッチ対象のパスを確認
   - インポート順序を確認

4. **カバレッジが低い**
   - 未テストのコードブロックを特定
   - 分岐条件の網羅性を確認

## パフォーマンステスト

時間のかかるテストには `@pytest.mark.slow` マーカーを付与：

```python
@pytest.mark.slow
def test_large_dataset_processing(self):
    # 大量データでのパフォーマンステスト
    pass
```

## セキュリティテスト

機密情報の漏洩防止テスト：

```python
def test_no_sensitive_data_in_logs(self):
    # ログに機密情報が含まれていないことを確認
    pass
```

このテストスイートにより、Jリーグ試合予想システムの品質と信頼性を確保できます。