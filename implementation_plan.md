# J-Swing Insight 実装計画

## 概要
日本株スイングトレード推奨Webサービス。テクニカル指標に基づき日経225銘柄を分析し、推奨銘柄を提示するStreamlitアプリ。

## 技術スタック
- Python 3.11+, Streamlit, pandas, pandas-ta-classic, yfinance, Plotly, SQLite
- 認証なし、GitHub Actions自動化なし（MVP）

## プロジェクト構成

```
finance_app/
├── src/
│   ├── config.py             # 定数・設定
│   ├── models.py             # DB スキーマ・CRUD
│   ├── data_fetcher.py       # yfinance ETL
│   ├── analyzer.py           # 分析エンジン（Strategy パターン）
│   ├── calculator.py         # 利確/損切り計算
│   ├── charts.py             # Plotly チャート生成
│   ├── app.py                # Streamlit エントリポイント
│   └── pages/
│       ├── 1_dashboard.py    # 推奨銘柄一覧
│       ├── 2_stock_detail.py # 銘柄詳細・チャート
│       ├── 3_backtest.py     # バックテスト結果
│       └── 4_settings.py     # 設定
├── data/
│   └── nikkei225.csv         # 日経225銘柄マスタ
├── requirements.txt
├── .env.template
└── .gitignore
```

## 実装フェーズ（段階的に構築）

### Phase 1: 基盤 — config.py, models.py, nikkei225.csv, requirements.txt, .gitignore, .env.template

**`src/config.py`**
- DB_PATH, CSV_PATH,各種デフォルト値（EMA期間、RSI期間、ATR乗数、閾値0.4など）
- ストラテジー重み: EMA=0.30, RSI=0.25, BB=0.25, Volume=0.20

**`src/models.py`** — `DatabaseManager` クラス
- 4テーブル:
  - `stocks`: ticker_code (PK), company_name, sector, is_active
  - `daily_prices`: ticker_code, trade_date, OHLCV (UNIQUE(ticker_code, trade_date))
  - `recommendations`: ticker_code, recommendation_date, signal_score, signal_details(JSON), entry_price, ATR/PCT損切り利確値, status
  - `backtest_results`: recommendation_id, 5日間の最高/最低/終値、リターン率、勝敗
- WALモード有効化、接続はメソッドごとに作成・クローズ
- メソッド: _init_db, load_stocks_from_csv, upsert_daily_prices, get_price_history, save_recommendation, get_recommendations_by_date, save_backtest_result, get_backtest_results

**`data/nikkei225.csv`** — ticker_code,company_name,sector の225行

**`requirements.txt`**
```
streamlit>=1.41.0,<2.0
pandas>=2.2.0,<3.0
pandas-ta-classic>=0.3.59
yfinance>=0.2.36,<1.0
plotly>=6.0.0,<7.0
python-dotenv>=1.0.0
numpy>=1.26.0,<3.0
```

### Phase 2: データ取得 — data_fetcher.py

**`src/data_fetcher.py`** — `DataFetcher` クラス
- `fetch_all_stocks(progress_callback)`: 225銘柄を50銘柄ずつバッチでyf.download()。バッチ間に1秒sleep
- `fetch_single_stock(ticker_code)`: 単一銘柄の取得（オンデマンド）
- `_parse_batch_result()`: MultiIndex DataFrame を銘柄ごとに分割
- `_normalize_columns()`: カラム名統一（open, high, low, close, volume）
- エッジケース対応: 上場廃止銘柄の空DataFrame、レート制限、auto_adjust=True

### Phase 3: 分析エンジン — analyzer.py, calculator.py

**`src/analyzer.py`**

抽象基底クラス `BaseStrategy` + 4つの具象ストラテジー:

1. **EMAPerfectOrderStrategy** — EMA(25,75)パーフェクトオーダー（close > ema25 > ema75）。直近3日以内に成立するほど強度高
2. **RSIReversalStrategy** — RSI(14)が30-40で反転上昇（直近3本でRSI上昇中）
3. **BollingerBandStrategy** — BB(20,2.0)の-2σ接触後に陽線確認
4. **VolumeSpikeStrategy** — 5日平均出来高の1.5倍以上（2倍=0.75, 3倍+=1.0）

`SignalResult` dataclass: name, triggered, strength(0-1), description(日本語)

`SwingAnalyzer` クラス:
- `analyze_stock()`: 4ストラテジーを実行し加重スコア算出
- `analyze_all_stocks(db, risk_tolerance)`: 全銘柄分析。risk_toleranceで閾値調整（低い許容度=高い閾値=少ない推奨）
- スコア >= 0.4 の銘柄のみ推奨（設定画面で調整可能）

`StockRecommendation` dataclass: ticker_code, company_name, current_price, signal_score, signals, triggered_signals, recommendation_reason

**`src/calculator.py`** — `ExpectedValueCalculator` クラス
- `calculate_atr_based()`: ATR(14)ベース。損切り=現在値-2ATR, 利確=現在値+3ATR
- `calculate_pct_based()`: 固定率。デフォルト損切り-3%, 利確+5%（サイドバーで調整可）
- `calculate_position_size()`: 100株単位（単元株）で投資金額に合わせた株数算出
- `RiskRewardResult` dataclass: method, entry_price, stop_loss, take_profit, risk_reward_ratio, position_size, total_risk_yen

### Phase 4: チャート — charts.py

**`src/charts.py`**
- `create_candlestick_chart()`: make_subplots(3行)
  - Row1 (60%): ローソク足 + EMA(25,75) + ボリンジャーバンド(fill付き) + 損切り/利確の水平線
  - Row2 (20%): RSI(14) + 30/40/70基準線
  - Row3 (20%): 出来高バー（陽線=緑、陰線=赤）
- `create_backtest_histogram()`: 最大リターン分布
- `create_cumulative_return_chart()`: 累積リターン推移

### Phase 5: UI組み立て — app.py + pages/

**`src/app.py`**
- `st.set_page_config(page_title="J-Swing Insight", layout="wide")`
- `st.navigation()` で4ページ構成
- 共通サイドバー: データ更新ボタン、リスク許容度スライダー(0-1)、投資金額入力、損切り方式切替(ATR/固定率)、固定率の場合はSL%/TP%スライダー
- session_state: risk_tolerance, investment_amount, sl_method, pct_sl, pct_tp, selected_ticker, recommendations, last_update, db

**`src/pages/1_dashboard.py`** — 推奨銘柄ダッシュボード
- 上部: メトリクス（推奨数、平均スコア、最終更新）
- メイン: 銘柄テーブル（コード、銘柄名、現在値、スコア、推奨理由、詳細ボタン）
- データ更新ボタン → DataFetcher → SwingAnalyzer → 結果表示
- 詳細ボタン → session_state.selected_ticker設定 → 銘柄詳細ページへ遷移

**`src/pages/2_stock_detail.py`** — 銘柄詳細
- Plotlyフルチャート（全指標オーバーレイ）
- 期待値計算: ATR方式とパーセント方式を2カラムで並列表示
- シグナル詳細: 各ストラテジーの発動状況・強度

**`src/pages/3_backtest.py`** — バックテスト
- 実行ボタン → 過去3ヶ月の推奨を検証（5営業日後のパフォーマンス）
- 集計メトリクス: 勝率、平均リターン、対象銘柄数
- チャート: リターン分布ヒストグラム + 累積リターン推移
- 結果テーブル: 日付、コード、銘柄名、推奨時価格、5日最高値、最大リターン%

**`src/pages/4_settings.py`** — 設定
- ストラテジー重み調整（4つのスライダー、正規化）
- 推奨閾値スライダー
- ATR期間・乗数設定
- DB統計情報、DBリセットボタン

## 重要な設計ポイント

### データフロー
```
yfinance → DataFetcher → SQLite(daily_prices)
                              ↓
                        SwingAnalyzer → SQLite(recommendations)
                              ↓
                   Dashboard/StockDetail ← charts.py, calculator.py
                              ↓
                   Backtest → SQLite(backtest_results) → charts.py
```

### エッジケース対処
- 上場廃止・取引停止銘柄: 空DataFrameをスキップしログ出力
- yfinanceレート制限: 50銘柄バッチ + 1秒sleep
- データ不足（<75日）: 該当銘柄をスキップ
- 株式分割: auto_adjust=True で対処、大幅な乖離時はフル再取得
- SQLiteスレッド安全性: メソッドごとに接続作成、WALモード
- 100株単位（単元株）: position_sizeは必ず100の倍数に切り捨て

### UI言語
- ラベル・メッセージ: すべて日本語
- コード・コメント・変数名: 英語

## 検証方法

1. **requirements.txt から依存インストール**: `pip install -r requirements.txt`
2. **アプリ起動**: `streamlit run src/app.py`
3. **データ取得テスト**: ダッシュボードの「データ更新」ボタンをクリック → 225銘柄のデータがSQLiteに保存されることを確認
4. **分析テスト**: データ更新後、推奨銘柄一覧が表示されることを確認（閾値0.4以上のスコアを持つ銘柄）
5. **チャートテスト**: 推奨銘柄の「詳細」→ ローソク足チャートに全指標（EMA, BB, RSI, 出来高）が正しく描画されることを確認
6. **計算テスト**: ATR方式・固定率方式の両方で損切り/利確ラインが表示され、サイドバーで切り替え可能なことを確認
7. **バックテストテスト**: バックテストページで「実行」→ 集計メトリクスとチャートが表示されることを確認
8. **サイドバー連動テスト**: リスク許容度・投資金額を変更 → 推奨銘柄数とポジションサイズが連動して変化することを確認
