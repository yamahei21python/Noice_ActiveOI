# -*- coding: utf-8 -*-
"""
Coinalyze APIから指定された仮想通貨の市場データを取得・分析し、
結果をグラフ化してDiscordに投稿するスクリプト。

VPSなどでの常時稼働を想定し、メモリ効率とデータ永続性を考慮した設計です。

■ 必要なライブラリ
- requests: HTTPリクエスト用
- pandas: データ分析用
- matplotlib: グラフ描画用
- pyarrow: データ保存用 (Parquet形式)
- python-dotenv: 環境変数読み込み用

■ 実行前の設定
1. 必要なライブラリをインストールします。
   pip install requests pandas matplotlib pyarrow python-dotenv
2. スクリプトと同じ階層に`.env`ファイルを作成し、以下を記述します。
   API_KEY="CoinalyzeのAPIキー"
   DISCORD_WEBHOOK_URL="グラフ投稿用のDiscord Webhook URL"
   DISCORD_ALERT_WEBHOOK_URL="アラート通知用のDiscord Webhook URL"
   DISCORD_ALERT_END_WEBHOOK_URL="アラート解除通知専用のDiscord Webhook URL (オプション)"
"""
from dotenv import load_dotenv
load_dotenv() # .envファイルから環境変数を読み込む

import datetime
import gc
import os
import time
import json
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests

# --- グローバル定数・設定 ---

# API関連
API_KEY = os.environ.get("API_KEY")
OI_API_URL = "https://api.coinalyze.net/v1/open-interest-history"
PRICE_API_URL = "https://api.coinalyze.net/v1/ohlcv-history"

# Discord通知関連
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL") # 通常のグラフ投稿用
DISCORD_ALERT_WEBHOOK_URL = os.environ.get("DISCORD_ALERT_WEBHOOK_URL") # アラート発生通知用
DISCORD_ALERT_END_WEBHOOK_URL = os.environ.get("DISCORD_ALERT_END_WEBHOOK_URL") # アラート解除通知用 (オプション)

# 分析対象の仮想通貨リスト
TARGET_COINS = ["BTC", "ETH", "SOL"]

# データ処理・分析パラメータ
DATA_FETCH_PERIOD_SECONDS = 864000  # データ取得期間 (秒数で指定、864000秒 = 10日間)
API_REQUEST_INTERVAL = "5min"      # データ取得間隔 (Coinalyze APIの仕様に合わせる)
ROLLING_WINDOW_BINS = 12 * 24 * 3  # 3日間のローリングウィンドウ (5分足の場合、12(1時間) * 24(1日) * 3(日数))
API_WAIT_SECONDS = 30              # 複数通貨を処理する際のAPI負荷軽減のための待機時間 (秒)

# アラート判定の閾値
ALERT_THRESHOLD_LOWER = -3.5 # 下落アラートのMerge_STD閾値
ALERT_THRESHOLD_UPPER = 5.0  # 上昇アラートのMerge_STD閾値

# 高度なアラート制御用の定数
ALERT_END_COUNT_THRESHOLD = 10           # 終了通知を発動させるためのアラート回数閾値 (直近1時間以内)
ALERT_END_COOLDOWN_SECONDS = 12 * 3600   # 終了通知のクールダウン時間 (12時間)

# データ保存ディレクトリ
# スクリプトファイルと同じ階層に 'data' ディレクトリを作成して使用
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# --- アラート状態管理用のヘルパー関数 ---
STATUS_FILE_PATH = os.path.join(DATA_DIR, 'alert_status.json')

def load_alert_status() -> Dict:
    """
    アラート状態を記録したJSONファイルを読み込みます。
    ファイルが存在しない、または中身が不正な場合は空の辞書を返します。
    """
    if not os.path.exists(STATUS_FILE_PATH):
        return {}
    try:
        with open(STATUS_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # ファイルが空だったり、JSONとして不正な形式だった場合に備える
        return {}

def save_alert_status(status_data: Dict):
    """
    現在のアラート状態をJSONファイルに書き込みます。
    """
    with open(STATUS_FILE_PATH, 'w', encoding='utf-8') as f:
        # indent=2 で人間が読みやすい形式で保存
        json.dump(status_data, f, indent=2, ensure_ascii=False)


# --- 動的設定生成 ---

def get_exchange_config(coin: str) -> Dict[str, Dict[str, any]]:
    """通貨シンボルに基づいて、分析対象の取引所設定を動的に生成します。"""
    return {
        'Binance': {'code': 'A', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']},
        'Bybit': {'code': '6', 'contracts': [f'{coin}USD.', f'{coin}USDT.']},
        'OKX': {'code': '3', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']},
        'BitMEX': {'code': '0', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']}
    }


# --- データ取得 ---

def build_symbol_string(exchange_config: Dict[str, Dict[str, any]]) -> str:
    """APIリクエスト用に、取引所設定からシンボル文字列を構築します。"""
    symbols = []
    for config in exchange_config.values():
        for contract in config['contracts']:
            # "BTCUSDT.A" のような形式のシンボルを作成
            symbols.append(f"{contract}{config['code']}")
    return ','.join(symbols)

def fetch_data_from_api(url: str, params: Dict[str, any], headers: Dict[str, str]) -> Optional[List[Dict]]:
    """汎用的なAPIデータ取得関数。失敗時に最大1回リトライします。"""
    max_retries = 2  # 最大試行回数 (初回 + リトライ1回)
    retry_delay_seconds = 30
    
    for attempt in range(max_retries):
        try:
            print(f"APIにリクエストを送信します... (試行 {attempt + 1}/{max_retries})")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # 200番台以外のステータスコードの場合、HTTPErrorを発生させる
            print("APIリクエスト成功。")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"APIリクエストに失敗しました: {e}")
            if attempt < max_retries - 1:
                print(f"{retry_delay_seconds}秒後にリトライします...")
                time.sleep(retry_delay_seconds)
            else:
                print("最大リトライ回数に達しました。")
    
    return None

def fetch_open_interest_data(exchange_config: Dict) -> List[Dict]:
    """Coinalyzeから建玉(Open Interest)の履歴データを取得します。"""
    if not API_KEY: return []
    end_time = int(time.time())
    start_time = end_time - DATA_FETCH_PERIOD_SECONDS
    headers = {"api-key": API_KEY}
    params = {
        "symbols": build_symbol_string(exchange_config), "interval": API_REQUEST_INTERVAL,
        "from": start_time, "to": end_time, "convert_to_usd": "true"
    }
    data = fetch_data_from_api(OI_API_URL, params, headers)
    return data if data else []

def fetch_price_data(price_symbol: str) -> List[Dict]:
    """Coinalyzeから価格(OHLCV)の履歴データを取得します。"""
    if not API_KEY: return []
    end_time = int(time.time())
    start_time = end_time - DATA_FETCH_PERIOD_SECONDS
    headers = {"api-key": API_KEY}
    params = {"symbols": price_symbol, "interval": API_REQUEST_INTERVAL, "from": start_time, "to": end_time}
    api_response = fetch_data_from_api(PRICE_API_URL, params, headers)
    # 価格データはレスポンスのリストの最初の要素に格納されている
    if api_response and isinstance(api_response, list) and api_response[0]:
        return api_response[0].get("history", [])
    return []


# --- データ処理 ---

def process_oi_api_data(api_data: List[Dict], code_to_name_map: Dict[str, str]) -> pd.DataFrame:
    """APIから取得した建玉データを整形し、取引所と通貨タイプ(USD/USDT)ごとに集計したDataFrameを返します。"""
    if not api_data: return pd.DataFrame()
    all_dfs = []
    for item in api_data:
        symbol, history = item.get("symbol"), item.get("history")
        if not (symbol and history): continue
        contract_name, exchange_code = symbol.rsplit('.', 1)
        exchange_name = code_to_name_map.get(exchange_code)
        if not exchange_name: continue
        currency_type = 'USDT' if 'USDT' in contract_name else 'USD'
        group_name = f"{exchange_name}_{currency_type}"
        df = pd.DataFrame(history)
        df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'}).set_index('Datetime')
        # メモリ効率化のためデータ型をfloat32に最適化
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        ohlc_df = df[['Open', 'High', 'Low', 'Close']]
        ohlc_df.columns = pd.MultiIndex.from_product([[group_name], ohlc_df.columns])
        all_dfs.append(ohlc_df)
    if not all_dfs: return pd.DataFrame()
    # 複数のDataFrameを結合し、同じ取引所・通貨タイプ・指標のデータを合計する
    temp_df = pd.concat(all_dfs, axis=1)
    combined_df = temp_df.T.groupby(level=[0, 1]).sum(min_count=1).T
    combined_df.columns = [f"{ex_type}_{met}" for ex_type, met in combined_df.columns]
    # 欠損値を線形補間
    return combined_df.interpolate().reset_index().dropna(how='all', subset=combined_df.columns).reset_index(drop=True)

def process_price_data(price_history: List[Dict]) -> pd.DataFrame:
    """APIから取得した価格データを整形し、DataFrameを返します。"""
    if not price_history: return pd.DataFrame()
    df = pd.DataFrame(price_history)
    df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
    df['c'] = pd.to_numeric(df['c'], errors='coerce').astype('float32')
    return df.rename(columns={'c': 'Bybit_Price_Close'})[['Datetime', 'Bybit_Price_Close']]

def calculate_active_oi(df: pd.DataFrame, group_names: List[str]) -> pd.DataFrame:
    """3日間の安値からの差分を「Active OI」として計算します。"""
    df = df.set_index('Datetime')
    active_oi_df = pd.DataFrame(index=df.index)
    for name in group_names:
        low_col, close_col = f'{name}_Low', f'{name}_Close'
        if low_col in df.columns and close_col in df.columns:
            # 3日間のローリングウィンドウで安値の最小値を取得
            min_3day = df[low_col].rolling(window=ROLLING_WINDOW_BINS, min_periods=1).min()
            active_oi_df[f'{name}_Active_OI_5min'] = (df[close_col] - min_3day).astype('float32')
    return active_oi_df.dropna(how='all').reset_index()

def aggregate_and_standardize_oi(df: pd.DataFrame) -> pd.DataFrame:
    """全取引所のActive OIを合計し、標準化（Z-score）します。"""
    df = df.set_index('Datetime')
    active_oi_cols = [col for col in df.columns if 'Active_OI_5min' in col]
    df['ALL_Active_OI_5min'] = df[active_oi_cols].sum(axis=1)
    # 3日間のローリングで平均と標準偏差を計算
    rolling_stats = df['ALL_Active_OI_5min'].rolling(window=ROLLING_WINDOW_BINS)
    mean, std = rolling_stats.mean(), rolling_stats.std()
    result_df = pd.DataFrame(index=df.index)
    # Z-score = (現在の値 - 平均) / 標準偏差
    result_df['STD_Active_OI'] = ((df['ALL_Active_OI_5min'] - mean) / std.replace(0, pd.NA)).astype('float32')
    return result_df.reset_index()

def calculate_price_std(df: pd.DataFrame) -> pd.DataFrame:
    """Bybitの価格を標準化（Z-score）します。"""
    if 'Bybit_Price_Close' not in df.columns: return pd.DataFrame(columns=['Datetime', 'Bybit_price_STD'])
    price_df = df[['Datetime', 'Bybit_Price_Close']].copy().set_index('Datetime')
    rolling_stats = price_df['Bybit_Price_Close'].rolling(window=ROLLING_WINDOW_BINS)
    mean, std = rolling_stats.mean(), rolling_stats.std()
    result_df = pd.DataFrame(index=price_df.index)
    result_df['Bybit_price_STD'] = ((price_df['Bybit_Price_Close'] - mean) / std.replace(0, pd.NA)).astype('float32')
    return result_df.reset_index()

# --- ここから変更 ---
def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """5分足と15分足のボリンジャーバンドを計算します。"""
    if 'Bybit_Price_Close' not in df.columns:
        return df

    # 5分足のボリンジャーバンド計算
    rolling_5min = df['Bybit_Price_Close'].rolling(window=20)
    df['bb_sma_5min'] = rolling_5min.mean()
    df['bb_std_5min'] = rolling_5min.std()
    df['bb_upper_5min'] = df['bb_sma_5min'] + (df['bb_std_5min'] * 2)
    df['bb_lower_5min'] = df['bb_sma_5min'] - (df['bb_std_5min'] * 2)

    # 15分足の価格データをリサンプリングして作成
    df_15min = df.set_index('Datetime')[['Bybit_Price_Close']].resample('15T').last()
    
    # 15分足のボリンジャーバンド計算
    rolling_15min = df_15min['Bybit_Price_Close'].rolling(window=20)
    df_15min['bb_sma_15min'] = rolling_15min.mean()
    df_15min['bb_std_15min'] = rolling_15min.std()
    df_15min['bb_upper_15min'] = df_15min['bb_sma_15min'] + (df_15min['bb_std_15min'] * 2)
    df_15min['bb_lower_15min'] = df_15min['bb_sma_15min'] - (df_15min['bb_std_15min'] * 2)

    # 元のDataFrameに15分足のデータを結合
    # how='left'で結合し、前方フィルで欠損値を埋めることで階段状のデータを表現
    df = pd.merge(df, df_15min[['bb_upper_15min', 'bb_lower_15min']], on='Datetime', how='left')
    df['bb_upper_15min'].fillna(method='ffill', inplace=True)
    df['bb_lower_15min'].fillna(method='ffill', inplace=True)

    return df
# --- ここまで変更 ---


# --- グラフ描画 & Discord通知 ---

def plot_figure(df: pd.DataFrame, save_path: str, coin: str, group_names: List[str]):
    """分析結果を3段のグラフとして描画し、ファイルに保存します。"""
    df_plot = df.dropna(subset=['Merge_STD', 'Bybit_Price_Close']).reset_index(drop=True)
    if df_plot.empty:
        print(f"[{coin}] グラフ描画用のデータが存在しないため、スキップします。")
        return
    # 下落・上昇アラートの条件を定義 (グラフの背景色変更用)
    down_alert_condition = ((df_plot['Merge_STD'] < ALERT_THRESHOLD_LOWER) & (df_plot['Bybit_price_STD'] < -1) & (df_plot['STD_Active_OI'] < -1))
    up_alert_condition = ((df_plot['Merge_STD'] > ALERT_THRESHOLD_UPPER) & (df_plot['Bybit_price_STD'] > 1.5) & (df_plot['STD_Active_OI'] > 1.5))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    latest_row = df_plot.iloc[-1]
    fig.suptitle(f"{coin} OI Analysis ({latest_row['Datetime'].strftime('%Y-%m-%d %H:%M %Z')})", fontsize=16)
    
    # 1段目: 価格チャート
    title_text = (f"Bybit Price: {latest_row['Bybit_Price_Close']:,.2f}\n"
                  f"Merge_STD: {latest_row['Merge_STD']:.2f} | "
                  f"Price_STD: {latest_row['Bybit_price_STD']:.2f} | "
                  f"Active_OI_STD: {latest_row['STD_Active_OI']:.2f}")
    ax1.set_title(title_text, loc='right', color='darkred', fontsize=10)
    
    # BTC/ETHの場合は価格を1000で割ってk USD単位で表示
    price_divisor = 1000 if coin in ["BTC", "ETH"] else 1
    price_label = "Price (k USD)" if coin in ["BTC", "ETH"] else "Price (USD)"
    
    # メインの価格ライン
    ax1.plot(df_plot['Datetime'], df_plot['Bybit_Price_Close'] / price_divisor, label=price_label, color='orangered', linewidth=1.5)
    
    # 5分足ボリンジャーバンド (±2σ)
    if 'bb_upper_5min' in df_plot.columns and 'bb_lower_5min' in df_plot.columns:
        ax1.plot(df_plot['Datetime'], df_plot['bb_upper_5min'] / price_divisor, label='5min 2σ', color='gray', linestyle='--', linewidth=0.8)
        ax1.plot(df_plot['Datetime'], df_plot['bb_lower_5min'] / price_divisor, color='gray', linestyle='--', linewidth=0.8)

    # 15分足ボリンジャーバンド (±2σ)
    if 'bb_upper_15min' in df_plot.columns and 'bb_lower_15min' in df_plot.columns:
        ax1.plot(df_plot['Datetime'], df_plot['bb_upper_15min'] / price_divisor, label='15min 2σ', color='dimgray', linestyle=':', linewidth=1.2)
        ax1.plot(df_plot['Datetime'], df_plot['bb_lower_15min'] / price_divisor, color='dimgray', linestyle=':', linewidth=1.2)

    # --- ここから変更 ---
    # ボリンジャーバンドのブレイク状態を表示するテキストを生成
    status_5min = "-"
    if 'bb_upper_5min' in latest_row and latest_row['Bybit_Price_Close'] > latest_row['bb_upper_5min']:
        status_5min = "🟢" # 上抜け
    elif 'bb_lower_5min' in latest_row and latest_row['Bybit_Price_Close'] < latest_row['bb_lower_5min']:
        status_5min = "🔴" # 下抜け
        
    status_15min = "-"
    if 'bb_upper_15min' in latest_row and latest_row['Bybit_Price_Close'] > latest_row['bb_upper_15min']:
        status_15min = "🟢" # 上抜け
    elif 'bb_lower_15min' in latest_row and latest_row['Bybit_Price_Close'] < latest_row['bb_lower_15min']:
        status_15min = "🔴" # 下抜け
    
    # 表示用のテキストを組み立て
    # f-string内で f"..." と {} を使うために、中括弧を二重にする {{}}
    bb_status_text = f"5min:  {status_5min}\n15min: {status_15min}"
    
    # テキストをグラフの右上に描画
    ax1.text(0.99, 0.95, bb_status_text,
             transform=ax1.transAxes, # 座標を軸の相対位置で指定
             fontsize=8,
             fontweight='bold',
             verticalalignment='top',   # テキストボックスの上辺を基準に配置
             horizontalalignment='right', # テキストボックスの右辺を基準に配置
             bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.5)) # 見やすいように背景ボックスを追加
    # --- ここまで変更 ---

    ax1.set_ylabel(price_label); ax1.legend(loc='upper left'); ax1.grid(True, which="both"); ax1.yaxis.tick_right(); ax1.yaxis.set_label_position('right')

    # 2段目: 標準化された指標
    ax2.plot(df_plot['Datetime'], df_plot['Merge_STD'], label='Merge_STD', color='orangered')
    ax2.plot(df_plot['Datetime'], df_plot['Bybit_price_STD'], label='Price_STD', color='aqua')
    ax2.plot(df_plot['Datetime'], df_plot['STD_Active_OI'], label='Active_OI_STD', color='green')
    ax2.set_ylabel("Z-Score"); ax2.legend(loc='upper left'); ax2.grid(True, which="both"); ax2.yaxis.tick_right(); ax2.yaxis.set_label_position('right')

    # 3段目: Active OIの内訳 (積み上げグラフ)
    color_map = {'Binance': {'USDT': '#00529B', 'USD': '#65A9E0'}, 'Bybit': {'USDT': '#FF8C00', 'USD': '#FFD699'},
                 'OKX': {'USDT': '#006400', 'USD': '#66CDAA'}, 'BitMEX': {'USDT': '#B22222', 'USD': '#F08080'}}
    active_oi_cols_exist = [c for c in [f'{n}_Active_OI_5min' for n in group_names] if c in df_plot.columns]
    if active_oi_cols_exist:
        stack_data = [df_plot[c] / 1_000_000 for c in active_oi_cols_exist] # 単位をM USDに
        labels = [c.replace('_Active_OI_5min', '') for c in active_oi_cols_exist]
        plot_colors = [color_map.get(l.split('_')[0], {}).get(l.split('_')[1], '#808080') for l in labels]
        ax3.stackplot(df_plot['Datetime'], stack_data, labels=labels, colors=plot_colors)
    
    # アラート期間の背景を塗りつぶし
    y_min1, y_max1 = ax1.get_ylim(); y_min2, y_max2 = ax2.get_ylim(); y_min3, y_max3 = ax3.get_ylim()
    ax1.fill_between(df_plot['Datetime'], y_min1, y_max1, where=up_alert_condition, facecolor='lightblue', alpha=0.3, interpolate=True)
    ax2.fill_between(df_plot['Datetime'], y_min2, y_max2, where=up_alert_condition, facecolor='lightblue', alpha=0.3, interpolate=True)
    ax3.fill_between(df_plot['Datetime'], y_min3, y_max3, where=up_alert_condition, facecolor='lightblue', alpha=0.3, interpolate=True)
    ax1.fill_between(df_plot['Datetime'], y_min1, y_max1, where=down_alert_condition, facecolor='lightcoral', alpha=0.3, interpolate=True)
    ax2.fill_between(df_plot['Datetime'], y_min2, y_max2, where=down_alert_condition, facecolor='lightcoral', alpha=0.3, interpolate=True)
    ax3.fill_between(df_plot['Datetime'], y_min3, y_max3, where=down_alert_condition, facecolor='lightcoral', alpha=0.3, interpolate=True)

    ax3.set_ylabel("Active OI (M USD)"); ax3.legend(loc='upper left', fontsize='small'); ax3.grid(True, which="both"); ax3.yaxis.tick_right(); ax3.yaxis.set_label_position('right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close() # メモリ解放のため明示的に閉じる
    print(f"グラフを '{save_path}' に保存しました。")

def send_to_discord(message: str, image_path: str, webhook_url: Optional[str]):
    """Discordにメッセージと画像をWebhookで投稿します。"""
    if not webhook_url:
        print("Discordへの通知設定（Webhook URL）が不十分です。")
        return
    try:
        with open(image_path, "rb") as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            payload = {"content": message}
            response = requests.post(webhook_url, data=payload, files=files)
            response.raise_for_status()
            print(f"Discordに「{message}」を投稿しました。")
    except FileNotFoundError:
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
    except requests.exceptions.RequestException as e:
        print(f"Discordへの投稿に失敗しました: {e}")


# --- メイン処理 ---
def run_analysis_for_coin(coin: str):
    """単一の通貨に対して分析パイプライン全体を実行します。"""
    jst = datetime.timezone(datetime.timedelta(hours=9))
    print(f"--- [{coin}] 処理開始: {datetime.datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

    # 1. ファイルパス定義
    figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')
    data_path = os.path.join(DATA_DIR, f'{coin.lower()}_analysis_data.parquet')

    # 2. データ取得
    exchange_config = get_exchange_config(coin)
    code_to_name_map = {v['code']: k for k, v in exchange_config.items()}
    price_symbol = f"{coin}USDT.6" # BybitのUSDT無期限先物を価格の基準とする
    raw_oi_data = process_oi_api_data(fetch_open_interest_data(exchange_config), code_to_name_map)
    price_data = process_price_data(fetch_price_data(price_symbol))
    if raw_oi_data.empty or price_data.empty:
        print(f"[{coin}] データ取得に失敗、またはデータが空のため処理を中断します。")
        return
        
    # 3. データ処理と各種指標の計算
    base_df = pd.merge(raw_oi_data, price_data, on='Datetime', how='left').interpolate()
    if base_df.empty:
        print(f"[{coin}] 初期マージ後のデータが空のため処理を中断します。")
        return
    exchange_currency_groups = [f"{n}_{ct}" for n, conf in exchange_config.items() for ct in (['USD'] if any('USDT' not in c for c in conf['contracts']) else []) + (['USDT'] if any('USDT' in c for c in conf['contracts']) else [])]
    active_oi_data = calculate_active_oi(base_df, exchange_currency_groups)
    standardized_oi_data = aggregate_and_standardize_oi(active_oi_data)
    price_std_data = calculate_price_std(base_df)
    
    # 計算済みのDataFrameをマージしていく
    all_data = pd.merge(base_df, active_oi_data, on='Datetime', how='inner')
    all_data = pd.merge(all_data, standardized_oi_data, on='Datetime', how='inner')
    all_data = pd.merge(all_data, price_std_data, on='Datetime', how='inner')
    
    # メモリ解放
    del raw_oi_data, price_data, base_df, active_oi_data, standardized_oi_data, price_std_data
    gc.collect()

    # --- ここから変更 ---
    # ボリンジャーバンドを計算
    all_data = calculate_bollinger_bands(all_data)
    # --- ここまで変更 ---

    # 最終的な分析指標を計算
    if 'STD_Active_OI' in all_data.columns and 'Bybit_price_STD' in all_data.columns:
        all_data['Merge_STD'] = all_data['Bybit_price_STD'] + all_data['STD_Active_OI']
    all_data.dropna(inplace=True)
    
    # 4. グラフ生成 & 通常のDiscord投稿
    plot_figure(all_data, figure_path, coin, exchange_currency_groups)
    if os.path.exists(figure_path) and DISCORD_WEBHOOK_URL:
        send_to_discord(f"📈 **{coin}** 分析グラフ", figure_path, DISCORD_WEBHOOK_URL)
    
    # 5. アラート判定と通知
    if all_data.empty:
        print(f"[{coin}] 最終データが空のため処理を中断します。")
        return
    
    now_timestamp = int(time.time())
    all_statuses = load_alert_status()
    # 状態ファイルに通貨の情報がなければ、デフォルト値を設定
    coin_status = all_statuses.get(coin, {"status": "NORMAL", "alert_timestamps": [], "last_end_notification_timestamp": None, "last_alert_details": None})
    previous_status = coin_status.get("status", "NORMAL")
    
    latest = all_data.iloc[-1]
    now_merge_std, now_price_std, now_oi_std = latest.get('Merge_STD'), latest.get('Bybit_price_STD'), latest.get('STD_Active_OI')
    
    current_status, alert_message = "NORMAL", None
    
    # 最新のデータでアラート条件をチェック
    if all(v is not None for v in [now_merge_std, now_price_std, now_oi_std]):
        if now_merge_std < ALERT_THRESHOLD_LOWER and now_price_std < -1 and now_oi_std < -1:
            current_status = "DOWN_ALERT"
            alert_message = (f"**🚨 [下落アラート] {coin} Alert** ({latest['Datetime'].strftime('%Y/%m/%d %H:%M')})\n"
                           f"> **Merge_STD: {now_merge_std:.2f}** (条件: < {ALERT_THRESHOLD_LOWER})\n"
                           f"> **Price_STD: {now_price_std:.2f}** (条件: < -1)\n"
                           f"> **Active_OI_STD: {now_oi_std:.2f}** (条件: < -1)\n"
                           f"Price: {latest['Bybit_Price_Close']:,.2f}")
        elif now_merge_std > ALERT_THRESHOLD_UPPER and now_price_std > 1.5 and now_oi_std > 1.5:
            current_status = "UP_ALERT"
            alert_message = (f"**🚨 [上昇アラート] {coin} Alert** ({latest['Datetime'].strftime('%Y/%m/%d %H:%M')})\n"
                           f"> **Merge_STD: {now_merge_std:.2f}** (条件: > {ALERT_THRESHOLD_UPPER})\n"
                           f"> **Price_STD: {now_price_std:.2f}** (条件: > 1.5)\n"
                           f"> **Active_OI_STD: {now_oi_std:.2f}** (条件: > 1.5)\n"
                           f"Price: {latest['Bybit_Price_Close']:,.2f}")
    
    # アラート発生時刻の履歴を更新
    one_hour_ago = now_timestamp - 3600
    timestamps = [t for t in coin_status.get("alert_timestamps", []) if t > one_hour_ago] # 1時間より古い記録は削除
    if current_status != "NORMAL":
        timestamps.append(now_timestamp) # アラート状態なら現在時刻を追加
    
    # 新規アラート通知（即時通知）
    if alert_message and os.path.exists(figure_path) and DISCORD_ALERT_WEBHOOK_URL:
        print(f"[{coin}] アラート発生条件を満たしました。通知を送信します。")
        send_to_discord(alert_message, figure_path, DISCORD_ALERT_WEBHOOK_URL)
        # アラート発生時の詳細情報を、終了通知で使えるように保存
        coin_status['last_alert_details'] = {
            "datetime_str": latest['Datetime'].strftime('%Y/%m/%d %H:%M'),
            "merge_std": float(now_merge_std),
            "price_std": float(now_price_std),
            "oi_std": float(now_oi_std),
            "price": float(latest['Bybit_Price_Close'])
        }

    # アラート終了通知の判定
    if current_status == "NORMAL" and len(timestamps) >= ALERT_END_COUNT_THRESHOLD:
        last_end_time = coin_status.get("last_end_notification_timestamp")
        # クールダウン期間が過ぎているかチェック
        if last_end_time is None or (now_timestamp - last_end_time) > ALERT_END_COOLDOWN_SECONDS:
            print(f"[{coin}] アラート終了条件（頻発後の収束）を満たしました。通知を送信します。")
            
            # 終了通知用のメッセージを生成
            end_message_type = previous_status.replace('_', ' ') if previous_status != "NORMAL" else "アラート"
            end_message = f"✅ **[{end_message_type} 解除]** {coin} は通常状態に戻りました。(直近1時間のアラート回数: {len(timestamps)}回)\n"
            
            # 保存しておいた最後のアラート詳細情報をメッセージに追加
            last_details = coin_status.get('last_alert_details')
            if last_details:
                is_up_alert = "UP" in previous_status
                merge_std_threshold = ALERT_THRESHOLD_UPPER if is_up_alert else ALERT_THRESHOLD_LOWER
                price_std_threshold = 1.5 if is_up_alert else -1
                oi_std_threshold = 1.5 if is_up_alert else -1
                op = ">" if is_up_alert else "<"

                details_message = (
                    f"> 以下のアラートが収束しました (最終検知: {last_details.get('datetime_str', 'N/A')})\n"
                    f"> **Merge_STD: {last_details.get('merge_std', 0):.2f}** (条件: {op} {merge_std_threshold})\n"
                    f"> **Price_STD: {last_details.get('price_std', 0):.2f}** (条件: {op} {price_std_threshold})\n"
                    f"> **Active_OI_STD: {last_details.get('oi_std', 0):.2f}** (条件: {op} {oi_std_threshold})\n"
                    f"> Price: {last_details.get('price', 0):,.2f}"
                )
                end_message += details_message

            # 解除専用URLが設定されていればそちらを使い、なければ通常のアラート用URLを使う
            webhook_for_end_alert = DISCORD_ALERT_END_WEBHOOK_URL or DISCORD_ALERT_WEBHOOK_URL
            if webhook_for_end_alert:
                # 終了通知はテキストのみ送信
                requests.post(webhook_for_end_alert, data={"content": end_message})
            
            coin_status["last_end_notification_timestamp"] = now_timestamp
            timestamps = [] # 終了通知を出したら、タイムスタンプリストをクリア
        else:
            print(f"[{coin}] アラートは収束しましたが、クールダウン期間中のため終了通知はスキップします。")

    # 今回の状態をファイルに保存
    coin_status["status"] = current_status
    coin_status["alert_timestamps"] = timestamps
    all_statuses[coin] = coin_status
    save_alert_status(all_statuses)
    
    # 6. 処理済みデータをParquet形式で保存
    try:
        all_data.to_parquet(data_path, index=False)
        print(f"処理済みデータを '{data_path}' に保存しました。")
    except Exception as e:
        print(f"データ保存中にエラーが発生しました: {e}")
    print(f"--- [{coin}] 処理完了 ---\n")


def main():
    """スクリプトのメインエントリポイント。"""
    # 必須の環境変数をチェック
    if not API_KEY:
        print("エラー: 環境変数 `API_KEY` が設定されていません。処理を終了します。")
        return
    if not DISCORD_WEBHOOK_URL:
        print("警告: 環境変数 `DISCORD_WEBHOOK_URL` が設定されていません。グラフ投稿はスキップされます。")
    if not DISCORD_ALERT_WEBHOOK_URL:
        print("警告: 環境変数 `DISCORD_ALERT_WEBHOOK_URL` が設定されていません。アラート通知はスキップされます。")
    if not DISCORD_ALERT_END_WEBHOOK_URL:
        print("情報: 環境変数 `DISCORD_ALERT_END_WEBHOOK_URL` は未設定です。アラート解除通知は `DISCORD_ALERT_WEBHOOK_URL` に送信されます。")

    # データ保存ディレクトリが存在しない場合は作成
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"データ保存ディレクトリ '{DATA_DIR}' を作成しました。")
    
    # ターゲット通貨を順番に処理
    for i, coin in enumerate(TARGET_COINS):
        run_analysis_for_coin(coin)
        # 最後の通貨でなければ、API負荷軽減のために待機
        if i < len(TARGET_COINS) - 1:
            print(f"API負荷軽減のため {API_WAIT_SECONDS} 秒待機します...")
            time.sleep(API_WAIT_SECONDS)

if __name__ == "__main__":
    main()
