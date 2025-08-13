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

■ 実行前の設定
1. 必要なライブラリをインストールします。
   pip install requests pandas matplotlib pyarrow
2. 以下の環境変数を設定します。
   - API_KEY: CoinalyzeのAPIキー
   - DISCORD_WEBHOOK_URL: グラフ投稿用のDiscord Webhook URL
   - (オプション) DISCORD_BOT_TOKEN: アラート通知用Botのトークン
   - (オプション) DISCORD_CHANNEL_ID: アラート通知先チャンネルID
"""

import datetime
import gc
import os
import time
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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "https://discordapp.com/api/webhooks/1395228757585035374/p4UYCIgmELzTG3-6MX6UTc0ihlf6isXi-_8FIZfTjqzuVbU415JMZGWvEP32e81lXlJA")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID")

# 分析対象
TARGET_COINS = ["BTC", "ETH", "SOL"]

# データ処理・分析パラメータ
DATA_FETCH_PERIOD_SECONDS = 864000  # データ取得期間 (10日間)
API_REQUEST_INTERVAL = "5min"      # データ取得間隔
ROLLING_WINDOW_BINS = 12 * 24 * 3  # 3日間のローリングウィンドウ (5分足換算)
API_WAIT_SECONDS = 30              # 通貨ごとの処理の合間の待機時間 (秒)

# アラート閾値
ALERT_THRESHOLD_LOWER = -3.5
ALERT_THRESHOLD_UPPER = 5.0

# データ保存ディレクトリ
# スクリプトファイルと同じ階層に 'data' ディレクトリを作成
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


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
            symbols.append(f"{contract}{config['code']}")
    return ','.join(symbols)

def fetch_data_from_api(url: str, params: Dict[str, any], headers: Dict[str, str]) -> Optional[List[Dict]]:
    """汎用的なAPIデータ取得関数。"""
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"APIリクエストに失敗しました: {e}")
        return None

def fetch_open_interest_data(exchange_config: Dict) -> List[Dict]:
    """Coinalyzeから建玉(Open Interest)の履歴データを取得します。"""
    if not API_KEY:
        return []
    
    end_time = int(time.time())
    start_time = end_time - DATA_FETCH_PERIOD_SECONDS
    
    headers = {"api-key": API_KEY}
    params = {
        "symbols": build_symbol_string(exchange_config),
        "interval": API_REQUEST_INTERVAL,
        "from": start_time,
        "to": end_time,
        "convert_to_usd": "true"
    }
    
    data = fetch_data_from_api(OI_API_URL, params, headers)
    return data if data else []

def fetch_price_data(price_symbol: str) -> List[Dict]:
    """Coinalyzeから価格(OHLCV)の履歴データを取得します。"""
    if not API_KEY:
        return []

    end_time = int(time.time())
    start_time = end_time - DATA_FETCH_PERIOD_SECONDS
    
    headers = {"api-key": API_KEY}
    params = {
        "symbols": price_symbol,
        "interval": API_REQUEST_INTERVAL,
        "from": start_time,
        "to": end_time
    }
    
    api_response = fetch_data_from_api(PRICE_API_URL, params, headers)
    # 価格データはレスポンスのリストの最初の要素に格納されている
    if api_response and isinstance(api_response, list) and api_response[0]:
        return api_response[0].get("history", [])
    return []


# --- データ処理 ---

def process_oi_api_data(api_data: List[Dict], code_to_name_map: Dict[str, str]) -> pd.DataFrame:
    """APIから取得した建玉データを整形し、取引所ごとに集計したDataFrameを返します。"""
    if not api_data:
        return pd.DataFrame()

    all_dfs = []
    for item in api_data:
        symbol, history = item.get("symbol"), item.get("history")
        if not (symbol and history):
            continue

        _, exchange_code = symbol.rsplit('.', 1)
        exchange_name = code_to_name_map.get(exchange_code)
        if not exchange_name:
            continue

        df = pd.DataFrame(history)
        df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'}).set_index('Datetime')
        
        # メモリ効率化のためデータ型を最適化
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        
        ohlc_df = df[['Open', 'High', 'Low', 'Close']]
        ohlc_df.columns = pd.MultiIndex.from_product([[exchange_name], ohlc_df.columns])
        all_dfs.append(ohlc_df)

    if not all_dfs:
        return pd.DataFrame()

    # 複数のDataFrameを結合し、同じ取引所・指標のデータを合計する
    temp_df = pd.concat(all_dfs, axis=1)
    combined_df = temp_df.T.groupby(level=[0, 1]).sum(min_count=1).T
    combined_df.columns = [f"{ex}_{met}" for ex, met in combined_df.columns]
    
    # 欠損値を補間し、全列がNaNの行を削除
    return combined_df.interpolate().reset_index().dropna(how='all', subset=combined_df.columns).reset_index(drop=True)

def process_price_data(price_history: List[Dict]) -> pd.DataFrame:
    """APIから取得した価格データを整形し、DataFrameを返します。"""
    if not price_history:
        return pd.DataFrame()

    df = pd.DataFrame(price_history)
    df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
    # メモリ効率化のためデータ型を最適化
    df['c'] = pd.to_numeric(df['c'], errors='coerce').astype('float32')
    return df.rename(columns={'c': 'Bybit_Price_Close'})[['Datetime', 'Bybit_Price_Close']]

def calculate_active_oi(df: pd.DataFrame, exchange_names: List[str]) -> pd.DataFrame:
    """3日間の安値からの差分を「Active OI」として計算します。"""
    df = df.set_index('Datetime')
    active_oi_df = pd.DataFrame(index=df.index)
    
    for name in exchange_names:
        low_col, close_col = f'{name}_Low', f'{name}_Close'
        if low_col in df.columns and close_col in df.columns:
            min_3day = df[low_col].rolling(window=ROLLING_WINDOW_BINS, min_periods=1).min()
            active_oi_df[f'{name}_Active_OI_5min'] = (df[close_col] - min_3day).astype('float32')
            
    return active_oi_df.dropna(how='all').reset_index()

def aggregate_and_standardize_oi(df: pd.DataFrame) -> pd.DataFrame:
    """全取引所のActive OIを合計し、標準化（Z-score）します。"""
    df = df.set_index('Datetime')
    active_oi_cols = [col for col in df.columns if 'Active_OI_5min' in col]
    df['ALL_Active_OI_5min'] = df[active_oi_cols].sum(axis=1)
    
    rolling_stats = df['ALL_Active_OI_5min'].rolling(window=ROLLING_WINDOW_BINS)
    mean, std = rolling_stats.mean(), rolling_stats.std()
    
    result_df = pd.DataFrame(index=df.index)
    # 0除算を避けるため、stdが0の場合はNAとして扱う
    result_df['STD_Active_OI'] = ((df['ALL_Active_OI_5min'] - mean) / std.replace(0, pd.NA)).astype('float32')
    return result_df.reset_index()

def calculate_price_std(df: pd.DataFrame) -> pd.DataFrame:
    """Bybitの価格を標準化（Z-score）します。"""
    if 'Bybit_Price_Close' not in df.columns:
        return pd.DataFrame(columns=['Datetime', 'Bybit_price_STD'])

    price_df = df[['Datetime', 'Bybit_Price_Close']].copy().set_index('Datetime')
    rolling_stats = price_df['Bybit_Price_Close'].rolling(window=ROLLING_WINDOW_BINS)
    mean, std = rolling_stats.mean(), rolling_stats.std()
    
    result_df = pd.DataFrame(index=price_df.index)
    # 0除算を避けるため、stdが0の場合はNAとして扱う
    result_df['Bybit_price_STD'] = ((price_df['Bybit_Price_Close'] - mean) / std.replace(0, pd.NA)).astype('float32')
    return result_df.reset_index()


# --- グラフ描画 & Discord通知 ---

def plot_figure(df: pd.DataFrame, save_path: str, coin: str, exchange_names: List[str]):
    """分析結果を3段のグラフとして描画し、ファイルに保存します。"""
    df_plot = df.dropna(subset=['Merge_STD', 'Bybit_Price_Close']).reset_index(drop=True)
    if df_plot.empty:
        print(f"[{coin}] グラフ描画用のデータが存在しないため、スキップします。")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    
    latest_price = df_plot['Bybit_Price_Close'].iloc[-1]
    latest_datetime = df_plot['Datetime'].iloc[-1]
    fig.suptitle(f"{coin} OI Analysis ({latest_datetime.strftime('%Y-%m-%d %H:%M %Z')})", fontsize=16)

    # 1段目: 価格チャート
    ax1.set_title(f"Bybit Price: {latest_price:,.2f}", loc='right', color='darkred')
    price_label, price_data = ("Price (k USD)", df_plot['Bybit_Price_Close'] / 1000) if coin in ["BTC", "ETH"] else ("Price (USD)", df_plot['Bybit_Price_Close'])
    ax1.plot(df_plot['Datetime'], price_data, label=price_label, color='orangered')
    ax1.set_ylabel(price_label)
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

    # 2段目: 標準化された指標
    ax2.plot(df_plot['Datetime'], df_plot['Merge_STD'], label='Merge_STD', color='orangered')
    ax2.plot(df_plot['Datetime'], df_plot['Bybit_price_STD'], label='Price_STD', color='aqua')
    ax2.plot(df_plot['Datetime'], df_plot['STD_Active_OI'], label='Active_OI_STD', color='green')
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc='upper left')
    ax2.grid(True, which="both")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')

    # 3段目: Active OIの内訳 (USD)
    active_oi_cols_exist = [c for c in [f'{n}_Active_OI_5min' for n in exchange_names] if c in df_plot.columns]
    if active_oi_cols_exist:
        stack_data = [df_plot[c] / 1_000_000 for c in active_oi_cols_exist] # M USD単位に変換
        labels = [c.split('_')[0] for c in active_oi_cols_exist]
        ax3.stackplot(df_plot['Datetime'], stack_data, labels=labels)
    
    ax3.set_ylabel("Active OI (M USD)")
    ax3.legend(loc='upper left')
    ax3.grid(True, which="both")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"グラフを '{save_path}' に保存しました。")

def send_to_discord(message: str, image_path: str, webhook_url: Optional[str] = None, bot_token: Optional[str] = None, channel_id: Optional[str] = None):
    """Discordにメッセージと画像を投稿します。WebhookかBot Tokenのいずれかを使用します。"""
    if not (webhook_url or (bot_token and channel_id)):
        print("Discordへの通知設定が不十分です。")
        return
        
    try:
        with open(image_path, "rb") as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            payload = {"content": message}
            
            if webhook_url:
                response = requests.post(webhook_url, data=payload, files=files)
            else: # Bot Tokenを使用
                url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
                headers = {"Authorization": f"Bot {bot_token}"}
                response = requests.post(url, headers=headers, data=payload, files=files)
            
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
    price_symbol = f"{coin}USDT.6"  # BybitのUSDT-Perpを価格指標とする

    raw_oi_data = process_oi_api_data(fetch_open_interest_data(exchange_config), code_to_name_map)
    price_data = process_price_data(fetch_price_data(price_symbol))
    
    if raw_oi_data.empty or price_data.empty:
        print(f"[{coin}] データ取得に失敗、またはデータが空のため処理を中断します。")
        return
        
    # 3. データ処理とメモリ管理
    # メモリ効率を考慮し、不要になったDataFrameは都度解放する
    dfs_to_merge = [raw_oi_data, price_data]
    del raw_oi_data, price_data
    
    base_df = pd.merge(*dfs_to_merge, on='Datetime', how='left').interpolate()
    if base_df.empty:
        print(f"[{coin}] 初期マージ後のデータが空のため処理を中断します。")
        return
    
    exchange_names = list(exchange_config.keys())
    
    # 各指標を計算
    active_oi_data = calculate_active_oi(base_df, exchange_names)
    standardized_oi_data = aggregate_and_standardize_oi(active_oi_data)
    price_std_data = calculate_price_std(base_df)
    
    # 逐次マージでメモリ効率を維持
    all_data = pd.merge(base_df, active_oi_data, on='Datetime', how='inner')
    del base_df, active_oi_data
    
    all_data = pd.merge(all_data, standardized_oi_data, on='Datetime', how='inner')
    del standardized_oi_data
    
    all_data = pd.merge(all_data, price_std_data, on='Datetime', how='inner')
    del price_std_data
    
    # ガベージコレクションを明示的に実行
    gc.collect()

    # 最終指標の計算と不要データの削除
    if 'STD_Active_OI' in all_data.columns and 'Bybit_price_STD' in all_data.columns:
        all_data['Merge_STD'] = all_data['Bybit_price_STD'] + all_data['STD_Active_OI']
    all_data.dropna(inplace=True)
    
    if all_data.empty:
        print(f"[{coin}] 最終データが空のため処理を中断します。")
        return
    
    # 4. グラフ生成 & Discord投稿
    plot_figure(all_data, figure_path, coin, exchange_names)
    if os.path.exists(figure_path) and DISCORD_WEBHOOK_URL:
        send_to_discord(
            message=f"📈 **{coin}** 分析グラフ",
            image_path=figure_path,
            webhook_url=DISCORD_WEBHOOK_URL
        )
    
    # 5. アラート判定と通知
    latest = all_data.iloc[-1]
    now_merge_std = latest.get('Merge_STD')
    if now_merge_std is not None and (now_merge_std < ALERT_THRESHOLD_LOWER or now_merge_std > ALERT_THRESHOLD_UPPER):
        alert_message = (f"**🚨 {coin} Alert** ({latest['Datetime'].strftime('%Y/%m/%d %H:%M')})\n"
                         f"**Merge_STD: {now_merge_std:.2f}**\nPrice: {latest['Bybit_Price_Close']:,.2f}")
        if os.path.exists(figure_path) and DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID:
            send_to_discord(
                message=alert_message,
                image_path=figure_path,
                bot_token=DISCORD_BOT_TOKEN,
                channel_id=DISCORD_CHANNEL_ID
            )
    
    # 6. 処理済みデータをParquet形式で保存
    try:
        all_data.to_parquet(data_path, index=False)
        print(f"処理済みデータを '{data_path}' に保存しました。")
    except ImportError:
        print("警告: 'pyarrow'が未インストールのため、データは保存されませんでした。`pip install pyarrow` を実行してください。")
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
