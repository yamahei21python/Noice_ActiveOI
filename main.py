# -*- coding: utf-8 -*-
"""
Coinalyze APIからデータを取得しグラフを生成後、dataフォルダを一旦削除し、
新しいグラフ画像のみをGitHubにプッシュし、生成された全画像をDiscordに投稿する統合スクリプト。
"""

import requests
import time
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from functools import reduce
import subprocess
import shutil

# --- グローバル設定項目 ---
API_KEY = os.environ.get("API_KEY")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN") # 既存のアラート通知用
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID") # 既存のアラート通知用
GITHUB_PAT = os.environ.get("GITHUB_PAT")

# ★★★ ユーザー指定のWebhook URLを追加 ★★★
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1395228757585035374/p4UYCIgmELzTG3-6MX6UTc0ihlf6isXi-_8FIZfTjqzuVbU415JMZGWvEP32e81lXlJA"

TARGET_COINS = ["BTC", "ETH", "SOL"]

OI_API_URL = "https://api.coinalyze.net/v1/open-interest-history"
PRICE_API_URL = "https://api.coinalyze.net/v1/ohlcv-history"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# --- 動的設定生成関数 ---
def get_exchange_config(coin: str) -> dict:
    """通貨シンボルに基づいて取引所の設定を動的に生成する。"""
    return {
        'Binance': {'code': 'A', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']},
        'Bybit': {'code': '6', 'contracts': [f'{coin}USD.', f'{coin}USDT.']},
        'OKX': {'code': '3', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']},
        'BitMEX': {'code': '0', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']}
    }

# --- データ取得・処理関数 ---
def build_symbol_string(exchange_config: dict) -> str:
    """APIリクエスト用のシンボル文字列を構築する"""
    symbols = []
    for config in exchange_config.values():
        for contract in config['contracts']:
            symbols.append(f"{contract}{config['code']}")
    return ','.join(symbols)

def fetch_open_interest_data(exchange_config: dict) -> list:
    if not API_KEY: return []
    headers = {"api-key": API_KEY}
    params = {"symbols": build_symbol_string(exchange_config), "interval": "5min", "from": int(time.time()) - 864000, "to": int(time.time()), "convert_to_usd": "true"}
    try:
        response = requests.get(OI_API_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"OI APIリクエストに失敗: {e}")
        return []

def fetch_price_data(price_symbol: str) -> list:
    if not API_KEY: return []
    headers = {"api-key": API_KEY}
    params = {"symbols": price_symbol, "interval": "5min", "from": int(time.time()) - 864000, "to": int(time.time())}
    try:
        response = requests.get(PRICE_API_URL, headers=headers, params=params)
        response.raise_for_status()
        api_response = response.json()
        if api_response: return api_response[0].get("history", [])
    except Exception as e:
        print(f"価格APIリクエストに失敗: {e}")
        return []
    return []

def process_oi_api_data(api_data: list, code_to_name_map: dict) -> pd.DataFrame:
    if not api_data: return pd.DataFrame()
    all_dfs = []
    for item in api_data:
        symbol, history = item.get("symbol"), item.get("history")
        if not symbol or not history: continue
        _, exchange_code = symbol.rsplit('.', 1)
        exchange_name = code_to_name_map.get(exchange_code)
        if not exchange_name: continue
        df = pd.DataFrame(history)
        df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'}).set_index('Datetime')
        ohlc_df = df[['Open', 'High', 'Low', 'Close']]
        ohlc_df.columns = pd.MultiIndex.from_product([[exchange_name], ohlc_df.columns])
        all_dfs.append(ohlc_df)
    if not all_dfs: return pd.DataFrame()
    temp_df = pd.concat(all_dfs, axis=1)
    combined_df = temp_df.T.groupby(level=[0, 1]).sum(min_count=1).T
    combined_df.columns = [f"{ex}_{met}" for ex, met in combined_df.columns]
    return combined_df.interpolate().reset_index().dropna(how='all', subset=[c for c in combined_df.columns if c != 'Datetime']).reset_index(drop=True)

def process_price_data(price_history: list) -> pd.DataFrame:
    if not price_history: return pd.DataFrame()
    df = pd.DataFrame(price_history)
    df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
    return df.rename(columns={'c': 'Bybit_Price_Close'})[['Datetime', 'Bybit_Price_Close']]

def calculate_active_oi(df: pd.DataFrame, exchange_names: list) -> pd.DataFrame:
    df = df.set_index('Datetime')
    active_oi_df = pd.DataFrame(index=df.index)
    rolling_window_3d = 12 * 24 * 3
    for name in exchange_names:
        low_col, close_col = f'{name}_Low', f'{name}_Close'
        if low_col in df.columns and close_col in df.columns:
            min_3day = df[low_col].rolling(window=rolling_window_3d, min_periods=1).min()
            active_oi_df[f'{name}_Active_OI_5min'] = df[close_col] - min_3day
    return active_oi_df.dropna(how='all').reset_index()

def aggregate_and_standardize_oi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('Datetime')
    active_oi_cols = [col for col in df.columns if 'Active_OI_5min' in col]
    df['ALL_Active_OI_5min'] = df[active_oi_cols].sum(axis=1)
    rolling_window_3d = 12 * 24 * 3
    rolling_stats = df['ALL_Active_OI_5min'].rolling(window=rolling_window_3d)
    mean, std = rolling_stats.mean(), rolling_stats.std()
    result_df = pd.DataFrame(index=df.index)
    result_df['STD_Active_OI'] = (df['ALL_Active_OI_5min'] - mean) / std.replace(0, pd.NA)
    return result_df.reset_index()

def calculate_price_std(df: pd.DataFrame) -> pd.DataFrame:
    if 'Bybit_Price_Close' not in df.columns: return pd.DataFrame(columns=['Datetime', 'Bybit_price_STD'])
    price_df = df[['Datetime', 'Bybit_Price_Close']].copy().set_index('Datetime')
    rolling_window_3d = 12 * 24 * 3
    rolling_stats = price_df['Bybit_Price_Close'].rolling(window=rolling_window_3d)
    mean, std = rolling_stats.mean(), rolling_stats.std()
    result_df = pd.DataFrame(index=price_df.index)
    result_df['Bybit_price_STD'] = (price_df['Bybit_Price_Close'] - mean) / std.replace(0, pd.NA)
    return result_df.reset_index()

# --- グラフ描画 & Discord通知関数 ---
def plot_figure(df: pd.DataFrame, save_path: str, coin: str, exchange_names: list):
    df_plot = df.dropna(subset=['Merge_STD', 'Bybit_Price_Close']).reset_index(drop=True)
    if df_plot.empty:
        print(f"[{coin}] グラフ描画用のデータがありません。スキップします。")
        return
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    latest_price = df_plot['Bybit_Price_Close'].iloc[-1]
    latest_datetime = df_plot['Datetime'].iloc[-1]
    fig.suptitle(f"{coin} OI Analysis ({latest_datetime.strftime('%Y-%m-%d %H:%M %Z')})", fontsize=16)
    ax1.set_title(f"Bybit Price: {latest_price:,.2f}", loc='right', color='darkred')
    if coin in ["BTC", "ETH"]:
        ax1.plot(df_plot['Datetime'], df_plot['Bybit_Price_Close'] / 1000, label='Price (k USD)', color='orangered')
        ax1.set_ylabel("Price (k USD)")
    else:
        ax1.plot(df_plot['Datetime'], df_plot['Bybit_Price_Close'], label='Price (USD)', color='orangered')
        ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper left'); ax1.grid(True, which="both"); ax1.yaxis.tick_right(); ax1.yaxis.set_label_position('right')
    ax2.plot(df_plot['Datetime'], df_plot['Merge_STD'], label='Merge_STD', color='orangered')
    ax2.plot(df_plot['Datetime'], df_plot['Bybit_price_STD'], label='Price_STD', color='aqua')
    ax2.plot(df_plot['Datetime'], df_plot['STD_Active_OI'], label='Active_OI_STD', color='green')
    ax2.legend(loc='upper left'); ax2.grid(True, which="both"); ax2.yaxis.tick_right(); ax2.yaxis.set_label_position('right'); ax2.set_ylabel("Z-Score")
    active_oi_cols_exist = [c for c in [f'{n}_Active_OI_5min' for n in exchange_names] if c in df_plot.columns]
    if active_oi_cols_exist:
        ax3.stackplot(df_plot['Datetime'], [df_plot[c] / 1_000_000 for c in active_oi_cols_exist], labels=[c.split('_')[0] for c in active_oi_cols_exist])
    ax3.set_ylabel("Active OI (M USD)"); ax3.legend(loc='upper left'); ax3.grid(True, which="both"); ax3.yaxis.tick_right(); ax3.yaxis.set_label_position('right')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(save_path); plt.close()
    print(f"グラフを '{save_path}' に保存しました。")

# ★★★ Webhookで画像を投稿する新しい関数を追加 ★★★
def send_image_via_webhook(image_path: str, coin: str):
    """指定されたDiscord Webhook URLに画像を投稿する"""
    if not DISCORD_WEBHOOK_URL:
        print("Webhook URLが設定されていません。")
        return
    
    try:
        with open(image_path, "rb") as f:
            # Webhookに送信するデータを作成
            files = {'file1': (os.path.basename(image_path), f, 'image/png')}
            payload = {"content": f"📈 **{coin}** 分析グラフ"}
            
            response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
            response.raise_for_status() # エラーがあれば例外を発生させる
            print(f"Webhook経由で {coin} の画像をDiscordに投稿しました。")
    except Exception as e:
        print(f"Webhook経由でのDiscordへの投稿に失敗しました: {e}")

# (既存のアラート通知用関数は変更なし)
def send_discord_message(message: str, image_path: str):
    """Bot Tokenを使用してアラートを通知する"""
    if not DISCORD_BOT_TOKEN or not DISCORD_CHANNEL_ID: return
    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            response = requests.post(url, headers=headers, data={"content": message}, files=files)
            response.raise_for_status()
            print("Discordへのアラート通知が正常に送信されました。")
    except Exception as e:
        print(f"Discordへのアラート通知送信に失敗: {e}")

# --- Git操作の関数（変更なし） ---
def setup_and_push_to_github():
    """dataフォルダ全体をGitHubにプッシュする"""
    if not GITHUB_PAT:
        print("環境変数 GITHUB_PAT が設定されていません。")
        return

    # Gitのユーザー情報とリモートURLを設定
    subprocess.run(["git", "config", "--global", "user.name", "Render Bot"])
    subprocess.run(["git", "config", "--global", "user.email", "bot@render.com"])
    repo_url = f"https://{GITHUB_PAT}@github.com/yamahei21python/Noice_ActiveOI.git"
    
    try:
        # --- リポジトリの準備 ---
        subprocess.run(["git", "remote", "remove", "origin"], stderr=subprocess.DEVNULL)
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "checkout", "-B", "main"], check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
        
        # --- dataフォルダの削除と再作成 ---
        print(f"既存のdataフォルダを削除します: {DATA_DIR}")
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # --- 全通貨の分析とグラフ生成 ---
        for i, coin in enumerate(TARGET_COINS):
            run_analysis_for_coin(coin)
            if i < len(TARGET_COINS) - 1:
                print(f"API負荷軽減のため30秒待機...")
                time.sleep(30)
        
        # --- 生成されたファイルをまとめてプッシュ ---
        print("生成されたファイルをGitHubに追加します...")
        subprocess.run(["git", "add", DATA_DIR], check=True)
        
        commit_message = f"Update analysis charts at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        result = subprocess.run(["git", "commit", "-m", commit_message], capture_output=True, text=True)

        if "nothing to commit" in result.stdout:
            print("更新する変更はありませんでした。")
            return
            
        print("GitHubにプッシュします...")
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        print("GitHubへのプッシュが完了しました。")

    except subprocess.CalledProcessError as e:
        print(f"Git操作中にエラー: {e}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")

# --- 分析実行関数（Webhook呼び出しを追加） ★★★ ---
def run_analysis_for_coin(coin: str):
    """単一の通貨に対して分析とグラフ生成を行い、結果を投稿する"""
    jst = datetime.timezone(datetime.timedelta(hours=9))
    print(f"--- [{coin}] 処理開始: {datetime.datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

    figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')
    
    price_symbol = f"{coin}USDT.6"
    exchange_config = get_exchange_config(coin)
    code_to_name_map = {v['code']: k for k, v in exchange_config.items()}
    raw_oi_data = process_oi_api_data(fetch_open_interest_data(exchange_config), code_to_name_map)
    price_data = process_price_data(fetch_price_data(price_symbol))
    if raw_oi_data.empty or price_data.empty:
        print(f"[{coin}] データ取得に失敗したため処理を中断します。")
        return
        
    raw_data_df = pd.merge(raw_oi_data, price_data, on='Datetime', how='left').interpolate()
    if raw_data_df.empty: return
        
    active_oi_data = calculate_active_oi(raw_data_df, list(exchange_config.keys()))
    standardized_oi_data = aggregate_and_standardize_oi(active_oi_data)
    price_std_data = calculate_price_std(raw_data_df)
    all_data = reduce(lambda l, r: pd.merge(l, r, on='Datetime', how='inner'), [raw_data_df, active_oi_data, standardized_oi_data, price_std_data])
    if 'STD_Active_OI' in all_data.columns and 'Bybit_price_STD' in all_data.columns:
        all_data['Merge_STD'] = all_data['Bybit_price_STD'] + all_data['STD_Active_OI']
    all_data.dropna(inplace=True)
    
    if all_data.empty:
        print(f"[{coin}] 最終データが空のため処理を中断します。")
        return
    
    plot_figure(all_data, figure_path, coin, list(exchange_config.keys()))
    
    # ★★★ Webhookで全ての生成画像を投稿する処理を追加 ★★★
    if os.path.exists(figure_path):
        send_image_via_webhook(figure_path, coin)
    
    # 既存のアラート通知ロジック (変更なし)
    latest = all_data.iloc[-1]
    now_merge_std = latest.get('Merge_STD')
    if now_merge_std is not None and (now_merge_std < -3.5 or now_merge_std > 5.0):
        message = (f"**🚨 {coin} Alert** ({latest['Datetime'].strftime('%Y/%m/%d %H:%M')})\n"
                   f"**Merge_STD: {now_merge_std:.2f}**\nPrice: {latest['Bybit_Price_Close']:,.2f}")
        send_discord_message(message, figure_path)
        
    print(f"--- [{coin}] 処理完了 ---\n")

# --- メイン実行部（変更なし） ---
if __name__ == "__main__":
    # Webhook URLは必須ではないため、チェックから除外
    if not all([API_KEY, GITHUB_PAT]):
        print("エラー: 必要な環境変数 (API_KEY, GITHUB_PAT) が設定されていません。")
    else:
        setup_and_push_to_github()
