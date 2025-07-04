# -*- coding: utf-8 -*-
"""
Coinalyze APIから指定した複数の通貨のOIデータを取得・加工し、
価格データと組み合わせてグラフを生成、指定した条件でDiscordに通知し、
生成したグラフをGitHubリポジトリにプッシュする統合スクリプト。
"""

import requests
import time
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from functools import reduce
import subprocess # ★★★ Git操作のために追加 ★★★

# --- グローバル設定項目 (Global Configuration) ---

# ★★★ 環境変数からキーとトークンを読み込むように修正 ★★★
API_KEY = os.environ.get("API_KEY")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID")

# 分析したい通貨のリスト
TARGET_COINS = ["BTC", "ETH", "SOL"]

# Coinalyze APIエンドポイント
OI_API_URL = "https://api.coinalyze.net/v1/open-interest-history"
PRICE_API_URL = "https://api.coinalyze.net/v1/ohlcv-history"

# データおよび画像ファイルの保存先ディレクトリ
# Renderのファイルシステムは一時的なものですが、実行中に読み書きできれば問題ありません
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# --- 動的設定生成関数 ---

def get_exchange_config(coin: str) -> dict:
    """通貨シンボルに基づいて取引所の設定を動的に生成する。"""
    return {
        'Binance': {'code': 'A', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']},
        'Bybit': {'code': '6', 'contracts': [f'{coin}USD.', f'{coin}USDT.']},
        'OKX': {'code': '3', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']},
        'BitMEX': {'code': '0', 'contracts': [f'{coin}USD_PERP.', f'{coin}USDT_PERP.', f'{coin}USD.', f'{coin}USDT.']}
    }


# --- データ取得・処理関数 (変更なし) ---

def build_symbol_string(exchange_config: dict) -> str:
    """APIリクエスト用のシンボル文字列を構築する"""
    symbols = []
    for config in exchange_config.values():
        for contract in config['contracts']:
            symbols.append(f"{contract}{config['code']}")
    return ','.join(symbols)


def fetch_open_interest_data(exchange_config: dict) -> list:
    """Coinalyze APIからOI（Open Interest）の履歴データを取得する。"""
    if not API_KEY:
        print("エラー: API_KEYが設定されていません。")
        return []

    symbols_str = build_symbol_string(exchange_config)
    headers = {"api-key": API_KEY}
    params = {
        "symbols": symbols_str,
        "interval": "5min",
        "from": int(time.time()) - 864000,  # 過去10日分
        "to": int(time.time()),
        "convert_to_usd": "true"
    }
    try:
        response = requests.get(OI_API_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"OI APIリクエストに失敗しました: {e}")
    except ValueError as e:
        print(f"OI JSONの解析に失敗しました: {e}")
    return []


def fetch_price_data(price_symbol: str) -> list:
    """Coinalyze APIから価格の履歴データを取得する。"""
    if not API_KEY:
        return []
    headers = {"api-key": API_KEY}
    params = {
        "symbols": price_symbol, "interval": "5min",
        "from": int(time.time()) - 864000, "to": int(time.time()),
    }
    try:
        response = requests.get(PRICE_API_URL, headers=headers, params=params)
        response.raise_for_status()
        api_response = response.json()
        if api_response and isinstance(api_response, list) and len(api_response) > 0:
            return api_response[0].get("history", [])
    except requests.exceptions.RequestException as e:
        print(f"価格APIリクエストに失敗しました: {e}")
    except (ValueError, IndexError, KeyError) as e:
        print(f"価格JSONの解析または構造に問題がありました: {e}")
    return []


def process_oi_api_data(api_data: list, code_to_name_map: dict) -> pd.DataFrame:
    """OI APIレスポンスを整形し、取引所ごとのOHLCデータを持つDataFrameに変換する。"""
    if not api_data:
        return pd.DataFrame()

    all_dfs = []
    for item in api_data:
        symbol = item.get("symbol")
        history = item.get("history")
        if not symbol or not isinstance(history, list) or not history:
            continue

        _, exchange_code = symbol.rsplit('.', 1)
        exchange_name = code_to_name_map.get(exchange_code)
        if not exchange_name:
            continue

        df = pd.DataFrame(history)
        df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'})
        df = df.set_index('Datetime')

        ohlc_df = df[['Open', 'High', 'Low', 'Close']]
        ohlc_df.columns = pd.MultiIndex.from_product([[exchange_name], ohlc_df.columns])
        all_dfs.append(ohlc_df)

    if not all_dfs:
        return pd.DataFrame()

    temp_df = pd.concat(all_dfs, axis=1)
    combined_df = temp_df.T.groupby(level=[0, 1]).sum(min_count=1).T
    combined_df.columns = [f"{exchange}_{metric}" for exchange, metric in combined_df.columns]

    final_df = combined_df.interpolate().reset_index()
    return final_df.dropna(how='all', subset=[col for col in final_df.columns if col != 'Datetime']).reset_index(
        drop=True)


def process_price_data(price_history: list) -> pd.DataFrame:
    """価格APIレスポンスをDataFrameに変換する。"""
    if not price_history:
        return pd.DataFrame()
    df = pd.DataFrame(price_history)
    df['Datetime'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo')
    df = df.rename(columns={'c': 'Bybit_Price_Close'})
    return df[['Datetime', 'Bybit_Price_Close']]


def calculate_active_oi(df: pd.DataFrame, exchange_names: list) -> pd.DataFrame:
    """OIデータからActive OIを計算する。"""
    df = df.set_index('Datetime')
    active_oi_df = pd.DataFrame(index=df.index)
    rolling_window_3d = 12 * 24 * 3

    for name in exchange_names:
        low_col = f'{name}_Low'
        close_col = f'{name}_Close'
        if low_col in df.columns and close_col in df.columns:
            min_3day = df[low_col].rolling(window=rolling_window_3d, min_periods=1).min()
            active_oi_df[f'{name}_Active_OI_5min'] = df[close_col] - min_3day

    return active_oi_df.dropna(how='all').reset_index()


def aggregate_and_standardize_oi(df: pd.DataFrame) -> pd.DataFrame:
    """各取引所のActive OIを合計し、標準化（Z-score）する。"""
    df = df.set_index('Datetime')
    active_oi_cols = [col for col in df.columns if 'Active_OI_5min' in col]
    df['ALL_Active_OI_5min'] = df[active_oi_cols].sum(axis=1)

    rolling_window_3d = 12 * 24 * 3
    rolling_stats = df['ALL_Active_OI_5min'].rolling(window=rolling_window_3d)
    mean = rolling_stats.mean()
    std = rolling_stats.std()

    result_df = pd.DataFrame(index=df.index)
    result_df['STD_Active_OI'] = (df['ALL_Active_OI_5min'] - mean) / std.replace(0, pd.NA)
    return result_df.reset_index()


def calculate_price_std(df: pd.DataFrame) -> pd.DataFrame:
    """Bybitの価格データからZ-scoreを計算する。"""
    if 'Bybit_Price_Close' not in df.columns or 'Datetime' not in df.columns:
        return pd.DataFrame(columns=['Datetime', 'Bybit_price_STD'])

    price_df = df[['Datetime', 'Bybit_Price_Close']].copy().set_index('Datetime')
    rolling_window_3d = 12 * 24 * 3
    rolling_stats = price_df['Bybit_Price_Close'].rolling(window=rolling_window_3d)
    mean = rolling_stats.mean()
    std = rolling_stats.std()

    result_df = pd.DataFrame(index=price_df.index)
    result_df['Bybit_price_STD'] = (price_df['Bybit_Price_Close'] - mean) / std.replace(0, pd.NA)
    return result_df.reset_index()


# --- グラフ描画 & Discord通知関数 (変更なし) ---

def plot_figure(df: pd.DataFrame, save_path: str, coin: str, exchange_names: list):
    """3つのパネルを持つグラフを生成して保存する。"""
    df_plot = df.dropna(subset=['Merge_STD', 'Bybit_Price_Close']).reset_index(drop=True)
    if df_plot.empty:
        print("グラフ描画用の有効なデータがありません。")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    latest_price = df_plot['Bybit_Price_Close'].iloc[-1]
    latest_datetime = df_plot['Datetime'].iloc[-1]
    main_title = f"{coin} Open Interest Analysis ({latest_datetime.strftime('%Y-%m-%d %H:%M %Z')})"
    fig.suptitle(main_title, fontsize=16)

    ax1.set_title(f"Bybit Price: {latest_price:,.2f}", loc='right', fontsize=12, color='darkred')
    if coin in ["BTC", "ETH"]:
        ax1.plot(df_plot['Datetime'], df_plot['Bybit_Price_Close'] / 1000, label='Bybit Price Close (k USD)',
                 color='orangered')
        ax1.set_ylabel("Price (k USD)")
    else:
        ax1.plot(df_plot['Datetime'], df_plot['Bybit_Price_Close'], label='Bybit Price Close (USD)', color='orangered')
        ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which="both")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

    ax2.plot(df_plot['Datetime'], df_plot['Merge_STD'], label='Merge_STD', color='orangered')
    ax2.plot(df_plot['Datetime'], df_plot['Bybit_price_STD'], label='Bybit_price_STD', color='aqua')
    ax2.plot(df_plot['Datetime'], df_plot['STD_Active_OI'], label='STD_Active_OI', color='green')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, which="both")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel("Z-Score")

    active_oi_cols = [f'{name}_Active_OI_5min' for name in exchange_names]
    active_oi_cols_exist = [col for col in active_oi_cols if col in df_plot.columns]
    if active_oi_cols_exist:
        labels = [col.split('_')[0] for col in active_oi_cols_exist]
        ax3.stackplot(df_plot['Datetime'], [df_plot[col] / 1_000_000 for col in active_oi_cols_exist], labels=labels)
    ax3.set_ylabel("Active OI (M USD)")
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, which="both")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, top=0.94)
    plt.savefig(save_path)
    plt.close()
    print(f"グラフを '{save_path}' に保存しました。")


def send_discord_message(message: str, image_path: str):
    """Discordにメッセージと画像を送信する。"""
    if not DISCORD_BOT_TOKEN or not DISCORD_CHANNEL_ID:
        print("Discordの通知設定が不完全なため、通知をスキップします。")
        return

    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    data = {"content": message}
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        print("Discordへの通知が正常に送信されました。")
    except requests.exceptions.RequestException as e:
        print(f"Discordへの通知送信に失敗しました: {e.text}")
    except FileNotFoundError:
        print(f"画像ファイルが見つかりません: {image_path}")


# --- ★★★ Git操作用の関数を新たに追加 ★★★ ---
def git_push_image(image_path: str, coin: str):
    """生成した画像をGitHubにプッシュする"""
    token = os.environ.get("GITHUB_PAT")
    if not token:
        print("環境変数 GITHUB_PAT が設定されていません。")
        return

    # Gitのユーザー情報を設定
    subprocess.run(["git", "config", "--global", "user.name", "Render Bot"])
    subprocess.run(["git", "config", "--global", "user.email", "bot@render.com"])

    # トークンを使ってリモートURLを再設定（リポジトリのURLはご自身のものに合わせてください）
    repo_url = f"https://{token}@github.com/yamauchiz/Noice_ActiveOI.git"
    subprocess.run(["git", "remote", "set-url", "origin", repo_url])

    try:
        print(f"[{coin}] Gitリポジトリに画像を追加します: {image_path}")
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        subprocess.run(["git", "add", image_path], check=True)
        
        commit_message = f"Update {coin} analysis chart"
        result = subprocess.run(["git", "commit", "-m", commit_message], capture_output=True, text=True)

        # 変更がない場合はコミットが失敗するので、その場合は正常終了とする
        if "nothing to commit" in result.stdout or "no changes added to commit" in result.stderr:
            print(f"[{coin}] 更新する変更はありませんでした。")
            return

        print(f"[{coin}] GitHubにプッシュします...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"[{coin}] GitHubへのプッシュが完了しました。")

    except subprocess.CalledProcessError as e:
        print(f"[{coin}] Git操作中にエラーが発生しました: {e}")
        print(f"Stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")


# --- 通貨ごとの分析実行関数 (Gitプッシュの呼び出しを追加) ---

def run_analysis_for_coin(coin: str):
    """単一の通貨に対して分析から通知までの全プロセスを実行する。"""
    jst = datetime.timezone(datetime.timedelta(hours=9))
    print(f"--- [{coin}] 処理開始: {datetime.datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

    price_symbol = f"{coin}USDT.6"
    figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')
    exchange_config = get_exchange_config(coin)
    exchange_names = list(exchange_config.keys())
    code_to_name_map = {v['code']: k for k, v in exchange_config.items()}

    raw_oi_data = process_oi_api_data(fetch_open_interest_data(exchange_config), code_to_name_map)
    price_data = process_price_data(fetch_price_data(price_symbol))

    if raw_oi_data.empty or price_data.empty:
        print(f"[{coin}] 有効なOIデータまたは価格データが取得できなかったため、処理をスキップします。")
        return

    raw_data_df = pd.merge(raw_oi_data, price_data, on='Datetime', how='left')
    raw_data_df['Bybit_Price_Close'] = raw_data_df['Bybit_Price_Close'].interpolate()
    raw_data_df.dropna(subset=['Bybit_Price_Close'], inplace=True)

    active_oi_data = calculate_active_oi(raw_data_df, exchange_names)
    standardized_oi_data = aggregate_and_standardize_oi(active_oi_data)
    price_std_data = calculate_price_std(raw_data_df)

    data_frames_to_merge = [raw_data_df, active_oi_data, standardized_oi_data, price_std_data]
    all_data = reduce(lambda left, right: pd.merge(left, right, on='Datetime', how='inner'), data_frames_to_merge)

    if 'STD_Active_OI' in all_data.columns and 'Bybit_price_STD' in all_data.columns:
        all_data['Merge_STD'] = all_data['Bybit_price_STD'] + all_data['STD_Active_OI']

    all_data.dropna(inplace=True)
    all_data = all_data.reset_index(drop=True)

    if all_data.empty:
        print(f"[{coin}] 最終的なデータが空になりました。処理をスキップします。")
        return

    print(f"\n--- [{coin}] 最終データ (末尾5行) ---")
    print(all_data.tail())

    # 8. グラフを生成
    plot_figure(all_data, figure_path, coin, exchange_names)

    # ★★★ 8.5. 生成した画像をGitHubにプッシュする ★★★
    if os.path.exists(figure_path):
        git_push_image(figure_path, coin)
    else:
        print(f"[{coin}] グラフファイルが見つからなかったため、Gitプッシュをスキップします。")

    # 9. 条件に基づいてDiscordに通知
    latest = all_data.iloc[-1]
    now_merge_std = latest.get('Merge_STD')

    if now_merge_std is not None and (now_merge_std < -2.5 or now_merge_std > 5.0):
        print(f"[{coin}] 閾値超えを検出 (Merge_STD: {now_merge_std:.2f})。Discordに通知します。")
        dt_now = latest['Datetime'].strftime("%Y/%m/%d %H:%M")
        message = (
            f"**{coin}** Alert ({dt_now})\n"
            f"**Merge_STD: {now_merge_std:.2f}** (閾値: < -2.5 or > 5.0)\n"
            f"STD_Active_OI: {latest['STD_Active_OI']:.2f}\n"
            f"Bybit_price_STD: {latest['Bybit_price_STD']:.2f}\n"
            f"Price: {latest['Bybit_Price_Close']:,.2f}"
        )
        send_discord_message(message, figure_path)
    elif now_merge_std is not None:
        print(f"[{coin}] 現在のMerge_STDは {now_merge_std:.2f} で、通知の閾値内です。")
    else:
        print(f"[{coin}] Merge_STDが計算できませんでした。")


# --- メイン実行部 (変更なし) ---

def main():
    """TARGET_COINSリスト内の各通貨について分析を実行する。"""
    for i, coin in enumerate(TARGET_COINS):
        run_analysis_for_coin(coin)
        print(f"--- [{coin}] 処理完了 ---\n")

        if i < len(TARGET_COINS) - 1:
            print(f"API負荷軽減のため、次の通貨へ移る前に30秒待機します...")
            time.sleep(30)


if __name__ == "__main__":
    main()
