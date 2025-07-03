# -*- coding: utf-8 -*-
"""
Coinalyze APIからOIデータを取得・加工し、価格データと組み合わせてグラフを生成、
指定した条件でDiscordに通知する統合スクリプト。

このスクリプトはデータベースを使用せず、実行の都度APIから取得したデータのみで処理を行います。

処理フロー：
1.  Coinalyze APIから複数の取引所のOIデータを5分足で取得します。
2.  取得したデータから各取引所のActive OIを計算します。
3.  全取引所のActive OIを合計し、その標準偏差（Z-score）を算出します。
4.  Bybitの終値から価格の標準偏差（Z-score）を算出します。
5.  Active OIデータと価格データを結合し、最終的な統合データを作成します。
6.  最終的な統合データフレームから3つのグラフ（価格、各種Z-score、取引所別OI）を生成し、
    画像ファイルとして保存します。
7.  算出された指標（Merge_STD）が特定の閾値を超えた場合、生成したグラフと共に
    Discordチャンネルへ通知を送信します。
"""

import requests
import time
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from functools import reduce

# --- 設定項目 (Configuration) ---

# APIキーとDiscordボットトークン (GitHub Secretsから環境変数として読み込む)
API_KEY = "e217c670-0033-47c4-af1e-d1ff2ea71954"
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

# Coinalyze APIエンドポイント
API_URL = "https://api.coinalyze.net/v1/open-interest-history"

# データおよび画像ファイルの保存先ディレクトリ
# スクリプトファイルと同じ場所に 'data' フォルダを作成して保存
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 画像ファイルのパス
FIGURE_PATH = os.path.join(DATA_DIR, 'oi_analysis_figure.png')

# 対象取引所の設定
EXCHANGE_CONFIG = {
    'Binance': {'code': 'A', 'contracts': ['BTCUSD_PERP.', 'BTCUSDT_PERP.', 'BTCUSD.', 'BTCUSDT.']},
    'Bybit': {'code': '6', 'contracts': ['BTCUSD.', 'BTCUSDT.']},
    'OKX': {'code': '3', 'contracts': ['BTCUSD_PERP.', 'BTCUSDT_PERP.', 'BTCUSD.', 'BTCUSDT.']},
    'BitMEX': {'code': '0', 'contracts': ['BTCUSD_PERP.', 'BTCUSDT_PERP.', 'BTCUSD.', 'BTCUSDT.']}
}
EXCHANGE_NAMES = list(EXCHANGE_CONFIG.keys())
CODE_TO_NAME_MAP = {v['code']: k for k, v in EXCHANGE_CONFIG.items()}

# OHLCカラム名の接尾辞
OHLC_SUFFIXES = ['_Open', '_High', '_Low', '_Close']


# --- データ取得・処理関数 ---

def build_symbol_string() -> str:
    """APIリクエスト用のシンボル文字列を構築する"""
    symbols = []
    for config in EXCHANGE_CONFIG.values():
        for contract in config['contracts']:
            symbols.append(f"{contract}{config['code']}")
    return ','.join(symbols)


def fetch_open_interest_data() -> list:
    """Coinalyze APIからOI（Open Interest）の履歴データを取得する。"""
    if not API_KEY:
        print("エラー: API_KEYが設定されていません。")
        return []

    symbols_str = build_symbol_string()
    headers = {"api-key": API_KEY}
    params = {
        "symbols": symbols_str,
        "interval": "5min",
        "from": int(time.time()) - 864000,  # 過去10日分
        "to": int(time.time()),
        "convert_to_usd": "true"
    }
    try:
        response = requests.get(API_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"APIリクエストに失敗しました: {e}")
    except ValueError as e:
        print(f"JSONの解析に失敗しました: {e}")
    return []


def process_api_data(api_data: list) -> pd.DataFrame:
    """APIレスポンスを整形し、取引所ごとのOHLCデータを持つDataFrameに変換する。"""
    if not api_data:
        print("APIデータが空のため、処理をスキップします。")
        return pd.DataFrame()

    all_dfs = []
    for item in api_data:
        symbol = item.get("symbol")
        history = item.get("history")
        if not symbol or not isinstance(history, list) or not history:
            continue

        # 'BTCUSD..A' のような不正なシンボルを避ける
        parts = symbol.rsplit('.', 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        contract_type, exchange_code = parts

        exchange_name = CODE_TO_NAME_MAP.get(exchange_code)
        if not exchange_name:
            continue

        df = pd.DataFrame(history)
        df['Datetime'] = pd.to_datetime(df['t'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'})
        df = df.set_index('Datetime')

        ohlc_df = df[['Open', 'High', 'Low', 'Close']]
        ohlc_df.columns = pd.MultiIndex.from_product(
            [[exchange_name], [contract_type], ohlc_df.columns]
        )
        all_dfs.append(ohlc_df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, axis=1)

    final_df = pd.DataFrame()
    for ex_name in EXCHANGE_NAMES:
        if ex_name in combined_df.columns.get_level_values(0):
            ex_df = combined_df[ex_name]
            summed_df = ex_df.T.groupby(level=1).sum(min_count=1).T
            summed_df.columns = [f"{ex_name}{suffix}" for suffix in OHLC_SUFFIXES]
            final_df = pd.concat([final_df, summed_df], axis=1)

    final_df = final_df.interpolate().reset_index()
    final_df = final_df.dropna().reset_index(drop=True)
    return final_df


def calculate_active_oi(df: pd.DataFrame) -> pd.DataFrame:
    """OIデータからActive OIを計算する。"""
    df = df.set_index('Datetime')
    active_oi_df = pd.DataFrame(index=df.index)
    rolling_window_3d = 12 * 24 * 3  # 3日間の5分足データ数

    for name in EXCHANGE_NAMES:
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
    all_active_oi = df[active_oi_cols].sum(axis=1)

    rolling_window_3d = 12 * 24 * 3
    rolling_stats = all_active_oi.rolling(window=rolling_window_3d)
    mean = rolling_stats.mean()
    std = rolling_stats.std()

    result_df = pd.DataFrame(index=df.index)
    result_df['STD_Active_OI'] = (all_active_oi - mean) / std.replace(0, pd.NA)
    return result_df.reset_index()


def calculate_price_std(df: pd.DataFrame) -> pd.DataFrame:
    """Bybitの価格データからZ-scoreを計算する。"""
    if 'Bybit_Close' not in df.columns or 'Datetime' not in df.columns:
        return pd.DataFrame(columns=['Datetime', 'Bybit_price_STD'])

    price_df = df[['Datetime', 'Bybit_Close']].copy()
    price_df = price_df.set_index('Datetime')

    rolling_window_3d = 12 * 24 * 3
    rolling_stats = price_df['Bybit_Close'].rolling(window=rolling_window_3d)
    mean = rolling_stats.mean()
    std = rolling_stats.std()

    result_df = pd.DataFrame(index=price_df.index)
    result_df['Bybit_price_STD'] = (price_df['Bybit_Close'] - mean) / std.replace(0, pd.NA)
    return result_df.reset_index()


# --- グラフ描画 & Discord通知関数 ---

def plot_figure(df: pd.DataFrame, save_path: str):
    """3つのパネルを持つグラフを生成して保存する。"""
    df_plot = df.dropna().reset_index(drop=True)
    if df_plot.empty:
        print("グラフ描画用のデータがありません。")
        return

    fig = plt.figure(figsize=(15, 8))
    span = (df_plot['Datetime'].iloc[0], df_plot['Datetime'].iloc[-1])

    # 1. 価格グラフ
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df_plot['Datetime'], df_plot['Bybit_Close'] / 1000, label='Bybit Close (k USD)', color='orangered')
    ax1.set_xlim(span)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which="both")
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

    # 2. Z-scoreグラフ
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(df_plot['Datetime'], df_plot['Merge_STD'], label='Merge_STD', color='orangered')
    ax2.plot(df_plot['Datetime'], df_plot['Bybit_price_STD'], label='Bybit_price_STD', color='aqua')
    ax2.plot(df_plot['Datetime'], df_plot['STD_Active_OI'], label='STD_Active_OI', color='green')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, which="both")
    ax2.tick_params(labelbottom=False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')

    # 3. Active OI 積み上げグラフ
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    active_oi_cols = [f'{name}_Active_OI_5min' for name in EXCHANGE_NAMES]
    active_oi_cols_exist = [col for col in active_oi_cols if col in df_plot.columns]
    labels = [col.split('_')[0] for col in active_oi_cols_exist]

    ax3.stackplot(df_plot['Datetime'],
                  [df_plot[col] / 1_000_000 for col in active_oi_cols_exist],
                  labels=labels)
    ax3.set_ylabel("Active OI (M USD)")
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, which="both")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
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
        print(f"Discordへの通知が正常に送信されました。")
    except requests.exceptions.RequestException as e:
        print(f"Discordへの通知送信に失敗しました: {e.text}")
    except FileNotFoundError:
        print(f"画像ファイルが見つかりません: {image_path}")


# --- メイン実行部 ---

def main():
    """メインの実行関数"""
    print(f"--- 処理開始: {datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

    # 1. APIからOIデータを取得し、整形
    raw_data_df = process_api_data(fetch_open_interest_data())
    if raw_data_df.empty:
        print("有効なデータが取得できなかったため、処理を終了します。")
        return

    # 2. Active OIを計算
    active_oi_data = calculate_active_oi(raw_data_df)

    # 3. Active OIを集計・標準化
    standardized_oi_data = aggregate_and_standardize_oi(active_oi_data)

    # 4. 価格の標準偏差を計算
    price_std_data = calculate_price_std(raw_data_df)

    # 5. 全てのデータを結合
    data_frames_to_merge = [
        raw_data_df,
        active_oi_data,
        standardized_oi_data,
        price_std_data
    ]
    all_data = reduce(lambda left, right: pd.merge(left, right, on='Datetime', how='inner'), data_frames_to_merge)

    # 6. 最終的な指標を計算
    all_data['Merge_STD'] = all_data['Bybit_price_STD'] + all_data['STD_Active_OI']

    # 7. 最終的なデータクリーンアップ
    all_data.dropna(inplace=True)
    all_data = all_data.reset_index(drop=True)

    if all_data.empty:
        print("最終的なデータが空になりました。処理を終了します。")
        return

    print("\n--- 最終データ (末尾5行) ---")
    print(all_data.tail())

    # 8. グラフを生成
    plot_figure(all_data, FIGURE_PATH)

    # 9. 条件に基づいてDiscordに通知
    if not all_data.empty:
        latest = all_data.iloc[-1]
        now_merge_std = latest['Merge_STD']

        if now_merge_std < -2.5 or now_merge_std > 5.0:
            print(f"閾値超えを検出 (Merge_STD: {now_merge_std:.2f})。Discordに通知します。")
            dt_now = latest['Datetime'].strftime("%Y/%m/%d %H:%M")
            message = (
                f"{dt_now}\n"
                f"**Merge_STD: {now_merge_std:.2f}** (閾値: < -2.5 or > 5.0)\n"
                f"STD_Active_OI: {latest['STD_Active_OI']:.2f}\n"
                f"Bybit_price_STD: {latest['Bybit_price_STD']:.2f}\n"
                f"Bybit_Close: {latest['Bybit_Close']:,.2f}"
            )
            send_discord_message(message, FIGURE_PATH)
        else:
            print(f"現在のMerge_STDは {now_merge_std:.2f} で、通知の閾値内です。")

    print(f"\n--- 処理完了: {datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")


if __name__ == "__main__":
    main()
