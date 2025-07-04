# -*- coding: utf-8 -*-
# ファイル名: dashboard.py
import streamlit as st
import os
from PIL import Image # ★★★ ライブラリをインポート ★★★

# --- ページ設定 ---
st.set_page_config(page_title="OI Analysis Dashboard", layout="wide")
st.title("📈 Multi-Coin OI Analysis Dashboard")

# --- 設定 ---
# 分析対象の通貨リスト（分析スクリプトと合わせる）
TARGET_COINS = ["BTC", "ETH", "SOL"]
# 画像が保存されているディレクトリ
DATA_DIR = "data"

# --- UI表示 ---

# 更新ボタンを一番上に配置
if st.button('🔄 Refresh All Charts'):
    # ページを再実行し、ディスク上の最新画像を読み込む
    st.rerun()

# ターゲットの通貨数に合わせてカラムを作成
cols = st.columns(len(TARGET_COINS))

# 各通貨のグラフを、対応するカラムに表示
for i, coin in enumerate(TARGET_COINS):
    with cols[i]:
        # 通貨名のヘッダーを表示
        st.subheader(f"{coin} Analysis")

        # 通貨ごとの画像ファイルパスを動的に生成
        figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')

        # 画像が存在するかチェックして表示
        if os.path.exists(figure_path):
            # ★★★ 変更箇所 ★★★
            # PILで画像ファイルを開く
            image = Image.open(figure_path)
            # 開いた画像オブジェクトをst.imageに渡す
            st.image(image, caption=f"Latest {coin} OI Analysis", use_container_width=True)
        else:
            st.warning(f"{coin}のグラフファイルが見つかりません。")

st.info("このダッシュボードは、バックグラウンドで実行されている分析スクリプトの結果を表示します。「Refresh」ボタンでいつでも最新のグラフを読み込めます。")
