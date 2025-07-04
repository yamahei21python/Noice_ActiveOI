# -*- coding: utf-8 -*-
# ファイル名: dashboard.py
import streamlit as st
import os
from PIL import Image

# --- ページ設定 ---
st.set_page_config(page_title="OI Analysis Dashboard", layout="wide")
st.title("📈 Multi-Coin OI Analysis Dashboard")

# --- 設定 ---
TARGET_COINS = ["BTC", "ETH", "SOL"]
DATA_DIR = "data"

# --- UI表示 ---

if st.button('🔄 Refresh All Charts'):
    st.rerun()

cols = st.columns(len(TARGET_COINS))

# 各通貨のグラフを、対応するカラムに表示
for i, coin in enumerate(TARGET_COINS):
    with cols[i]:
        st.subheader(f"{coin} Analysis")
        figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')

        if os.path.exists(figure_path):
            image = Image.open(figure_path)
            st.image(image, caption=f"Latest {coin} OI Analysis", use_container_width=True)

            # ★★★ 追加: 拡大表示ボタンとダイアログ表示 ★★★
            if st.button(f"🔍 {coin}を拡大表示", key=f"zoom_{coin}"):
                with st.dialog():
                    st.header(f"{coin} Analysis - 拡大図")
                    st.image(image, use_container_width=True)
        else:
            st.warning(f"{coin}のグラフファイルが見つかりません。")

st.info("このダッシュボードは、バックグラウンドで実行されている分析スクリプトの結果を表示します。「Refresh」ボタンでいつでも最新のグラフを読み込めます。")
