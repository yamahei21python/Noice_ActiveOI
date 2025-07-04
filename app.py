# -*- coding: utf-8 -*-
# ファイル名: app.py
import streamlit as st
import os
from PIL import Image

# --- ページ設定 ---
st.set_page_config(page_title="OI Analysis Dashboard", layout="wide")

# --- 設定 ---
TARGET_COINS = ["BTC", "ETH", "SOL"]
DATA_DIR = "data"

# --- Session Stateの初期化 ---
# 拡大表示するコインを記憶する場所を準備
if "maximized_coin" not in st.session_state:
    st.session_state.maximized_coin = None

# --- 表示ロジックの分岐 ---

# 1. 拡大表示モードの場合
if st.session_state.maximized_coin:
    coin = st.session_state.maximized_coin
    st.header(f"{coin} Analysis - 拡大図")

    # 「一覧に戻る」ボタン
    if st.button("⬅️ 一覧に戻る"):
        st.session_state.maximized_coin = None
        st.rerun()

    # 画像の表示
    figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')
    if os.path.exists(figure_path):
        image = Image.open(figure_path)
        st.image(image, use_container_width=True)
    else:
        st.error(f"{coin}の画像ファイルが見つかりません。")

# 2. 通常の一覧表示モードの場合
else:
    st.title("📈 Multi-Coin OI Analysis Dashboard")

    if st.button('🔄 Refresh All Charts'):
        st.rerun()

    cols = st.columns(len(TARGET_COINS))
    for i, coin in enumerate(TARGET_COINS):
        with cols[i]:
            st.subheader(f"{coin} Analysis")
            figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')

            if os.path.exists(figure_path):
                image = Image.open(figure_path)
                st.image(image, caption=f"Latest {coin} OI Analysis", use_container_width=True)
                
                # このボタンが押されると、session_stateにコイン名がセットされ、再実行される
                if st.button(f"🔍 {coin}を拡大表示", key=f"zoom_{coin}"):
                    st.session_state.maximized_coin = coin
                    st.rerun()
            else:
                st.warning(f"{coin}のグラフファイルが見つかりません。")
    
    st.info("このダッシュボードは、バックグラウンドで実行されている分析スクリプトの結果を表示します。「Refresh」ボタンでいつでも最新のグラフを読み込めます。")
