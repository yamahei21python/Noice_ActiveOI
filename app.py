# -*- coding: utf-8 -*-
# ファイル名: app.py
import streamlit as st
import os
from PIL import Image

# --- ページ設定 ---
st.set_page_config(page_title="OI Analysis Dashboard", layout="wide")
st.title("📈 Multi-Coin OI Analysis Dashboard")

# --- 設定 ---
# 選択肢のリスト。このリスト以外の入力はできません。
VIEW_OPTIONS = ["一覧表示", "BTC", "ETH", "SOL"]
DATA_DIR = "data"

# --- UI: 表示モードを選択するセレクトボックス ---
c1, c2 = st.columns([1, 3])
with c1:
    selected_view = st.selectbox(
        "表示モードを選択してください",
        VIEW_OPTIONS,
        #label_visibility="collapsed"
    )

with c2:
    if st.button('🔄 Refresh All Charts'):
        st.rerun()

st.divider()

# --- 表示ロジックの分岐 ---

# 1. 「一覧表示」が選択された場合
if selected_view == "一覧表示":
    cols = st.columns(len(VIEW_OPTIONS) - 1)
    for i, coin in enumerate(VIEW_OPTIONS[1:]):
        with cols[i]:
            st.subheader(f"{coin} Analysis")
            figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')

            if os.path.exists(figure_path):
                image = Image.open(figure_path)
                st.image(image, caption=f"Latest {coin} OI Analysis", use_container_width=True)
            else:
                st.warning(f"{coin}のグラフファイルが見つかりません。")
    
    st.info("このダッシュボードは、バックグラウンドで実行されている分析スクリプトの結果を表示します。「Refresh」ボタンでいつでも最新のグラフを読み込めます。")

# 2. 特定の通貨が選択された場合
else:
    coin = selected_view
    st.header(f"{coin} Analysis - 詳細表示")

    figure_path = os.path.join(DATA_DIR, f'{coin.lower()}_oi_analysis_figure.png')
    if os.path.exists(figure_path):
        image = Image.open(figure_path)
        st.image(image, use_container_width=True)
    else:
        st.error(f"{coin}の画像ファイルが見つかりません。")
