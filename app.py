import streamlit as st
from PIL import Image
import os

# ページのタイトルと設定
st.set_page_config(page_title="OI Analysis Dashboard", layout="wide")
st.title("📈 Coinalyze OI Analysis")

# 表示する画像のパス
FIGURE_PATH = os.path.join('data', 'oi_analysis_figure.png')

# 画像が存在するかチェック
if os.path.exists(FIGURE_PATH):
    # 画像を読み込んで表示
    image = Image.open(FIGURE_PATH)
    st.image(image, caption="Latest OI Analysis Chart", use_column_width=True)

    # 更新ボタン
    if st.button('🔄 Refresh'):
        st.rerun()
else:
    st.warning("グラフファイルが見つかりません。GitHub Actionsが実行されるまでお待ちください。")

st.info("このグラフはGitHub Actionsによって5分ごとに自動更新されます。")
