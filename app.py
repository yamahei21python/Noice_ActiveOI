# -*- coding: utf-8 -*-
# ファイル名: dashboard.py
import streamlit as st
import os

# ページのタイトルと設定
st.set_page_config(page_title="OI Analysis Dashboard", layout="wide")
st.title("📈 Coinalyze OI Analysis")

# 表示する画像のパス
# 'analysis_script.py' と同じ階層に 'data' フォルダがあることを想定
FIGURE_PATH = os.path.join('data', 'oi_analysis_figure.png')

# 画像が存在するかチェック
if os.path.exists(FIGURE_PATH):
    # PILで開くのではなく、画像のパスを直接st.imageに渡します。
    # これにより、Streamlitがファイルのキャッシュ管理を適切に行い、
    # 更新された画像が正しく表示されるようになります。
    st.image(FIGURE_PATH, caption="Latest OI Analysis Chart", use_column_width=True)

    # 更新ボタン
    if st.button('🔄 Refresh'):
        # このボタンが押されると、ページが再実行（rerun）され、
        # st.imageはディスク上のファイルを再度チェックします。
        st.rerun()
else:
    st.warning("グラフファイルが見つかりません。分析スクリプトが実行されるまでお待ちください。")

st.info("このグラフはバックグラウンドで5分ごとに更新される可能性があります。「Refresh」ボタンを押すと最新のグラフを読み込みます。")
