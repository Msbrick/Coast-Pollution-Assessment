import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # ✅ 이 줄 꼭 있어야 함

st.set_page_config(page_title="Shore Pollution Analysis", layout="wide")

st.title("🌊 해안 오염 수준별 환경 특성 분석")
uploaded_file = st.file_uploader("📁 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    env_columns = [
        'Organic matter%',
        'Water pH', 'Soil pH',
        'Water Salinity', 'Soil Salinity',
        'Conductivity', 'Total dissolved solids ',
        'Conduction', 'Specific resistance ', 'Temp ©'
    ]
    available_columns = [col for col in env_columns if col in df.columns]

    selected = st.multiselect("비교할 변수 선택", available_columns, default=available_columns[:2])

    if selected:
        fig = make_subplots(
            rows=1, cols=len(selected),
            subplot_titles=selected
        )

        for idx, col in enumerate(selected):
            temp_df = df[[col, 'Pollution Level']].dropna()
            box = px.box(temp_df, x="Pollution Level", y=col, points="outliers")
            for trace in box.data:
                fig.add_trace(trace, row=1, col=idx + 1)

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
