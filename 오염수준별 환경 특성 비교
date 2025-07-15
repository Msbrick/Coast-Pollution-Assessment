import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Shore Pollution Analysis", layout="wide")

st.title("🌊 해안 오염 수준별 환경 특성 분석")
st.markdown("CSV 파일을 업로드하고 오염 수준별로 다양한 환경 지표를 시각화해보세요.")

# 파일 업로드
uploaded_file = st.file_uploader("📁 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    # 파일 읽기
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    except:
        st.error("파일을 읽는 데 실패했습니다. 인코딩을 확인해주세요.")
    else:
        # 컬럼 확인
        env_columns = [
            'Organic matter%',
            'Water pH', 'Soil pH',
            'Water Salinity', 'Soil Salinity',
            'Conductivity', 'Total dissolved solids ',
            'Conduction', 'Specific resistance ', 'Temp ©'
        ]
        available_columns = [col for col in env_columns if col in df.columns]

        st.success(f"데이터가 성공적으로 로드되었습니다! 총 {len(df)}개 샘플")

        # 변수 선택
        selected = st.multiselect(
            "🔍 비교할 환경 변수 선택", 
            options=available_columns, 
            default=available_columns[:3]
        )

        if selected:
            fig = make_subplots(
                rows=1, cols=len(selected),
                subplot_titles=selected,
                shared_yaxes=False
            )

            for idx, col in enumerate(selected):
                temp_df = df[[col, 'Pollution Level']].dropna()
                box = px.box(temp_df, x="Pollution Level", y=col, points="all")
                for trace in box.data:
                    fig.add_trace(trace, row=1, col=idx+1)

            fig.update_layout(height=500, showlegend=False, title="오염 수준별 환경 변수 비교 (Box Plot)")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("비교할 변수들을 선택해주세요.")
else:
    st.warning("먼저 CSV 파일을 업로드해주세요.")
