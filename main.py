import streamlit as st
from main import ShorePollutionPredictor  # ShorePollutionPredictor 클래스가 main.py 안에 있는 경우 유지
import pandas as pd

st.set_page_config(page_title="해안 오염 예측", layout="wide")
st.title("🌊 해안 오염 예측 시스템")

predictor = ShorePollutionPredictor()

uploaded_file = st.file_uploader("📂 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    try:
        # 업로드된 CSV 파일 읽기
        data = pd.read_csv(uploaded_file)
        predictor.data = data

        st.success("✅ 데이터 로드 완료")
        with st.expander("🔍 데이터 미리보기"):
            st.dataframe(data.head())

        # 모델 훈련
        predictor.train_model()

        # 데이터 개요 시각화
        st.subheader("📊 데이터 개요")
        fig1 = predictor.plot_data_overview()
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)

        # 특성 중요도 시각화
        st.subheader("🎯 특성 중요도")
        fig2 = predictor.plot_feature_importance()
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

        # 히트맵 시각화
        st.subheader("🗺️ 해안별-계절별 히트맵")
        fig3 = predictor.plot_pollution_heatmap()
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

        # 상관관계 매트릭스
        st.subheader("🔗 주요 변수 상관관계")
        fig4 = predictor.plot_correlation_matrix()
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)

        # 리포트 생성
        with st.expander("📄 리포트 출력"):
            predictor.generate_report()

    except Exception as e:
        st.error(f"❌ 처리 중 오류 발생: {e}")
else:
    st.info("왼쪽에 CSV 데이터를 업로드해 주세요.")
