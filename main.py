import streamlit as st
from main import ShorePollutionPredictor
import pandas as pd

st.title("🌊 해안 오염 예측 대시보드")

# 모델 인스턴스 생성
predictor = ShorePollutionPredictor()

# 파일 업로드
uploaded_file = st.file_uploader("CSV 데이터 업로드", type="csv")

if uploaded_file:
    data = predictor.load_data(uploaded_file)
    if data is not None:
        st.success("✅ 데이터 로드 완료")
        
        if st.button("모델 훈련"):
            predictor.train_model()
            st.success("✅ 모델 훈련 완료")

            # 시각화 출력
            st.subheader("📈 데이터 개요")
            fig1 = predictor.plot_data_overview()
            if fig1:
                st.plotly_chart(fig1)

            st.subheader("🎯 특성 중요도")
            fig2 = predictor.plot_feature_importance()
            if fig2:
                st.plotly_chart(fig2)

            st.subheader("🗺️ 오염 히트맵")
            fig3 = predictor.plot_pollution_heatmap()
            if fig3:
                st.plotly_chart(fig3)

            st.subheader("🔗 변수 상관관계")
            fig4 = predictor.plot_correlation_matrix()
            if fig4:
                st.plotly_chart(fig4)

# 예측 섹션
st.header("🔍 새 데이터 예측")
sample_input = {}
for feature in [
    'Month', 'Season', 'Shore',
    'Mean Number of Nematode species 1 per gram soil',
    'Mean Number of Turbillaria per gram soil',
    'Water pH', 'Soil pH', 'Water Salinity', 'Soil Salinity',
    'Total dissolved solids', 'Conduction', 'ORP'
]:
    sample_input[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("예측 실행"):
    try:
        pred, prob = predictor.predict(sample_input)
        if pred is not None:
            st.success(f"예측된 오염 수준: {pred[0]}")
            st.write("📊 예측 확률:")
            st.write(prob[0])
    except Exception as e:
        st.error(f"예측 실패: {e}")
