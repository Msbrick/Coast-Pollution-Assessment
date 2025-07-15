# streamlit_app.py
import streamlit as st
from main import ShorePollutionPredictor
import pandas as pd

st.set_page_config(page_title="Coast Pollution Dashboard", layout="wide")
st.title("🌊 해안 오염 예측 대시보드")

predictor = ShorePollutionPredictor()

uploaded_file = st.file_uploader("CSV 데이터 파일 업로드", type="csv")

if uploaded_file:
    try:
        data = predictor.load_data(uploaded_file)
        st.success(f"✅ {len(data)}개의 데이터 로드됨")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"❌ 데이터 로딩 오류: {e}")

    if st.button("모델 훈련"):
        try:
            acc = predictor.train_model()
            st.success(f"✅ 모델 훈련 완료 - 정확도: {acc:.3f}")

            st.subheader("🎯 특성 중요도")
            fig_imp = predictor.plot_feature_importance()
            st.plotly_chart(fig_imp)

            st.subheader("📊 오염 수준 분포")
            fig_overview = predictor.plot_data_overview()
            st.plotly_chart(fig_overview)
        except Exception as e:
            st.error(f"❌ 훈련 실패: {e}")

st.divider()
st.header("🔍 새로운 샘플 예측")

sample = {}
features = [
    'Month', 'Season', 'Shore',
    'Mean Number of Nematode species 1 per gram soil',
    'Mean Number of Turbillaria per gram soil',
    'Water pH', 'Soil pH', 'Water Salinity', 'Soil Salinity',
    'Total dissolved solids', 'Conduction', 'ORP'
]
for feature in features:
    sample[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("예측 실행"):
    try:
        pred, prob = predictor.predict(sample)
        st.success(f"예측 결과: 오염 수준 {pred[0]}")
        st.write("예측 확률:", prob[0])
    except Exception as e:
        st.error(f"❌ 예측 실패: {e}")
