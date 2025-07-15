# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("해양 오염도 예측 프로그램")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("coast/Shore_Pollution.csv")
    return df

df = load_data()
st.subheader("원본 데이터 미리보기")
st.write(df.head())

# 입력 및 타깃 변수 설정
st.subheader("모델 학습 및 예측")
if 'Pollution' not in df.columns:
    st.error("❌ 'Pollution'이라는 컬럼이 존재해야 예측 가능합니다.")
else:
    X = df.drop(columns=["Pollution"])
    y = df["Pollution"]

    # 사용자 선택으로 입력 변수 조정
    selected_features = st.multiselect("예측에 사용할 특성 선택", X.columns.tolist(), default=X.columns.tolist())
    if not selected_features:
        st.warning("적어도 하나의 특성을 선택하세요.")
    else:
        X = X[selected_features]

        # 학습
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # 예측
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.success(f"✅ 모델 학습 완료 (평균제곱오차: {mse:.2f})")

        # 새로운 값 입력
        st.subheader("새로운 데이터로 예측하기")
        input_data = {}
        for col in selected_features:
            val = st.number_input(f"{col} 값 입력", value=float(X[col].mean()))
            input_data[col] = val

        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]
        st.write(f"🌊 예측된 오염도: **{result:.2f}**")
