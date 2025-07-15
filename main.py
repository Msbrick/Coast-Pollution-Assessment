import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("🌊 해양 오염도 예측 프로그램")

# 데이터 불러오기 함수
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "coast", "Shore_Pollution - Shore_Pollution.csv")
    df = pd.read_csv(file_path)
    return df

# 데이터 불러오기
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ CSV 파일을 찾을 수 없습니다. 'coast/Shore_Pollution - Shore_Pollution.csv' 경로를 확인하세요.")
    st.stop()

st.subheader("📊 원본 데이터 미리보기")
st.write(df.head())

# 오염도 예측
if 'Pollution' not in df.columns:
    st.error("❌ 'Pollution'이라는 컬럼이 존재해야 예측이 가능합니다.")
else:
    X = df.drop(columns=["Pollution"])
    y = df["Pollution"]

    st.subheader("⚙️ 예측에 사용할 입력 변수 선택")
    selected_features = st.multiselect("특성 선택", X.columns.tolist(), default=X.columns.tolist())

    if not selected_features:
        st.warning("하나 이상의 특성을 선택하세요.")
    else:
        X = X[selected_features]

        # 모델 학습
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # 예측 및 평가
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.success(f"✅ 모델 학습 완료 - 평균제곱오차(MSE): {mse:.2f}")

        st.subheader("🔮 새로운 데이터 입력")
        input_data = {}
        for col in selected_features:
            default_val = float(X[col].mean()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0
            val = st.number_input(f"{col} 값 입력", value=default_val)
            input_data[col] = val

        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]
        st.write(f"🌡️ 예측된 해양 오염도: **{result:.2f}**")
