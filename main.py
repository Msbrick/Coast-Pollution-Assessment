
import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("🌊 전체 해양 오염 데이터 기반 예측 프로그램")

DATA_PATH = os.path.join(os.path.dirname(__file__), "coast_all", "Shore_Pollution.csv")

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH, encoding="utf-8")
    except:
        return pd.read_csv(DATA_PATH, encoding="latin1")

df = load_data()

if "Pollution" not in df.columns:
    st.error("'Pollution' 컬럼이 존재하지 않습니다. CSV 내용을 확인하세요.")
    st.stop()

st.subheader("📊 원본 데이터 미리보기")
st.dataframe(df.head())

y = df["Pollution"]
X = df.drop(columns=["Pollution"])

st.subheader("⚙️ 예측에 사용할 입력 특성 선택")
selected_features = st.multiselect("사용할 특성", X.columns.tolist(), default=X.columns.tolist())

if not selected_features:
    st.warning("하나 이상의 특성을 선택해야 합니다.")
    st.stop()

X = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.success(f"✅ 모델 학습 완료! 평균제곱오차(MSE): {mse:.2f}")

st.subheader("🧪 사용자 입력 기반 예측")
user_input = {}
for feature in selected_features:
    default_val = float(X[feature].mean()) if pd.api.types.is_numeric_dtype(X[feature]) else 0.0
    user_input[feature] = st.number_input(f"{feature}", value=default_val)

input_df = pd.DataFrame([user_input])
predicted = model.predict(input_df)[0]
st.write(f"🌡️ 예측된 오염도: **{predicted:.2f}**")
