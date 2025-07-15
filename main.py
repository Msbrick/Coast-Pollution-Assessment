import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("🌊 해양 오염도 예측 프로그램")

# CSV 경로 (coast.zip 압축 해제 후 들어간 곳)
DATA_PATH = os.path.join(os.path.dirname(__file__), "coast", "Shore_Pollution.csv")

# 데이터 로드
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(DATA_PATH, encoding="latin1")

df = load_data()

# 타겟 컬럼 확인
if "Pollution" not in df.columns:
    st.error("'Pollution' 컬럼이 없습니다. 예측 대상 컬럼을 확인하세요.")
    st.stop()

st.subheader("📊 데이터 미리보기")
st.dataframe(df.head())

# 예측 대상 및 입력 변수 설정
y = df["Pollution"]
X = df.drop(columns=["Pollution"])

st.subheader("⚙️ 예측에 사용할 특성 선택")
selected_features = st.multiselect("사용할 입력 변수", X.columns.tolist(), default=X.columns.tolist())

if not selected_features:
    st.warning("적어도 하나의 입력 특성을 선택해야 합니다.")
    st.stop()

X = X[selected_features]

# 학습/평가
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.success(f"✅ 모델 학습 완료! MSE(평균 제곱 오차): {mse:.2f}")

# 사용자 입력 예측
st.subheader("🧪 새로운 데이터 입력")
user_input = {}
for feature in selected_features:
    default_val = float(X[feature].mean()) if pd.api.types.is_numeric_dtype(X[feature]) else 0.0
    user_input[feature] = st.number_input(f"{feature}", value=default_val)

user_df = pd.DataFrame([user_input])
prediction = model.predict(user_df)[0]
st.write(f"🌡️ 예측된 오염도: **{prediction:.2f}**")
