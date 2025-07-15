# main.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

class ShorePollutionPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        self.is_trained = False
        self.data = None

    def load_data(self, filepath_or_buffer):
        self.data = pd.read_csv(filepath_or_buffer)
        return self.data

    def preprocess_data(self):
        required = ['Pollution Level']
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        potential_features = [
            'Month', 'Season', 'Shore',
            'Mean Number of Nematode species 1 per gram soil',
            'Mean Number of Turbillaria per gram soil',
            'Water pH', 'Soil pH', 'Water Salinity', 'Soil Salinity',
            'Total dissolved solids', 'Conduction', 'ORP'
        ]
        self.feature_columns = [f for f in potential_features if f in self.data.columns]
        X = self.data[self.feature_columns].copy()
        y = self.data['Pollution Level']
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median() if X[col].dtype != 'object' else X[col].mode()[0])
        return X, y

    def train_model(self, test_size=0.2):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        self.is_trained = True
        return accuracy_score(y_test, y_pred)

    def predict(self, new_data: dict):
        df = pd.DataFrame([new_data])
        for col in self.feature_columns:
            df[col] = df[col].fillna(self.data[col].median() if df[col].dtype != 'object' else self.data[col].mode()[0])
        df_scaled = self.scaler.transform(df[self.feature_columns])
        return self.model.predict(df_scaled), self.model.predict_proba(df_scaled)

    def plot_feature_importance(self):
        return px.bar(self.feature_importance.head(10), x="importance", y="feature", orientation="h")

    def plot_data_overview(self):
        pollution_counts = self.data['Pollution Level'].value_counts()
        return px.bar(pollution_counts, title="Pollution Level Distribution")

# 제거 또는 주석 처리
# if __name__ == "__main__":
#     ...
