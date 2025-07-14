import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ShorePollutionPredictor:
    """
    해안 오염 예측 모델 클래스
   
    이 모델은 해안의 생물학적, 화학적, 물리적 지표를 기반으로
    오염 수준(0: 낮음, 1: 보통, 2: 높음)을 예측합니다.
    """
   
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        self.is_trained = False
       
    def load_data(self, filepath):
        """데이터 로드 및 전처리"""
        self.data = pd.read_csv(filepath)
        print(f"📊 데이터 로드 완료: {len(self.data)}개 샘플")
        return self.data
   
    def preprocess_data(self):
        """데이터 전처리"""
        # 주요 특성 선택 (결측값이 적고 중요한 변수들)
        self.feature_columns = [
            'Month', 'Season', 'Shore',
            'Mean Number of Nematode species 1 per gram soil',
            'Mean Number of Turbillaria per gram soil',
            'Water pH', 'Soil pH', 'Water Salinity', 'Soil Salinity',
            'Total dissolved solids', 'Conduction', 'ORP'
        ]
       
        # 특성과 타겟 분리
        X = self.data[self.feature_columns].copy()
        y = self.data['Pollution Level']
       
        # 결측값 처리 (중앙값으로 대체)
        X = X.fillna(X.median())
       
        return X, y
   
    def train_model(self, test_size=0.2):
        """모델 훈련"""
        X, y = self.preprocess_data()
       
        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
       
        # 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
       
        # 모델 훈련
        self.model.fit(X_train_scaled, y_train)
       
        # 예측 및 평가
        y_pred = self.model.predict(X_test_scaled)
       
        # 특성 중요도 저장
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
       
        self.is_trained = True
       
        # 성능 지표 출력
        print("🎯 모델 성능 평가:")
        print(f"정확도: {accuracy_score(y_test, y_pred):.3f}")
        print("\n분류 보고서:")
        print(classification_report(y_test, y_pred))
       
        return X_train, X_test, y_train, y_test, y_pred
   
    def predict(self, new_data):
        """새로운 데이터 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. train_model()을 먼저 실행하세요.")
       
        # 입력 데이터 전처리
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
       
        # 필요한 컬럼만 선택하고 결측값 처리
        X_new = new_data[self.feature_columns].fillna(self.data[self.feature_columns].median())
        X_new_scaled = self.scaler.transform(X_new)
       
        # 예측
        prediction = self.model.predict(X_new_scaled)
        probability = self.model.predict_proba(X_new_scaled)
       
        return prediction, probability
   
    def plot_data_overview(self):
        """데이터 개요 시각화"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('오염 수준별 분포', '해안별 샘플 수', '계절별 분포', '월별 트렌드'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
       
        # 오염 수준별 분포
        pollution_counts = self.data['Pollution Level'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=pollution_counts.index, y=pollution_counts.values,
                   name='오염 수준', marker_color='lightcoral'),
            row=1, col=1
        )
       
        # 해안별 샘플 수
        shore_counts = self.data['Shore'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=shore_counts.index, y=shore_counts.values,
                   name='해안별', marker_color='lightblue'),
            row=1, col=2
        )
       
        # 계절별 분포
        season_counts = self.data['Season'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=season_counts.index, y=season_counts.values,
                   name='계절별', marker_color='lightgreen'),
            row=2, col=1
        )
       
        # 월별 트렌드
        monthly_pollution = self.data.groupby('Month')['Pollution Level'].mean()
        fig.add_trace(
            go.Scatter(x=monthly_pollution.index, y=monthly_pollution.values,
                      mode='lines+markers', name='월별 평균 오염도',
                      line=dict(color='orange', width=3)),
            row=2, col=2
        )
       
        fig.update_layout(
            title_text="🌊 해안 오염 데이터 개요",
            showlegend=False,
            height=600
        )
       
        return fig
   
    def plot_feature_importance(self):
        """특성 중요도 시각화"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
       
        fig = px.bar(
            self.feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='🎯 특성 중요도 (Top 10)',
            labels={'importance': '중요도', 'feature': '특성'}
        )
       
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
       
        return fig
   
    def plot_pollution_heatmap(self):
        """해안별 오염 수준 히트맵"""
        # 해안별, 계절별 평균 오염도
        heatmap_data = self.data.pivot_table(
            values='Pollution Level',
            index='Shore',
            columns='Season',
            aggfunc='mean'
        )
       
        fig = px.imshow(
            heatmap_data,
            title='🗺️ 해안별-계절별 평균 오염도',
            labels=dict(x="계절", y="해안", color="오염 수준"),
            color_continuous_scale='Reds'
        )
       
        return fig
   
    def plot_correlation_matrix(self):
        """주요 변수들 간 상관관계 매트릭스"""
        # 수치형 변수들만 선택
        numeric_cols = [
            'Pollution Level', 'Water pH', 'Soil pH',
            'Water Salinity', 'Soil Salinity',
            'Mean Number of Nematode species 1 per gram soil',
            'Mean Number of Turbillaria per gram soil'
        ]
       
        correlation_matrix = self.data[numeric_cols].corr()
       
        fig = px.imshow(
            correlation_matrix,
            title='🔗 주요 변수들 간 상관관계',
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
       
        return fig
   
    def generate_report(self):
        """종합 리포트 생성"""
        if not self.is_trained:
            self.train_model()
       
        print("=" * 60)
        print("🌊 해안 오염 예측 모델 리포트")
        print("=" * 60)
       
        print(f"\n📊 데이터 개요:")
        print(f"• 총 샘플 수: {len(self.data)}")
        print(f"• 해안 수: {self.data['Shore'].nunique()}")
        print(f"• 조사 기간: {self.data['Month'].min()}월 ~ {self.data['Month'].max()}월")
       
        print(f"\n⚠️ 오염 수준별 분포:")
        for level in sorted(self.data['Pollution Level'].unique()):
            count = (self.data['Pollution Level'] == level).sum()
            print(f"• 수준 {level}: {count}개 ({count/len(self.data)*100:.1f}%)")
       
        print(f"\n🎯 상위 5개 중요 특성:")
        for i, row in self.feature_importance.head(5).iterrows():
            print(f"• {row['feature']}: {row['importance']:.3f}")
       
        return self.feature_importance

# 사용 예제
def main():
    """메인 실행 함수"""
    # 모델 초기화
    predictor = ShorePollutionPredictor()
   
    # 데이터 로드 (파일 경로를 실제 경로로 변경하세요)
    # data = predictor.load_data('Shore_Pollution.csv')
   
    # 모델 훈련
    # predictor.train_model()
   
    # 시각화
    # fig1 = predictor.plot_data_overview()
    # fig1.show()
   
    # fig2 = predictor.plot_feature_importance()
    # fig2.show()
   
    # fig3 = predictor.plot_pollution_heatmap()
    # fig3.show()
   
    # 새로운 데이터 예측 예제
    # new_sample = {
    #     'Month': 8,
    #     'Season': 4,
    #     'Shore': 1,
    #     'Mean Number of Nematode species 1 per gram soil': 5.7,
    #     'Mean Number of Turbillaria per gram soil': 0.15,
    #     'Water pH': 8.0,
    #     'Soil pH': 8.2,
    #     'Water Salinity': 38.5,
    #     'Soil Salinity': 11.0,
    #     'Total dissolved solids': 57500,
    #     'Conduction': 57400,
    #     'ORP': -79.9
    # }
   
    # prediction, probability = predictor.predict(new_sample)
    # print(f"예측 결과: 오염 수준 {prediction[0]}")
    # print(f"예측 확률: {probability[0]}")
   
    # 리포트 생성
    # predictor.generate_report()

if __name__ == "__main__":
    main()
