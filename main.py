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
import os
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
        self.data = None
       
    def load_data(self, filepath):
        """데이터 로드 및 전처리"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
            
            self.data = pd.read_csv(filepath)
            
            # 데이터 기본 검증
            if self.data.empty:
                raise ValueError("데이터가 비어있습니다.")
                
            # 필수 컬럼 존재 확인
            required_columns = ['Pollution Level']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
            
            print(f"📊 데이터 로드 완료: {len(self.data)}개 샘플")
            print(f"📊 컬럼 수: {len(self.data.columns)}")
            print(f"📊 결측값 현황:")
            print(self.data.isnull().sum().sort_values(ascending=False).head(10))
            
            return self.data
            
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {str(e)}")
            return None
   
    def preprocess_data(self):
        """데이터 전처리"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
        
        # 사용 가능한 컬럼 확인 및 주요 특성 선택
        available_columns = self.data.columns.tolist()
        
        # 우선순위에 따른 특성 선택
        potential_features = [
            'Month', 'Season', 'Shore',
            'Mean Number of Nematode species 1 per gram soil',
            'Mean Number of Turbillaria per gram soil',
            'Water pH', 'Soil pH', 'Water Salinity', 'Soil Salinity',
            'Total dissolved solids', 'Conduction', 'ORP'
        ]
        
        # 실제 존재하는 컬럼만 선택
        self.feature_columns = [col for col in potential_features if col in available_columns]
        
        if not self.feature_columns:
            raise ValueError("사용 가능한 특성 컬럼이 없습니다.")
        
        print(f"📋 사용할 특성 컬럼: {len(self.feature_columns)}개")
        for col in self.feature_columns:
            print(f"  - {col}")
       
        # 특성과 타겟 분리
        X = self.data[self.feature_columns].copy()
        y = self.data['Pollution Level']
        
        # 타겟 변수 검증
        if y.isnull().sum() > 0:
            print(f"⚠️ 타겟 변수에 결측값이 {y.isnull().sum()}개 있습니다. 해당 행을 제거합니다.")
            valid_idx = ~y.isnull()
            X = X[valid_idx]
            y = y[valid_idx]
       
        # 결측값 처리 (수치형은 중앙값, 범주형은 최빈값으로 대체)
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        # 무한값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"📊 전처리 완료: {len(X)}개 샘플, {len(X.columns)}개 특성")
        
        return X, y
   
    def train_model(self, test_size=0.2):
        """모델 훈련"""
        try:
            X, y = self.preprocess_data()
            
            # 최소 샘플 수 확인
            if len(X) < 10:
                raise ValueError("훈련을 위한 최소 샘플 수가 부족합니다.")
            
            # 클래스 분포 확인
            class_counts = y.value_counts()
            print(f"📊 클래스 분포:")
            for class_val, count in class_counts.items():
                print(f"  클래스 {class_val}: {count}개")
            
            # 각 클래스에 최소 2개 이상의 샘플이 있는지 확인
            if any(count < 2 for count in class_counts.values()):
                print("⚠️ 일부 클래스의 샘플이 부족합니다. stratify 옵션을 비활성화합니다.")
                stratify_param = None
            else:
                stratify_param = y
       
            # 훈련/테스트 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify_param
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
            print(f"훈련 세트 크기: {len(X_train)}")
            print(f"테스트 세트 크기: {len(X_test)}")
            print("\n분류 보고서:")
            print(classification_report(y_test, y_pred, zero_division=0))
       
            return X_train, X_test, y_train, y_test, y_pred
            
        except Exception as e:
            print(f"❌ 모델 훈련 중 오류 발생: {str(e)}")
            return None, None, None, None, None
   
    def predict(self, new_data):
        """새로운 데이터 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. train_model()을 먼저 실행하세요.")
        
        if self.data is None:
            raise ValueError("원본 데이터가 없습니다. 예측을 위해 필요합니다.")
       
        try:
            # 입력 데이터 전처리
            if isinstance(new_data, dict):
                new_data = pd.DataFrame([new_data])
            
            # 필요한 컬럼만 선택
            missing_cols = [col for col in self.feature_columns if col not in new_data.columns]
            if missing_cols:
                raise ValueError(f"예측에 필요한 컬럼이 누락되었습니다: {missing_cols}")
            
            X_new = new_data[self.feature_columns].copy()
            
            # 결측값 처리
            for col in X_new.columns:
                if X_new[col].dtype in ['int64', 'float64']:
                    X_new[col] = X_new[col].fillna(self.data[col].median())
                else:
                    mode_val = self.data[col].mode()
                    X_new[col] = X_new[col].fillna(mode_val[0] if not mode_val.empty else 0)
            
            # 무한값 처리
            X_new = X_new.replace([np.inf, -np.inf], np.nan)
            for col in X_new.columns:
                if X_new[col].dtype in ['int64', 'float64']:
                    X_new[col] = X_new[col].fillna(self.data[col].median())
            
            X_new_scaled = self.scaler.transform(X_new)
       
            # 예측
            prediction = self.model.predict(X_new_scaled)
            probability = self.model.predict_proba(X_new_scaled)
       
            return prediction, probability
            
        except Exception as e:
            print(f"❌ 예측 중 오류 발생: {str(e)}")
            return None, None
   
    def plot_data_overview(self):
        """데이터 개요 시각화"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
            
        try:
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
           
            # 해안별 샘플 수 (컬럼이 존재하는 경우만)
            if 'Shore' in self.data.columns:
                shore_counts = self.data['Shore'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(x=shore_counts.index, y=shore_counts.values,
                           name='해안별', marker_color='lightblue'),
                    row=1, col=2
                )
           
            # 계절별 분포 (컬럼이 존재하는 경우만)
            if 'Season' in self.data.columns:
                season_counts = self.data['Season'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(x=season_counts.index, y=season_counts.values,
                           name='계절별', marker_color='lightgreen'),
                    row=2, col=1
                )
           
            # 월별 트렌드 (컬럼이 존재하는 경우만)
            if 'Month' in self.data.columns:
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
            
        except Exception as e:
            print(f"❌ 시각화 중 오류 발생: {str(e)}")
            return None
   
    def plot_feature_importance(self):
        """특성 중요도 시각화"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        if self.feature_importance is None:
            raise ValueError("특성 중요도 정보가 없습니다.")
        
        try:
            fig = px.bar(
                self.feature_importance.head(min(10, len(self.feature_importance))),
                x='importance',
                y='feature',
                orientation='h',
                title='🎯 특성 중요도 (Top 10)',
                labels={'importance': '중요도', 'feature': '특성'}
            )
           
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(self.feature_importance.head(10)) * 50)
            )
           
            return fig
            
        except Exception as e:
            print(f"❌ 특성 중요도 시각화 중 오류 발생: {str(e)}")
            return None
   
    def plot_pollution_heatmap(self):
        """해안별 오염 수준 히트맵"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 필요한 컬럼 확인
        required_cols = ['Shore', 'Season', 'Pollution Level']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            print(f"⚠️ 히트맵 생성에 필요한 컬럼이 누락되었습니다: {missing_cols}")
            return None
        
        try:
            # 해안별, 계절별 평균 오염도
            heatmap_data = self.data.pivot_table(
                values='Pollution Level',
                index='Shore',
                columns='Season',
                aggfunc='mean'
            )
           
            if heatmap_data.empty:
                print("⚠️ 히트맵 생성을 위한 데이터가 부족합니다.")
                return None
            
            fig = px.imshow(
                heatmap_data,
                title='🗺️ 해안별-계절별 평균 오염도',
                labels=dict(x="계절", y="해안", color="오염 수준"),
                color_continuous_scale='Reds'
            )
           
            return fig
            
        except Exception as e:
            print(f"❌ 히트맵 생성 중 오류 발생: {str(e)}")
            return None
   
    def plot_correlation_matrix(self):
        """주요 변수들 간 상관관계 매트릭스"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        try:
            # 수치형 변수들만 선택
            numeric_cols = [
                'Pollution Level', 'Water pH', 'Soil pH',
                'Water Salinity', 'Soil Salinity',
                'Mean Number of Nematode species 1 per gram soil',
                'Mean Number of Turbillaria per gram soil'
            ]
            
            # 실제 존재하는 컬럼만 선택
            available_numeric_cols = [col for col in numeric_cols if col in self.data.columns]
            
            if len(available_numeric_cols) < 2:
                print("⚠️ 상관관계 분석을 위한 수치형 컬럼이 부족합니다.")
                return None
            
            correlation_matrix = self.data[available_numeric_cols].corr()
           
            fig = px.imshow(
                correlation_matrix,
                title='🔗 주요 변수들 간 상관관계',
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1
            )
           
            return fig
            
        except Exception as e:
            print(f"❌ 상관관계 매트릭스 생성 중 오류 발생: {str(e)}")
            return None
   
    def generate_report(self):
        """종합 리포트 생성"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        if not self.is_trained:
            print("모델이 훈련되지 않았습니다. 훈련을 시작합니다...")
            result = self.train_model()
            if result[0] is None:  # 훈련 실패
                print("❌ 모델 훈련에 실패했습니다. 리포트 생성을 중단합니다.")
                return None
       
        try:
            print("=" * 60)
            print("🌊 해안 오염 예측 모델 리포트")
            print("=" * 60)
           
            print(f"\n📊 데이터 개요:")
            print(f"• 총 샘플 수: {len(self.data)}")
            
            if 'Shore' in self.data.columns:
                print(f"• 해안 수: {self.data['Shore'].nunique()}")
            
            if 'Month' in self.data.columns:
                print(f"• 조사 기간: {self.data['Month'].min()}월 ~ {self.data['Month'].max()}월")
           
            print(f"\n⚠️ 오염 수준별 분포:")
            for level in sorted(self.data['Pollution Level'].unique()):
                count = (self.data['Pollution Level'] == level).sum()
                print(f"• 수준 {level}: {count}개 ({count/len(self.data)*100:.1f}%)")
           
            if self.feature_importance is not None:
                print(f"\n🎯 상위 5개 중요 특성:")
                for i, row in self.feature_importance.head(5).iterrows():
                    print(f"• {row['feature']}: {row['importance']:.3f}")
           
            return self.feature_importance
            
        except Exception as e:
            print(f"❌ 리포트 생성 중 오류 발생: {str(e)}")
            return None

# 사용 예제
def main():
    """메인 실행 함수"""
    # 모델 초기화
    predictor = ShorePollutionPredictor()
   
    # 데이터 로드 (파일 경로를 실제 경로로 변경하세요)
    # data = predictor.load_data('Shore_Pollution.csv')
    
    # 데이터 로드가 성공한 경우에만 계속 진행
    # if data is not None:
    #     # 모델 훈련
    #     predictor.train_model()
    #     
    #     # 시각화
    #     fig1 = predictor.plot_data_overview()
    #     if fig1 is not None:
    #         fig1.show()
    #     
    #     fig2 = predictor.plot_feature_importance()
    #     if fig2 is not None:
    #         fig2.show()
    #     
    #     fig3 = predictor.plot_pollution_heatmap()
    #     if fig3 is not None:
    #         fig3.show()
    #     
    #     # 새로운 데이터 예측 예제
    #     new_sample = {
    #         'Month': 8,
    #         'Season': 4,
    #         'Shore': 1,
    #         'Mean Number of Nematode species 1 per gram soil': 5.7,
    #         'Mean Number of Turbillaria per gram soil': 0.15,
    #         'Water pH': 8.0,
    #         'Soil pH': 8.2,
    #         'Water Salinity': 38.5,
    #         'Soil Salinity': 11.0,
    #         'Total dissolved solids': 57500,
    #         'Conduction': 57400,
    #         'ORP': -79.9
    #     }
    #     
    #     prediction, probability = predictor.predict(new_sample)
    #     if prediction is not None:
    #         print(f"예측 결과: 오염 수준 {prediction[0]}")
    #         print(f"예측 확률: {probability[0]}")
    #     
    #     # 리포트 생성
    #     predictor.generate_report()
    
    print("모델 초기화 완료. 데이터 파일을 로드하여 사용하세요.")

if __name__ == "__main__":
    main()
