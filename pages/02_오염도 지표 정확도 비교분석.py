import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class PollutionIndicatorAnalyzer:
    """
    생물 다양성 지표 vs 환경 지표의 오염도 설명력 비교 분석 시스템
    Streamlit과 Plotly를 사용한 대화형 분석 도구
    """
    
    def __init__(self):
        self.data = None
        self.bio_indicators = None
        self.env_indicators = None
        self.pollution_target = 'Pollution Level'
        self.results = {}
        
    def load_data(self, data):
        """데이터 로드 및 지표 분류"""
        self.data = data
        
        # 생물 다양성 지표 정의
        self.bio_indicators = [
            'Mean Number of Nematode species 1 per gram soil',
            'Mean Number of Turbillaria per gram soil',
            'Mean Number of foraminefera per gram soil',
            'Mean Number of Nematode species 2 per gram soil',
            'Organic matter%'
        ]
        
        # 환경 지표 정의 (화학적 + 물리적)
        self.env_indicators = [
            'Water pH', 'Soil pH', 'Water Salinity', 'Soil Salinity',
            'Total dissolved solids', 'Conduction', 'ORP', 
            'Specific resistance', 'Temp ©', 'Conductivity',
            'P', 'PP', 'OC', 'H', 'C-A', 'C-B', 'C-C'
        ]
        
        # 실제 존재하는 컬럼만 필터링
        self.bio_indicators = [col for col in self.bio_indicators if col in self.data.columns]
        self.env_indicators = [col for col in self.env_indicators if col in self.data.columns]
        
        return len(self.bio_indicators), len(self.env_indicators)
    
    def calculate_correlation_analysis(self):
        """상관관계 분석"""
        results = {
            'biological': {'correlations': [], 'indicators': [], 'p_values': []},
            'environmental': {'correlations': [], 'indicators': [], 'p_values': []}
        }
        
        # 생물학적 지표 상관관계
        for indicator in self.bio_indicators:
            clean_data = self.data[[indicator, self.pollution_target]].dropna()
            if len(clean_data) > 10:
                corr, p_val = pearsonr(clean_data[indicator], clean_data[self.pollution_target])
                results['biological']['correlations'].append(abs(corr))
                results['biological']['indicators'].append(indicator.split(' ')[0:3])
                results['biological']['p_values'].append(p_val)
        
        # 환경 지표 상관관계
        for indicator in self.env_indicators:
            clean_data = self.data[[indicator, self.pollution_target]].dropna()
            if len(clean_data) > 10:
                corr, p_val = pearsonr(clean_data[indicator], clean_data[self.pollution_target])
                results['environmental']['correlations'].append(abs(corr))
                results['environmental']['indicators'].append(indicator)
                results['environmental']['p_values'].append(p_val)
        
        self.results['correlation'] = results
        return results
    
    def calculate_predictive_power(self):
        """예측력 비교 분석"""
        target = self.data[self.pollution_target].dropna()
        
        # 생물학적 지표 데이터
        bio_data = self.data[self.bio_indicators].fillna(self.data[self.bio_indicators].median())
        bio_data = bio_data.loc[target.index]
        
        # 환경 지표 데이터  
        env_data = self.data[self.env_indicators].fillna(self.data[self.env_indicators].median())
        env_data = env_data.loc[target.index]
        
        results = {}
        
        # Random Forest 분류 성능
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        if len(bio_data.columns) > 0:
            bio_scores = cross_val_score(rf_classifier, bio_data, target, cv=5, scoring='accuracy')
            results['bio_accuracy'] = bio_scores.mean()
            results['bio_accuracy_std'] = bio_scores.std()
        
        if len(env_data.columns) > 0:
            env_scores = cross_val_score(rf_classifier, env_data, target, cv=5, scoring='accuracy')
            results['env_accuracy'] = env_scores.mean()
            results['env_accuracy_std'] = env_scores.std()
        
        # Random Forest 회귀 성능 (R²)
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        if len(bio_data.columns) > 0:
            bio_r2 = cross_val_score(rf_regressor, bio_data, target, cv=5, scoring='r2')
            results['bio_r2'] = bio_r2.mean()
            results['bio_r2_std'] = bio_r2.std()
        
        if len(env_data.columns) > 0:
            env_r2 = cross_val_score(rf_regressor, env_data, target, cv=5, scoring='r2')
            results['env_r2'] = env_r2.mean()
            results['env_r2_std'] = env_r2.std()
        
        # 특성 중요도 분석
        combined_data = pd.concat([bio_data, env_data], axis=1)
        if len(combined_data.columns) > 0:
            rf_regressor.fit(combined_data, target)
            feature_importance = pd.DataFrame({
                'feature': combined_data.columns,
                'importance': rf_regressor.feature_importances_,
                'type': ['생물학적' if col in self.bio_indicators else '환경적' 
                        for col in combined_data.columns]
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = feature_importance
        
        self.results['predictive_power'] = results
        return results
    
    def calculate_group_differences(self):
        """오염도별 그룹 차이 분석"""
        results = {'biological': [], 'environmental': []}
        
        # 생물학적 지표 그룹 차이
        for indicator in self.bio_indicators:
            clean_data = self.data[[indicator, self.pollution_target]].dropna()
            if len(clean_data) > 20:
                groups = []
                for level in sorted(clean_data[self.pollution_target].unique()):
                    group_data = clean_data[clean_data[self.pollution_target] == level][indicator]
                    if len(group_data) >= 3:
                        groups.append(group_data)
                
                if len(groups) >= 2:
                    try:
                        statistic, p_value = stats.kruskal(*groups)
                        effect_size = self._calculate_effect_size(groups)
                        
                        results['biological'].append({
                            'indicator': indicator,
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'significant': p_value < 0.05
                        })
                    except:
                        pass
        
        # 환경 지표 그룹 차이
        for indicator in self.env_indicators:
            clean_data = self.data[[indicator, self.pollution_target]].dropna()
            if len(clean_data) > 20:
                groups = []
                for level in sorted(clean_data[self.pollution_target].unique()):
                    group_data = clean_data[clean_data[self.pollution_target] == level][indicator]
                    if len(group_data) >= 3:
                        groups.append(group_data)
                
                if len(groups) >= 2:
                    try:
                        statistic, p_value = stats.kruskal(*groups)
                        effect_size = self._calculate_effect_size(groups)
                        
                        results['environmental'].append({
                            'indicator': indicator,
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'significant': p_value < 0.05
                        })
                    except:
                        pass
        
        self.results['group_differences'] = results
        return results
    
    def _calculate_effect_size(self, groups):
        """효과 크기 계산 (eta-squared)"""
        all_values = np.concatenate(groups)
        overall_mean = np.mean(all_values)
        
        ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in groups)
        ss_total = sum((value - overall_mean)**2 for value in all_values)
        
        return ss_between / ss_total if ss_total > 0 else 0
    
    def plot_correlation_comparison(self):
        """상관관계 비교 시각화"""
        if 'correlation' not in self.results:
            self.calculate_correlation_analysis()
        
        corr_data = self.results['correlation']
        
        # 데이터 준비
        bio_corrs = corr_data['biological']['correlations']
        env_corrs = corr_data['environmental']['correlations']
        
        fig = go.Figure()
        
        # 박스플롯으로 분포 비교
        fig.add_trace(go.Box(
            y=bio_corrs,
            name='생물학적 지표',
            marker_color='lightgreen',
            boxpoints='all'
        ))
        
        fig.add_trace(go.Box(
            y=env_corrs,
            name='환경 지표',
            marker_color='lightblue',
            boxpoints='all'
        ))
        
        fig.update_layout(
            title='📊 오염도와의 상관관계 강도 비교',
            yaxis_title='절대 상관계수',
            xaxis_title='지표 유형',
            height=500
        )
        
        return fig
    
    def plot_predictive_performance(self):
        """예측 성능 비교 시각화"""
        if 'predictive_power' not in self.results:
            self.calculate_predictive_power()
        
        results = self.results['predictive_power']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('분류 정확도', 'R² 점수'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 분류 정확도
        categories = ['생물학적 지표', '환경 지표']
        accuracies = [results.get('bio_accuracy', 0), results.get('env_accuracy', 0)]
        accuracy_errors = [results.get('bio_accuracy_std', 0), results.get('env_accuracy_std', 0)]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=accuracies,
                error_y=dict(type='data', array=accuracy_errors),
                marker_color=['lightgreen', 'lightblue'],
                name='정확도'
            ),
            row=1, col=1
        )
        
        # R² 점수
        r2_scores = [results.get('bio_r2', 0), results.get('env_r2', 0)]
        r2_errors = [results.get('bio_r2_std', 0), results.get('env_r2_std', 0)]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=r2_scores,
                error_y=dict(type='data', array=r2_errors),
                marker_color=['lightgreen', 'lightblue'],
                name='R² 점수'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='🎯 오염도 예측 성능 비교',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self):
        """특성 중요도 시각화"""
        if 'predictive_power' not in self.results or 'feature_importance' not in self.results['predictive_power']:
            self.calculate_predictive_power()
        
        importance_df = self.results['predictive_power']['feature_importance']
        top_features = importance_df.head(15)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            color='type',
            orientation='h',
            title='🔍 개별 지표 중요도 Top 15',
            labels={'importance': '중요도', 'feature': '지표', 'type': '유형'},
            color_discrete_map={'생물학적': 'lightgreen', '환경적': 'lightblue'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        return fig
    
    def plot_pollution_distribution(self):
        """오염도별 지표 분포 비교"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('선충류 vs 오염도', 'pH vs 오염도', '터빌라리아 vs 오염도', '염분 vs 오염도')
        )
        
        pollution_levels = self.data[self.pollution_target].unique()
        colors = ['green', 'orange', 'red']
        
        # 선충류 vs 오염도
        if 'Mean Number of Nematode species 1 per gram soil' in self.data.columns:
            for i, level in enumerate(sorted(pollution_levels)):
                level_data = self.data[self.data[self.pollution_target] == level]
                fig.add_trace(
                    go.Box(
                        y=level_data['Mean Number of Nematode species 1 per gram soil'],
                        name=f'수준 {level}',
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # pH vs 오염도
        if 'Water pH' in self.data.columns:
            for i, level in enumerate(sorted(pollution_levels)):
                level_data = self.data[self.data[self.pollution_target] == level]
                fig.add_trace(
                    go.Box(
                        y=level_data['Water pH'],
                        name=f'수준 {level}',
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 터빌라리아 vs 오염도
        if 'Mean Number of Turbillaria per gram soil' in self.data.columns:
            for i, level in enumerate(sorted(pollution_levels)):
                level_data = self.data[self.data[self.pollution_target] == level]
                fig.add_trace(
                    go.Box(
                        y=level_data['Mean Number of Turbillaria per gram soil'],
                        name=f'수준 {level}',
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 염분 vs 오염도
        if 'Water Salinity' in self.data.columns:
            for i, level in enumerate(sorted(pollution_levels)):
                level_data = self.data[self.data[self.pollution_target] == level]
                fig.add_trace(
                    go.Box(
                        y=level_data['Water Salinity'],
                        name=f'수준 {level}',
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='📈 오염도별 주요 지표 분포',
            height=600
        )
        
        return fig
    
    def plot_indicator_correlation_heatmap(self):
        """지표간 상관관계 히트맵"""
        # 주요 지표들 선택
        key_indicators = []
        if self.bio_indicators:
            key_indicators.extend(self.bio_indicators[:3])
        if self.env_indicators:
            key_indicators.extend(self.env_indicators[:5])
        key_indicators.append(self.pollution_target)
        
        # 상관관계 매트릭스 계산
        corr_matrix = self.data[key_indicators].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='🔗 주요 지표간 상관관계 히트맵',
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        
        return fig
    
    def generate_summary_report(self):
        """종합 분석 리포트 생성"""
        if not all(key in self.results for key in ['correlation', 'predictive_power']):
            self.calculate_correlation_analysis()
            self.calculate_predictive_power()
            self.calculate_group_differences()
        
        report = {
            'correlation_summary': {},
            'prediction_summary': {},
            'recommendation': ""
        }
        
        # 상관관계 요약
        corr_data = self.results['correlation']
        bio_corr_avg = np.mean(corr_data['biological']['correlations']) if corr_data['biological']['correlations'] else 0
        env_corr_avg = np.mean(corr_data['environmental']['correlations']) if corr_data['environmental']['correlations'] else 0
        
        report['correlation_summary'] = {
            'bio_avg_correlation': bio_corr_avg,
            'env_avg_correlation': env_corr_avg,
            'better_correlator': '생물학적 지표' if bio_corr_avg > env_corr_avg else '환경 지표'
        }
        
        # 예측 성능 요약
        pred_data = self.results['predictive_power']
        bio_accuracy = pred_data.get('bio_accuracy', 0)
        env_accuracy = pred_data.get('env_accuracy', 0)
        bio_r2 = pred_data.get('bio_r2', 0)
        env_r2 = pred_data.get('env_r2', 0)
        
        report['prediction_summary'] = {
            'bio_accuracy': bio_accuracy,
            'env_accuracy': env_accuracy,
            'bio_r2': bio_r2,
            'env_r2': env_r2,
            'better_predictor': '생물학적 지표' if (bio_accuracy + bio_r2) > (env_accuracy + env_r2) else '환경 지표'
        }
        
        # 권장사항 생성
        if bio_corr_avg > env_corr_avg and (bio_accuracy + bio_r2) > (env_accuracy + env_r2):
            report['recommendation'] = "생물학적 지표가 오염도를 더 잘 설명합니다. 생물다양성 모니터링에 집중하는 것을 권장합니다."
        elif env_corr_avg > bio_corr_avg and (env_accuracy + env_r2) > (bio_accuracy + bio_r2):
            report['recommendation'] = "환경 지표가 오염도를 더 잘 설명합니다. 화학적·물리적 환경 모니터링에 집중하는 것을 권장합니다."
        else:
            report['recommendation'] = "생물학적 지표와 환경 지표 모두 중요합니다. 통합적인 모니터링 접근법을 권장합니다."
        
        return report

# Streamlit 앱 메인 함수
def main():
    st.set_page_config(
        page_title="오염도 지표 비교 분석",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("🌊 생물 다양성 vs 환경 지표: 오염도 설명력 비교")
    st.markdown("---")
    
    # 사이드바
    st.sidebar.title("📋 분석 설정")
    
    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일 업로드",
        type=['csv'],
        help="Shore_Pollution.csv 파일을 업로드하세요"
    )
    
    if uploaded_file is not None:
        # 데이터 로드
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ 데이터 로드 성공: {len(data)}개 샘플")
            
            # 분석기 초기화
            analyzer = PollutionIndicatorAnalyzer()
            bio_count, env_count = analyzer.load_data(data)
            
            st.sidebar.info(f"🦠 생물 지표: {bio_count}개")
            st.sidebar.info(f"🌡️ 환경 지표: {env_count}개")
            
            # 분석 옵션
            analysis_options = st.sidebar.multiselect(
                "분석 유형 선택",
                ["상관관계 분석", "예측 성능 비교", "특성 중요도", "분포 비교", "종합 리포트"],
                default=["상관관계 분석", "예측 성능 비교"]
            )
            
            # 메인 컨텐츠
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("📊 분석 결과")
                
                if "상관관계 분석" in analysis_options:
                    st.subheader("1. 상관관계 강도 비교")
                    with st.spinner("상관관계 분석 중..."):
                        fig_corr = analyzer.plot_correlation_comparison()
                        st.plotly_chart(fig_corr, use_container_width=True)
                
                if "예측 성능 비교" in analysis_options:
                    st.subheader("2. 오염도 예측 성능")
                    with st.spinner("예측 성능 계산 중..."):
                        fig_pred = analyzer.plot_predictive_performance()
                        st.plotly_chart(fig_pred, use_container_width=True)
                
                if "특성 중요도" in analysis_options:
                    st.subheader("3. 개별 지표 중요도")
                    with st.spinner("특성 중요도 분석 중..."):
                        fig_importance = analyzer.plot_feature_importance()
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                if "분포 비교" in analysis_options:
                    st.subheader("4. 오염도별 지표 분포")
                    with st.spinner("분포 분석 중..."):
                        fig_dist = analyzer.plot_pollution_distribution()
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        fig_heatmap = analyzer.plot_indicator_correlation_heatmap()
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                st.header("📋 요약 정보")
                
                # 데이터 개요
                st.subheader("데이터 개요")
                st.metric("총 샘플 수", len(data))
                st.metric("오염 수준", len(data['Pollution Level'].unique()))
                st.metric("조사 해안", len(data['Shore'].unique()))
                
                # 오염도 분포
                st.subheader("오염도 분포")
                pollution_dist = data['Pollution Level'].value_counts().sort_index()
                for level, count in pollution_dist.items():
                    st.metric(f"수준 {level}", f"{count}개", f"{count/len(data)*100:.1f}%")
                
                if "종합 리포트" in analysis_options:
                    st.subheader("🎯 종합 분석 결과")
                    with st.spinner("종합 리포트 생성 중..."):
                        report = analyzer.generate_summary_report()
                        
                        st.write("**상관관계 분석:**")
                        st.write(f"• 생물학적 지표 평균: {report['correlation_summary']['bio_avg_correlation']:.3f}")
                        st.write(f"• 환경 지표 평균: {report['correlation_summary']['env_avg_correlation']:.3f}")
                        st.write(f"• 더 강한 상관관계: {report['correlation_summary']['better_correlator']}")
                        
                        st.write("**예측 성능:**")
                        st.write(f"• 생물학적 정확도: {report['prediction_summary']['bio_accuracy']:.3f}")
                        st.write(f"• 환경 지표 정확도: {report['prediction_summary']['env_accuracy']:.3f}")
                        st.write(f"• 더 나은 예측력: {report['prediction_summary']['better_predictor']}")
                        
                        st.success(f"**권장사항:** {report['recommendation']}")
            
            # 상세 통계 정보
            if st.checkbox("상세 통계 정보 보기"):
                st.subheader("📈 상세 통계")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**생물학적 지표:**")
                    for indicator in analyzer.bio_indicators:
                        clean_data = data[[indicator, 'Pollution Level']].dropna()
                        if len(clean_data) > 0:
                            corr, p_val = pearsonr(clean_data[indicator], clean_data['Pollution Level'])
                            st.write(f"• {indicator[:20]}...: r={corr:.3f}, p={p_val:.3f}")
                
                with col2:
                    st.write("**환경 지표:**")
                    for indicator in analyzer.env_indicators[:5]:  # 상위 5개만 표시
                        clean_data = data[[indicator, 'Pollution Level']].dropna()
                        if len(clean_data) > 0:
                            corr, p_val = pearsonr(clean_data[indicator], clean_data['Pollution Level'])
                            st.write(f"• {indicator}: r={corr:.3f}, p={p_val:.3f}")
                        
        except Exception as e:
            st.error(f"❌ 파일 처리 중 오류 발생: {str(e)}")
    
    else:
        st.info("👆 사이드바에서 CSV 파일을 업로드하여 분석을 시작하세요.")
        
        # 샘플 데이터 구조 설명
        st.subheader("📋 필요한 데이터 구조")
        st.write("""
        **필수 컬럼:**
        - `Pollution Level`: 오염 수준 (0, 1, 2)
        - `Shore`: 해안 지점
        - `Month`, `Season`: 시간 정보
        
        **생물학적 지표:**
        - `Mean Number of Nematode species 1 per gram soil`
        - `Mean Number of Turbillaria per gram soil`
        - `Mean Number of foraminefera per gram soil`
        
        **환경 지표:**
        - `Water pH`, `Soil pH`
        - `Water Salinity`, `Soil Salinity`
        - `Total dissolved solids`
        - `Conduction`, `ORP`, `Temp ©`
        """)

if __name__ == "__main__":
    main()
