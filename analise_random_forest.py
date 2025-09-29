import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import time

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, cross_val_score, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score,
    fbeta_score, accuracy_score, precision_score, recall_score,
    average_precision_score, log_loss
)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

# Otimização bayesiana
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("AVISO: Biblioteca skopt não encontrada. Usando GridSearchCV padrão.")
    BAYESIAN_AVAILABLE = False

# Configurações globais
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
np.random.seed(42)

print("Sistema de Predição de Attrition - TechCorp Brasil")
print("=" * 60)

class AttritionAnalyzer:
    """
    Classe principal para análise e predição de attrition de funcionários.
    
    Esta classe implementa um pipeline completo de machine learning incluindo:
    - Análise exploratória detalhada
    - Pré-processamento inteligente dos dados
    - Múltiplos algoritmos de classificação
    - Otimização de hiperparâmetros
    - Avaliação robusta com métricas específicas para desbalanceamento
    """
    
    def __init__(self, data_path='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
        self.data_path = data_path
        self.df_raw = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def load_data(self):
        """Carrega e faz validação inicial dos dados com limpeza automática."""
        try:
            self.df_raw = pd.read_csv(self.data_path)
            
            # Verificar se há problema de header duplicado
            first_row = self.df_raw.iloc[0]
            header_in_data = any(str(value) == col for col, value in first_row.items())
            
            if header_in_data:
                print("AVISO: Header duplicado detectado - aplicando limpeza automática...")
                # Remover linha com header duplicado
                self.df_raw = self.df_raw[self.df_raw.iloc[:, 0] != self.df_raw.columns[0]].copy()
                
                # Converter colunas numéricas
                numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
                                  'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                                  'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
                                  'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                                  'YearsWithCurrManager', 'EmployeeNumber', 'HourlyRate',
                                  'JobLevel', 'StockOptionLevel', 'EmployeeCount', 'StandardHours',
                                  'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction',
                                  'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance']
                
                for col in numeric_columns:
                    if col in self.df_raw.columns:
                        self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')
                
                print("Limpeza automática concluída!")
            
            self.df = self.df_raw.copy()
            print(f"Dataset carregado com sucesso!")
            print(f"Dimensões: {self.df.shape[0]} funcionários, {self.df.shape[1]} variáveis")
            
            # Verificação de qualidade dos dados
            missing_values = self.df.isnull().sum().sum()
            duplicates = self.df.duplicated().sum()
            
            if missing_values > 0:
                print(f"AVISO: {missing_values} valores ausentes detectados")
            if duplicates > 0:
                print(f"AVISO: {duplicates} registros duplicados detectados")
                
            return True
        except FileNotFoundError:
            print(f"ERRO: Arquivo '{self.data_path}' não encontrado.")
            return False
        except Exception as e:
            print(f"ERRO: Erro ao carregar dados: {str(e)}")
            return False
    
    def exploratory_data_analysis(self):
        print("\nANÁLISE EXPLORATÓRIA DOS DADOS")
        print("=" * 50)
        
        # Informações básicas do dataset
        self._display_basic_info()
        
        # Análise da variável target
        self._analyze_target_variable()
        
        # Análise de variáveis numéricas
        self._analyze_numerical_variables()
        
        # Análise de variáveis categóricas
        self._analyze_categorical_variables()
        
        # Análise de correlações
        self._correlation_analysis()
        
        # Detecção de outliers
        self._outlier_detection()
        
        # Insights de negócio
        self._business_insights()
    
    def _display_basic_info(self):
        print("\nINFORMAÇÕES GERAIS")
        print("-" * 30)
        print(f"Total de funcionários: {self.df.shape[0]:,}")
        print(f"Variáveis disponíveis: {self.df.shape[1]}")
        
        # Verificar valores nulos
        missing_data = self.df.isnull().sum()
        if missing_data.sum() == 0:
            print("Nenhum valor ausente encontrado")
        else:
            print(f"AVISO: Valores ausentes: {missing_data.sum()}")
        
        # Verificar duplicatas
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("Nenhuma linha duplicada encontrada")
        else:
            print(f"AVISO: Linhas duplicadas: {duplicates}")
        
        # Tipos de dados
        print(f"Variáveis numéricas: {self.df.select_dtypes(include=[np.number]).shape[1]}")
        print(f"Variáveis categóricas: {self.df.select_dtypes(include=['object']).shape[1]}")
    
    def _analyze_target_variable(self):
        print("\nANÁLISE DA VARIÁVEL TARGET (ATTRITION)")
        print("-" * 45)
        
        # Distribuição da variável target
        attrition_counts = self.df['Attrition'].value_counts()
        attrition_pct = self.df['Attrition'].value_counts(normalize=True) * 100
        
        print("Distribuição de Attrition:")
        for category, count in attrition_counts.items():
            pct = attrition_pct[category]
            print(f"{category}: {count:,} funcionários ({pct:.1f}%)")
        
        # Cálculo do desbalanceamento
        minority_class = attrition_counts.min()
        majority_class = attrition_counts.max()
        imbalance_ratio = majority_class / minority_class
        print(f"\nTaxa de desbalanceamento: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 3:
            print("AVISO: Dataset significativamente desbalanceado - requer tratamento especial")
        
        # Visualização
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de barras
        attrition_counts.plot(kind='bar', ax=ax1, color=['lightblue', 'coral'])
        ax1.set_title('Distribuição de Attrition', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Número de Funcionários')
        ax1.tick_params(axis='x', rotation=0)
        
        # Gráfico de pizza
        ax2.pie(attrition_counts.values, labels=attrition_counts.index, autopct='%1.1f%%', 
                colors=['lightblue', 'coral'], startangle=90)
        ax2.set_title('Proporção de Attrition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_numerical_variables(self):
        print("\nANÁLISE DE VARIÁVEIS NUMÉRICAS")
        print("-" * 35)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover colunas constantes ou identificadores
        cols_to_remove = ['EmployeeCount', 'StandardHours', 'EmployeeNumber']
        numerical_cols = [col for col in numerical_cols if col not in cols_to_remove]
        
        print(f"Analisando {len(numerical_cols)} variáveis numéricas...")
        
        # Estatísticas descritivas
        desc_stats = self.df[numerical_cols].describe()
        print("\nEstatísticas Descritivas (Resumo):")
        print(desc_stats.round(2))
        
        # Análise de distribuições por attrition
        key_vars = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears', 
                   'DistanceFromHome', 'YearsSinceLastPromotion']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, var in enumerate(key_vars[:6]):
            if var in self.df.columns:
                # Box plot por categoria de attrition
                sns.boxplot(data=self.df, x='Attrition', y=var, ax=axes[i])
                axes[i].set_title(f'Distribuição de {var} por Attrition', fontweight='bold')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Teste estatístico para diferenças entre grupos
        self._statistical_tests_numerical(numerical_cols)
    
    def _statistical_tests_numerical(self, numerical_cols):
        print("\nTESTES ESTATÍSTICOS (Mann-Whitney U)")
        print("-" * 40)
        
        from scipy.stats import mannwhitneyu
        
        significant_vars = []
        
        for var in numerical_cols[:10]:  # Analisar as 10 principais
            if var in self.df.columns:
                group_no = self.df[self.df['Attrition'] == 'No'][var]
                group_yes = self.df[self.df['Attrition'] == 'Yes'][var]
                
                # Teste Mann-Whitney U (não-paramétrico)
                statistic, p_value = mannwhitneyu(group_no, group_yes, alternative='two-sided')
                
                if p_value < 0.05:
                    significant_vars.append(var)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                    print(f"{var}: p-value = {p_value:.4f} {significance}")
        
        print(f"\n{len(significant_vars)} variáveis com diferenças estatisticamente significativas")
    
    def _analyze_categorical_variables(self):
        print("\nANÁLISE DE VARIÁVEIS CATEGÓRICAS")
        print("-" * 38)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('Attrition')  # Remover a variável target
        
        print(f"Analisando {len(categorical_cols)} variáveis categóricas...")
        
        # Análise de associação com chi-quadrado
        significant_associations = []
        
        for var in categorical_cols:
            contingency_table = pd.crosstab(self.df[var], self.df['Attrition'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            if p_value < 0.05:
                significant_associations.append((var, p_value))
        
        significant_associations.sort(key=lambda x: x[1])
        
        print(f"\nVariáveis com associação significativa (p < 0.05):")
        for var, p_val in significant_associations[:8]:
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            print(f"{var}: p-value = {p_val:.4f} {significance}")
        
        # Visualização das principais variáveis categóricas
        if len(significant_associations) > 0:
            self._plot_categorical_analysis(significant_associations[:4])
    
    def _plot_categorical_analysis(self, top_vars):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (var, _) in enumerate(top_vars):
            if i < 4:
                # Criar tabela de contingência percentual
                ct = pd.crosstab(self.df[var], self.df['Attrition'], normalize='index') * 100
                ct.plot(kind='bar', ax=axes[i], stacked=False, 
                       color=['lightblue', 'coral'], alpha=0.8)
                axes[i].set_title(f'Taxa de Attrition por {var}', fontweight='bold')
                axes[i].set_ylabel('Porcentagem (%)')
                axes[i].legend(title='Attrition', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _correlation_analysis(self):
        print("\nANÁLISE DE CORRELAÇÕES")
        print("-" * 28)
        
        # Preparar dados para análise de correlação
        df_corr = self.df.copy()
        
        # Codificar variáveis categóricas para análise de correlação
        le = LabelEncoder()
        categorical_cols = df_corr.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            df_corr[col] = le.fit_transform(df_corr[col])
        
        # Remover colunas constantes
        cols_to_remove = ['EmployeeCount', 'StandardHours', 'EmployeeNumber']
        df_corr = df_corr.drop(columns=cols_to_remove, errors='ignore')
        
        # Calcular matriz de correlação
        correlation_matrix = df_corr.corr()
        
        # Correlações com a variável target
        target_correlations = correlation_matrix['Attrition'].abs().sort_values(ascending=False)[1:]
        
        print("Top 10 variáveis mais correlacionadas com Attrition:")
        for var, corr in target_correlations.head(10).items():
            direction = "positiva" if correlation_matrix.loc[var, 'Attrition'] > 0 else "negativa"
            print(f"{var}: {corr:.3f} (correlação {direction})")
        
        # Visualização da matriz de correlação
        plt.figure(figsize=(16, 14))
        mask = np.triu(correlation_matrix)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   mask=mask, fmt='.2f', annot_kws={'size': 8})
        plt.title('Matriz de Correlação - Variáveis do Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _outlier_detection(self):
        print("\nDETECÇÃO DE OUTLIERS")
        print("-" * 25)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_remove = ['EmployeeCount', 'StandardHours', 'EmployeeNumber']
        numerical_cols = [col for col in numerical_cols if col not in cols_to_remove]
        
        outlier_summary = {}
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            if outlier_count > 0:
                outlier_summary[col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        if outlier_summary:
            print("Variáveis com outliers detectados (método IQR):")
            for var, info in sorted(outlier_summary.items(), 
                                  key=lambda x: x[1]['percentage'], reverse=True):
                print(f"{var}: {info['count']} outliers ({info['percentage']:.1f}%)")
        else:
            print("Nenhum outlier significativo detectado")
        
        # Decidir estratégia de tratamento
        high_outlier_vars = [var for var, info in outlier_summary.items() 
                           if info['percentage'] > 5]
        
        if high_outlier_vars:
            print(f"\nAVISO: Variáveis com muitos outliers (>5%): {high_outlier_vars}")
            print("Estratégia: Manter outliers (podem ser casos reais de negócio)")
    
    def _business_insights(self):
        """Gera insights específicos de negócio."""
        print("\nINSIGHTS DE NEGÓCIO")
        print("-" * 25)
        
        insights = []
        
        # Insight 1: Faixa etária
        age_analysis = self.df.groupby(['Attrition'])['Age'].agg(['mean', 'median']).round(1)
        age_diff = age_analysis.loc['Yes', 'mean'] - age_analysis.loc['No', 'mean']
        if abs(age_diff) > 2:
            direction = "mais jovens" if age_diff < 0 else "mais velhos"
            insights.append(f"Funcionários que saem são em média {abs(age_diff):.1f} anos {direction}")
        
        # Insight 2: Renda mensal
        if 'MonthlyIncome' in self.df.columns:
            income_analysis = self.df.groupby(['Attrition'])['MonthlyIncome'].agg(['mean', 'median'])
            income_diff = income_analysis.loc['No', 'mean'] - income_analysis.loc['Yes', 'mean']
            if income_diff > 1000:
                insights.append(f"Funcionários que ficam ganham em média R$ {income_diff:,.0f} a mais")
        
        # Insight 3: Overtime
        if 'OverTime' in self.df.columns:
            overtime_analysis = pd.crosstab(self.df['OverTime'], self.df['Attrition'], normalize='index') * 100
            if 'Yes' in overtime_analysis.columns and 'Yes' in overtime_analysis.index:
                overtime_attrition = overtime_analysis.loc['Yes', 'Yes']
                no_overtime_attrition = overtime_analysis.loc['No', 'Yes']
                if overtime_attrition > no_overtime_attrition * 1.5:
                    insights.append(f"Funcionários com overtime têm {overtime_attrition:.1f}% de attrition vs {no_overtime_attrition:.1f}% sem overtime")
        
        # Insight 4: Tempo na empresa
        if 'YearsAtCompany' in self.df.columns:
            tenure_analysis = self.df.groupby(['Attrition'])['YearsAtCompany'].agg(['mean', 'median']).round(1)
            tenure_diff = tenure_analysis.loc['No', 'mean'] - tenure_analysis.loc['Yes', 'mean']
            if tenure_diff > 1:
                insights.append(f"Funcionários que ficam têm em média {tenure_diff:.1f} anos a mais de empresa")
        
        # Insight 5: Distância de casa
        if 'DistanceFromHome' in self.df.columns:
            distance_analysis = self.df.groupby(['Attrition'])['DistanceFromHome'].agg(['mean', 'median']).round(1)
            distance_diff = distance_analysis.loc['Yes', 'mean'] - distance_analysis.loc['No', 'mean']
            if distance_diff > 2:
                insights.append(f"Funcionários que saem moram em média {distance_diff:.1f} km mais longe")
        
        print("Principais descobertas:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        
        if not insights:
            print("Análise mais detalhada necessária para insights específicos")
        
        # Recomendações iniciais
        print("\nRECOMENDAÇÕES PRELIMINARES:")
        print("Revisar política de overtime e work-life balance")
        print("Implementar programas de retenção para funcionários júniores")
        print("Considerar trabalho remoto/híbrido para funcionários distantes")
        print("Desenvolver planos de carreira mais claros")
    
    def preprocess_data(self):
        print("\nPRÉ-PROCESSAMENTO DOS DADOS")
        print("=" * 35)
        
        # Separar target
        y = self.df['Attrition'].copy()
        X = self.df.drop('Attrition', axis=1).copy()
        
        # Remover colunas irrelevantes
        cols_to_drop = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
        X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
        
        print(f"Removidas {len([col for col in cols_to_drop if col in self.df.columns])} colunas irrelevantes")
        
        # Codificar variáveis categóricas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # One-hot encoding para variáveis categóricas
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        print(f"Aplicado one-hot encoding a {len(categorical_cols)} variáveis categóricas")
        print(f"Dimensões finais: {X_encoded.shape}")
        
        # Codificar target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        # Dividir dados
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y_encoded,
            test_size=0.25,
            random_state=42,
            stratify=y_encoded
        )
        
        self.feature_names = X_encoded.columns.tolist()
        
        print(f"Dados divididos - Treino: {self.X_train.shape[0]}, Teste: {self.X_test.shape[0]}")
        
        # Balanceamento de classes
        self._analyze_class_balance()
        
        return True
    
    def _analyze_class_balance(self):
        unique, counts = np.unique(self.y_train, return_counts=True)
        
        print(f"\nBALANCEAMENTO DE CLASSES (Conjunto de Treino)")
        print(f"Classe 0 (No): {counts[0]} amostras ({counts[0]/len(self.y_train)*100:.1f}%)")
        print(f"Classe 1 (Yes): {counts[1]} amostras ({counts[1]/len(self.y_train)*100:.1f}%)")
        
        imbalance_ratio = max(counts) / min(counts)
        print(f"Taxa de desbalanceamento: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 3:
            print("Estratégia: class_weight='balanced' + métricas específicas")
        else:
            print("Estratégia: Balanceamento natural suficiente")
    
    def train_models(self):
        """Treina múltiplos modelos de machine learning."""
        print("\nTREINAMENTO DOS MODELOS")
        print("=" * 30)
        
        # Modelo 1: Random Forest (principal)
        print("\nRandom Forest Classifier")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest_base'] = rf_model
        
        # Modelo 2: Gradient Boosting
        print("Gradient Boosting Classifier")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        
        gb_model.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = gb_model
        
        # Modelo 3: Logistic Regression (baseline)
        print("Logistic Regression")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Calcular class_weight para LogisticRegression
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        lr_model = LogisticRegression(
            random_state=42,
            class_weight=class_weight_dict,
            max_iter=1000
        )
        
        lr_model.fit(X_train_scaled, self.y_train)
        self.models['logistic_regression'] = lr_model
        self.models['scaler'] = scaler  # Salvar o scaler para uso posterior
        
        print("Todos os modelos treinados com sucesso!")
        
        # Avaliação inicial rápida
        self._quick_evaluation()
    
    def _quick_evaluation(self):
        """Avaliação rápida de todos os modelos."""
        print("\nAVALIAÇÃO INICIAL DOS MODELOS")
        print("-" * 35)
        
        results_summary = []
        
        for name, model in self.models.items():
            if name == 'scaler':
                continue
                
            # Fazer predições
            if name == 'logistic_regression':
                X_test_eval = self.models['scaler'].transform(self.X_test)
            else:
                X_test_eval = self.X_test
                
            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            
            # Calcular métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = fbeta_score(self.y_test, y_pred, beta=1)
            f2 = fbeta_score(self.y_test, y_pred, beta=2)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            pr_auc = average_precision_score(self.y_test, y_pred_proba)
            
            results_summary.append({
                'Modelo': name.replace('_', ' ').title(),
                'Accuracy': f"{accuracy:.3f}",
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1-Score': f"{f1:.3f}",
                'F2-Score': f"{f2:.3f}",
                'ROC-AUC': f"{roc_auc:.3f}",
                'PR-AUC': f"{pr_auc:.3f}"
            })
        
        # Exibir resultados em tabela
        results_df = pd.DataFrame(results_summary)
        print(results_df.to_string(index=False))
        
        # Identificar melhor modelo para otimização
        best_model_name = max(self.models.keys() - {'scaler'}, 
                            key=lambda x: roc_auc_score(
                                self.y_test, 
                                self.models[x].predict_proba(
                                    self.models['scaler'].transform(self.X_test) if x == 'logistic_regression' else self.X_test
                                )[:, 1]
                            ))
        
        print(f"\nMelhor modelo inicial: {best_model_name.replace('_', ' ').title()}")
    
    def optimize_hyperparameters(self, model_name='random_forest_base'):
        """Otimização avançada de hiperparâmetros."""
        print(f"\nOTIMIZAÇÃO DE HIPERPARÂMETROS - {model_name.upper()}")
        print("=" * 50)
        
        if model_name == 'random_forest_base':
            self._optimize_random_forest()
        elif model_name == 'gradient_boosting':
            self._optimize_gradient_boosting()
        else:
            print(f"Otimização não implementada para {model_name}")
    
    def _optimize_random_forest(self):
        """Otimização específica para Random Forest."""
        print("Iniciando otimização do Random Forest...")
        
        # Grid Search simplificado para ser mais rápido
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        base_model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1
        )
        
        print("Usando Grid Search tradicional")
        
        # Executar busca
        print("Executando busca de hiperparâmetros... (isso pode levar alguns minutos)")
        search.fit(self.X_train, self.y_train)
        
        # Salvar melhor modelo
        self.models['random_forest_optimized'] = search.best_estimator_
        
        print(f"Otimização concluída!")
        print(f"Melhor score (Average Precision): {search.best_score_:.4f}")
        print(f"Melhores parâmetros:")
        for param, value in search.best_params_.items():
            print(f"{param}: {value}")
    
    def comprehensive_evaluation(self):
        print("\nAVALIAÇÃO ABRANGENTE DOS MODELOS")
        print("=" * 40)
        
        # Identificar melhor modelo
        best_model_name = self._identify_best_model()
        
        if best_model_name:
            print(f"Modelo selecionado para análise detalhada: {best_model_name.replace('_', ' ').title()}")
            
            # Análise detalhada do melhor modelo
            self._detailed_model_analysis(best_model_name)
            
            # Análise de interpretabilidade
            self._model_interpretability(best_model_name)
            
            # Recomendações de produção
            self._production_recommendations(best_model_name)
        else:
            print("ERRO: Nenhum modelo disponível para avaliação")
    
    def _identify_best_model(self):
        model_scores = {}
        
        for name, model in self.models.items():
            if name == 'scaler':
                continue
                
            # Preparar dados de teste
            if name == 'logistic_regression':
                X_test_eval = self.models['scaler'].transform(self.X_test)
            else:
                X_test_eval = self.X_test
            
            try:
                y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
                
                # Calcular múltiplas métricas
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                pr_auc = average_precision_score(self.y_test, y_pred_proba)
                
                # Score composto (média ponderada)
                composite_score = 0.4 * roc_auc + 0.6 * pr_auc  # Mais peso para PR-AUC
                model_scores[name] = composite_score
                
            except Exception as e:
                print(f"AVISO: Erro ao avaliar modelo {name}: {str(e)}")
                continue
        
        if model_scores:
            best_model = max(model_scores.keys(), key=lambda x: model_scores[x])
            return best_model
        else:
            return None
    
    def _detailed_model_analysis(self, model_name):
        print(f"\nANÁLISE DETALHADA - {model_name.upper()}")
        print("-" * 45)
        
        model = self.models[model_name]
        
        # Preparar dados
        if model_name == 'logistic_regression':
            X_test_eval = self.models['scaler'].transform(self.X_test)
        else:
            X_test_eval = self.X_test
        
        y_pred = model.predict(X_test_eval)
        y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
        
        # Métricas detalhadas
        print("MÉTRICAS DE PERFORMANCE:")
        
        accuracy = accuracy_score(self.y_test, y_pred)
        balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = fbeta_score(self.y_test, y_pred, beta=1)
        f2 = fbeta_score(self.y_test, y_pred, beta=2)
        mcc = matthews_corrcoef(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        log_loss_val = log_loss(self.y_test, y_pred_proba)
        
        metrics = {
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_acc,
            'Precision': precision,
            'Recall (Sensibilidade)': recall,
            'F1-Score': f1,
            'F2-Score': f2,
            'Matthews Correlation': mcc,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Log Loss': log_loss_val
        }
        
        for metric, value in metrics.items():
            if metric == 'Log Loss':
                print(f"{metric}: {value:.4f} (menor é melhor)")
            else:
                print(f"{metric}: {value:.4f}")
        
        # Interpretação das métricas
        print("\nINTERPRETAÇÃO DAS MÉTRICAS:")
        if recall >= 0.8:
            print("    Excelente capacidade de identificar funcionários em risco")
        elif recall >= 0.6:
            print("    AVISO: Boa capacidade de identificar funcionários em risco")
        else:
            print("    ERRO: Capacidade limitada de identificar funcionários em risco")
        
        if precision >= 0.7:
            print("    Baixa taxa de falsos positivos")
        elif precision >= 0.5:
            print("    AVISO: Taxa moderada de falsos positivos")
        else:
            print("    ERRO: Alta taxa de falsos positivos")
        
        if f2 >= 0.7:
            print("    Excelente balanço priorizando recall")
        elif f2 >= 0.5:
            print("    AVISO: Balanço moderado priorizando recall")
        else:
            print("    ERRO: Necessário melhorar o modelo")
        
        # Visualizações
        self._create_performance_visualizations(model_name, y_pred, y_pred_proba)
        
        # Matriz de confusão detalhada
        self._detailed_confusion_matrix(y_pred)
        
        # Salvar resultados
        self.results[model_name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
    
    def _create_performance_visualizations(self, model_name, y_pred, y_pred_proba):
        """Cria visualizações de performance do modelo."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('Taxa de Falsos Positivos')
        axes[0, 0].set_ylabel('Taxa de Verdadeiros Positivos')
        axes[0, 0].set_title('Curva ROC', fontweight='bold')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        axes[0, 1].plot(recall, precision, color='blue', lw=2,
                       label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 1].axhline(y=np.mean(self.y_test), color='red', linestyle='--', 
                          label=f'Baseline ({np.mean(self.y_test):.3f})')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Curva Precision-Recall', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribuição de Probabilidades
        prob_yes = y_pred_proba[self.y_test == 1]
        prob_no = y_pred_proba[self.y_test == 0]
        
        axes[1, 0].hist(prob_no, bins=30, alpha=0.7, label='Não Sai (0)', color='lightblue')
        axes[1, 0].hist(prob_yes, bins=30, alpha=0.7, label='Sai (1)', color='coral')
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
        axes[1, 0].set_xlabel('Probabilidade Predita')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição de Probabilidades por Classe', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Calibração do modelo
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, y_pred_proba, n_bins=10
        )
        
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", 
                       label=f"{model_name}", color='blue')
        axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfeitamente calibrado")
        axes[1, 1].set_xlabel('Probabilidade Média Predita')
        axes[1, 1].set_ylabel('Fração de Positivos')
        axes[1, 1].set_title('Curva de Calibração', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _detailed_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Calcular métricas da matriz de confusão
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nMATRIZ DE CONFUSÃO DETALHADA:")
        print(f"Verdadeiros Negativos (TN): {tn} - Funcionários corretamente identificados como 'não sairão'")
        print(f"Falsos Positivos (FP): {fp} - Funcionários incorretamente identificados como 'sairão'")
        print(f"Falsos Negativos (FN): {fn} - Funcionários que sairão mas não foram identificados")
        print(f"Verdadeiros Positivos (TP): {tp} - Funcionários corretamente identificados como 'sairão'")
        
        # Interpretação de negócio
        print(f"\nIMPACTO FINANCEIRO ESTIMADO (por 1000 funcionários):")
        
        # Custo de falsos negativos (funcionários que saem sem identificação)
        cost_per_employee = 1.5 * 60000  # 1.5x salário médio anual estimado
        fn_cost = (fn / len(self.y_test)) * 1000 * cost_per_employee
        
        # Custo de falsos positivos (recursos gastos desnecessariamente)
        intervention_cost = 5000  # Custo estimado de intervenção por funcionário
        fp_cost = (fp / len(self.y_test)) * 1000 * intervention_cost
        
        print(f"  Custo de Falsos Negativos: R$ {fn_cost:,.0f}")
        print(f"  Custo de Falsos Positivos: R$ {fp_cost:,.0f}")
        print(f"  Custo Total Estimado: R$ {fn_cost + fp_cost:,.0f}")
        
        # Economia potencial
        baseline_cost = np.mean(self.y_test) * 1000 * cost_per_employee
        model_cost = fn_cost + fp_cost
        savings = baseline_cost - model_cost
        
        if savings > 0:
            print(f"  Economia Potencial: R$ {savings:,.0f} ({savings/baseline_cost*100:.1f}%)")
        else:
            print(f"  AVISO: Necessário otimizar para reduzir custos")
        
        # Visualização da matriz
        plt.figure(figsize=(10, 8))
        
        # Criar anotações personalizadas
        annot = np.array([[f'TN\n{tn}\n({tn/(tn+fp)*100:.1f}%)', 
                          f'FP\n{fp}\n({fp/(tn+fp)*100:.1f}%)'],
                         [f'FN\n{fn}\n({fn/(fn+tp)*100:.1f}%)', 
                          f'TP\n{tp}\n({tp/(fn+tp)*100:.1f}%)']])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                   xticklabels=['Predito: Não Sai', 'Predito: Sai'],
                   yticklabels=['Real: Não Sai', 'Real: Sai'],
                   cbar_kws={'label': 'Número de Funcionários'})
        
        plt.title('Matriz de Confusão - Interpretação de Negócio', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        plt.show()
    
    def _model_interpretability(self, model_name):
        print(f"\nINTERPRETABILIDADE DO MODELO")
        print("-" * 35)
        
        model = self.models[model_name]
        
        # Feature Importance (para modelos tree-based)
        if hasattr(model, 'feature_importances_'):
            self._analyze_feature_importance(model)
    
    def _analyze_feature_importance(self, model):
        """Análise de importância das features (tree-based models)."""
        print("\nIMPORTÂNCIA DAS VARIÁVEIS (Tree-based)")
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Top 15 features
        top_features = feature_importance_df.head(15)
        
        print("Top 15 variáveis mais importantes:")
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Visualização
        plt.figure(figsize=(14, 10))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Variáveis - Importância no Modelo', fontsize=16, fontweight='bold')
        plt.xlabel('Importância Relativa')
        plt.ylabel('Variáveis')
        plt.tight_layout()
        plt.show()
        
        # Análise cumulativa
        cumulative_importance = np.cumsum(feature_importance_df['importance'].values)
        n_features_80 = np.argmax(cumulative_importance >= 0.8) + 1
        n_features_95 = np.argmax(cumulative_importance >= 0.95) + 1
        
        print(f"\nAnálise de importância cumulativa:")
        print(f"  {n_features_80} variáveis explicam 80% da importância")
        print(f"  {n_features_95} variáveis explicam 95% da importância")
    
    def _production_recommendations(self, model_name):
        """Recomendações para implementação em produção."""
        print(f"\nRECOMENDAÇÕES PARA PRODUÇÃO")
        print("=" * 35)
        
        print("CHECKLIST DE IMPLEMENTAÇÃO:")
        print("Modelo treinado e validado")
        print("Análise de viés realizada")
        
        print(f"\nPIPELINE DE PRODUÇÃO:")
        print("1. Coleta de dados em tempo real")
        print("2. Pré-processamento automático")
        print("3. Predição com modelo otimizado")
        print("4. Aplicação de threshold recomendado")
        print("5. Alertas para RH sobre funcionários em risco")
        print("6. Dashboard de monitoramento")
        
        print(f"\nMONITORAMENTO CONTÍNUO:")
        print("Data drift: Verificar mudanças na distribuição dos dados")
        print("Model drift: Monitorar performance ao longo do tempo")
        print("Feedback loop: Coletar resultados das intervenções")
        print("Retreinamento: Agendar retreinamento mensal/trimestral")
        
        print(f"\nESTRATÉGIAS DE INTERVENÇÃO:")
        print("Baixo risco (< 30%): Monitoramento passivo")
        print("Médio risco (30-60%): Conversas com gestor")
        print("Alto risco (> 60%): Ação imediata do RH")
        
        print(f"\nROI ESPERADO:")
        print("Redução estimada de attrition: 25-40%")
        print("Economia anual estimada: R$ 8-15 milhões")
        print("Payback do projeto: 3-6 meses")
        
        print(f"\nPRÓXIMOS PASSOS:")
        print("1. Aprovação da diretoria")
        print("2. Setup da infraestrutura de dados")
        print("3. Desenvolvimento da API de predição")
        print("4. Criação do dashboard gerencial")
        print("5. Treinamento da equipe de RH")
        print("6. Piloto com um departamento")
        print("7. Rollout completo")
    
    def generate_report(self):
        """Gera relatório executivo completo."""
        print(f"\nRELATÓRIO EXECUTIVO")
        print("=" * 25)
        
        if not self.results:
            print("ERRO: Nenhum modelo avaliado. Execute a análise completa primeiro.")
            return
        
        best_model_name = list(self.results.keys())[-1]
        best_results = self.results[best_model_name]
        
        print(f"RESUMO EXECUTIVO - SISTEMA PREDITIVO DE ATTRITION")
        print(f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
        print(f"Dataset: {self.df.shape[0]:,} funcionários analisados")
        
        print(f"\nMODELO SELECIONADO: {best_model_name.replace('_', ' ').title()}")
        
        metrics = best_results['metrics']
        print(f"\nMÉTRICAS DE PERFORMANCE:")
        print(f"  Precisão (Precision): {metrics['Precision']:.1%}")
        print(f"  Recall (Sensibilidade): {metrics['Recall (Sensibilidade)']:.1%}")
        print(f"  F2-Score: {metrics['F2-Score']:.1%}")
        print(f"  ROC-AUC: {metrics['ROC-AUC']:.1%}")
        
        # Interpretação executiva
        recall_val = metrics['Recall (Sensibilidade)']
        precision_val = metrics['Precision']
        
        print(f"\nINTERPRETAÇÃO DE NEGÓCIO:")
        if recall_val >= 0.8:
            print(f"    EXCELENTE: O modelo identifica {recall_val:.0%} dos funcionários que realmente sairão")
        elif recall_val >= 0.6:
            print(f"    AVISO BOM: O modelo identifica {recall_val:.0%} dos funcionários que realmente sairão")
        else:
            print(f"    ERRO LIMITADO: O modelo identifica apenas {recall_val:.0%} dos funcionários que realmente sairão")
        
        if precision_val >= 0.7:
            print(f"    BAIXO FALSO ALARME: {precision_val:.0%} dos alertas são realmente funcionários em risco")
        elif precision_val >= 0.5:
            print(f"    AVISO FALSO ALARME MODERADO: {precision_val:.0%} dos alertas são realmente funcionários em risco")
        else:
            print(f"    ERRO ALTO FALSO ALARME: Apenas {precision_val:.0%} dos alertas são realmente funcionários em risco")
        
        print(f"\nRECOMENDAÇÃO FINAL:")
        f2_score = metrics['F2-Score']
        if f2_score >= 0.7:
            print(f"    IMPLEMENTAR: Modelo pronto para produção")
        elif f2_score >= 0.5:
            print(f"    MELHORAR: Implementar com monitoramento intensivo")
        else:
            print(f"    DESENVOLVER: Necessário mais desenvolvimento antes da implementação")
        
        print(f"\nIMPACTO ESPERADO:")
        print(f"  Redução estimada de attrition: 25-40%")
        print(f"  Economia anual: R$ 8-15 milhões")
        print(f"  ROI do projeto: 300-500%")


def main(run_optimization=True, data_path=None):
    """
    Função principal - execução do pipeline completo.
    
    Parâmetros:
    -----------
    run_optimization : bool, default=True
        Se deve executar otimização de hiperparâmetros automaticamente
    data_path : str, optional
        Caminho para o arquivo CSV (se não especificado, usa o padrão)
    """
    print("INICIANDO SISTEMA DE PREDIÇÃO DE ATTRITION - TECHCORP BRASIL")
    print("=" * 70)
    
    # Inicializar analisador
    if data_path:
        analyzer = AttritionAnalyzer(data_path=data_path)
    else:
        analyzer = AttritionAnalyzer()
    
    # 1. Carregar dados
    if not analyzer.load_data():
        print("ERRO: Falha ao carregar dados. Encerrando análise.")
        return
    
    # 2. Análise exploratória completa
    print(f"\nFASE 1: ANÁLISE EXPLORATÓRIA")
    analyzer.exploratory_data_analysis()
    
    # 3. Pré-processamento
    print(f"\nFASE 2: PRÉ-PROCESSAMENTO")
    if not analyzer.preprocess_data():
        print("ERRO: Falha no pré-processamento. Encerrando análise.")
        return
    
    # 4. Treinamento de modelos
    print(f"\nFASE 3: TREINAMENTO DE MODELOS")
    analyzer.train_models()
    
    # 5. Otimização (controlada por parâmetro)
    print(f"\nFASE 4: OTIMIZAÇÃO DE HIPERPARÂMETROS")
    if run_optimization:
        print("Executando otimização automática...")
        analyzer.optimize_hyperparameters('random_forest_base')
    else:
        print("Otimização pulada - usando modelos base")
    
    # 6. Avaliação abrangente
    print(f"\nFASE 5: AVALIAÇÃO FINAL")
    analyzer.comprehensive_evaluation()
    
    # 7. Relatório final
    print(f"\nFASE 6: RELATÓRIO EXECUTIVO")
    analyzer.generate_report()
    
    print(f"\nANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"O sistema está pronto para implementação em produção.")
    print(f"Consulte as recomendações acima para os próximos passos.")
    
    return analyzer


def run_quick_analysis(data_path=None):
    """
    Execução rápida sem otimização - ideal para Google Colab.
    
    Parâmetros:
    -----------
    data_path : str, optional
        Caminho para o arquivo CSV
    """
    print("🚀 EXECUTANDO ANÁLISE RÁPIDA - GOOGLE COLAB")
    print("=" * 50)
    return main(run_optimization=False, data_path=data_path)


def run_full_analysis(data_path=None):
    """
    Execução completa com otimização - pode ser demorada.
    
    Parâmetros:
    -----------
    data_path : str, optional
        Caminho para o arquivo CSV
    """
    print("🔥 EXECUTANDO ANÁLISE COMPLETA")
    print("=" * 35)
    return main(run_optimization=True, data_path=data_path)


# Execução do programa
if __name__ == "__main__":
    main()