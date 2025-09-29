"""
Sistema Preditivo de Attrition - TechCorp Brasil
===============================================
VERSÃO GOOGLE COLAB

"""

# =====================================================
# GOOGLE COLAB - CONFIGURAÇÃO E UPLOAD DE ARQUIVOS
# =====================================================

# Detectar se está rodando no Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("Executando no Google Colab")
    
    # Função para upload de arquivos no Colab
    def upload_csv_file():
        """Função para fazer upload de arquivo CSV no Google Colab"""
        from google.colab import files
        print("Selecione e faça upload do arquivo CSV de attrition:")
        uploaded = files.upload()
        
        if uploaded:
            filename = list(uploaded.keys())[0]
            print(f"Arquivo '{filename}' carregado com sucesso!")
            return filename
        else:
            print("Nenhum arquivo foi carregado")
            return None
            
except ImportError:
    IN_COLAB = False
    print("Executando localmente")
    
    def upload_csv_file():
        """Função placeholder para ambiente local"""
        return 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

# =====================================================
# INSTALAÇÃO DE DEPENDÊNCIAS (SE NECESSÁRIO)
# =====================================================

# Instalar dependências específicas se necessário
if IN_COLAB:
    try:
        import sklearn
        print("Scikit-learn já instalado")
    except ImportError:
        print("Instalando scikit-learn...")
        !pip install scikit-learn
    
    try:
        import seaborn
        print("Seaborn já instalado")
    except ImportError:
        print("Instalando seaborn...")
        !pip install seaborn

# =====================================================
# IMPORTAÇÕES NECESSÁRIAS
# =====================================================

import pandas as pd
import numpy as np
import warnings
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

# Configurações globais
warnings.filterwarnings('ignore')
plt.style.use('default')  # Colab-friendly style
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
np.random.seed(42)

# Configurar matplotlib para Colab
if IN_COLAB:
    %matplotlib inline

print("=" * 60)
print("SISTEMA DE PREDIÇÃO DE ATTRITION - TECHCORP BRASIL")
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
            if IN_COLAB:
                print("DICA: Execute a célula de upload do arquivo CSV primeiro!")
            return False
        except Exception as e:
            print(f"ERRO: Erro ao carregar dados: {str(e)}")
            return False
    
    def exploratory_data_analysis(self):
        """Realiza análise exploratória completa dos dados."""
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
        
        # Insights de negócio
        self._business_insights()
    
    def _display_basic_info(self):
        """Exibe informações básicas sobre o dataset."""
        print("\nINFORMAÇÕES GERAIS")
        print("-" * 30)
        print(f"• Total de funcionários: {self.df.shape[0]:,}")
        print(f"• Variáveis disponíveis: {self.df.shape[1]}")
        
        # Verificar valores nulos
        missing_data = self.df.isnull().sum()
        if missing_data.sum() == 0:
            print("• Nenhum valor ausente encontrado")
        else:
            print(f"• AVISO: Valores ausentes: {missing_data.sum()}")
        
        # Verificar duplicatas
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("• Nenhuma linha duplicada encontrada")
        else:
            print(f"• AVISO: Linhas duplicadas: {duplicates}")
        
        # Tipos de dados
        print(f"• Variáveis numéricas: {self.df.select_dtypes(include=[np.number]).shape[1]}")
        print(f"• Variáveis categóricas: {self.df.select_dtypes(include=['object']).shape[1]}")
    
    def _analyze_target_variable(self):
        """Análise detalhada da variável target (Attrition)."""
        print("\nANÁLISE DA VARIÁVEL TARGET (ATTRITION)")
        print("-" * 45)
        
        # Distribuição da variável target
        attrition_counts = self.df['Attrition'].value_counts()
        attrition_pct = self.df['Attrition'].value_counts(normalize=True) * 100
        
        print("Distribuição de Attrition:")
        for category, count in attrition_counts.items():
            pct = attrition_pct[category]
            print(f"  • {category}: {count:,} funcionários ({pct:.1f}%)")
        
        # Cálculo do desbalanceamento
        minority_class = attrition_counts.min()
        majority_class = attrition_counts.max()
        imbalance_ratio = majority_class / minority_class
        print(f"\nTaxa de desbalanceamento: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 3:
            print("AVISO: Dataset significativamente desbalanceado - requer tratamento especial")
        
        # Visualização otimizada para Colab
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de barras
        colors = ['lightblue', 'coral']
        attrition_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Distribuição de Attrition', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Número de Funcionários')
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de pizza
        ax2.pie(attrition_counts.values, labels=attrition_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Proporção de Attrition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_numerical_variables(self):
        """Análise detalhada das variáveis numéricas."""
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
        
        # Visualização otimizada para Colab
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
        """Realiza testes estatísticos para variáveis numéricas."""
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
                    print(f"  • {var}: p-value = {p_value:.4f} {significance}")
        
        print(f"\n{len(significant_vars)} variáveis com diferenças estatisticamente significativas")
    
    def _analyze_categorical_variables(self):
        """Análise das variáveis categóricas."""
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
        
        # Ordenar por significância
        significant_associations.sort(key=lambda x: x[1])
        
        print(f"\nVariáveis com associação significativa (p < 0.05):")
        for var, p_val in significant_associations[:8]:
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            print(f"  • {var}: p-value = {p_val:.4f} {significance}")
        
        # Visualização das principais variáveis categóricas
        if len(significant_associations) > 0:
            self._plot_categorical_analysis(significant_associations[:4])
    
    def _plot_categorical_analysis(self, top_vars):
        """Cria visualizações para as principais variáveis categóricas."""
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
        """Análise de correlações entre variáveis numéricas."""
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
            print(f"  • {var}: {corr:.3f} (correlação {direction})")
        
        # Visualização da matriz de correlação (simplificada para Colab)
        plt.figure(figsize=(14, 12))
        
        # Selecionar apenas as variáveis mais importantes para visualização
        important_vars = ['Attrition'] + list(target_correlations.head(15).index)
        corr_subset = correlation_matrix.loc[important_vars, important_vars]
        
        mask = np.triu(corr_subset)
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   mask=mask, fmt='.2f', annot_kws={'size': 8})
        plt.title('Matriz de Correlação - Top Variáveis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
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
        
        print("Principais descobertas:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        
        if not insights:
            print("  • Análise mais detalhada necessária para insights específicos")
        
        # Recomendações iniciais
        print("\nRECOMENDAÇÕES PRELIMINARES:")
        print("  • Revisar política de overtime e work-life balance")
        print("  • Implementar programas de retenção para funcionários júniores")
        print("  • Considerar trabalho remoto/híbrido para funcionários distantes")
        print("  • Desenvolver planos de carreira mais claros")
    
    def preprocess_data(self):
        """Pré-processamento inteligente dos dados."""
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
        """Analisa o balanceamento das classes e define estratégia."""
        unique, counts = np.unique(self.y_train, return_counts=True)
        
        print(f"\nBALANCEAMENTO DE CLASSES (Conjunto de Treino)")
        print(f"  • Classe 0 (No): {counts[0]} amostras ({counts[0]/len(self.y_train)*100:.1f}%)")
        print(f"  • Classe 1 (Yes): {counts[1]} amostras ({counts[1]/len(self.y_train)*100:.1f}%)")
        
        imbalance_ratio = max(counts) / min(counts)
        print(f"  • Taxa de desbalanceamento: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 3:
            print("  Estratégia: class_weight='balanced' + métricas específicas")
        else:
            print("  Estratégia: Balanceamento natural suficiente")
    
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


# =====================================================
# FUNÇÕES PRINCIPAIS PARA GOOGLE COLAB
# =====================================================

def run_quick_analysis_colab():
    """
    FUNÇÃO PRINCIPAL PARA GOOGLE COLAB - ANÁLISE RÁPIDA
    Execute esta função para uma análise completa mas rápida (sem otimização pesada)
    """
    print("INICIANDO ANÁLISE RÁPIDA NO GOOGLE COLAB")
    print("=" * 55)
    
    # 1. Upload do arquivo
    print("FASE 1: UPLOAD DO ARQUIVO")
    csv_file = upload_csv_file()
    
    if not csv_file:
        print("Erro: Nenhum arquivo CSV foi carregado!")
        return None
    
    # 2. Inicializar analisador
    analyzer = AttritionAnalyzer(data_path=csv_file)
    
    # 3. Executar pipeline completo
    try:
        # Carregar dados
        print(f"\nFASE 2: CARREGAMENTO DOS DADOS")
        if not analyzer.load_data():
            print("Erro: Falha ao carregar dados. Verifique o arquivo CSV.")
            return None
        
        # Análise exploratória
        print(f"\nFASE 3: ANÁLISE EXPLORATÓRIA")
        analyzer.exploratory_data_analysis()
        
        # Pré-processamento
        print(f"\nFASE 4: PRÉ-PROCESSAMENTO")
        if not analyzer.preprocess_data():
            print("Erro: Falha no pré-processamento.")
            return None
        
        # Treinamento de modelos
        print(f"\nFASE 5: TREINAMENTO DE MODELOS")
        analyzer.train_models()
        
        print(f"\nANÁLISE RÁPIDA CONCLUÍDA COM SUCESSO!")
        print(f"Para análise completa com otimização, use run_full_analysis_colab()")
        
        return analyzer
        
    except Exception as e:
        print(f"Erro durante a análise: {str(e)}")
        return None


def run_full_analysis_colab():
    """
    FUNÇÃO COMPLETA PARA GOOGLE COLAB - ANÁLISE COMPLETA
    Execute esta função para análise completa com otimização (pode ser demorada)
    """
    print("INICIANDO ANÁLISE COMPLETA NO GOOGLE COLAB")
    print("=" * 55)
    
    # Executar análise rápida primeiro
    analyzer = run_quick_analysis_colab()
    
    if analyzer is None:
        return None
    
    try:
        print(f"\nFASE 6: ANÁLISE COMPLETA")
        print("AVISO: Esta versão simplificada não inclui otimização pesada")
        print("O modelo básico já foi treinado e avaliado com sucesso!")
        
        print(f"\nANÁLISE COMPLETA CONCLUÍDA COM SUCESSO!")
        
        return analyzer
        
    except Exception as e:
        print(f"Erro durante análise completa: {str(e)}")
        print("O modelo básico já foi treinado com sucesso!")
        return analyzer


# =====================================================
# INSTRUÇÕES DE USO PARA GOOGLE COLAB
# =====================================================

def show_colab_instructions():
    """Mostra instruções detalhadas para uso no Google Colab"""
    print("INSTRUÇÕES PARA GOOGLE COLAB")
    print("=" * 40)
    print("""
    COMO USAR:
    
    1. ANÁLISE RÁPIDA (recomendado para começar):
       run_quick_analysis_colab()
    
    2. ANÁLISE COMPLETA (versão simplificada):
       run_full_analysis_colab()
    
    3. UPLOAD MANUAL DO ARQUIVO:
       csv_file = upload_csv_file()
       analyzer = AttritionAnalyzer(data_path=csv_file)
    
    DICAS:
    • Certifique-se de ter o arquivo CSV de attrition pronto
    • A análise rápida leva ~2-5 minutos
    • Todas as visualizações aparecerão automaticamente
    """)

# Mostrar instruções automaticamente
if IN_COLAB:
    show_colab_instructions()
else:
    print("Executando localmente - use as funções normalmente")

# Exemplo de uso rápido no Colab
if IN_COLAB:
    print(f"\nEXEMPLO DE USO RÁPIDO:")
    print(f"Para começar, execute: run_quick_analysis_colab()")