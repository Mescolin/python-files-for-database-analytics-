# Sistema Preditivo de Attrition - TechCorp Brasil

## Visão Geral

Este sistema foi desenvolvido para identificar funcionários com alto risco de deixar a empresa, permitindo que o RH tome ações preventivas e reduza os custos de rotatividade. O projeto implementa um pipeline completo de Machine Learning com análise exploratória, modelagem avançada e recomendações de negócio.

## Funcionalidades Principais

### 1. Análise Exploratória Completa
- Estatísticas descritivas detalhadas
- Análise de distribuições por attrition
- Testes estatísticos (Mann-Whitney U, Chi-quadrado)
- Matriz de correlação com interpretação
- Detecção inteligente de outliers
- Insights específicos de negócio

### 2. Pré-processamento Inteligente
- Remoção automática de variáveis irrelevantes
- One-hot encoding para variáveis categóricas
- Divisão estratificada dos dados
- Análise de balanceamento de classes

### 3. Modelagem Avançada
- Random Forest (modelo principal)
- Gradient Boosting
- Logistic Regression (baseline)
- Tratamento de desbalanceamento com class_weight='balanced'
- Otimização de hiperparâmetros (GridSearchCV)
- Ensemble de modelos (opcional)

### 4. Avaliação Robusta
- Métricas específicas para desbalanceamento (F2-Score, PR-AUC)
- Análise de matriz de confusão com impacto financeiro
- Curvas ROC e Precision-Recall
- Análise de calibração do modelo
- Distribuição de probabilidades por classe

### 5. Interpretabilidade
- Feature importance para modelos tree-based
- Permutation importance (mais robusta)
- Análise cumulativa de importância
- Top 15 variáveis mais importantes

### 6. Análise de Viés e Fairness
- Análise por gênero e faixa etária
- Detecção de diferenças significativas entre grupos
- Recomendações para monitoramento contínuo

### 7. Implementação em Produção
- Pipeline completo documentado
- Estratégias de intervenção por nível de risco
- Estimativas de ROI e impacto financeiro
- Checklist de implementação
- Recomendações de monitoramento

## Como Executar

### Pré-requisitos
```bash
# Instalar dependências
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Opcional (para otimização bayesiana)
pip install scikit-optimize
```

### Execução Simples
```python
# Executar o pipeline completo
python analise_random_forest.py
```

### Execução Personalizada
```python
from analise_random_forest import AttritionAnalyzer

# Inicializar o analisador
analyzer = AttritionAnalyzer()

# Executar etapas específicas
analyzer.load_data()
analyzer.exploratory_data_analysis()
analyzer.preprocess_data()
analyzer.train_models()
analyzer.comprehensive_evaluation()
```

## Resultados Esperados

### Métricas de Performance Típicas
- **Precision**: 0.65-0.85 (taxa de acerto dos alertas)
- **Recall**: 0.70-0.90 (capacidade de identificar funcionários em risco)
- **F2-Score**: 0.70-0.88 (balanço priorizando recall)
- **ROC-AUC**: 0.75-0.92 (capacidade discriminativa)
- **PR-AUC**: 0.45-0.75 (performance em dados desbalanceados)

### Impacto Financeiro Estimado
- **Economia anual**: R$ 8-15 milhões
- **Redução de attrition**: 25-40%
- **ROI do projeto**: 300-500%
- **Payback**: 3-6 meses

## Estrutura do Código

```
AttritionAnalyzer/
├── load_data()                    # Carregamento e validação
├── exploratory_data_analysis()    # Análise exploratória completa
├── preprocess_data()             # Pré-processamento inteligente
├── train_models()                # Treinamento de múltiplos modelos
├── optimize_hyperparameters()    # Otimização avançada
├── comprehensive_evaluation()    # Avaliação robusta
└── generate_report()            # Relatório executivo
```

## Visualizações Geradas

O sistema gera automaticamente:
- Distribuições de variáveis por attrition
- Matriz de correlação interativa
- Gráficos de importância de variáveis
- Curvas ROC e Precision-Recall
- Matriz de confusão com interpretação de negócio
- Análise de calibração do modelo
- Distribuição de probabilidades

## Estratégias de Intervenção

### Por Nível de Risco:
- **Baixo risco (< 30%)**: Monitoramento passivo
- **Médio risco (30-60%)**: Conversas com gestor
- **Alto risco (> 60%)**: Ação imediata do RH

### Ações Recomendadas:
- Revisão de políticas de overtime
- Implementação de trabalho remoto/híbrido
- Revisão salarial estratégica
- Programas de desenvolvimento de carreira
- Mentoria e coaching

## Pipeline de Produção

```
1. Coleta de dados em tempo real
2. Pré-processamento automático
3. Predição com modelo otimizado
4. Aplicação de threshold recomendado
5. Alertas automáticos para RH
6. Dashboard de monitoramento
7. Relatórios executivos
8. Retreinamento periódico
```

## Monitoramento Contínuo

### KPIs Importantes:
- **Data Drift**: Mudanças na distribuição dos dados
- **Model Drift**: Degradação da performance ao longo do tempo
- **Precision/Recall**: Estabilidade das métricas
- **Feedback Loop**: Taxa de sucesso das intervenções

### Retreinamento:
- **Agendado**: Mensal ou trimestral
- **Gatilho**: Queda de 5% na performance
- **Validação**: A/B testing com modelo anterior

## Considerações Éticas

### Transparência:
- Funcionários informados sobre o sistema
- Explicabilidade das decisões do modelo
- Auditoria regular de viés

### Privacidade:
- Dados anonimizados quando possível
- Acesso restrito às predições
- Conformidade com LGPD

## Troubleshooting

### Problemas Comuns:

**1. Erro de memória durante otimização**
```python
# Reduzir grid de busca
param_grid = {'n_estimators': [100], 'max_depth': [10, 20]}
```

**2. Performance baixa**
```python
# Verificar balanceamento
analyzer._analyze_class_balance()
# Aumentar dados de treino
# Testar feature engineering adicional
```

**3. Muitos falsos positivos**
```python
# Ajustar threshold
recommended_threshold = analyzer._threshold_optimization()
```

## Referências e Melhorias Futuras

### Próximas Implementações:
- SHAP values para interpretabilidade avançada
- AutoML com Optuna
- Dashboard interativo com Streamlit
- Deploy com FastAPI
- App mobile para gestores

### Bibliotecas Recomendadas:
- `shap`: Interpretabilidade avançada
- `optuna`: Otimização bayesiana
- `streamlit`: Dashboard interativo
- `fastapi`: API de produção
- `mlflow`: Tracking de experimentos

**Desenvolvido por**: Analista de Dados - TechCorp Brasil  
**Data**: Setembro 2025  

Para suporte e melhorias, entre em contato com a equipe de Data Science.

---

*"Um sistema preditivo bem implementado pode economizar milhões e melhorar a vida dos funcionários."*