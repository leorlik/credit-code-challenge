# Code Challenge 

## 1. Apresentação

Este projeto visa um problema de predizer o target da variável y de um dataset mascarado.

## 2. Estrutura do projeto:

- **artifacts**: 

Contém artefatos gerados durante o projeto que não são gráficos ou modelos, como scalers, encoders e outros objetos.

- **data**:

O diretório data contém as camadas SOR, SOT e SPEC do projeto, em que cada camada representa um estado dos dados consumidos e/ou produzidos durante o projeto. 

-   *SOR*:
    Camada que contém cópias brutas e sem tratamento dos dados originais do desafio.
-   *SOT*:
    Camada que contém dados com tratamento inicial, como tratamento de missing, bem como seleção básica de colunas.
-   *SPEC*:
    Camada que contém dados trabalhados, com previsão e probabilidade, bem como as colunas utilizadas no modelo.

- **environment**:

Diretório que contém arquivo yml que representa o ambiente virtual utilizado, para fins de reprodutibilidade com versionamento do python. Este arquivo foi gerado pelo Anaconda.

Para reproduzir o ambiente virtual, rode no diretório environment:

```bash
conda env create -f environment.yml
```

- **graficos**:

Diretório que contem os gráficos gerados durante o projeto. Estes gráficos são utilizados no README.md

- **models**:

Diretório para guardar os arquivos que representam artefatos dos modelos gerados e guardados.

- **notebooks**:

Diretório que contém os Jupyter Notebooks utilizados durante o projeto. Úteis para seguir o raciocínio de quem fez o projeto.

- **src**:

Diretório que contém o código fonte de classes, bibliotecas e exemplares de modelo em produção.

## 3. Premissas:

- Arquivos Parquet: A preferência por arquivos do tipo parquet para guardar os dados é devido também a eficiência, pensando possivelmente em particionar dentro de um bucket as safras e escalabilidade. Porém, há ainda outro motivo pela escolha dos arquivos parquet: eles se demonstram responsivos. Relatos de arquivos ".pkl" e ".csv" em máquinas diferentes, com ênfase em sistemas operacionais diversos (embora tenha preferência por sistemas Linux, muito desse projeto foi desenvolvido em sistema Windows por falta de tempo de instalar Linux no meu notebook) são muito mais comuns do que experiências desse tipo com arquivos parquet.

Outra opção seria armazenar estes dados em um banco de dados simples, como postgrees, porém a abordagem aqui também reflete simplicidade. Se quiser conhecer um pouco mais de como lido com banco de dados, recomendo [este projeto](https://github.com/leorlik/king-county-houses).

- Problema: Devido à natureza da [base de dados](https://github.com/leorlik/credit-code-challenge/blob/main/data/SOR/base_modelo.csv), e y (variável target) ser uma variável com dois possíveis valores (0 e 1), este problema foi tratado como um problema de classificação binária. 

- Variáveis: Por todas as variáveis serem números, as variáveis foram tratadas como numéricas. Por mais que seja possível que algumas variáveis sejam, na verdade, categóricas ordinais, nenhum indício foi encontrado nas 78 variáveis do dataset.

## 4. Análise inicial:

### Escolha de conjuntos de treino, teste e validação

O dataframe inicial é dividido em safras que refletem o mês e ano de coleta dos dados. Devido à natureza temporal dos dados, e a divergência não tão evidente entre a quantidade de dados, conforme o gráfico abaixo, a penúltima safra foi escolhida para teste e a última como validação. 

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/safras.png)

Esta escolha é devido a dados sazonais terem uma possível identidade. Um exemplo de como a coleta dos dados pode influenciar um modelo está disponível neste meu [outro projeto](https://github.com/leorlik/mfccnn), em que o locutor influenciava no acerto do modelo.

### Seleção inicial de variáveis

O notebook referência nesta etapa é [este](https://github.com/leorlik/credit-code-challenge/blob/main/notebooks/tratamento_de_base_e_correlacao.ipynb).

#### Valores faltantes

No dataframe, 19 colunas possuíam mais de 60% dos valores faltantes. Portanto, foram eliminadas, restando 59 colunas além da variável target.

#### Correlação entre variáveis

Enquanto a correlação com a variável resposta não foi esclarecedora, muitas variáveis possuem alta correlação entre si, conforme a matriz de correlação abaixo.

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/matriz_alta_correlacao.png)

Devido à uma correlação absoluta acima de 0.8 (embora 0.7 já seria alta, porém serei conservador), as variáveis que mais apresentavam correlação com outras foram eliminadas. Os pares de variáveis 57 e 60, 25 e 28, 45 e 13 e 42 e 46 possuíam alta correlação somente entre si, assim como porcentagem semelhante de dados faltantes. A tabela abaixo sintetiza a variância e a porcentagem de valores faltantes destes dados:

| Variável | var         | % de NAs |
|----------|-------------|----------|
| VAR_57   | 163.812942  | 0.0%     |
| VAR_60   | 0.111383    | 0.0%     |
| VAR_25   | 14.762769   | 0.5%     |
| VAR_28   | 20.196434   | 0.5%     |
| VAR_45   | 92095.586633| 48.3%    |
| VAR_13   | 5884.994715 | 48.3%    |
| VAR_42   | 614.861688  | 54.5%    |
| VAR_46   | 953.636707  | 53.5%    |

Devido à semelhante falta de dados, a maior variância entre o par foi mantida, por possívelmente conter mais informações.   

#### Transformações logarítmicas

A tabela da sessão anterior já mostra alta variância principalmente nas variáveis 13 e 45. Embora alguns modelos não sejam sensíveis a esta alta variância quando normalizados, como árvores e florestas, outros modelos lineares como regressão logística e vetores de suporte são. A tabela abaixo ilustra a variância e o deslocamento das distribuições, em que o deslocamento positivo indica deslocamento para a direita, enquanto o negativo indica deslocamento à esquerda.

| Variável | Deslocamento da distribuição | Variância
|--------|----------|----------------|
| VAR_6  | 63.89    | 42,433,191.72  |
| VAR_8  | 14.85    | 4,737.69       |
| VAR_46 | 7.08     | 953.64         |
| VAR_11 | 6.00     | 3,866.81       |
| VAR_67 | -5.30    | 234,959.71     |
| VAR_3  | 4.99     | 0.73           |
| VAR_37 | -4.85    | 526.00         |
| VAR_2  | 4.56     | 1.63           |
| VAR_31 | 4.34     | 666.73         |
| VAR_76 | 4.12     | 324,673.31     |
| VAR_74 | 3.59     | 2.50           |
| VAR_72 | 3.50     | 91,870.33      |
| VAR_30 | 3.50     | 2,087,365.45   |
| VAR_33 | 3.47     | 32.59          |
| VAR_4  | 3.29     | 1.51           |
| VAR_17 | 3.26     | 4,761,217.84   |
| VAR_34 | 3.24     | 83,230.32      |
| VAR_77 | 3.07     | 10,124.69      |
| VAR_66 | 2.90     | 229.44         |
| VAR_7  | 2.89     | 3,115.08       |
| VAR_53 | 2.84     | 5,545,212.73   |
| VAR_65 | 2.70     | 2,025,585.44   |
| VAR_19 | 2.70     | 254.44         |
| VAR_15 | 2.66     | 2,893.88       |
| VAR_54 | 2.61     | 2,650,152.14   |
| VAR_35 | 2.56     | 68,954.97      |
| VAR_52 | 2.56     | 52,454.22      |
| VAR_5  | 2.46     | 5,866.73       |
| VAR_38 | 2.38     | 57,465.99      |
| VAR_24 | 2.37     | 90,290.65      |
| VAR_45 | 2.26     | 92,095.59      |
| VAR_51 | 2.10     | 45,681.28      |
| VAR_1  | 1.72     | 2,292.29       |
| VAR_59 | 1.68     | 65,057.02      |
| VAR_20 | -1.59    | 7.30           |
| VAR_32 | -1.39    | 0.00           |
| VAR_28 | 1.17     | 20.20          |
| VAR_14 | 1.04     | 156,884.62     |
| VAR_9  | 0.96     | 433,893.55     |
| VAR_27 | 0.32     | 66,166.88      |
| VAR_57 | 0.10     | 163.81         |

Devido a esta alta despadronização, uma versão do dataframe foi criado em que as variàveis com variância acima de 1500 e deslocamento acima de 1.5 tiveram transformação logaritmica aplicada.

#### Preenchimento de variáveis faltantes:

A mediana foi escolhida devida a presença considerável de outliers em algumas colunas, para evitar deslocar a mediana em direção a média e, por consequência, acentuar o impacto de outliers. Porém, para preservar a identidade de cada safra, cada safra teve sua mediana retirada e colocada como preenchimento dos valores faltantes.

#### Distribuição da variável resposta:

O gráfico abaixo indica a distribuição da variável resposta:

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/proporcao_y.png)

O problema é desbalanceado, e as safras de treinamento seguem de forma geral a divisão de 72/28% do DataFrame como um todo com dispersões não tão altas. Dito isso, é seguro indicar que é um problema moderadamente desbalanceado, e, portanto, técnicas como SMOTE são uma opção para balancear o problema, assim como usar pesos nas métricas objetivo.

#### Análise de variáveis

Observando as variáveis mais correlacionadas (ainda sim que uma correlação abaixo de 0.4), é possível ver claras diferenças entre as distribuições para 0 e 1, exceto a variável 32, que possui uma distribuição semelhante.

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/boxplot_variaveis_y_dif.png)

Já quando observamos as variáveis 11, 35, e 52, podemos ver que sua distribuição normalizada não só é muito parecida entre 0 e 1, como extremamente parecidas entre si mesmas, conforme os boxplots abaixo:

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/boxplot_variaveis_y_igual.png)

#### Tratamento de outliers

Por ser um problema extremamente multivariado, o tratamento de outliers pode ser complexo. Para isto, foi utilizada a distância de Malahanobis para ver as amostras que distoavam muito, e 270 linhas foram excluídas.

## 5. Modelagem:

### Escolha do Modelo

A Regressão Logística foi escolhida para o modelo pois é simples e obteve uma das melhores performances, prezando pela explicabilidade, além de ser rápida de treinar.

### Normalização

Os dados foram normalizados utilizando Z-Score devido à sensibilidade da escala e boas práticas. Com média 0 e desvio padrão 1, nenhum dado domina outro.

### PCA

Foi utilizado o PCA para redução de componentes. O principal motivo, além de reduzir a complexidade, foi que aumentou levemente as métricas.

### Balanceamento de classes

Para mitigação do problema de desbalanceamento, o ADASYN (Adaptive Synthetic Sampling) foi utilizado. O ADASYN adapta a geração de exemplos com base na dificuldade da classificação de pontos próximos a fronteira de decisão, dando ênfase para o modelo aprender mais com a classe menos presente.

### Parâmetros

A escolha dos parâmetros foi feita utilizando GridSearch, em que o relevante é a regularização inversa (C) igual a 0.1 e a regularização l2, esta última suspeita, pois este problema se apresentou, nos testes, supeitos ao overfitting.

### Código

O código da classe do modelo se encontra [aqui](https://github.com/leorlik/credit-code-challenge/blob/main/src/modeloClassificacao.py), enquanto a validação e execução estão [neste notebook](https://github.com/leorlik/credit-code-challenge/blob/main/notebooks/prototipo.ipynb).
 
# 6. Resultados

## Teste

Fazer o modelo enxergar a classe desbalanceada foi um desafio, pois a decisão parecia deveras complexa para quaisquer modelo. No fim, a regressão obteve os seguintes resultados no teste:

| Classe | Precisão | Recall | F1-Score | Suporte |
|--------|----------|--------|----------|---------|
| 0      | 0.80     | 0.74   | 0.77     | 539     |
| 1      | 0.55     | 0.63   | 0.59     | 269     |
| **Acurácia** | -        | -      | 0.70     | 808     |
| **Macro avg** | 0.67     | 0.69   | 0.68     | 808     |
| **Weighted avg** | 0.72     | 0.70   | 0.71     | 808     |

E abaixo a matriz de confusão:

|               | Predito 0 | Predito 1 |
|---------------|-----------|-----------|
| **Real 0**    | 399       | 140       |
| **Real 1**    | 99        | 170       |

Prever a classe minoritária se mostrou difícil para todos os modelos testados, sendo este o modelo que melhor equilibra a previsão das classes. Naturalmente, este é o tipo de trade-off que deve ser validado com o negócio para priorizar a assertividade em uma classe, se necessário. 

Enquanto na classe 0 a precisão e o recall (0.8 e 0.74), com f1-score de 0.77, o modelo indica bom desempenho na classe majoritária. Já na classe 1, a precisão foi de 0.55 e o recall de 0.63. Com um f1-score de 0.59, ainda existe dificuldade na previsão da classe minoritária. Como o f1-score ponderado é 0.71 e a acurácia geral (esta influenciada pela classe maior) é de 70%, mesmo com desequilibrio nas classes, se o objetivo do modelo, como foi aqui, é ser o mais justo possível com as duas classes, ele alcança desempenho razoável.

## Validação

Em validação, os seguintes resultados se apresentaram:

| Classe | Precisão | Recall | F1-Score | Suporte |
|--------|----------|--------|----------|---------|
| 0      | 0.76     | 0.68   | 0.72     | 509     |
| 1      | 0.50     | 0.60   | 0.55     | 277     |
| **Acurácia** | -        | -      | 0.65     | 786     |
| **Macro avg** | 0.63     | 0.64   | 0.63     | 786     |
| **Weighted avg** | 0.67     | 0.65   | 0.66     | 786     |

Com a seguinte matriz de confusão:

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/confusion_matrix.png)

Enquanto na classe 0, a precisão e o recall foram de 0.76 e 0.68, com f1-score de 0.72, o modelo indica bom desempenho na classe majoritária. Já na classe 1, a precisão foi de 0.50 e o recall de 0.60. Com um f1-score de 0.55, ainda existe dificuldade na previsão da classe minoritária. Como o f1-score ponderado é 0.66 e a acurácia geral (influenciada pela classe maior) é de 65%, o modelo se demonstrou piorar um pouco no conjunto de validação, como se esperar, mas não o suficiente para indicar overfitting grotesto.

Abaixo a curva de evolução das métricas:

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/linha_metricas.png)

O recall decai com o aumento de threshold, como esperado nos conjuntos desbalanceados. O F1 tem um pico intermediário próximo do padrão de 0.5, o que indica que o ajuste de threshold não é necessário neste caso. O trade-off entre precisão e recall fica bem claro no gráfico.

Já a curva ROC:

![](https://github.com/leorlik/credit-code-challenge/blob/main/graficos/curva_roc.png)

Indica proximidade com a linha aleatória, mostrando dificuldades no modelo de distinguir as classes, havendo espaço para melhora, porém considerando a dificuldade em separar certas amostras, o modelo proposto é um bom ponto de partida.