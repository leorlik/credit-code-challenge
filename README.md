# Code Challenge 

## 1. Apresentação

## 2. Estrutura do projeto:

- ** artifacts**: 

Contém artefatos gerados durante o projeto que não são gráficos ou modelos, como scalers, encoders e outros objetos.

- ** data**:

O diretório data contém as camadas SOR, SOT e SPEC do projeto, em que cada camada representa um estado dos dados consumidos e/ou produzidos durante o projeto. 
    - *SOR*:

    Camada que contém cópias brutas e sem tratamento dos dados originais do desafio.

    - *SOT*:

    Camada que contém dados com tratamento inicial, como tratamento de missing, bem como seleção básica de colunas.
    
    - *SPEC*:
    
    Camada que contém dados trabalhados, com previsão e probabilidade, bem como as colunas utilizadas no modelo.

- ** environment**:

Diretório que contém arquivo yml que representa o ambiente virtual utilizado, para fins de reprodutibilidade com versionamento do python. Este arquivo foi gerado pelo Anaconda.

Para reproduzir o ambiente virtual, rode no diretório:

```bash
conda env create -f environment.yml
```

- ** ***

## 3. Premissas