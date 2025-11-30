import yaml
import numpy as np
import pandas as pd
from sklearn import datasets
import sys
import os


def load_config(config_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    if not os.path.exists(config_path):
        print(f"Erro Crítico: O arquivo '{config_path}' não foi encontrado.")
        print("Certifique-se de estar executando o script no mesmo diretório do arquivo config.yaml.")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Erro ao ler o arquivo YAML: {e}")
        sys.exit(1)


def load_data(cfg):
    """Carrega os dados baseado na configuração (sklearn ou csv)."""
    source = cfg['data'].get('source_type', 'sklearn')
    
    if source == 'sklearn':
        return _load_sklearn_data(cfg)
    elif source == 'csv':
        return _load_csv_data(cfg)
    else:
        print(f"Erro: source_type '{source}' desconhecido. Use 'sklearn' ou 'csv'.")
        sys.exit(1)


def _load_sklearn_data(cfg):
    """Carrega datasets internos do Scikit-Learn."""
    name = cfg['data']['dataset_name'].lower()
    print(f"--- Carregando dataset interno: {name} ---")
    
    loaders = {
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'iris': datasets.load_iris,
        'digits': datasets.load_digits
    }
    
    if name not in loaders:
        print(f"Erro: Dataset '{name}' não suportado. Tente: {list(loaders.keys())}")
        sys.exit(1)
        
    data = loaders[name]()
    print(f"Dataset carregado. Features: {data.feature_names}")
    return data.data, list(data.feature_names)


def _load_csv_data(cfg):
    """Carrega dados de um arquivo CSV externo."""
    try:
        path = cfg['data']['filepath']
        print(f"--- Carregando dados de {path} ---")
        df = pd.read_csv(path)
        numeric_df = df.select_dtypes(include=[np.number])
        X = numeric_df.values
        feature_names = list(numeric_df.columns)
        print(f"Dataset carregado. Shape: {X.shape}")
        return X, feature_names
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        sys.exit(1)
