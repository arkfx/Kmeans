import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_loader import load_config, load_data
from visualizer import Visualizer


def main():
    # 1. Carregar Configurações e Dados
    config = load_config()
    X, feature_names = load_data(config)
    print(f"Dados prontos. Formato: {X.shape}")

    # 2. Inicializar Visualizador
    viz = Visualizer(
        style=config['visualization']['style'], 
        palette=config['visualization']['palette']
    )

    # 3. Parâmetros do modelo
    k_min = config['model']['k_min']
    k_max = config['model']['k_max']
    k_range = range(k_min, k_max + 1)
    
    silhouette_scores = []

    print(f"\nIniciando varredura de k={k_min} até k={k_max}...")

    # 4. Loop de Treinamento e Avaliação
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k, 
            init=config['model']['init'],
            n_init=config['model']['n_init'],
            max_iter=config['model']['max_iter'],
            random_state=config['data']['random_state']
        )
        kmeans.fit(X)
        
        # Calcula o coeficiente de silhueta
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        
        print(f"k={k} | Silhueta={score:.4f}")

    # 5. Calcula o melhor k baseado no coeficiente de silhueta
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_k_idx]
    
    print(f"\n--- Conclusão ---")
    print(f"O melhor k baseado na Silhueta é: {best_k}")
    print(f"Score máximo: {silhouette_scores[best_k_idx]:.4f}")

    # 6. Visualização das Métricas
    viz.plot_silhouette_evaluation(k_range, silhouette_scores)

    # 7. Treinamento Final e Visualização dos Clusters
    print(f"Gerando visualização final para k={best_k}...")
    final_model = KMeans(
        n_clusters=best_k,
        init=config['model']['init'],
        n_init=config['model']['n_init'],
        max_iter=config['model']['max_iter'],
        random_state=config['data']['random_state']
    )
    labels = final_model.fit_predict(X)
    centers = final_model.cluster_centers_
    
    # Visualização 2D (usa as 2 primeiras features)
    viz.plot_final_clusters(X, labels, centers, best_k, feature_names)
    
    # Visualização 3D (usa as 3 primeiras features se disponíveis)
    if X.shape[1] >= 3:
        viz.plot_final_clusters_3d(X, labels, centers, best_k, feature_names)
    else:
        print("Não é possível visualizar em 3D, pois o dataset tem menos de 3 features.")

if __name__ == "__main__":
    main()