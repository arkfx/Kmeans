import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self, style='seaborn-v0_8-darkgrid', palette='viridis'):
        """
        Inicializa o visualizador com configurações estéticas.
        Não confie no estilo padrão do matplotlib, ele é feio.
        """
        plt.style.use(style)
        self.palette = palette

    def plot_silhouette_evaluation(self, k_range, silhouette_scores):
        """
        Plota o Coeficiente de Silhueta para avaliação do melhor k.
        """
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(x=list(k_range), y=silhouette_scores, marker='o', color='green')
        
        # Destacar o melhor score
        best_k_idx = np.argmax(silhouette_scores)
        best_k = list(k_range)[best_k_idx]
        best_score = silhouette_scores[best_k_idx]
        
        plt.axvline(x=best_k, linestyle='--', color='red', alpha=0.7, label=f'Melhor k={best_k}')
        plt.scatter(best_k, best_score, color='red', s=100, zorder=5)
        
        plt.title('Coeficiente de Silhueta (Quanto maior, melhor)')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Score Médio de Silhueta')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        print("Feche a janela do gráfico para continuar...")
        plt.show()

    def plot_final_clusters(self, X, labels, centers, k, feature_names=None):
        """
        Plota a dispersão dos dados com os clusters finais definidos.
        """
        plt.figure(figsize=(10, 8))
        
        # Plot dos pontos de dados
        sns.scatterplot(
            x=X[:, 0], 
            y=X[:, 1], 
            hue=labels, 
            palette=self.palette, 
            s=60, 
            alpha=0.7,
            legend='full'
        )
        
        # Plot dos centróides (o "X" da questão)
        plt.scatter(
            centers[:, 0], 
            centers[:, 1], 
            c='red', 
            s=200, 
            alpha=0.9, 
            marker='X', 
            label='Centróides'
        )
        
        plt.title(f'Resultado Final do Clustering K-Means (k={k})')
        
        # Usar nomes das features se disponíveis
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
        
        plt.legend()
        
        print("Exibindo clusters finais (2D)...")
        plt.show()

    def plot_final_clusters_3d(self, X, labels, centers, k, feature_names=None):
        """
        Plota a dispersão 3D dos dados com os clusters finais definidos.
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Obter cores do palette
        colors = sns.color_palette(self.palette, n_colors=k)
        
        # Plot dos pontos de dados
        for cluster_id in range(k):
            mask = labels == cluster_id
            ax.scatter(
                X[mask, 0], 
                X[mask, 1], 
                X[mask, 2],
                c=[colors[cluster_id]], 
                s=60, 
                alpha=0.7,
                label=f'Cluster {cluster_id}'
            )
        
        # Plot dos centróides
        ax.scatter(
            centers[:, 0], 
            centers[:, 1], 
            centers[:, 2],
            c='red', 
            s=200, 
            alpha=0.9, 
            marker='X', 
            label='Centróides'
        )
        
        ax.set_title(f'Resultado Final do Clustering K-Means 3D (k={k})')
        
        # Usar nomes das features se disponíveis
        if feature_names and len(feature_names) >= 3:
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_zlabel(feature_names[2])
        else:
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
        
        ax.legend()
        
        print("Exibindo clusters finais (3D)...")
        plt.show()