import json
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import webbrowser

class QuestionClusterer:
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 n_neighbors: int = 15,
                 min_cluster_size: int = 5,
                 min_samples: int = 3):
        """
        Initialize the question clusterer.
        
        Args:
            embedding_model: Name of the sentence transformer model
            n_neighbors: Number of neighbors for UMAP
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.embeddings = None
        self.reduced_embeddings = None
        self.clusters = None
        self.clusterer = None
        
    def load_questions(self, json_path: str) -> List[str]:
        """Load questions from JSON file."""
        # Convert to absolute path if relative
        if not os.path.isabs(json_path):
            json_path = os.path.abspath(json_path)
            
        print(f"Loading questions from: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['questions']
    
    def create_embeddings(self, questions: List[str]) -> np.ndarray:
        """Create BERT embeddings for questions."""
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(questions, show_progress_bar=True)
        return self.embeddings
    
    def reduce_dimensions(self) -> np.ndarray:
        """Reduce dimensions using UMAP."""
        print("Reducing dimensions...")
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        return self.reduced_embeddings
    
    def cluster_questions(self) -> np.ndarray:
        """Cluster questions using HDBSCAN."""
        print("Clustering questions...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        self.clusters = self.clusterer.fit_predict(self.reduced_embeddings)
        return self.clusters
    
    def visualize_clusters(self, questions: List[str], output_path: str = None):
        """Visualize clusters using plotly."""
        print("Visualizing clusters...")
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'cluster': self.clusters,
            'question': questions
        })
        
        # Create interactive plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['question'],
            title='Question Clusters',
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
        )
        
        if output_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            print(f"Plot saved to {output_path}")
        
        return fig
    
    def analyze_clusters(self, questions: List[str]) -> Dict:
        """Analyze cluster characteristics."""
        print("Analyzing clusters...")
        
        # Calculate silhouette score
        if len(np.unique(self.clusters)) > 1:
            silhouette = silhouette_score(self.reduced_embeddings, self.clusters)
        else:
            silhouette = -1
            
        # Get cluster statistics
        cluster_stats = {}
        for cluster_id in np.unique(self.clusters):
            if cluster_id != -1:  # Skip noise points
                cluster_questions = [q for q, c in zip(questions, self.clusters) if c == cluster_id]
                cluster_stats[cluster_id] = {
                    'size': len(cluster_questions),
                    'sample_questions': cluster_questions[:3]  # Show first 3 questions
                }
        
        return {
            'silhouette_score': silhouette,
            'n_clusters': len(np.unique(self.clusters)) - 1,  # Exclude noise cluster
            'cluster_stats': cluster_stats
        }
    
    def create_cluster_table(self, questions: List[str], output_path: str = None) -> go.Figure:
        """Create a detailed table visualization of clusters and their questions."""
        print("Creating cluster table visualization...")
        
        # Create DataFrame with cluster information
        data = []
        for cluster_id in np.unique(self.clusters):
            if cluster_id != -1:  # Skip noise points
                cluster_questions = [q for q, c in zip(questions, self.clusters) if c == cluster_id]
                for i, question in enumerate(cluster_questions):
                    data.append({
                        'Cluster': f'Cluster {cluster_id}',
                        'Question Number': i + 1,
                        'Question': question,
                        'Cluster Size': len(cluster_questions)
                    })
        
        df = pd.DataFrame(data)
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Cluster', 'Question Number', 'Question', 'Cluster Size'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[df['Cluster'], df['Question Number'], df['Question'], df['Cluster Size']],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        # Update layout
        fig.update_layout(
            title='Question Clusters - Detailed View',
            title_x=0.5,
            margin=dict(l=20, r=20, t=50, b=20),
            height=800
        )
        
        if output_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            print(f"Table visualization saved to {output_path}")
        
        return fig
    
    def run_pipeline(self, 
                    json_path: str, 
                    output_path: str = None,
                    table_output_path: str = None) -> Tuple[np.ndarray, Dict]:
        """Run the complete clustering pipeline."""
        # Load and process questions
        questions = self.load_questions(json_path)
        self.create_embeddings(questions)
        self.reduce_dimensions()
        self.cluster_questions()
        
        # Visualize and analyze
        fig = self.visualize_clusters(questions, output_path)
        table_fig = self.create_cluster_table(questions, table_output_path)
        analysis = self.analyze_clusters(questions)
        
        return self.clusters, analysis

    def open_visualizations(self, scatter_path: str, table_path: str):
        """Open the visualization files in the default web browser."""
        print("\nOpening visualizations in your default web browser...")
        
        # Convert paths to absolute paths
        scatter_path = os.path.abspath(scatter_path)
        table_path = os.path.abspath(table_path)
        
        # Open scatter plot
        print(f"Opening scatter plot: {scatter_path}")
        webbrowser.open(f'file://{scatter_path}')
        
# Example usage
if __name__ == "__main__":
    # Initialize clusterer
    clusterer = QuestionClusterer(
        embedding_model='all-MiniLM-L6-v2',
        n_neighbors=15,
        min_cluster_size=5,
        min_samples=3
    )
    
    # Get the absolute path to the JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'tokenized_questions.json')
    output_path = os.path.join(current_dir, '..', '..', 'results', 'cluster_visualization.html')
    
    # Run pipeline
    clusters, analysis = clusterer.run_pipeline(
        json_path=json_path,
        output_path=output_path,
    )
    
    # Print analysis
    print("\nCluster Analysis:")
    print(f"Number of clusters: {analysis['n_clusters']}")
    print(f"Silhouette score: {analysis['silhouette_score']:.3f}")
    print("\nCluster Statistics:")
    for cluster_id, stats in analysis['cluster_stats'].items():
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {stats['size']}")
        print("Sample questions:")
        for q in stats['sample_questions']:
            print(f"- {q[:100]}...")
    
    # Open visualizations
    clusterer.open_visualizations(output_path) 