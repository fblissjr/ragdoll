# src/utils/visualization_utils.py
# Utilities for generating visualization data, primarily UMAP projections.

import os
import json
import numpy as np

# Project-specific config for default filenames, UMAP params
from .. import ragdoll_config # If utils is a sub-package of src
# from ragdoll_config import VISUALIZATION_DATA_FILENAME, DEFAULT_UMAP_NEIGHBORS etc. if src is directly in PYTHONPATH

# UMAP library for dimensionality reduction
try: 
    import umap
    UMAP_AVAILABLE = True
except ImportError: 
    UMAP_AVAILABLE = False
    print("Visualization Utils: Warning - UMAP library not available. Visualization data generation will be skipped.")


def save_data_for_visualization( 
    vector_data_dir: str, 
    chunk_vectors: np.ndarray, 
    metadata_list: list[dict], 
    chunk_texts_list: list[str],
    output_viz_filename: str = ragdoll_config.VISUALIZATION_DATA_FILENAME, 
    umap_n_neighbors: int = ragdoll_config.DEFAULT_UMAP_NEIGHBORS, 
    umap_min_dist_val: float = ragdoll_config.DEFAULT_UMAP_MIN_DIST, 
    umap_metric_val: str = ragdoll_config.DEFAULT_UMAP_METRIC
):
    """
    Generates 2D UMAP embeddings from high-dimensional chunk vectors for visualization 
    and saves them to a JSON file.
    """
    if not UMAP_AVAILABLE: 
        print("Visualization Utils: UMAP library not found. Skipping visualization data generation."); return
    
    # Validate input data consistency: number of vectors, metadata entries, and text snippets must match.
    num_vectors = chunk_vectors.shape[0]
    if not (num_vectors == len(metadata_list) == len(chunk_texts_list)): 
        print(f"Visualization Utils Error: Mismatch in lengths for visualization data. "
              f"Vectors: {num_vectors}, Metadata: {len(metadata_list)}, Texts: {len(chunk_texts_list)}. "
              "Skipping visualization data generation.")
        return
    if num_vectors == 0: 
        print("Visualization Utils: No vector data available for visualization. Skipping."); return
    
    print(f"\nVisualization Utils: Generating UMAP for {num_vectors} points (Neighbors: {umap_n_neighbors}, MinDist: {umap_min_dist_val}, Metric: {umap_metric_val})...")
    try:
        # Adjust UMAP's n_neighbors parameter for small datasets to prevent errors.
        # UMAP requires n_neighbors to be less than the number of samples.
        actual_n_neighbors = umap_n_neighbors
        if num_vectors <= umap_n_neighbors : 
             # Set to num_samples - 1, but ensure it's at least 2 if num_vectors > 1.
             actual_n_neighbors = max(2, num_vectors - 1) if num_vectors > 2 else (2 if num_vectors == 2 else 1)
             print(f"  UMAP Info: Dataset size ({num_vectors}) is small. Adjusting n_neighbors from {umap_n_neighbors} to {actual_n_neighbors}.")
        
        if num_vectors == 1: # UMAP cannot run on a single sample.
            print("  UMAP Info: Only 1 data point found. UMAP cannot generate a 2D projection. Skipping visualization.")
            return

        # Initialize UMAP reducer with specified parameters.
        reducer = umap.UMAP(
            n_neighbors=actual_n_neighbors, 
            min_dist=umap_min_dist_val, 
            n_components=2, # Fixed to 2D for scatter plot visualization
            metric=umap_metric_val, 
            random_state=42, # For reproducibility of UMAP layout
            verbose=False    # Set to True for detailed UMAP logs during fitting
        ) 
        # Fit UMAP and transform the high-dimensional chunk vectors to 2D.
        embedding_2d = reducer.fit_transform(chunk_vectors)
        
        # Prepare data points for JSON serialization, to be used by the frontend.
        viz_data_points_list = []
        for i, (metadata_item, chunk_text_item) in enumerate(zip(metadata_list, chunk_texts_list)):
            viz_data_points_list.append({
                'id': metadata_item.get('vicinity_item_id', f'chunk_fallback_id_{i}'), # Unique ID for the point
                'x': float(embedding_2d[i,0]), # X-coordinate from UMAP
                'y': float(embedding_2d[i,1]), # Y-coordinate from UMAP
                'source_file': metadata_item.get('source_file', 'N/A'), 
                'display_source': metadata_item.get('display_source_name', 'N/A'), # User-friendly source name
                'classification': metadata_item.get('top_label', 'N/A'), # Top classification label
                'snippet': chunk_text_item[:200] + "..." if len(chunk_text_item) > 200 else chunk_text_item # Text snippet
            })
            
        output_file_full_path = os.path.join(vector_data_dir, output_viz_filename)
        with open(output_file_full_path, 'w', encoding='utf-8') as f: 
            json.dump(viz_data_points_list, f, indent=2) # Save with indentation for better human readability
        print(f"Visualization data with {len(viz_data_points_list)} points saved successfully to: {output_file_full_path}")
    except Exception as e_umap: # Catch any errors during UMAP processing or file saving
        print(f"Visualization Utils Error: Exception during UMAP generation or saving data: {e_umap}")