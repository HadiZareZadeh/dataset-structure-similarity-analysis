"""
Reference implementation of similarity-aware split function (NON-OPTIMIZED VERSION).

WARNING: This is the original slow implementation that computes a full O(n²) pairwise 
distance matrix. It is kept here for reference only.

DO NOT USE THIS VERSION - it is extremely slow for large datasets (>5000 samples).
Use similarity_split_optimized.py instead, which is 100-1000x faster.

Performance Issues:
1. Computes full pairwise distance matrix: O(n²) memory and time complexity
2. Nested loops in assignment: O(n²) time complexity
3. For 32,561 samples, this creates a 32,561 × 32,561 matrix (~1 billion elements)

Use similarity_aware_split_optimized() from similarity_split_optimized.py instead.
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def similarity_aware_split(X, y, test_size=0.2, similarity_threshold=None):
    """
    Split dataset by grouping similar samples together.
    
    WARNING: This is the SLOW, non-optimized version. 
    Use similarity_aware_split_optimized() from similarity_split_optimized.py instead.
    
    This version has severe performance issues:
    - Computes full O(n²) pairwise distance matrix
    - Uses nested loops for assignment
    - Extremely slow for datasets > 5000 samples
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target vector
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    similarity_threshold : float, optional
        Distance threshold for similarity (auto-computed if None)
    
    Returns:
    --------
    train_indices : array
        Indices of training samples
    test_indices : array
        Indices of test samples
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Compute pairwise distances - THIS IS THE BOTTLENECK!
    # For 32,561 samples, this creates a 32,561 × 32,561 matrix
    distances = euclidean_distances(X)
    
    # Initialize
    assigned = np.zeros(n_samples, dtype=bool)
    test_indices = []
    
    # Start with a random sample
    current_idx = np.random.randint(0, n_samples)
    test_indices.append(current_idx)
    assigned[current_idx] = True
    
    # Find similar samples
    if similarity_threshold is None:
        similarity_threshold = np.percentile(distances[distances > 0], 20)
    
    # Nested loops - ANOTHER BOTTLENECK!
    # O(n²) complexity where n grows as test samples are added
    while len(test_indices) < n_test:
        # Find unassigned samples similar to current test set
        unassigned = np.where(~assigned)[0]
        if len(unassigned) == 0:
            break
        
        # Find closest unassigned sample to any test sample
        min_dist = float('inf')
        next_idx = None
        
        # Nested loops - very slow!
        for test_idx in test_indices:
            for unassigned_idx in unassigned:
                dist = distances[test_idx, unassigned_idx]
                if dist < min_dist:
                    min_dist = dist
                    next_idx = unassigned_idx
        
        if next_idx is not None:
            test_indices.append(next_idx)
            assigned[next_idx] = True
        else:
            # If no similar sample found, pick random
            next_idx = np.random.choice(unassigned)
            test_indices.append(next_idx)
            assigned[next_idx] = True
    
    train_indices = np.where(~assigned)[0]
    
    return train_indices, np.array(test_indices)
